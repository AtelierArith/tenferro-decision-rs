#![cfg(feature = "cuda")]

use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_ext::{EagerSessionErfExt, cpu_extensions_supported};
use tenferro_gpu::cuda::{CudaBackend, CudaDeviceId, download_tensor};

#[test]
#[ignore = "requires CUDA hardware; run explicitly with --ignored"]
fn cuda_erf_and_gelu_use_native_operations() {
    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA device required");
    let runtime = EagerRuntime::with_cuda_backend(backend.clone()).unwrap();
    let input: Vec<f32> = (-2000..=2000).map(|i| i as f32 * 0.005).collect();
    let mut cache = tenferro_infer::TensorCache::new();
    let mut retained = Vec::new();
    let mut constants = 0;
    for scale in [1.0f32, 0.5, -0.75] {
        let changed: Vec<f32> = input.iter().map(|x| x * scale).collect();
        let (erf, gelu, cached_erf, cached_gelu) = runtime
            .with_eager_session(|session| {
                assert!(!cpu_extensions_supported(session));
                let x = session.constant_from_host(Tensor::from_vec_col_major(
                    vec![changed.len()],
                    changed.clone(),
                )?)?;
                let erf = session.erf(&x)?;
                let gelu = session.gelu_erf(&x)?;
                let cached_erf = tenferro_ext::erf_tensor_native_cached(session, &mut cache, &x)?;
                let cached_gelu =
                    tenferro_ext::gelu_erf_tensor_native_cached(session, &mut cache, &x)?;
                Ok::<_, tenferro_ad::Error>((
                    session.duplicate_value(&erf)?,
                    session.duplicate_value(&gelu)?,
                    session.duplicate_value(&cached_erf)?,
                    session.duplicate_value(&cached_gelu)?,
                ))
            })
            .unwrap()
            .unwrap();
        if constants == 0 {
            constants = cache.len();
        }
        assert!(constants > 0);
        assert_eq!(cache.len(), constants);
        retained.push((changed, erf, gelu, cached_erf, cached_gelu));
    }
    // Keep every output alive through later cache reuse. No intermediate download.
    for (input, erf, gelu, cached_erf, cached_gelu) in retained {
        let erf = download_tensor(backend.runtime(), &erf).unwrap();
        let gelu = download_tensor(backend.runtime(), &gelu).unwrap();
        let cached_erf = download_tensor(backend.runtime(), &cached_erf).unwrap();
        let cached_gelu = download_tensor(backend.runtime(), &cached_gelu).unwrap();
        assert_eq!(
            erf.as_slice::<f32>().unwrap(),
            cached_erf.as_slice::<f32>().unwrap()
        );
        assert_eq!(
            gelu.as_slice::<f32>().unwrap(),
            cached_gelu.as_slice::<f32>().unwrap()
        );
        for ((&x, &actual_erf), &actual_gelu) in input
            .iter()
            .zip(erf.as_slice::<f32>().unwrap())
            .zip(gelu.as_slice::<f32>().unwrap())
        {
            let expected_erf = cpu_kernels::erf_f32(x);
            let expected_gelu =
                x * 0.5 * (1.0 + cpu_kernels::erf_f32(x * std::f32::consts::FRAC_1_SQRT_2));
            assert!(
                actual_erf.is_finite() && (actual_erf - expected_erf).abs() <= 2e-6,
                "x={x}, erf={actual_erf}, expected={expected_erf}"
            );
            assert!(
                actual_gelu.is_finite() && (actual_gelu - expected_gelu).abs() <= 4e-6,
                "x={x}, gelu={actual_gelu}, expected={expected_gelu}"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly with --ignored"]
fn cuda_output_boundary_downloads_strided_retained_outputs() {
    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA device required");
    let runtime = EagerRuntime::with_cuda_backend(backend).unwrap();
    let mut retained = Vec::new();
    for factor in [1.0f32, -0.5, 3.0] {
        let output = runtime
            .with_eager_session(|session| {
                let input = session.constant_from_host(Tensor::from_vec_col_major(
                    vec![2, 3],
                    vec![1.0f32, 2., 3., 4., 5., 6.],
                )?)?;
                let factor = session
                    .constant_from_host(Tensor::from_vec_col_major(vec![], vec![factor])?)?;
                let changed = session.mul(&input, &factor)?;
                session.transpose(&changed, &[1, 0])
            })
            .unwrap()
            .unwrap();
        retained.push((factor, output));
    }
    // Transfer only after subsequent requests, preserving all original outputs.
    for (factor, output) in retained {
        runtime
            .with_eager_session(|session| {
                for _ in 0..2 {
                    let host = tenferro_infer::output::host_value(session, &output)?;
                    assert_eq!(host.shape(), &[3, 2]);
                    let expected = [1., 3., 5., 2., 4., 6.].map(|value| value * factor);
                    assert_eq!(host.as_slice::<f32>()?, &expected);
                }
                Ok::<_, tenferro_ad::Error>(())
            })
            .unwrap()
            .unwrap();
    }
}
