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
    let (erf, gelu) = runtime
        .with_eager_session(|session| {
            assert!(!cpu_extensions_supported(session));
            let x = session.constant_from_host(Tensor::from_vec_col_major(
                vec![input.len()],
                input.clone(),
            )?)?;
            let erf = session.erf(&x)?;
            let gelu = session.gelu_erf(&x)?;
            Ok::<_, tenferro_ad::Error>((
                session.duplicate_value(&erf)?,
                session.duplicate_value(&gelu)?,
            ))
        })
        .unwrap()
        .unwrap();
    let erf = download_tensor(backend.runtime(), &erf).unwrap();
    let gelu = download_tensor(backend.runtime(), &gelu).unwrap();
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
