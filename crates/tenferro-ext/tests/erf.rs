//! Numerical tests for the self-hosted `erf` extension op and exact GELU.

use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_ext::EagerSessionErfExt;

fn run<F>(shape: Vec<usize>, data: Vec<f32>, f: F) -> Vec<f32>
where
    F: FnOnce(&mut EagerSession<'_>, &EagerTensor) -> tenferro_ad::Result<EagerTensor> + Send,
{
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let out = runtime
        .with_eager_session(|s| {
            let x = s.constant_from(Tensor::from_vec_col_major(shape, data)?)?;
            f(s, &x)
        })
        .unwrap()
        .unwrap();
    out.value().unwrap().as_slice::<f32>().unwrap().to_vec()
}

#[test]
fn erf_matches_mlx_reference_values() {
    let values = run(
        vec![7],
        vec![-3.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0],
        |s, x| s.erf(x),
    );
    // MLX `erff` (mathfns.jl) values.
    let expected = [
        -0.999_977_9f32,
        -0.842_700_8,
        -0.520_499_9,
        0.0,
        0.520_499_9,
        0.842_700_8,
        0.999_977_9,
    ];
    for (got, want) in values.iter().zip(expected) {
        assert!((got - want).abs() < 1e-6, "erf {got} vs {want}");
    }
}

#[test]
fn gelu_erf_matches_reference_values() {
    let values = run(vec![5], vec![-3.0f32, -1.0, 0.0, 1.0, 2.0], |s, x| {
        s.gelu_erf(x)
    });
    // Exact GELU: `x * Phi(x)`.
    let expected = [-0.004_049_7f32, -0.158_655_3, 0.0, 0.841_344_8, 1.954_499_7];
    for (got, want) in values.iter().zip(expected) {
        assert!((got - want).abs() < 1e-6, "gelu_erf {got} vs {want}");
    }
}

#[test]
fn native_erf_f32_matches_cpu_polynomial_across_branches() {
    let mut input: Vec<f32> = (-20000..=20000).map(|i| i as f32 * 0.0003).collect();
    let boundary = 0.927_734_4f32;
    input.extend([
        f32::from_bits(boundary.to_bits() - 1),
        boundary,
        f32::from_bits(boundary.to_bits() + 1),
        -f32::from_bits(boundary.to_bits() - 1),
        -boundary,
        -f32::from_bits(boundary.to_bits() + 1),
        0.0,
        -0.0,
        1e-30,
        -1e-30,
    ]);
    let actual = run(vec![input.len()], input.clone(), |s, x| {
        tenferro_ext::erf_tensor_native(s, x)
    });
    let mut max_error = 0.0f32;
    for (&x, &got) in input.iter().zip(&actual) {
        let expected = cpu_kernels::erf_f32(x);
        max_error = max_error.max((got - expected).abs());
        assert!(got.is_finite(), "x={x}, got={got}");
    }
    eprintln!("native erf F32 max absolute error: {max_error}");
    assert!(max_error <= 1e-6);
}

#[test]
fn native_erf_preserves_special_values_and_signed_zero() {
    let input = vec![
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
        f32::NAN,
    ];
    let actual = run(vec![input.len()], input, |s, x| {
        tenferro_ext::erf_tensor_native(s, x)
    });
    assert_eq!(actual[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(actual[1].to_bits(), (-0.0f32).to_bits());
    assert_eq!(&actual[2..6], &[1.0, -1.0, 1.0, -1.0]);
    assert!(actual[6].is_nan());
}

#[test]
fn native_erf_f64_matches_cpu_extension() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let input: Vec<f64> = (-1000..=1000).map(|i| i as f64 * 0.006).collect();
    let (native, cpu) = runtime
        .with_eager_session(|s| {
            let x = s.constant_from(Tensor::from_vec_col_major(vec![input.len()], input)?)?;
            Ok::<_, tenferro_ad::Error>((tenferro_ext::erf_tensor_native(s, &x)?, s.erf(&x)?))
        })
        .unwrap()
        .unwrap();
    let native = native.value().unwrap();
    let cpu = cpu.value().unwrap();
    for (&got, &expected) in native
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(cpu.as_slice::<f64>().unwrap())
    {
        assert!(
            (got - expected).abs() <= 5e-16,
            "got={got}, expected={expected}"
        );
    }
}

#[test]
fn native_gelu_matches_cpu_erf_form() {
    let input: Vec<f32> = (-20000..=20000).map(|i| i as f32 * 0.0005).collect();
    let expected = run(vec![input.len()], input.clone(), |s, x| s.gelu_erf(x));
    let actual = run(vec![input.len()], input, |s, x| {
        tenferro_ext::gelu_erf_tensor_native(s, x)
    });
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("native GELU F32 max absolute error: {max_error}");
    assert!(max_error <= 2e-6);
}

#[test]
fn cached_native_gelu_reuses_constants_across_changed_inputs_and_runtime() {
    let runtimes = [EagerRuntime::new().unwrap(), EagerRuntime::new().unwrap()];
    let mut cache = tenferro_infer::TensorCache::new();
    let mut retained = Vec::new();
    let mut count = 0;
    for runtime in &runtimes {
        for shift in [0.0f32, 0.25, -0.5] {
            let (cached, plain) = runtime
                .with_eager_session(|session| {
                    let data = vec![-2.0 + shift, 0.0 + shift, 2.0 + shift, 3.0 + shift];
                    let x = session
                        .constant_from_host(Tensor::from_vec_col_major(vec![2, 2], data)?)?;
                    // Exercise a strided feature-first view as well as changed data.
                    let x = session.transpose(&x, &[1, 0])?;
                    let output =
                        tenferro_ext::gelu_erf_tensor_native_cached(session, &mut cache, &x)?;
                    let plain = tenferro_ext::gelu_erf_tensor_native(session, &x)?;
                    Ok::<_, tenferro_ad::Error>((output, plain))
                })
                .unwrap()
                .unwrap();
            let actual = cached.value().unwrap().as_slice::<f32>().unwrap().to_vec();
            assert_eq!(actual, plain.value().unwrap().as_slice::<f32>().unwrap());
            if count == 0 {
                count = cache.len();
            }
            assert!(count > 0);
            assert_eq!(cache.len(), count);
            retained.push((cached, actual));
            for (old, expected) in &retained {
                assert_eq!(old.value().unwrap().as_slice::<f32>().unwrap(), expected);
            }
        }
    }
}
