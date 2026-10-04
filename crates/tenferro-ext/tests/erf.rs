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
