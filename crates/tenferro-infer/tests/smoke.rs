//! Phase 0 smoke test: proves the tenferro dependency wiring builds and runs
//! basic CPU operations inside one backend session.
//!
//! These are wiring checks, not primitive tests; the `tenferro-infer`
//! primitives and their numerical tests arrive in Phase 1.

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};
use tenferro_tensor::BackendSessionHost;

#[test]
fn cpu_matmul_runs() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    // Column-major storage: [1,3,2,4] is the matrix [[1,2],[3,4]].
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0])?;

    let c = backend.with_backend_session(|session| a.matmul(&b, session))??;

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);
    Ok(())
}

#[test]
fn cpu_elementwise_and_session_scope() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let x = Tensor::from_vec_col_major(vec![3], vec![0.0_f64, 1.0, 2.0])?;
    let one = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;

    // One session, two ops: broadcast add then elementwise exp.
    let y = backend.with_backend_session(|session| {
        let shifted = x.add(&one, session)?;
        shifted.exp(session)
    })??;

    let values = y.as_slice::<f64>().unwrap();
    assert_eq!(values.len(), 3);
    assert!((values[0] - std::f64::consts::E).abs() < 1e-12);
    assert!((values[1] - std::f64::consts::E.powi(2)).abs() < 1e-12);
    assert!((values[2] - std::f64::consts::E.powi(3)).abs() < 1e-12);
    Ok(())
}
