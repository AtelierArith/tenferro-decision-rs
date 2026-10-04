//! Phase 0/1 smoke test: the tenferro eager runtime builds and runs CPU
//! operations in one session entry.

use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_cpu::CpuBackend;

#[test]
fn eager_session_matmul() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;

    let out = ctx.with_eager_session(|session| {
        let a = session.constant_from(Tensor::from_vec_col_major(
            vec![2, 2],
            vec![1.0_f64, 3.0, 2.0, 4.0],
        )?)?;
        let b = session.constant_from(Tensor::from_vec_col_major(
            vec![2, 2],
            vec![5.0_f64, 7.0, 6.0, 8.0],
        )?)?;
        let c = session.matmul(&a, &b)?;
        session.duplicate_value(&c)
    })??;

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);
    Ok(())
}
