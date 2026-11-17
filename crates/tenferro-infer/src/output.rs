//! Explicit output transfer from an admitted eager backend to host storage.

use tenferro_ad::{EagerSession, EagerTensor, Result, Tensor};
use tenferro_tensor::TensorRead;

/// Materialize a tensor in column-major order and download device storage.
///
/// The session validates runtime ownership before copying. Host storage keeps
/// the existing single materialization; device storage is downloaded through
/// its admitted backend. This is a request output boundary, not an implicit
/// transfer in a model operation. Unsupported transfers return backend errors.
pub fn host_value(session: &mut EagerSession<'_>, tensor: &EagerTensor) -> Result<Tensor> {
    let materialized = session.duplicate_value(tensor)?;
    if TensorRead::from_tensor(&materialized)
        .backend_family()
        .is_some()
    {
        Ok(session
            .backend_session()
            .download_to_host(TensorRead::from_tensor(&materialized))?)
    } else {
        Ok(materialized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_ad::EagerRuntime;
    use tenferro_cpu::CpuEngine;

    #[test]
    fn output_materializes_strided_views_without_changing_retained_values() {
        let runtime = EagerRuntime::new().unwrap();
        let mut engine = CpuEngine::serial();
        let (original, transposed) = runtime
            .with_eager_session(&mut engine, |session| {
                let original = session.constant_from_host(Tensor::from_vec_col_major(
                    vec![2, 3],
                    vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
                )?)?;
                let transposed = session.transpose(&original, &[1, 0])?;
                Ok::<_, tenferro_ad::Error>((original, transposed))
            })
            .unwrap();
        runtime
            .with_eager_session(&mut engine, |session| {
                for _ in 0..2 {
                    let output = host_value(session, &transposed)?;
                    assert_eq!(output.shape(), &[3, 2]);
                    assert_eq!(output.as_slice::<f32>()?, &[1., 3., 5., 2., 4., 6.]);
                    assert!(TensorRead::from_tensor(&output).backend_family().is_none());
                }
                let output = host_value(session, &original)?;
                assert_eq!(output.as_slice::<f32>()?, &[1., 2., 3., 4., 5., 6.]);
                Ok::<_, tenferro_ad::Error>(())
            })
            .unwrap();
    }

    #[test]
    fn output_rejects_a_foreign_runtime_before_transfer() {
        let first = EagerRuntime::new().unwrap();
        let second = EagerRuntime::new().unwrap();
        let mut engine = CpuEngine::serial();
        let tensor = first
            .with_eager_session(&mut engine, |session| {
                session.constant_from_host(Tensor::from_vec_col_major(vec![], vec![1.0f32])?)
            })
            .unwrap();
        let error = second
            .with_eager_session(&mut engine, |session| host_value(session, &tensor))
            .unwrap_err();
        assert!(matches!(error, tenferro_ad::Error::ContextMismatch { .. }));
    }
}
