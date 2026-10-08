//! Explicit input construction through backend-portable eager operations.

use tenferro_ad::{CompareDir, EagerSession, EagerTensor, Result, Tensor};

/// Upload a host mask as numeric values and construct Bool on the backend.
///
/// The pinned CUDA backend can compare F32 values to produce Bool, but its
/// eager leaf materialization does not support an uploaded Bool tensor. No
/// Bool leaf import or host fallback is used here. Unsupported comparison or
/// upload operations propagate their typed backend error.
pub fn bool_tensor_native(
    session: &mut EagerSession<'_>,
    shape: Vec<usize>,
    values: &[bool],
) -> Result<EagerTensor> {
    let encoded: Vec<f32> = values
        .iter()
        .map(|value| if *value { 1. } else { 0. })
        .collect();
    let numeric = session.constant_from_host(Tensor::from_vec_col_major(shape, encoded)?)?;
    let zero = session.constant_from_host(Tensor::from_vec_col_major(vec![], vec![0.0f32])?)?;
    session.compare(&numeric, &zero, CompareDir::Gt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_ad::EagerRuntime;
    use tenferro_cpu::CpuBackend;

    #[test]
    fn native_masks_select_columns_and_retain_previous_requests() {
        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        let mut retained = Vec::new();
        for keep in [
            [true, false, false, true, true, false],
            [false; 6],
            [true; 6],
        ] {
            let selected = runtime
                .with_eager_session(|session| {
                    let mask = bool_tensor_native(session, vec![2, 3], &keep)?;
                    let values = session.constant_from_host(Tensor::from_vec_col_major(
                        vec![2, 3],
                        vec![1.0f32, 2., 3., 4., 5., 6.],
                    )?)?;
                    let replacement = session
                        .constant_from_host(Tensor::from_vec_col_major(vec![], vec![-100.0f32])?)?;
                    let selected = session.where_select(&mask, &values, &replacement)?;
                    session.transpose(&selected, &[1, 0])
                })
                .unwrap()
                .unwrap();
            retained.push((keep, selected));
        }
        for (keep, selected) in retained {
            let output = runtime
                .with_eager_session(|session| crate::output::host_value(session, &selected))
                .unwrap()
                .unwrap();
            let expected: Vec<f32> = [0usize, 2, 4, 1, 3, 5]
                .into_iter()
                .map(|index| {
                    if keep[index] {
                        (index + 1) as f32
                    } else {
                        -100.
                    }
                })
                .collect();
            assert_eq!(output.as_slice::<f32>().unwrap(), expected);
        }
    }

    #[test]
    fn native_masks_reject_invalid_shape_and_support_scalar_inputs() {
        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        runtime
            .with_eager_session(|session| {
                assert!(bool_tensor_native(session, vec![2, 3], &[true]).is_err());
                for value in [false, true] {
                    let scalar = bool_tensor_native(session, vec![], &[value])?;
                    assert_eq!(scalar.shape(), &[] as &[usize]);
                    assert_eq!(scalar.value()?.as_slice::<bool>()?, &[value]);
                }
                Ok::<_, tenferro_ad::Error>(())
            })
            .unwrap()
            .unwrap();
    }
}
