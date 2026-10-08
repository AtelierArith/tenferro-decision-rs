use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_ext::{
    EagerSessionGegluExt, EagerSessionGemmBiasExt, EagerSessionGemmExt, EagerSessionGemmGegluExt,
};

#[test]
fn fused_projection_matches_separate_operations_for_strides_batches_and_bias() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    for (input, hidden, length, batch) in [
        (7usize, 5usize, 1usize, 1usize),
        (7, 5, 3, 2),
        (33, 17, 65, 2),
        (129, 129, 129, 1),
        (7, 5, 0, 2),
    ] {
        for has_bias in [false, true] {
            for strided in [false, true] {
                runtime
                    .with_eager_session(|session| {
                        let data: Vec<f32> = (0..input * length * batch)
                            .map(|i| ((i % 23) as f32 - 11.) * 0.031)
                            .collect();
                        let mut x = session.constant_from_host(Tensor::from_vec_col_major(
                            if strided {
                                vec![length, input, batch]
                            } else {
                                vec![input, length, batch]
                            },
                            data,
                        )?)?;
                        if strided {
                            x = session.transpose(&x, &[1, 0, 2])?;
                        }
                        if length == 1 && batch == 1 {
                            x = session.reshape(&x, vec![input])?;
                        }
                        let weights: Vec<f32> = (0..input * 2 * hidden)
                            .map(|i| ((i % 19) as f32 - 9.) * 0.027)
                            .collect();
                        let weight = session.constant_from_host(Tensor::from_vec_col_major(
                            vec![input, 2 * hidden],
                            weights,
                        )?)?;
                        let bias = if has_bias {
                            Some(session.constant_from_host(Tensor::from_vec_col_major(
                                vec![2 * hidden],
                                (0..2 * hidden).map(|i| (i % 7) as f32 * 0.019).collect(),
                            )?)?)
                        } else {
                            None
                        };
                        let fused = session.gemm_geglu(&x, &weight, bias.as_ref(), hidden)?;
                        let flat = session.reshape(&x, vec![input, length * batch])?;
                        let projected = if let Some(bias) = &bias {
                            session.gemm_bias(&flat, &weight, bias)?
                        } else {
                            session.gemm(&flat, &weight)?
                        };
                        let expected = session.geglu(&projected, hidden)?;
                        let repeated = session.gemm_geglu(&x, &weight, bias.as_ref(), hidden)?;
                        let mut shape = x.shape().to_vec();
                        shape[0] = hidden;
                        assert_eq!(fused.shape(), shape);
                        let expected = session.duplicate_value(&expected)?;
                        for value in [&fused, &repeated] {
                            let actual = session.duplicate_value(value)?;
                            assert_eq!(
                                actual
                                    .as_slice::<f32>()?
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                expected
                                    .as_slice::<f32>()?
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>()
                            );
                        }
                        Ok::<_, tenferro_ad::Error>(())
                    })
                    .unwrap()
                    .unwrap();
            }
        }
    }
}

#[test]
fn fused_projection_rejects_dtype_width_and_bias_errors() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    runtime
        .with_eager_session(|session| {
            let x = session
                .constant_from_host(Tensor::from_vec_col_major(vec![3, 2], vec![1.0f32; 6])?)?;
            let w = session
                .constant_from_host(Tensor::from_vec_col_major(vec![3, 8], vec![0.1f32; 24])?)?;
            let bias = session
                .constant_from_host(Tensor::from_vec_col_major(vec![7], vec![0.0f32; 7])?)?;
            assert!(session.gemm_geglu(&x, &w, None, 3).is_err());
            assert!(session.gemm_geglu(&x, &w, Some(&bias), 4).is_err());
            assert!(session.gemm_geglu(&x, &w, None, usize::MAX).is_err());
            let x = session
                .constant_from_host(Tensor::from_vec_col_major(vec![3, 2], vec![1.0f64; 6])?)?;
            let w = session
                .constant_from_host(Tensor::from_vec_col_major(vec![3, 8], vec![0.1f64; 24])?)?;
            assert!(session.gemm_geglu(&x, &w, None, 4).is_err());
            Ok::<_, tenferro_ad::Error>(())
        })
        .unwrap()
        .unwrap();
}

#[test]
fn retained_outputs_survive_workspace_growth_and_changed_inputs() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let mut retained = Vec::new();
    for (length, factor) in [(1usize, 1.0f32), (17, -0.5), (3, 2.0)] {
        let values = runtime
            .with_eager_session(|session| {
                let x = session.constant_from_host(Tensor::from_vec_col_major(
                    vec![3, length],
                    (0..3 * length)
                        .map(|i| factor * (i as f32 + 1.) * 0.02)
                        .collect(),
                )?)?;
                let w = session.constant_from_host(Tensor::from_vec_col_major(
                    vec![3, 8],
                    (0..24).map(|i| (i % 7) as f32 * 0.01 - 0.03).collect(),
                )?)?;
                let fused = session.gemm_geglu(&x, &w, None, 4)?;
                let projected = session.gemm(&x, &w)?;
                let separate = session.geglu(&projected, 4)?;
                let expected = session
                    .duplicate_value(&separate)?
                    .as_slice::<f32>()?
                    .to_vec();
                Ok::<_, tenferro_ad::Error>((fused, expected))
            })
            .unwrap()
            .unwrap();
        retained.push(values);
    }
    for (output, expected) in retained {
        runtime
            .with_eager_session(|session| {
                let actual = session.duplicate_value(&output)?;
                assert_eq!(actual.as_slice::<f32>()?, expected);
                Ok::<_, tenferro_ad::Error>(())
            })
            .unwrap()
            .unwrap();
    }
}
