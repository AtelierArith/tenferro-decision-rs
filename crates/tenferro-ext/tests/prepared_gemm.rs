#![cfg(feature = "onednn")]
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_ext::{EagerSessionPreparedGemmExt, PreparedGemm};
use tenferro_tensor::Tensor;

#[test]
fn prepared_op_matches_projection_bias_and_geglu_and_retains_outputs() {
    let (input, output) = (7usize, 10usize);
    let weights: Vec<f32> = (0..input * output)
        .map(|i| (i % 13) as f32 * 0.03 - 0.2)
        .collect();
    let prepared = PreparedGemm::new(&weights, input, output).unwrap();
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let mut outputs = Vec::new();
    for (length, batch) in [(1, 1), (3, 2), (17, 2), (65, 1), (1, 1), (0, 2)] {
        for biased in [false, true] {
            for geglu in [false, true] {
                let x: Vec<f32> = (0..input * length * batch)
                    .map(|i| (i % 11) as f32 * 0.02 - 0.1)
                    .collect();
                let bias: Vec<f32> = (0..output).map(|i| i as f32 * 0.01).collect();
                let mut projected = vec![0.; length * batch * output];
                if biased {
                    cpu_kernels::input_mul_weight_transpose_bias_into(
                        &x,
                        length * batch,
                        input,
                        &weights,
                        output,
                        &bias,
                        &mut projected,
                    );
                } else {
                    cpu_kernels::input_mul_weight_transpose_into(
                        &x,
                        length * batch,
                        input,
                        &weights,
                        output,
                        &mut projected,
                    );
                }
                let expected = if geglu {
                    {
                        let mut activated = vec![0.; length * batch * output / 2];
                        cpu_kernels::geglu_into(
                            &projected,
                            output / 2,
                            length * batch,
                            &mut activated,
                        );
                        activated
                    }
                } else {
                    projected
                };
                let result = runtime
                    .with_eager_session(|session| {
                        let x = session.constant_from(Tensor::from_vec_col_major(
                            vec![input, length, batch],
                            x.clone(),
                        )?)?;
                        let bias = if biased {
                            Some(session.constant_from(Tensor::from_vec_col_major(
                                vec![output],
                                bias.clone(),
                            )?)?)
                        } else {
                            None
                        };
                        session.prepared_gemm(&x, &prepared, bias.as_ref(), geglu)
                    })
                    .unwrap()
                    .unwrap();
                outputs.push((result, expected));
            }
        }
    }
    drop(prepared);
    for (actual, expected) in outputs {
        let actual = actual.value().unwrap();
        let actual = actual.as_slice::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}

#[test]
fn rejects_bad_dtype_dimensions_and_bias_before_execution() {
    let prepared = PreparedGemm::new(&[1.; 6], 2, 3).unwrap();
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    runtime
        .with_eager_session(|session| {
            let x =
                session.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![1.0f32; 2])?)?;
            assert!(session.prepared_gemm(&x, &prepared, None, true).is_err());
            let wrong =
                session.constant_from(Tensor::from_vec_col_major(vec![3, 1], vec![1.0f32; 3])?)?;
            assert!(
                session
                    .prepared_gemm(&wrong, &prepared, None, false)
                    .is_err()
            );
            let double =
                session.constant_from(Tensor::from_vec_col_major(vec![2, 1], vec![1.0f64; 2])?)?;
            assert!(
                session
                    .prepared_gemm(&double, &prepared, None, false)
                    .is_err()
            );
            let bias =
                session.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0f32; 2])?)?;
            assert!(
                session
                    .prepared_gemm(&x, &prepared, Some(&bias), false)
                    .is_err()
            );
            Ok::<_, tenferro_ad::Error>(())
        })
        .unwrap()
        .unwrap();
}

#[test]
fn strided_and_rank_one_inputs_can_share_preparation_across_runtimes() {
    let prepared = PreparedGemm::new(&[1., 2., 3., 4., 5., 6.], 3, 2).unwrap();
    for _ in 0..2 {
        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        runtime
            .with_eager_session(|session| {
                let base = session.constant_from(Tensor::from_vec_col_major(
                    vec![2, 3],
                    vec![1.0f32, 2., 3., 4., 5., 6.],
                )?)?;
                let strided = session.transpose(&base, &[1, 0])?;
                let output = session.prepared_gemm(&strided, &prepared, None, false)?;
                assert_eq!(output.shape(), [2, 2]);
                assert_eq!(
                    session.duplicate_value(&output)?.as_slice::<f32>()?,
                    [22., 49., 28., 64.]
                );
                let vector = session
                    .constant_from(Tensor::from_vec_col_major(vec![3], vec![1.0f32, 3., 5.])?)?;
                let output = session.prepared_gemm(&vector, &prepared, None, false)?;
                assert_eq!(output.shape(), [2]);
                assert_eq!(
                    session.duplicate_value(&output)?.as_slice::<f32>()?,
                    [22., 49.]
                );
                Ok::<_, tenferro_ad::Error>(())
            })
            .unwrap()
            .unwrap();
    }
}
