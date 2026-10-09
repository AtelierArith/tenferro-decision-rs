#![cfg(feature = "onednn")]
use cpu_kernels::prepared_projection::PreparedProjection;

#[test]
fn owns_weights_reuses_shapes_and_preserves_outputs() {
    for (input, output) in [(7usize, 5usize), (33, 17), (129, 129)] {
        let mut weights: Vec<f32> = (0..input * output)
            .map(|i| (i % 89) as f32 * 0.002 - 0.088)
            .collect();
        let expected_weights = weights.clone();
        let mut prepared = PreparedProjection::new(&weights, input, output).unwrap();
        // Mutation and deallocation of the source cannot invalidate preparation.
        weights.fill(100.0);
        drop(weights);
        let mut retained = Vec::new();
        for rows in [1usize, 3, 7, 8, 9, 16, 64, 65, 1] {
            let x: Vec<f32> = (0..rows * input)
                .map(|i| (i % 97) as f32 * 0.001 - 0.048)
                .collect();
            for accumulate in [false, true] {
                let initial = 0.13;
                let mut y = vec![initial; rows * output];
                prepared.run(&x, rows, &mut y, accumulate).unwrap();
                let mut expected = vec![0.0; y.len()];
                for row in 0..rows {
                    for feature in 0..output {
                        let dot: f64 = (0..input)
                            .map(|k| {
                                f64::from(x[row * input + k])
                                    * f64::from(expected_weights[feature * input + k])
                            })
                            .sum();
                        expected[row * output + feature] =
                            dot as f32 + if accumulate { initial } else { 0.0 };
                    }
                }
                for (a, b) in y.iter().zip(&expected) {
                    assert!((a - b).abs() < 1e-5);
                }
                retained.push((y, expected));
            }
        }
        assert!(prepared.retained_bytes() >= input * output * size_of::<f32>());
        prepared.run(&[], 0, &mut [], false).unwrap();
        drop(prepared);
        for (actual, expected) in retained {
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn rejects_invalid_extents_before_native_calls() {
    assert!(PreparedProjection::new(&[], 0, 1).is_err());
    assert!(PreparedProjection::new(&[1.0], 2, 1).is_err());
    assert!(PreparedProjection::new(&[], usize::MAX, 2).is_err());
    let mut prepared = PreparedProjection::new(&[1., 2., 3., 4.], 2, 2).unwrap();
    assert!(prepared.run(&[1.], 1, &mut [0.; 2], false).is_err());
    assert!(prepared.run(&[1., 2.], 1, &mut [0.; 1], false).is_err());
    assert!(prepared.run(&[], usize::MAX, &mut [], false).is_err());
    let mut warmed = [0.; 2];
    prepared.run(&[1., 2.], 1, &mut warmed, false).unwrap();
    let y = std::thread::spawn(move || {
        let mut y = [0.; 2];
        prepared.run(&[1., 2.], 1, &mut y, false).unwrap();
        y
    })
    .join()
    .unwrap();
    assert_eq!(y, [5., 11.]);
}

#[test]
fn changing_lengths_evict_plans_and_still_reuse_owned_weights() {
    let mut prepared = PreparedProjection::new(&[1., 2., 3., 4.], 2, 2).unwrap();
    let first_bytes = prepared.retained_bytes();
    for rows in (1..=40).chain([1, 40, 8]) {
        let x = vec![1.; rows * 2];
        let mut y = vec![0.; rows * 2];
        prepared.run(&x, rows, &mut y, false).unwrap();
        assert!(y.chunks_exact(2).all(|row| row == [3., 7.]));
    }
    assert!(prepared.retained_bytes() >= first_bytes);
}

#[test]
fn one_workspace_can_grow_shrink_and_move_between_projections() {
    use cpu_kernels::prepared_projection::ProjectionWorkspace;
    let mut workspace = ProjectionWorkspace::default();
    let mut prepared: Vec<_> = [(7usize, 5usize), (129, 129), (33, 17)]
        .into_iter()
        .map(|(input, output)| {
            let weights: Vec<f32> = (0..input * output)
                .map(|i| (i % 89) as f32 * 0.002 - 0.088)
                .collect();
            (
                PreparedProjection::new(&weights, input, output).unwrap(),
                weights,
                input,
                output,
            )
        })
        .collect();
    let mut high_water = 0;
    for (index, rows) in [(0, 1), (1, 65), (2, 3), (0, 17), (1, 8), (2, 0)] {
        let (projection, weights, input, output) = &mut prepared[index];
        let x: Vec<f32> = (0..rows * *input)
            .map(|i| (i % 97) as f32 * 0.001 - 0.048)
            .collect();
        let mut actual = vec![0.; rows * *output];
        projection
            .run_with_workspace(&x, rows, &mut actual, false, &mut workspace)
            .unwrap();
        let mut expected = actual.clone();
        cpu_kernels::input_mul_weight_transpose_into(
            &x,
            rows,
            *input,
            weights,
            *output,
            &mut expected,
        );
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
        assert!(workspace.retained_bytes() >= high_water);
        high_water = workspace.retained_bytes();
    }
    std::thread::spawn(move || {
        let (projection, weights, input, output) = &mut prepared[1];
        let x = vec![1.; 65 * *input];
        let mut actual = vec![0.; 65 * *output];
        projection
            .run_with_workspace(&x, 65, &mut actual, false, &mut workspace)
            .unwrap();
        let mut expected = actual.clone();
        cpu_kernels::input_mul_weight_transpose_into(
            &x,
            65,
            *input,
            weights,
            *output,
            &mut expected,
        );
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
    })
    .join()
    .unwrap();
}

#[test]
fn native_geglu_matches_oracle_across_ranges_and_shared_workspace_reuse() {
    use cpu_kernels::prepared_projection::ProjectionWorkspace;
    let hidden = 2624;
    let mut projection = PreparedProjection::new(&vec![0.; 2 * hidden], 1, 2 * hidden).unwrap();
    let mut workspace = ProjectionWorkspace::default();
    let values = [
        -1000.0f32, -40., -10., -3., -0.1, 0., 0.1, 3., 10., 40., 1000.,
    ];
    for rows in [1usize, 3, 8, 16, 64, 65, 1, 0] {
        let mut u = vec![0.; rows * hidden * 2];
        for row in 0..rows {
            for i in 0..hidden {
                u[row * hidden * 2 + i] = values[(row + i) % values.len()];
                u[row * hidden * 2 + hidden + i] = (i % 17) as f32 * 0.2 - 1.6;
            }
        }
        let mut expected = vec![0.; rows * hidden];
        cpu_kernels::geglu_into(&u, hidden, rows, &mut expected);
        let mut actual = expected.clone();
        projection
            .geglu_with_workspace(&u, rows, &mut actual, &mut workspace)
            .unwrap();
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() <= 1e-5 + 2e-6 * b.abs())
        );
    }
    std::thread::spawn(move || {
        let rows = 8;
        let u = vec![1.; rows * hidden * 2];
        let mut y = vec![0.; rows * hidden];
        projection
            .geglu_with_workspace(&u, rows, &mut y, &mut workspace)
            .unwrap();
        assert!(
            y.iter()
                .all(|v| (*v - cpu_kernels::gelu_erf_f32(1.)).abs() < 1e-6)
        );
        assert!(
            projection
                .geglu_with_workspace(&[1.], 1, &mut [], &mut workspace)
                .is_err()
        );
    })
    .join()
    .unwrap();
}

#[test]
fn native_layernorm_uses_current_inputs_and_parameters_after_thread_move() {
    use cpu_kernels::prepared_projection::{PreparedLayerNorm, ProjectionWorkspace};
    let mut workspace = ProjectionWorkspace::default();
    for width in [7usize, 1024] {
        for rows in [1usize, 16, 64] {
            for with_bias in [false, true] {
                let mut plan = PreparedLayerNorm::new(rows, width, 1e-5, with_bias).unwrap();
                for factor in [1.0f32, 0.5] {
                    let x: Vec<f32> = (0..rows * width)
                        .map(|i| (i % 97) as f32 * 0.08 * factor - 3.84)
                        .collect();
                    let weight: Vec<f32> = (0..width).map(|i| 1. + (i % 7) as f32 * 0.03).collect();
                    let bias = vec![0.1; width];
                    let bias = with_bias.then_some(bias.as_slice());
                    let mut expected = vec![0.; x.len()];
                    cpu_kernels::layer_norm_feature_first_into(
                        &x,
                        width,
                        rows,
                        &weight,
                        bias,
                        1e-5,
                        &mut expected,
                    );
                    let mut actual = expected.clone();
                    plan.run(&x, &weight, bias, &mut actual, &mut workspace)
                        .unwrap();
                    assert!(
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| (a - b).abs() < 2e-5)
                    );
                }
                std::thread::spawn(move || {
                    let x = vec![1.; rows * width];
                    let w = vec![2.; width];
                    let b = vec![0.1; width];
                    let b = with_bias.then_some(b.as_slice());
                    let mut y = vec![0.; x.len()];
                    let mut workspace = ProjectionWorkspace::default();
                    plan.run(&x, &w, b, &mut y, &mut workspace).unwrap();
                    assert!(
                        y.iter()
                            .all(|v| (*v - if with_bias { 0.1 } else { 0. }).abs() < 1e-6)
                    );
                })
                .join()
                .unwrap();
            }
        }
    }
    assert!(PreparedLayerNorm::new(0, 1024, 1e-5, false).is_err());
    assert!(PreparedLayerNorm::new(16, 1024, f32::NAN, false).is_err());
    let mut plan = PreparedLayerNorm::new(1, 2, 1e-5, false).unwrap();
    assert!(
        plan.run(&[1.], &[1.; 2], None, &mut [0.; 2], &mut workspace)
            .is_err()
    );
    assert!(
        plan.run(
            &[1.; 2],
            &[1.; 2],
            Some(&[1.; 2]),
            &mut [0.; 2],
            &mut workspace
        )
        .is_err()
    );
}
