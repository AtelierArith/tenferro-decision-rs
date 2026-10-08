#![cfg(feature = "cuda")]

//! Raw-stage parity gate. Explicit execution requires CUDA and NVRTC; absence
//! is a failure, never a skipped successful parity result.

use tenferro_gated_delta::{
    DeltaScanInputs,
    cuda::{CudaKernels, RecurrentGeometry, StageBuffers},
    delta_scan_reference,
    ops::{l2_normalize, sigmoid, silu, softplus},
};
use tenferro_gpu::cuda::{
    CudaBackend, CudaDeviceId, download_tensor, upload_tensor, with_cuda_exec_session,
};
use tenferro_tensor::{BackendSessionHost, Tensor};

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 37 + salt * 19) % 101) as f32 - 50.0) / 71.0)
        .collect()
}

#[test]
#[ignore = "requires CUDA hardware and NVRTC; run explicitly with --ignored"]
fn recurrent_stages_match_cpu_at_boundaries_and_grouped_heads() {
    let mut backend =
        CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA device required");
    let arch = std::env::var("TENFERRO_CUDA_ARCH").unwrap_or_else(|_| "compute_70".into());
    for (kd, vd, kh, vh) in [(1, 1, 1, 1), (33, 37, 2, 4), (256, 128, 1, 2)] {
        for length in [1usize, 63, 64, 65, 127, 128, 129] {
            let channels = 2 * kh * kd + vh * vd;
            let taps = 4;
            let x = values(length * channels, 1);
            let weight = values(channels * taps, 2);
            let a = values(length * vh, 3);
            let b = values(length * vh, 4);
            let decay = vec![-0.4; vh];
            let bias = values(vh, 5);
            let z = values(length * vh * vd, 6);
            let norm: Vec<_> = values(vd, 7).iter().map(|v| v + 1.0).collect();
            // Independent causal-convolution oracle, followed by the existing
            // CPU recurrent oracle (not the optimized CPU implementation).
            let mut mixed = vec![0.0; x.len()];
            for channel in 0..channels {
                for token in 0..length {
                    let sum = (0..taps)
                        .filter_map(|tap| {
                            token
                                .checked_sub(taps - 1 - tap)
                                .map(|t| x[channel * length + t] * weight[tap * channels + channel])
                        })
                        .sum();
                    mixed[channel * length + token] = silu(sum);
                }
            }
            let mut expected = Vec::new();
            for head in 0..vh {
                let key_head = head / (vh / kh);
                let mut q = mixed[key_head * kd * length..(key_head + 1) * kd * length].to_vec();
                let mut k = mixed
                    [(kh * kd + key_head * kd) * length..(kh * kd + (key_head + 1) * kd) * length]
                    .to_vec();
                for token in 0..length {
                    for (data, scale) in [(&mut q, 1.0 / (kd as f32).sqrt()), (&mut k, 1.0)] {
                        let mut column: Vec<_> =
                            (0..kd).map(|d| data[d * length + token]).collect();
                        l2_normalize(&mut column, 1e-6);
                        for d in 0..kd {
                            data[d * length + token] = column[d] * scale;
                        }
                    }
                }
                let beta: Vec<_> = (0..length).map(|t| sigmoid(b[head * length + t])).collect();
                let log_decay: Vec<_> = (0..length)
                    .map(|t| decay[head] * softplus(a[head * length + t] + bias[head]))
                    .collect();
                let start = (2 * kh * kd + head * vd) * length;
                expected.extend(delta_scan_reference(&DeltaScanInputs {
                    q: &q,
                    k: &k,
                    v: &mixed[start..start + vd * length],
                    beta: &beta,
                    decay: &log_decay,
                    z: &z[head * vd * length..(head + 1) * vd * length],
                    norm: &norm,
                    eps: 1e-5,
                    key_dim: kd,
                    value_dim: vd,
                    length,
                }));
            }
            let upload = |shape: Vec<usize>, data: Vec<f32>| {
                let host = Tensor::from_vec_col_major(shape, data).unwrap();
                upload_tensor(backend.runtime(), &host)
                    .unwrap()
                    .into_typed::<f32>()
                    .unwrap()
            };
            let inputs = [
                upload(vec![length, channels], x),
                upload(vec![channels, taps], weight),
                upload(vec![length, vh], a),
                upload(vec![length, vh], b),
                upload(vec![vh], decay),
                upload(vec![vh], bias),
                upload(vec![length, vh * vd], z),
                upload(vec![vd], norm),
            ];
            let geometry = RecurrentGeometry::new(length, kd, vd, kh, vh, taps).unwrap();
            let output = backend
                .with_backend_session(|session| {
                    with_cuda_exec_session(session, |cuda| {
                        cuda.with_raw("cuda_stage_parity", |raw| {
                            let kernels = CudaKernels::compile(raw, &arch)?;
                            let mut conv = raw.alloc_output::<f32>(&[length, channels])?;
                            let mut scan = raw.alloc_output::<f32>(&[length, vh * vd])?;
                            let mut out = raw.alloc_output::<f32>(&[length, vh * vd])?;
                            let mut retained = Vec::new();
                            for tensor in inputs.iter().chain([&conv, &scan, &out]) {
                                retained.push(raw.retain_tensor(tensor, "cuda_stage_parity")?);
                            }
                            // SAFETY: each upload/allocation is distinct; all resources
                            // are retained until the barrier, including on launch errors.
                            let launched = unsafe {
                                kernels.enqueue_stages(
                                    raw,
                                    &geometry,
                                    StageBuffers {
                                        mixed_input: &inputs[0],
                                        conv_weight: &inputs[1],
                                        a: &inputs[2],
                                        b: &inputs[3],
                                        a_decay: &inputs[4],
                                        dt_bias: &inputs[5],
                                        z: &inputs[6],
                                        norm: &inputs[7],
                                        convolved: &mut conv,
                                        scanned: &mut scan,
                                        output: &mut out,
                                    },
                                    1e-5,
                                )
                            };
                            if let Err(error) = raw.synchronize() {
                                // GPU completion is unknown: retain allocations/module
                                // permanently rather than racing resource reclamation.
                                std::mem::forget(retained);
                                std::mem::forget(kernels);
                                return Err(error);
                            }
                            launched?;
                            Ok(Tensor::from_typed(out))
                        })
                    })
                    .expect("CUDA execution session")
                })
                .unwrap()
                .unwrap();
            let downloaded = download_tensor(backend.runtime(), &output).unwrap();
            let actual = downloaded.as_slice::<f32>().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= 5e-3 + 5e-3 * expected.abs(),
                    "kd={kd} vd={vd} heads={kh}/{vh} L={length} index={index}: {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware and cuBLAS; run explicitly with --ignored"]
fn native_unit_lower_triangular_solve_supports_large_key_rhs() {
    use tenferro_linalg::TensorLinalgExt;
    let mut backend =
        CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA device required");
    // Chunk width is the triangular-system dimension; key width is the RHS
    // column count and may exceed the recurrent kernel's register-state limit.
    let key_dim = 257;
    for n in [1, 63, 64, 65] {
        let expected = values(n * key_dim, 11);
        let mut matrix = vec![999.0; n * n];
        let mut rhs = expected.clone();
        for row in 0..n {
            for column in 0..row {
                let coefficient = ((row + column) % 7) as f32 / (n as f32 * 20.0);
                matrix[row + column * n] = coefficient;
                for key in 0..key_dim {
                    rhs[row + key * n] += coefficient * expected[column + key * n];
                }
            }
        }
        // Deliberately non-unit diagonal and large upper-triangular entries:
        // the lower/unit flags must ignore both rather than solve a dense matrix.
        let matrix = Tensor::from_vec_col_major(vec![n, n], matrix).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![n, key_dim], rhs).unwrap();
        let matrix = upload_tensor(backend.runtime(), &matrix).unwrap();
        let rhs = upload_tensor(backend.runtime(), &rhs).unwrap();
        let solution = backend
            .with_backend_session(|session| {
                matrix.triangular_solve(&rhs, true, true, false, true, session)
            })
            .unwrap()
            .expect("CUDA unit triangular solve must be supported");
        assert!(
            solution.as_typed::<f32>().unwrap().host_data().is_err(),
            "native CUDA solve must return device storage"
        );
        let solution = download_tensor(backend.runtime(), &solution).unwrap();
        let actual = solution.as_slice::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert!(actual.is_finite() && (actual - expected).abs() <= 1e-4);
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware and NVRTC; run explicitly with --ignored"]
fn chunk_decay_matches_cpu_with_padding_and_underflow() {
    use tenferro_gated_delta::cuda::{ChunkDecayBuffers, ChunkDecayGeometry};
    let mut backend =
        CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA device required");
    let arch = std::env::var("TENFERRO_CUDA_ARCH").unwrap_or_else(|_| "compute_70".into());
    let heads = 2;
    let chunk_size = 64usize;
    for length in [1usize, 63, 64, 65, 127, 128, 129] {
        let chunks = length.div_ceil(chunk_size);
        let a = vec![0.75; length * heads];
        let b = values(length * heads, 13);
        let decay = vec![-0.4, -30.0];
        let bias = vec![0.1, -0.2];
        let mut cumulative = vec![0.0; length * heads];
        let beta: Vec<_> = b.iter().copied().map(sigmoid).collect();
        let mut pair = vec![0.0; chunk_size * chunk_size * chunks * heads];
        let mut tail = vec![0.0; length * heads];
        let mut final_factor = vec![0.0; chunks * heads];
        for head in 0..heads {
            for chunk in 0..chunks {
                let start = chunk * chunk_size;
                let n = chunk_size.min(length - start);
                let base = head * length + start;
                let mut sum = 0.0;
                for token in 0..n {
                    sum += decay[head] * softplus(a[base + token] + bias[head]);
                    cumulative[base + token] = sum;
                }
                let task = head * chunks + chunk;
                final_factor[task] = sum.exp();
                for row in 0..n {
                    tail[base + row] = (sum - cumulative[base + row]).exp();
                    for column in 0..=row {
                        pair[task * chunk_size * chunk_size + row + column * chunk_size] =
                            (cumulative[base + row] - cumulative[base + column]).exp();
                    }
                }
                assert_eq!(tail[base + n - 1], 1.0);
                if head == 1 && n >= 63 {
                    assert_eq!(final_factor[task], 0.0);
                }
            }
        }
        let expected = [cumulative, beta, pair, tail, final_factor];
        let upload = |shape: Vec<usize>, data: Vec<f32>| {
            let host = Tensor::from_vec_col_major(shape, data).unwrap();
            upload_tensor(backend.runtime(), &host)
                .unwrap()
                .into_typed::<f32>()
                .unwrap()
        };
        let inputs = [
            upload(vec![length, heads], a),
            upload(vec![length, heads], b),
            upload(vec![heads], decay),
            upload(vec![heads], bias),
        ];
        let geometry = ChunkDecayGeometry::new(length, heads, chunk_size).unwrap();
        let outputs = backend
            .with_backend_session(|session| {
                with_cuda_exec_session(session, |cuda| {
                    cuda.with_raw("cuda_chunk_decay", |raw| {
                        let kernels = CudaKernels::compile(raw, &arch)?;
                        let mut outputs = [
                            raw.alloc_output::<f32>(&[length, heads])?,
                            raw.alloc_output::<f32>(&[length, heads])?,
                            raw.alloc_output::<f32>(&[chunk_size, chunk_size, chunks, heads])?,
                            raw.alloc_output::<f32>(&[length, heads])?,
                            raw.alloc_output::<f32>(&[chunks, heads])?,
                        ];
                        let mut retained = Vec::new();
                        for tensor in inputs.iter().chain(outputs.iter()) {
                            retained.push(raw.retain_tensor(tensor, "cuda_chunk_decay")?);
                        }
                        let [cumulative, beta, pair, tail, final_factor] = &mut outputs;
                        // SAFETY: distinct owned uploads/outputs stay retained through
                        // synchronization, including when enqueueing returns an error.
                        let launched = unsafe {
                            kernels.enqueue_chunk_decay(
                                raw,
                                &geometry,
                                ChunkDecayBuffers {
                                    a: &inputs[0],
                                    b: &inputs[1],
                                    a_decay: &inputs[2],
                                    dt_bias: &inputs[3],
                                    cumulative,
                                    beta,
                                    pair,
                                    tail,
                                    final_factor,
                                },
                            )
                        };
                        if let Err(error) = raw.synchronize() {
                            std::mem::forget(retained);
                            std::mem::forget(kernels);
                            return Err(error);
                        }
                        launched?;
                        Ok(outputs.map(Tensor::from_typed))
                    })
                })
                .expect("CUDA execution session")
            })
            .unwrap()
            .unwrap();
        for (index, (output, expected)) in outputs.iter().zip(&expected).enumerate() {
            let host = download_tensor(backend.runtime(), output).unwrap();
            let actual = host.as_slice::<f32>().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (&actual, &expected) in actual.iter().zip(expected) {
                assert!(
                    actual.is_finite()
                        && (actual - expected).abs() <= 3e-4 * (1.0 + expected.abs()),
                    "L={length} output={index}: {actual} vs {expected}"
                );
            }
        }
    }
}
