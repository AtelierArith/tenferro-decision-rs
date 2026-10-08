#![cfg(feature = "cuda")]

//! Raw-stage parity gate. Explicit execution requires CUDA and NVRTC; absence
//! is a failure, never a skipped successful parity result.

use tenferro_gated_delta::{
    DeltaScanInputs,
    cuda::{CudaKernels, CudaStageRun, RecurrentGeometry},
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

// Construct inside the admitted eager callback: loaded modules cannot cross
// its Send boundary. Drop runs before the enclosing execution scope exits.
struct CudaScanResources {
    runtime: tenferro_gpu::cuda::CudaRuntime,
    inputs: Option<[tenferro_tensor::TypedTensor<f32>; 4]>,
    buffers: Option<[Tensor; 5]>,
    kernels: Option<CudaKernels>,
    completed: bool,
}

impl Drop for CudaScanResources {
    fn drop(&mut self) {
        if !self.completed && self.runtime.synchronize().is_err() {
            // Completion is unknown, including during unwinding. Retain every
            // allocation and module rather than returning them to an allocator.
            std::mem::forget(self.inputs.take());
            std::mem::forget(self.buffers.take());
            std::mem::forget(self.kernels.take());
        }
    }
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
                            let outputs = [
                                raw.alloc_output::<f32>(&[length, channels])?,
                                raw.alloc_output::<f32>(&[length, vh * vd])?,
                                raw.alloc_output::<f32>(&[length, vh * vd])?,
                            ];
                            // SAFETY: all uploads/allocations are distinct, and
                            // ownership prevents conflicting accesses until finish.
                            let pending = unsafe {
                                CudaStageRun::enqueue(
                                    raw, kernels, &geometry, inputs, outputs, 1e-5,
                                )?
                            };
                            let (kernels, inputs, outputs) = pending.finish()?;
                            // Reuse the loaded module and all workspaces. A second
                            // scan must restart its recurrent state and match the oracle.
                            let pending = unsafe {
                                CudaStageRun::enqueue(
                                    raw, kernels, &geometry, inputs, outputs, 1e-5,
                                )?
                            };
                            let (_, _, [_, _, out]) = pending.finish()?;
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

#[test]
#[ignore = "requires CUDA hardware, NVRTC, cuTENSOR and cuBLAS; run explicitly with --ignored"]
fn cuda_decay_preparation_feeds_native_large_key_scan_on_one_session() {
    use tenferro_ad::EagerRuntime;
    use tenferro_gated_delta::chunked::{PreparedChunkScan, delta_scan_prepared};
    use tenferro_gated_delta::cuda::{ChunkDecayBuffers, ChunkDecayGeometry};
    use tenferro_tensor::TensorRead;

    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA device required");
    let runtime = EagerRuntime::with_cuda_backend(backend.clone()).unwrap();
    let arch = std::env::var("TENFERRO_CUDA_ARCH").unwrap_or_else(|_| "compute_70".into());
    let length = 65usize;
    let kd = 257;
    let vd = 3;
    let chunk_size = 64;
    let chunks = length.div_ceil(chunk_size);
    let mut q = values(kd * length, 21);
    let mut k = values(kd * length, 22);
    for token in 0..length {
        for (data, scale) in [(&mut q, 1.0 / (kd as f32).sqrt()), (&mut k, 1.0)] {
            let mut column: Vec<_> = (0..kd).map(|d| data[d * length + token]).collect();
            l2_normalize(&mut column, 1e-6);
            for d in 0..kd {
                data[d * length + token] = column[d] * scale;
            }
        }
    }
    let v = values(vd * length, 23);
    let z = values(vd * length, 24);
    let a = values(length, 25);
    let b = values(length, 26);
    let decay = [-0.4f32];
    let bias = [0.1f32];
    let beta: Vec<_> = b.iter().copied().map(sigmoid).collect();
    let log_decay: Vec<_> = a.iter().map(|a| decay[0] * softplus(a + bias[0])).collect();
    let norm = vec![1.0; vd];
    let expected = delta_scan_reference(&DeltaScanInputs {
        q: &q,
        k: &k,
        v: &v,
        z: &z,
        beta: &beta,
        decay: &log_decay,
        norm: &norm,
        eps: 1e-5,
        key_dim: kd,
        value_dim: vd,
        length,
    });
    let upload = |shape, data| {
        let host = Tensor::from_vec_col_major(shape, data).unwrap();
        upload_tensor(backend.runtime(), &host)
            .unwrap()
            .into_typed::<f32>()
            .unwrap()
    };
    let inputs = [
        upload(vec![length, 1], a),
        upload(vec![length, 1], b),
        upload(vec![1], decay.to_vec()),
        upload(vec![1], bias.to_vec()),
    ];
    let runtime_handle = backend.runtime().clone();
    let output = runtime
        .with_eager_session(move |session| {
            let mut resources = CudaScanResources {
                runtime: runtime_handle,
                inputs: Some(inputs),
                buffers: None,
                kernels: None,
                completed: false,
            };
            // Prepare model inputs/constants before the raw producer is enqueued.
            let matrix = |session: &mut tenferro_ad::EagerSession<'_>, rows, data: &[f32]| {
                let col: Vec<_> = (0..length)
                    .flat_map(|t| (0..rows).map(move |r| data[r * length + t]))
                    .collect();
                session.constant_from_host(Tensor::from_vec_col_major(vec![rows, length], col)?)
            };
            let q = matrix(session, kd, &q)?;
            let k = matrix(session, kd, &k)?;
            let v = matrix(session, vd, &v)?;
            let z = matrix(session, vd, &z)?;
            let state = session.constant_from_host(Tensor::from_vec_col_major(
                vec![vd, kd],
                vec![0.0; vd * kd],
            )?)?;
            let norm = session.constant_from_host(Tensor::from_vec_col_major(vec![vd], norm)?)?;
            let eps =
                session.constant_from_host(Tensor::from_vec_col_major(vec![], vec![1e-5f32])?)?;
            let inverse = session
                .constant_from_host(Tensor::from_vec_col_major(vec![], vec![1.0 / vd as f32])?)?;
            let one =
                session.constant_from_host(Tensor::from_vec_col_major(vec![], vec![1.0f32])?)?;
            let geometry = ChunkDecayGeometry::new(length, 1, chunk_size).unwrap();
            resources.kernels = Some(
                with_cuda_exec_session(session.backend_session(), |cuda| {
                    cuda.with_raw("cuda_scan_compile", |raw| CudaKernels::compile(raw, &arch))
                })
                .expect("CUDA execution session")?,
            );
            // Install every output owner before launch, so an unwind from the
            // driver or later graph code cannot drop a raw allocation early.
            resources.buffers = Some(
                with_cuda_exec_session(session.backend_session(), |cuda| {
                    cuda.with_raw("cuda_scan_alloc", |raw| {
                        Ok([
                            raw.alloc_output::<f32>(&[length, 1])?,
                            raw.alloc_output::<f32>(&[length, 1])?,
                            raw.alloc_output::<f32>(&[chunk_size, chunk_size, chunks, 1])?,
                            raw.alloc_output::<f32>(&[length, 1])?,
                            raw.alloc_output::<f32>(&[chunks, 1])?,
                        ]
                        .map(Tensor::from_typed))
                    })
                })
                .expect("CUDA execution session")?,
            );
            let launched = with_cuda_exec_session(session.backend_session(), |cuda| {
                cuda.with_raw("cuda_scan_prepare", |raw| {
                    let inputs = resources.inputs.as_ref().expect("owned inputs");
                    let [cumulative, beta, pair, tail, final_factor] =
                        resources.buffers.as_mut().expect("owned outputs");
                    // SAFETY: distinct allocations/module are owned by the
                    // guard through final synchronization or error unwinding.
                    unsafe {
                        resources
                            .kernels
                            .as_ref()
                            .expect("loaded module")
                            .enqueue_chunk_decay(
                                raw,
                                &geometry,
                                ChunkDecayBuffers {
                                    a: &inputs[0],
                                    b: &inputs[1],
                                    a_decay: &inputs[2],
                                    dt_bias: &inputs[3],
                                    cumulative: cumulative.as_typed_mut::<f32>().unwrap(),
                                    beta: beta.as_typed_mut::<f32>().unwrap(),
                                    pair: pair.as_typed_mut::<f32>().unwrap(),
                                    tail: tail.as_typed_mut::<f32>().unwrap(),
                                    final_factor: final_factor.as_typed_mut::<f32>().unwrap(),
                                },
                            )
                    }
                })
            })
            .expect("CUDA execution session");
            // Never use `?` outside this result-producing scope after enqueue:
            // both the success and error paths must reach the final barrier.
            let computed: tenferro_ad::Result<Tensor> = (|| {
                launched?;
                let mut factors = Vec::new();
                for buffer in resources.buffers.as_ref().expect("owned outputs") {
                    // Make provider-owned device copies for eager registration;
                    // raw originals remain owned until the barrier even if import
                    // fails. There is no host download between producer and scan.
                    let copy = session
                        .backend_session()
                        .to_contiguous_read(TensorRead::from_tensor(buffer))?;
                    if !copy.is_backend_buffer() {
                        return Err(tenferro_tensor::Error::invalid_argument(
                            "cuda_scan",
                            "storage",
                            "factor copy must stay device-backed",
                        )
                        .into());
                    }
                    factors.push(session.constant_from(copy)?);
                }
                let cumulative = session.reshape(&factors[0], vec![length])?;
                let beta = session.reshape(&factors[1], vec![length])?;
                let pair = session.reshape(&factors[2], vec![chunk_size, chunk_size, chunks])?;
                let tail = session.reshape(&factors[3], vec![length])?;
                let final_decay = session.reshape(&factors[4], vec![chunks])?;
                let (_, output) = delta_scan_prepared(
                    session,
                    PreparedChunkScan {
                        state: &state,
                        q: &q,
                        k: &k,
                        v: &v,
                        z: &z,
                        beta: &beta,
                        cumulative: &cumulative,
                        pair: &pair,
                        tail: &tail,
                        final_decay: &final_decay,
                        norm_weight: &norm,
                        eps: &eps,
                        inverse_value_dim: &inverse,
                        one: &one,
                    },
                    chunk_size,
                )?;
                session.duplicate_value(&output)
            })();
            let synchronized = with_cuda_exec_session(session.backend_session(), |cuda| {
                cuda.with_raw("cuda_scan_finish", |raw| raw.synchronize())
            })
            .expect("CUDA execution session");
            if let Err(error) = synchronized {
                // The raw guard also protects its resources on return/unwind.
                std::mem::forget(computed);
                return Err(error.into());
            }
            resources.completed = true;
            computed
        })
        .unwrap()
        .unwrap();
    let output = download_tensor(backend.runtime(), &output).unwrap();
    let actual = output.as_slice::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len());
    for row in 0..vd {
        for token in 0..length {
            let actual = actual[row + token * vd];
            let expected = expected[row * length + token];
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 5e-3,
                "row={row} token={token}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware, cuBLAS and cuTENSOR; run explicitly with --ignored"]
fn native_full_layer_matches_cpu_with_masks_grouping_and_large_keys() {
    use tenferro_gated_delta::{
        Algorithm, GatedDeltaConfig, GatedDeltaWeights, delta_layer_reference,
        delta_layer_tenferro_prepared_mask, prepare_tensor_weights,
    };
    use tenferro_infer::TensorCache;

    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA device required");
    let runtime = tenferro_ad::EagerRuntime::with_cuda_backend(backend.clone()).unwrap();
    for (kd, kh, vh, lengths) in [
        (7, 2, 4, vec![1usize, 63, 64, 65, 127, 128, 129]),
        (257, 1, 2, vec![1usize, 63, 64, 65]),
    ] {
        let cfg = GatedDeltaConfig {
            hidden: 4,
            key_dim: kd,
            value_dim: 3,
            key_heads: kh,
            value_heads: vh,
            conv_taps: 4,
            chunk_size: 64,
            eps: 1e-5,
            algorithm: Algorithm::Chunked,
        };
        let channels = 2 * kd * kh + cfg.value_dim * vh;
        let width = cfg.value_dim * vh;
        let weights = GatedDeltaWeights {
            qkv: values(cfg.hidden * channels, 1),
            z: values(cfg.hidden * width, 2),
            a: values(cfg.hidden * vh, 3),
            b: values(cfg.hidden * vh, 4),
            conv: values(channels * cfg.conv_taps, 5),
            a_decay: vec![-0.4; vh],
            dt_bias: values(vh, 6),
            norm: vec![1.0; cfg.value_dim],
            out_proj: values(width * cfg.hidden, 7),
        };
        let mut cache = TensorCache::new();
        let prepared = runtime
            .with_eager_session(|session| {
                prepare_tensor_weights(session, &cfg, &weights, &mut cache)
            })
            .unwrap()
            .unwrap();
        assert!(prepared.qkv.tensor_read().backend_family().is_some());
        for length in lengths {
            let x = values(cfg.hidden * length, 8);
            for masked in [false, true] {
                let mask: Vec<f32> = (0..length)
                    .map(|t| {
                        if masked && (t < 2 || t % 11 == 5) {
                            0.0
                        } else {
                            1.0
                        }
                    })
                    .collect();
                let expected = delta_layer_reference(&cfg, &weights, &x, &mask).unwrap();
                let mut column_major = vec![0.0; x.len()];
                for h in 0..cfg.hidden {
                    for t in 0..length {
                        column_major[h + cfg.hidden * t] = x[h * length + t];
                    }
                }
                let output = runtime
                    .with_eager_session(|session| {
                        let x = session.constant_from_host(Tensor::from_vec_col_major(
                            vec![cfg.hidden, length],
                            column_major,
                        )?)?;
                        let mask = session
                            .constant_from_host(Tensor::from_vec_col_major(vec![length], mask)?)?;
                        let result = delta_layer_tenferro_prepared_mask(
                            session, &cfg, &prepared, &x, &mask,
                        )?;
                        let result = session.duplicate_value(&result)?;
                        assert!(
                            result.is_backend_buffer(),
                            "full layer must remain on device"
                        );
                        with_cuda_exec_session(session.backend_session(), |cuda| {
                            cuda.with_raw("cuda_full_layer_finish", |raw| raw.synchronize())
                        })
                        .expect("CUDA execution session")?;
                        Ok::<_, tenferro_ad::Error>(result)
                    })
                    .unwrap()
                    .unwrap();
                let host = download_tensor(backend.runtime(), &output).unwrap();
                let actual = host.as_slice::<f32>().unwrap();
                for h in 0..cfg.hidden {
                    for t in 0..length {
                        let actual = actual[h + cfg.hidden * t];
                        let expected = expected[h * length + t];
                        assert!(actual.is_finite());
                        assert!(
                            (actual - expected).abs() < 5e-3,
                            "kd={kd} length={length} masked={masked} h={h} t={t}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
    }
}
