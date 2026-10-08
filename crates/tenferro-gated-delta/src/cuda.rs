//! Raw CUDA stages for a device-resident Gated DeltaNet pipeline.
//!
//! This is the low-level kernel component, not an enabled layer backend.
//! The full layer adapter, chunked large-key path and device parity are pending.
//! Plan resolution continues to reject CUDA until those are implemented.

use tenferro_gpu::cuda::raw::{Function, Module, NvrtcOptions, Session};

/// CUDA source compiled by [`CudaKernels::compile`].
pub const KERNEL_SOURCE: &str = include_str!("cuda/kernels.cu");

/// Runtime-bound CUDA kernel handles. Naturally `!Send`/`!Sync`, following the
/// public tenferro raw-module boundary.
///
/// Keep this owner alive until all launches using its functions complete.
/// A raw launch's allocation/ABI/stream-lifetime obligations still apply; this
/// type does not enqueue or synchronize work and must not be dropped while a
/// kernel is in flight. Store it in the CUDA execution owner, not a `Send`
/// extension cache or a process-global cache.
pub struct CudaKernels {
    runtime: tenferro_gpu::cuda::CudaRuntimeIdentity,
    _module: Module,
    conv_silu: Function,
    recurrent: Function,
    norm_gate: Function,
}

impl CudaKernels {
    /// Compile and load on the supplied runtime. `arch` is an NVRTC virtual
    /// architecture such as `compute_80`; the caller selects one supported by
    /// its device. Compile once and retain this owner across requests.
    ///
    /// Returns tenferro's typed compiler/driver errors without a CPU fallback.
    pub fn compile(session: &Session<'_>, arch: &str) -> tenferro_tensor::Result<Self> {
        let options = NvrtcOptions {
            arch: Some(arch.into()),
            std: Some("c++14".into()),
            extra: vec!["--fmad=false".into()],
        };
        let module = session.compile_nvrtc(KERNEL_SOURCE, &options)?;
        Ok(Self {
            runtime: session.runtime_identity(),
            conv_silu: module.function("gated_delta_conv_silu")?,
            recurrent: module.function("gated_delta_recurrent")?,
            norm_gate: module.function("gated_delta_norm_gate")?,
            _module: module,
        })
    }

    /// Causal depthwise convolution plus SiLU. Flat launch over channel×token.
    /// The source documents exact argument order and buffer layout.
    pub fn conv_silu(&self) -> &Function {
        &self.conv_silu
    }

    /// Warp-owned register-state scan with fused Q/K normalization and gates.
    /// Launch exactly 32 threads per block; key width must be in `1..=256`.
    pub fn recurrent(&self) -> &Function {
        &self.recurrent
    }

    /// Per-token output RMSNorm and SiLU gate. Requires disjoint input/output,
    /// a power-of-two block size and dynamic shared memory as specified in source.
    pub fn norm_gate(&self) -> &Function {
        &self.norm_gate
    }
}

/// Validated indexing and launch geometry for the f32 register-state stages.
///
/// This validates scalar dimensions only. The eventual execution adapter must
/// additionally validate tensor layouts, device residency and allocation spans.
#[derive(Clone, Copy, Debug)]
pub struct RecurrentGeometry {
    length: i32,
    key_dim: i32,
    value_dim: i32,
    key_heads: i32,
    value_heads: i32,
    taps: i32,
    channels: i32,
}

impl RecurrentGeometry {
    /// Reject unsupported dimensions before narrowing to the CUDA integer ABI.
    /// Every intermediate index in the source must fit a signed 32-bit integer.
    pub fn new(
        length: usize,
        key_dim: usize,
        value_dim: usize,
        key_heads: usize,
        value_heads: usize,
        taps: usize,
    ) -> decision_core::Result<Self> {
        let unsupported = || {
            decision_core::DecisionError::unsupported(
                "CUDA recurrent dimensions exceed the supported f32 indexing or launch range",
            )
        };
        if [length, key_dim, value_dim, key_heads, value_heads, taps].contains(&0)
            || key_dim > 256
            || value_heads % key_heads != 0
            || value_dim > 65_535
        {
            return Err(unsupported());
        }
        let key_width = key_heads.checked_mul(key_dim).ok_or_else(unsupported)?;
        let value_width = value_heads.checked_mul(value_dim).ok_or_else(unsupported)?;
        let channels = key_width
            .checked_mul(2)
            .and_then(|n| n.checked_add(value_width))
            .ok_or_else(unsupported)?;
        for count in [
            channels.checked_mul(length),
            channels.checked_mul(taps),
            value_heads.checked_mul(length),
        ] {
            if count.is_none_or(|n| n > i32::MAX as usize) {
                return Err(unsupported());
            }
        }
        let narrow = |n| i32::try_from(n).map_err(|_| unsupported());
        Ok(Self {
            length: narrow(length)?,
            key_dim: narrow(key_dim)?,
            value_dim: narrow(value_dim)?,
            key_heads: narrow(key_heads)?,
            value_heads: narrow(value_heads)?,
            taps: narrow(taps)?,
            channels: narrow(channels)?,
        })
    }

    /// Scalar arguments for convolution, in source ABI order.
    pub fn conv_arguments(&self) -> [i32; 3] {
        [self.length, self.channels, self.taps]
    }

    /// Scalar arguments for the recurrent scan, in source ABI order.
    pub fn recurrent_arguments(&self) -> [i32; 5] {
        [
            self.length,
            self.key_dim,
            self.value_dim,
            self.key_heads,
            self.value_heads,
        ]
    }

    /// Launch shapes for convolution, scan and normalization, respectively.
    pub fn launches(&self) -> [tenferro_gpu::cuda::raw::LaunchConfig; 3] {
        use tenferro_gpu::cuda::raw::LaunchConfig;
        let elements = (self.length * self.channels) as u32;
        let norm_threads = (self.value_dim as u32).next_power_of_two().clamp(32, 1024);
        [
            LaunchConfig {
                grid: [elements.div_ceil(256), 1, 1],
                block: [256, 1, 1],
                shared_mem_bytes: 0,
            },
            LaunchConfig {
                grid: [self.value_heads as u32, self.value_dim as u32, 1],
                block: [32, 1, 1],
                shared_mem_bytes: 0,
            },
            LaunchConfig {
                grid: [(self.length * self.value_heads) as u32, 1, 1],
                block: [norm_threads, 1, 1],
                shared_mem_bytes: norm_threads * 4,
            },
        ]
    }
}

/// Device-resident inputs and distinct scratch/output tensors for the three
/// raw stages. All matrices have token as their first (contiguous) dimension.
pub struct StageBuffers<'a> {
    pub mixed_input: &'a tenferro_tensor::TypedTensor<f32>,
    pub conv_weight: &'a tenferro_tensor::TypedTensor<f32>,
    pub a: &'a tenferro_tensor::TypedTensor<f32>,
    pub b: &'a tenferro_tensor::TypedTensor<f32>,
    pub a_decay: &'a tenferro_tensor::TypedTensor<f32>,
    pub dt_bias: &'a tenferro_tensor::TypedTensor<f32>,
    pub z: &'a tenferro_tensor::TypedTensor<f32>,
    pub norm: &'a tenferro_tensor::TypedTensor<f32>,
    pub convolved: &'a mut tenferro_tensor::TypedTensor<f32>,
    pub scanned: &'a mut tenferro_tensor::TypedTensor<f32>,
    pub output: &'a mut tenferro_tensor::TypedTensor<f32>,
}

fn check_stage_tensor(
    tensor: &tenferro_tensor::TypedTensor<f32>,
    shape: &[usize],
) -> tenferro_tensor::Result<()> {
    if tensor.shape() != shape
        || !tensor.is_col_major_contiguous()?
        || tensor.layout_linear_offset(&vec![0; shape.len()])? != 0
    {
        return Err(tenferro_tensor::Error::invalid_argument(
            "gated_delta.cuda",
            "tensor",
            "expected zero-offset column-major stage shape",
        ));
    }
    Ok(())
}

impl CudaKernels {
    /// Enqueue convolution, recurrent scan and norm/gating on one raw stream.
    /// Performs no upload, download, allocation or synchronization.
    ///
    /// # Safety
    /// All underlying allocations must be disjoint, including allocations
    /// shared by tensor views/aliases. The caller must retain every buffer and
    /// this kernel owner, and prohibit conflicting access, until successful
    /// synchronization of this stream. This obligation also applies on error:
    /// a preceding stage may already be in flight. A failed synchronization
    /// requires retaining/leaking resources rather than freeing them early.
    pub unsafe fn enqueue_stages(
        &self,
        session: &Session<'_>,
        geometry: &RecurrentGeometry,
        buffers: StageBuffers<'_>,
        eps: f32,
    ) -> tenferro_tensor::Result<()> {
        use tenferro_gpu::cuda::raw::KernelArg;
        if self.runtime != session.runtime_identity() || !eps.is_finite() || eps <= 0.0 {
            return Err(tenferro_tensor::Error::invalid_argument(
                "gated_delta.cuda",
                "runtime/eps",
                "runtime must match and epsilon must be finite and positive",
            ));
        }
        let l = geometry.length as usize;
        let channels = geometry.channels as usize;
        let heads = geometry.value_heads as usize;
        let width = heads * geometry.value_dim as usize;
        for (tensor, shape) in [
            (buffers.mixed_input, vec![l, channels]),
            (buffers.conv_weight, vec![channels, geometry.taps as usize]),
            (buffers.a, vec![l, heads]),
            (buffers.b, vec![l, heads]),
            (buffers.a_decay, vec![heads]),
            (buffers.dt_bias, vec![heads]),
            (buffers.z, vec![l, width]),
            (buffers.norm, vec![geometry.value_dim as usize]),
            (&*buffers.convolved, vec![l, channels]),
            (&*buffers.scanned, vec![l, width]),
            (&*buffers.output, vec![l, width]),
        ] {
            check_stage_tensor(tensor, &shape)?;
        }
        // Obtain every checked device binding before enqueueing any work.
        let mixed = session.tensor(buffers.mixed_input)?;
        let weight = session.tensor(buffers.conv_weight)?;
        let a = session.tensor(buffers.a)?;
        let b = session.tensor(buffers.b)?;
        let decay = session.tensor(buffers.a_decay)?;
        let bias = session.tensor(buffers.dt_bias)?;
        let z = session.tensor(buffers.z)?;
        let norm = session.tensor(buffers.norm)?;
        let conv = session.tensor_mut(buffers.convolved)?;
        let scan = session.tensor_mut(buffers.scanned)?;
        let output = session.tensor_mut(buffers.output)?;
        for (actual, elements) in [
            (mixed.byte_len(), l * channels),
            (weight.byte_len(), channels * geometry.taps as usize),
            (a.byte_len(), l * heads),
            (b.byte_len(), l * heads),
            (decay.byte_len(), heads),
            (bias.byte_len(), heads),
            (z.byte_len(), l * width),
            (norm.byte_len(), geometry.value_dim as usize),
            (conv.byte_len(), l * channels),
            (scan.byte_len(), l * width),
            (output.byte_len(), l * width),
        ] {
            if elements
                .checked_mul(size_of::<f32>())
                .is_none_or(|bytes| actual < bytes)
            {
                return Err(tenferro_tensor::Error::invalid_argument(
                    "gated_delta.cuda",
                    "allocation",
                    "stage allocation is shorter than the kernel span",
                ));
            }
        }
        let launches = geometry.launches();
        let [length, channels, taps] = geometry.conv_arguments();
        // SAFETY: scalar ABI and geometry follow the compiled source; tensor
        // shape/layout/runtime are checked above. Allocation aliasing and
        // asynchronous lifetime are the caller's documented obligations.
        unsafe {
            session.launch(
                &self.conv_silu,
                launches[0],
                &[
                    KernelArg::tensor_mut(&conv),
                    KernelArg::tensor(&mixed),
                    KernelArg::tensor(&weight),
                    KernelArg::i32(length),
                    KernelArg::i32(channels),
                    KernelArg::i32(taps),
                ],
            )?;
            let mut args = vec![
                KernelArg::tensor_mut(&scan),
                KernelArg::tensor_mut(&conv),
                KernelArg::tensor(&a),
                KernelArg::tensor(&b),
                KernelArg::tensor(&decay),
                KernelArg::tensor(&bias),
            ];
            args.extend(geometry.recurrent_arguments().map(KernelArg::i32));
            session.launch(&self.recurrent, launches[1], &args)?;
            session.launch(
                &self.norm_gate,
                launches[2],
                &[
                    KernelArg::tensor_mut(&output),
                    KernelArg::tensor_mut(&scan),
                    KernelArg::tensor(&z),
                    KernelArg::tensor(&norm),
                    KernelArg::i32(length),
                    KernelArg::i32(geometry.value_dim),
                    KernelArg::f32(eps),
                ],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::RecurrentGeometry;

    #[test]
    fn stage_bindings_reject_wrong_shape() {
        let tensor = tenferro_tensor::TypedTensor::<f32>::zeros(vec![2, 3]).unwrap();
        assert!(super::check_stage_tensor(&tensor, &[2, 3]).is_ok());
        assert!(super::check_stage_tensor(&tensor, &[3, 2]).is_err());
    }

    #[test]
    fn grouped_heads_and_token_boundaries() {
        for length in [1, 63, 64, 65, 127, 128, 129] {
            let geometry = RecurrentGeometry::new(length, 256, 128, 4, 8, 4).unwrap();
            assert_eq!(geometry.conv_arguments(), [length as i32, 3072, 4]);
            let launches = geometry.launches();
            assert_eq!(launches[1].grid, [8, 128, 1]);
            assert_eq!(launches[1].block, [32, 1, 1]);
            assert_eq!(launches[2].grid, [length as u32 * 8, 1, 1]);
            assert_eq!(launches[2].shared_mem_bytes, 512);
            assert!(launches[0].grid[0] * 256 >= length as u32 * 3072);
        }
    }

    #[test]
    fn rejects_integer_overflow_and_unsupported_launches() {
        for dims in [
            [0, 128, 128, 4, 8, 4],
            [1, 257, 128, 4, 8, 4],
            [1, 128, 128, 3, 8, 4],
            [1, 128, 65_536, 4, 8, 4],
            [usize::MAX, 128, 128, 4, 8, 4],
            [1, 128, 128, usize::MAX, usize::MAX, 4],
            [1, 128, 128, 4, 8, usize::MAX],
            [i32::MAX as usize, 1, 1, 1, 1, 1],
        ] {
            let [l, k, v, kh, vh, taps] = dims;
            assert!(RecurrentGeometry::new(l, k, v, kh, vh, taps).is_err());
        }
    }
}
