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
