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

#[cfg(test)]
mod geometry_tests {
    use super::RecurrentGeometry;

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
