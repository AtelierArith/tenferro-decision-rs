//! Execution-device selection for the engines' tenferro runtimes.
//!
//! [`Device`] is always available so callers can name a device without
//! feature gates; constructing a CUDA runtime requires this crate's `cuda`
//! feature (forwarded by `laya-infer/cuda` and `jeff-infer/cuda`) and returns a
//! typed unsupported error otherwise. No device silently falls back to the CPU.

use std::sync::Arc;

use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;

/// The device a tenferro runtime executes on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    /// The tenferro CPU provider.
    #[default]
    Cpu,
    /// A CUDA device by process-visible ordinal (after `CUDA_VISIBLE_DEVICES`).
    Cuda(usize),
}

impl Device {
    /// Whether this is a CUDA device.
    pub fn is_cuda(self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Whether CUDA runtimes can be constructed in this build.
    pub fn cuda_compiled() -> bool {
        cfg!(feature = "cuda")
    }

    /// Create a fresh eager runtime on this device.
    ///
    /// Each call creates an independent runtime; weight tensors prepared on
    /// one runtime must be used with that runtime only.
    pub fn runtime(self) -> tenferro_ad::Result<Arc<EagerRuntime>> {
        match self {
            Device::Cpu => EagerRuntime::with_cpu_backend(CpuBackend::new()),
            Device::Cuda(ordinal) => cuda_runtime(ordinal),
        }
    }
}

/// How an engine runs its forward on a CUDA device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CudaPath {
    /// One raw single-stream scope per forward ([`crate::raw_exec`]): cuBLAS
    /// and NVRTC kernels on tenferro's stream, with weights and scratch
    /// resident. Engines fall back to [`Self::Native`] for models the raw
    /// path does not support. The default.
    #[default]
    Raw,
    /// The tenferro-native eager forward (backend-portable).
    Native,
}

impl CudaPath {
    /// [`CudaPath::Raw`], or [`CudaPath::Native`] when
    /// `TENFERRO_DECISION_CUDA_PATH=native`.
    pub fn from_env() -> Self {
        match std::env::var("TENFERRO_DECISION_CUDA_PATH").as_deref() {
            Ok("native") => CudaPath::Native,
            _ => CudaPath::Raw,
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => f.write_str("cpu"),
            Device::Cuda(ordinal) => write!(f, "cuda:{ordinal}"),
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_runtime(ordinal: usize) -> tenferro_ad::Result<Arc<EagerRuntime>> {
    use tenferro_gpu::cuda::{CudaBackend, CudaDeviceId};
    let ordinal = u32::try_from(ordinal).map_err(|_| {
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            "tenferro-ext::Device",
            "ordinal",
            "CUDA ordinal out of range",
        ))
    })?;
    // `cudarc` panics when the driver library cannot be loaded; report that as
    // a typed backend error instead of unwinding through the engine.
    let backend =
        std::panic::catch_unwind(|| CudaBackend::new(CudaDeviceId::from_ordinal(ordinal)))
            .map_err(|_| {
                tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::unsupported(
                    "tenferro-ext::Device",
                    "the CUDA driver library could not be loaded",
                ))
            })?
            .map_err(|error| {
                tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::unsupported(
                    "tenferro-ext::Device",
                    format!("failed to open CUDA device {ordinal}: {error}"),
                ))
            })?;
    EagerRuntime::with_cuda_backend(backend)
}

#[cfg(not(feature = "cuda"))]
fn cuda_runtime(_ordinal: usize) -> tenferro_ad::Result<Arc<EagerRuntime>> {
    Err(tenferro_ad::Error::TensorRuntime(
        tenferro_tensor::Error::unsupported(
            "tenferro-ext::Device",
            "CUDA support is not compiled in; enable the `cuda` feature",
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_runtime_and_display() {
        assert!(Device::Cpu.runtime().is_ok());
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(1).to_string(), "cuda:1");
        assert!(Device::Cuda(0).is_cuda());
        assert!(!Device::default().is_cuda());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_without_feature_is_unsupported() {
        assert!(Device::Cuda(0).runtime().is_err());
    }
}
