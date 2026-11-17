//! Execution-device selection for the engines' tenferro runtimes.
//!
//! tenferro 0.11 stores no engine in `EagerRuntime`: every eager session
//! borrows the engine the caller holds. [`DeviceRuntime`] therefore owns the
//! pair (eager runtime + engine) for one device, and the engines keep one of
//! these instead of a bare `Arc<EagerRuntime>`.
//!
//! [`Device`] is always available so callers can name a device without feature
//! gates; constructing a CUDA runtime requires this crate's `cuda` feature
//! (forwarded by `laya-infer/cuda` and `jeff-infer/cuda`) and returns a typed
//! unsupported error otherwise. No device silently falls back to the CPU.

use std::cell::RefCell;
use std::sync::{Arc, Mutex, MutexGuard};

use tenferro_ad::{EagerRuntime, EagerSession};
use tenferro_runtime::ExecutionEngine;

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

    /// Create a fresh eager runtime and its engine on this device.
    ///
    /// Each call creates an independent pair; weight tensors prepared on one
    /// runtime must be used with that runtime only.
    pub fn runtime(self) -> tenferro_ad::Result<Arc<DeviceRuntime>> {
        DeviceRuntime::new(self)
    }
}

/// An eager runtime plus the engine every session on it borrows.
///
/// The engine lives behind a [`Mutex`] because tenferro operations take it as
/// `&mut` while the engines are used through `&self`. One forward run holds the
/// lock for its whole session, which is also the only shape tenferro supports:
/// a second session on the same engine is not re-entrant.
pub struct DeviceRuntime {
    device: Device,
    runtime: Arc<EagerRuntime>,
    engine: Mutex<Engine>,
}

/// The caller-held engine of one device.
///
/// Both engines implement tenferro's [`ExecutionEngine`] and are `Send`, so the
/// pair can live behind one lock and be handed to `with_eager_session` as the
/// `&mut dyn ExecutionEngine` it takes.
type Engine = Box<dyn ExecutionEngine + Send>;

impl DeviceRuntime {
    fn new(device: Device) -> tenferro_ad::Result<Arc<Self>> {
        let runtime = EagerRuntime::new()?;
        let engine = match device {
            Device::Cpu => Box::new(tenferro_infer::cpu_engine()),
            Device::Cuda(ordinal) => cuda_engine(ordinal)?,
        };
        Ok(Arc::new(Self {
            device,
            runtime,
            engine: Mutex::new(engine),
        }))
    }

    /// The device this runtime executes on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The eager runtime, for values that must be created in its context
    /// (`EagerTensor::from_tensor_in`, context ids).
    pub fn eager(&self) -> &Arc<EagerRuntime> {
        &self.runtime
    }

    /// Run one eager session on this device's engine.
    ///
    /// The callback's error type must convert from `tenferro_ad::Error`, so a
    /// single `?` covers the engine lock and the operation.
    pub fn with_eager_session<T, E: From<tenferro_ad::Error>>(
        &self,
        f: impl FnOnce(&mut EagerSession<'_>) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        let _active = self.enter()?;
        let mut engine = self.lock()?;
        let engine: &mut dyn ExecutionEngine = engine.as_mut();
        self.runtime.with_eager_session(engine, f)
    }

    /// Mark this runtime busy on the current thread.
    ///
    /// tenferro's engine is held behind a `Mutex`, which is not re-entrant, so a
    /// session opened inside another session on the same runtime would block
    /// forever. Rejecting it keeps the old `Reentered` behaviour.
    fn enter(&self) -> tenferro_ad::Result<ActiveSession<'_>> {
        let key: *const Self = self;
        ACTIVE.with(|active| {
            if active.borrow().contains(&key) {
                return Err(tenferro_ad::Error::TensorRuntime(
                    tenferro_tensor::Error::runtime_state(
                        "tenferro-ext::DeviceRuntime",
                        "an eager session is already open on this runtime; run the work inside it",
                    ),
                ));
            }
            active.borrow_mut().push(key);
            Ok(())
        })?;
        Ok(ActiveSession {
            key,
            _marker: std::marker::PhantomData,
        })
    }

    fn lock(&self) -> tenferro_ad::Result<MutexGuard<'_, Engine>> {
        self.engine.lock().map_err(|_| {
            tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::unsupported(
                "tenferro-ext::DeviceRuntime",
                "the device engine lock is poisoned",
            ))
        })
    }
}

impl std::fmt::Debug for DeviceRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceRuntime")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

thread_local! {
    /// Runtimes with an open eager session on this thread.
    static ACTIVE: RefCell<Vec<*const DeviceRuntime>> = const { RefCell::new(Vec::new()) };
}

/// Releases the thread-local busy mark when a session ends, panics included.
struct ActiveSession<'a> {
    key: *const DeviceRuntime,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl Drop for ActiveSession<'_> {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if let Some(position) = active.iter().rposition(|key| *key == self.key) {
                active.remove(position);
            }
        });
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
fn cuda_engine(ordinal: usize) -> tenferro_ad::Result<Engine> {
    use tenferro_gpu::cuda::{CudaDeviceId, CudaEngine};
    let ordinal = u32::try_from(ordinal).map_err(|_| {
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            "tenferro-ext::Device",
            "ordinal",
            "CUDA ordinal out of range",
        ))
    })?;
    // `cudarc` panics when the driver library cannot be loaded; report that as
    // a typed backend error instead of unwinding through the engine.
    let engine = std::panic::catch_unwind(|| CudaEngine::open(CudaDeviceId::from_ordinal(ordinal)))
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
    Ok(Box::new(engine))
}

#[cfg(not(feature = "cuda"))]
fn cuda_engine(_ordinal: usize) -> tenferro_ad::Result<Engine> {
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

    #[test]
    fn cpu_session_runs_an_eager_op() {
        let runtime = Device::Cpu.runtime().unwrap();
        let value = runtime
            .with_eager_session(|session| {
                let one = session.constant_from_host(tenferro_ad::Tensor::from_vec_col_major(
                    vec![1],
                    vec![1.0f32],
                )?)?;
                session.add(&one, &one)
            })
            .unwrap();
        assert_eq!(value.value().unwrap().as_slice::<f32>().unwrap(), &[2.0]);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_without_feature_is_unsupported() {
        assert!(Device::Cuda(0).runtime().is_err());
    }
}
