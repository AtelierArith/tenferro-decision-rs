//! Local NVRTC/launch adapter over tenferro-gpu's public CUDA seam.
//!
//! tenferro 0.11 removed the raw `Module` / `Function` / `NvrtcOptions` /
//! `KernelArg` / `LaunchConfig` types and `Session::{compile_nvrtc, launch,
//! runtime_identity}`. The replacement contract is that a downstream crate owns
//! its own NVRTC compilation and driver launches through the re-exported
//! `cudarc` bindings, using tenferro's stream and device buffers
//! (`cuda::raw::Session`). This module restores the small surface the decision
//! engines' raw CUDA paths need, so kernel sources, launch geometry, and ABIs
//! are unchanged.
//!
//! The old `with_cuda_exec_session(session, |cuda| ...)` entry keeps its shape:
//! [`with_cuda_exec_session`] visits the caller-held [`CudaEngine`] and lends a
//! [`CudaExecSession`] whose `with_raw` opens tenferro's raw scope.

use std::ffi::{CString, c_void};

use tenferro_gpu::cuda::cudarc::driver::result as cu_result;
use tenferro_gpu::cuda::cudarc::driver::sys as cu_sys;
use tenferro_gpu::cuda::cudarc::nvrtc;
use tenferro_gpu::cuda::raw::{self, Session as RawSession, StreamRef, TensorMut, TensorRef};
use tenferro_gpu::cuda::{CudaEngine, CudaRuntime, CudaRuntimeIdentity};
use tenferro_tensor::{TensorScalar, TypedTensor};

const OP: &str = "tenferro-ext::cuda";

fn backend_error(error: impl std::error::Error + Send + Sync + 'static) -> tenferro_tensor::Error {
    tenferro_tensor::Error::backend_source(OP, error)
}

/// NVRTC options for [`CudaSession::compile_nvrtc`].
#[derive(Clone, Debug, Default)]
pub struct NvrtcOptions {
    /// Virtual architecture, for example `compute_80`.
    pub arch: Option<String>,
    /// `--std=` value, for example `c++14`.
    pub std: Option<String>,
    /// Extra compiler flags, verbatim.
    pub extra: Vec<String>,
}

/// A loaded CUDA module owned by the caller.
///
/// Dropping it unloads the module, so every [`Function`] taken from it must be
/// dropped first; the kernel caches in this workspace keep both together.
pub struct Module {
    module: cu_sys::CUmodule,
}

impl Module {
    /// Resolve one kernel entry point by name.
    pub fn function(&self, name: &str) -> tenferro_tensor::Result<Function> {
        let symbol = CString::new(name).map_err(|_| {
            tenferro_tensor::Error::invalid_argument(OP, "name", "kernel name contains a NUL")
        })?;
        // SAFETY: `self.module` is loaded and stays alive for `self`'s borrow.
        let function = unsafe { cu_result::module::get_function(self.module, symbol) }
            .map_err(backend_error)?;
        Ok(Function { function })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // SAFETY: the handle came from `cuModuleLoadData` and is unloaded
            // exactly once here.
            unsafe {
                let _ = cu_sys::cuModuleUnload(self.module);
            }
        }
    }
}

/// One kernel entry point resolved from a [`Module`].
#[derive(Clone, Copy)]
pub struct Function {
    function: cu_sys::CUfunction,
}

/// Launch geometry for one kernel.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem_bytes: u32,
}

/// One scalar or buffer kernel argument.
///
/// The device pointer must stay valid until the launch completes; that is part
/// of [`CudaSession::launch`]'s contract.
#[derive(Clone, Copy)]
pub enum KernelArg {
    /// A device address (`void*` ABI), e.g. from [`Self::tensor`].
    Pointer(*mut c_void),
    /// An `i32` scalar.
    I32(i32),
    /// An `i64` scalar.
    I64(i64),
    /// An `f32` scalar.
    F32(f32),
}

impl KernelArg {
    /// A read-only device buffer.
    pub fn tensor(reference: &TensorRef<'_, f32>) -> Self {
        Self::Pointer(reference.raw_ptr())
    }

    /// A writable device buffer.
    pub fn tensor_mut(reference: &TensorMut<'_, f32>) -> Self {
        Self::Pointer(reference.raw_ptr())
    }

    /// A device address carried as a `u64` (the raw-cache ABI's pointer form).
    pub fn u64(address: u64) -> Self {
        Self::Pointer(address as usize as *mut c_void)
    }

    /// An `i32` scalar.
    pub fn i32(value: i32) -> Self {
        Self::I32(value)
    }

    /// An `i64` scalar.
    pub fn i64(value: i64) -> Self {
        Self::I64(value)
    }

    /// An `f32` scalar.
    pub fn f32(value: f32) -> Self {
        Self::F32(value)
    }
}

/// tenferro's raw scope plus the runtime identity the kernels were compiled on.
pub struct CudaSession<'a, 's> {
    raw: &'a RawSession<'s>,
    identity: CudaRuntimeIdentity,
}

impl<'a, 's> CudaSession<'a, 's> {
    fn new(raw: &'a RawSession<'s>, identity: CudaRuntimeIdentity) -> Self {
        Self { raw, identity }
    }

    /// The runtime identity of the engine this session belongs to.
    pub fn runtime_identity(&self) -> &CudaRuntimeIdentity {
        &self.identity
    }

    /// The underlying tenferro raw session.
    pub fn raw(&self) -> &RawSession<'s> {
        self.raw
    }

    /// This engine's stream, for vendor calls to bind.
    pub fn stream(&self) -> StreamRef {
        self.raw.stream()
    }

    /// Compile `source` with NVRTC and load it on this session's context.
    pub fn compile_nvrtc(
        &self,
        source: &str,
        options: &NvrtcOptions,
    ) -> tenferro_tensor::Result<Module> {
        // `CompileOptions::arch` is `Option<&'static str>` because NVRTC reads
        // the string after the call returns. A compile happens once per engine,
        // so the leak is bounded by the number of loaded architectures.
        let arch = options
            .arch
            .as_deref()
            .map(|value| &*Box::leak(value.to_string().into_boxed_str()));
        let mut flags = Vec::new();
        if let Some(std) = &options.std {
            flags.push(format!("--std={std}"));
        }
        flags.extend(options.extra.iter().cloned());
        let compile_options = nvrtc::CompileOptions {
            arch,
            options: flags,
            ..Default::default()
        };
        let ptx = nvrtc::compile_ptx_with_opts(source, compile_options).map_err(backend_error)?;
        let image = CString::new(ptx.to_src()).map_err(|_| {
            tenferro_tensor::Error::invalid_argument(OP, "source", "PTX contains a NUL")
        })?;
        // SAFETY: the image is a NUL-terminated PTX text and outlives the call.
        // tenferro's raw scope keeps this runtime's context current, so the
        // module loads into the context the stream belongs to.
        let module = unsafe { cu_result::module::load_data(image.as_ptr() as *const c_void) }
            .map_err(backend_error)?;
        Ok(Module { module })
    }

    /// A read-only device view of `tensor`.
    pub fn tensor<T: TensorScalar>(
        &self,
        tensor: &TypedTensor<T>,
    ) -> tenferro_tensor::Result<TensorRef<'_, T>> {
        self.raw.tensor(tensor)
    }

    /// A writable device view of `tensor`.
    pub fn tensor_mut<T: TensorScalar>(
        &self,
        tensor: &TypedTensor<T>,
    ) -> tenferro_tensor::Result<TensorMut<'_, T>> {
        self.raw.tensor_mut(tensor)
    }

    /// A zero-filled device output in this runtime's domain.
    pub fn alloc_output<T: TensorScalar>(
        &self,
        shape: &[usize],
    ) -> tenferro_tensor::Result<TypedTensor<T>> {
        self.raw.alloc_output::<T>(shape)
    }

    /// Prove `tensor` is one of this runtime's allocations.
    ///
    /// # Safety
    ///
    /// The address stays valid only while `tensor`'s allocation is resident.
    pub unsafe fn retain_tensor<T: TensorScalar>(
        &self,
        tensor: &TypedTensor<T>,
        op: &'static str,
    ) -> tenferro_tensor::Result<raw::DeviceBytes<'static>> {
        // SAFETY: forwarded to the caller's contract.
        unsafe { self.raw.retain_tensor(tensor, op) }
    }

    /// Wait for work enqueued on this session's stream.
    pub fn synchronize(&self) -> tenferro_tensor::Result<()> {
        self.raw.synchronize()
    }

    /// Enqueue `function` on this session's stream.
    ///
    /// # Safety
    ///
    /// Every buffer must be resident, disjoint where the kernel requires it, and
    /// retained until stream completion. The argument list must match the
    /// kernel's ABI exactly.
    pub unsafe fn launch(
        &self,
        function: &Function,
        config: LaunchConfig,
        args: &[KernelArg],
    ) -> tenferro_tensor::Result<()> {
        // `cuLaunchKernel` takes, for every parameter, the address of a value
        // whose size the kernel ABI decides: eight bytes for a device pointer
        // and the low four bytes for an `i32`/`f32`. One aligned slot per
        // argument keeps every address stable (the vectors are pre-sized, so
        // pushing never reallocates) and every parameter value correctly
        // *pointed at* rather than passed by value.
        let mut pointers: Vec<*mut c_void> = Vec::with_capacity(args.len());
        let mut slots: Vec<u64> = Vec::with_capacity(args.len());
        for arg in args {
            // Write the value's own byte image at offset 0 so the parameter is
            // correct on either endianness; the slot's 8-byte alignment is what
            // the driver needs.
            let mut bytes = [0u8; 8];
            match arg {
                KernelArg::Pointer(pointer) => {
                    let address = *pointer as usize;
                    bytes[..std::mem::size_of::<usize>()].copy_from_slice(&address.to_ne_bytes());
                }
                KernelArg::I32(value) => bytes[..4].copy_from_slice(&value.to_ne_bytes()),
                KernelArg::I64(value) => bytes.copy_from_slice(&value.to_ne_bytes()),
                KernelArg::F32(value) => bytes[..4].copy_from_slice(&value.to_ne_bytes()),
            }
            slots.push(u64::from_ne_bytes(bytes));
            let slot: *mut c_void =
                std::ptr::from_mut(slots.last_mut().expect("pre-sized slot")).cast();
            pointers.push(slot);
        }
        let stream = self.raw.stream().raw_handle() as cu_sys::CUstream;
        // SAFETY: forwarded to the caller's contract.
        unsafe {
            cu_result::launch_kernel(
                function.function,
                (config.grid[0], config.grid[1], config.grid[2]),
                (config.block[0], config.block[1], config.block[2]),
                config.shared_mem_bytes,
                stream,
                &mut pointers,
            )
        }
        .map_err(backend_error)
    }
}

/// tenferro's caller-held CUDA engine, in the shape the raw paths expect.
pub struct CudaExecSession<'a> {
    engine: &'a mut CudaEngine,
    identity: CudaRuntimeIdentity,
}

impl CudaExecSession<'_> {
    /// The engine's runtime.
    pub fn runtime(&self) -> &CudaRuntime {
        self.engine.runtime()
    }

    /// Run `f` inside this engine's raw CUDA scope.
    pub fn with_raw<R>(
        &mut self,
        op: &'static str,
        f: impl FnOnce(&mut CudaSession<'_, '_>) -> tenferro_tensor::Result<R>,
    ) -> tenferro_tensor::Result<R> {
        let identity = self.identity.clone();
        self.engine
            .with_raw(op, |raw| f(&mut CudaSession::new(raw, identity)))
    }
}

/// Visit the CUDA engine behind an admitted eager session, if it is one.
///
/// Returns `None` for any other engine, so callers keep reporting their own
/// typed "CUDA execution session required" error.
pub fn with_cuda_exec_session<R>(
    engine: &mut dyn tenferro_tensor::Engine,
    f: impl FnOnce(&mut CudaExecSession<'_>) -> R,
) -> Option<R> {
    raw::with_cuda_engine(engine, |cuda| {
        let identity = cuda.runtime().identity().clone();
        f(&mut CudaExecSession {
            engine: cuda,
            identity,
        })
    })
}

/// Visit the caller-held CUDA engine without opening a raw scope.
pub fn with_cuda_engine<R>(
    engine: &mut dyn tenferro_tensor::Engine,
    f: impl FnOnce(&mut CudaEngine) -> R,
) -> Option<R> {
    raw::with_cuda_engine(engine, f)
}
