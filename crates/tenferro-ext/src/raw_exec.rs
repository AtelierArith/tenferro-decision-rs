//! Single-stream raw CUDA execution for the engines' fast device forward
//! (`cuda` feature).
//!
//! The tenferro-native device forward issues ~700 eager ops per call, each
//! with host-side provider bookkeeping, and every raw-kernel scope flushes
//! CubeCL with a host barrier. [`with_raw_exec`] instead runs a whole forward
//! inside one `CudaExecSession::with_raw` scope: raw NVRTC kernels and cuBLAS
//! GEMMs are enqueued on tenferro's own captured stream, inputs are copied in
//! stream order, and the only host barrier is the final download.
//!
//! Engines keep weights and scratch resident in [`DeviceBuffer`]s: plain
//! driver allocations on tenferro's primary context (retained by the
//! buffer), freed when the buffer drops. They bypass CubeCL's memory pool on
//! purpose: that pool is process-global per device and never returns memory
//! to the driver, so gigabytes of engine weights parked in it would stay
//! reserved after the engine is gone. Addresses are passed to kernels as
//! 64-bit scalars, which is ABI-identical to a pointer parameter.
//!
//! Compiled modules and cuBLAS handles are `!Send` / context-bound, so they
//! live in per-thread caches keyed by runtime identity (CUDA eager sessions
//! run their callback on the calling thread). cuBLAS handles are created on
//! the tenferro primary context and intentionally never destroyed (one per
//! thread and runtime; destroying them at thread exit could race context
//! teardown).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use cudarc::cublas::{result as cublas, sys as cublas_sys};
use cudarc::driver::{result as driver, sys as driver_sys};
use tenferro_ad::{EagerSession, Result};
use tenferro_gpu::cuda::raw::{Function, KernelArg, LaunchConfig, Module, NvrtcOptions, Session};
use tenferro_gpu::cuda::{CudaRuntimeIdentity, with_cuda_exec_session};

/// A device address (bytes).
pub type DevPtr = u64;

/// Byte address of float `index` after `base`.
#[inline]
pub fn at(base: DevPtr, index: usize) -> DevPtr {
    base + 4 * index as u64
}

fn backend(op: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::unsupported(op, message.into())
}

fn to_ad(error: tenferro_tensor::Error) -> tenferro_ad::Error {
    tenferro_ad::Error::TensorRuntime(error)
}

/// A dense f32 device allocation on tenferro's primary context (see the
/// module docs). Dropping it frees the memory; the owner must not drop it
/// while enqueued work still uses it (every [`with_raw_exec`] scope ends
/// synchronized).
pub struct DeviceBuffer {
    ptr: driver_sys::CUdeviceptr,
    ctx: driver_sys::CUcontext,
    device: driver_sys::CUdevice,
    floats: usize,
}

// SAFETY: the buffer is a device address plus a retained primary context;
// freeing pushes that context on whichever thread drops it.
unsafe impl Send for DeviceBuffer {}
// SAFETY: shared references expose only the length.
unsafe impl Sync for DeviceBuffer {}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: `ctx` is the primary context retained at allocation, so it
        // is alive; the allocation belongs to it and is no longer in use.
        unsafe {
            if driver_sys::cuCtxPushCurrent_v2(self.ctx).result().is_ok() {
                let _ = driver::free_sync(self.ptr);
                let mut previous = std::ptr::null_mut();
                let _ = driver_sys::cuCtxPopCurrent_v2(&mut previous);
            }
            let _ = driver::primary_ctx::release(self.device);
        }
    }
}

impl DeviceBuffer {
    /// Number of f32 elements.
    pub fn len(&self) -> usize {
        self.floats
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.floats == 0
    }
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("floats", &self.floats)
            .finish()
    }
}

/// A compiled NVRTC module with its functions by name.
pub struct RawModule {
    runtime: CudaRuntimeIdentity,
    source: usize,
    functions: HashMap<&'static str, Function>,
    _module: Module,
}

impl RawModule {
    /// A function of the module.
    pub fn function(&self, name: &str) -> tenferro_tensor::Result<&Function> {
        self.functions
            .get(name)
            .ok_or_else(|| backend("tenferro-ext::raw_exec", format!("no kernel `{name}`")))
    }
}

thread_local! {
    static MODULES: RefCell<Vec<Rc<RawModule>>> = const { RefCell::new(Vec::new()) };
    static BLAS: RefCell<Vec<(CudaRuntimeIdentity, usize)>> = const { RefCell::new(Vec::new()) };
}

const MAX_CACHED_MODULES: usize = 16;

/// cuBLAS transpose flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Use the matrix as stored (column-major).
    N,
    /// Use its transpose.
    T,
}

impl Op {
    fn sys(self) -> cublas_sys::cublasOperation_t {
        match self {
            Op::N => cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            Op::T => cublas_sys::cublasOperation_t::CUBLAS_OP_T,
        }
    }
}

/// A kernel argument.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    /// A device address.
    P(DevPtr),
    /// A 32-bit signed integer.
    I(i32),
    /// A 64-bit signed integer.
    L(i64),
    /// A 32-bit float.
    F(f32),
}

/// One strided-batched GEMM operand: base address, leading dimension and
/// batch stride (floats).
#[derive(Clone, Copy, Debug)]
pub struct Mat {
    /// Base device address.
    pub ptr: DevPtr,
    /// Leading dimension (floats).
    pub ld: usize,
    /// Batch stride (floats).
    pub stride: usize,
}

impl Mat {
    /// An operand with leading dimension `ld` and batch stride `stride`.
    pub fn new(ptr: DevPtr, ld: usize, stride: usize) -> Self {
        Self { ptr, ld, stride }
    }
}

/// The execution context of one raw scope (see the module docs).
pub struct RawExec<'r, 's> {
    raw: &'r Session<'s>,
    stream: u64,
    blas: cublas_sys::cublasHandle_t,
    arch: &'r str,
}

impl<'r, 's> RawExec<'r, 's> {
    /// The NVRTC architecture of this device (`compute_XY`).
    pub fn arch(&self) -> &str {
        self.arch
    }

    /// Allocate a dense f32 buffer of `floats` elements (uninitialized) on
    /// the scope's primary context.
    pub fn alloc(&self, floats: usize) -> tenferro_tensor::Result<DeviceBuffer> {
        let fail = |what: &str, error: driver::DriverError| {
            backend("tenferro-ext::raw_exec::alloc", format!("{what}: {error}"))
        };
        // SAFETY: the raw scope keeps tenferro's primary context current.
        unsafe {
            let mut device = 0;
            driver_sys::cuCtxGetDevice(&mut device)
                .result()
                .map_err(|e| fail("cuCtxGetDevice", e))?;
            let ctx = driver::primary_ctx::retain(device).map_err(|e| fail("retain", e))?;
            match driver::malloc_sync(4 * floats.max(1)) {
                Ok(ptr) => Ok(DeviceBuffer {
                    ptr,
                    ctx,
                    device,
                    floats,
                }),
                Err(error) => {
                    let _ = driver::primary_ctx::release(device);
                    Err(fail(&format!("cuMemAlloc of {} bytes", 4 * floats), error))
                }
            }
        }
    }

    /// The device address of `buffer`.
    pub fn ptr(&self, buffer: &DeviceBuffer) -> tenferro_tensor::Result<DevPtr> {
        Ok(buffer.ptr as DevPtr)
    }

    /// Compile (once per thread and runtime) and return the module of
    /// `source` with the listed kernels. `source` must be `'static` (its
    /// address keys the cache).
    pub fn module(
        &self,
        source: &'static str,
        kernels: &[&'static str],
    ) -> tenferro_tensor::Result<Rc<RawModule>> {
        let runtime = self.raw.runtime_identity();
        let key = source.as_ptr() as usize;
        if let Some(module) = MODULES.with(|cache| {
            cache
                .borrow()
                .iter()
                .find(|m| m.runtime == runtime && m.source == key)
                .cloned()
        }) {
            return Ok(module);
        }
        let options = NvrtcOptions {
            arch: Some(self.arch.to_string()),
            std: Some("c++14".into()),
            extra: vec![],
        };
        let module = self.raw.compile_nvrtc(source, &options)?;
        let mut functions = HashMap::new();
        for name in kernels {
            functions.insert(*name, module.function(name)?);
        }
        let module = Rc::new(RawModule {
            runtime,
            source: key,
            functions,
            _module: module,
        });
        MODULES.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() >= MAX_CACHED_MODULES {
                cache.remove(0);
            }
            cache.push(module.clone());
        });
        Ok(module)
    }

    /// Enqueue `function` with grid/block geometry and dynamic shared memory.
    ///
    /// # Safety
    ///
    /// `args` must match the kernel's parameter list, and every address must
    /// stay valid (and in range for the launch) until the scope synchronizes.
    pub unsafe fn launch(
        &self,
        function: &Function,
        grid: (u32, u32, u32),
        block: u32,
        shared: u32,
        args: &[Arg],
    ) -> tenferro_tensor::Result<()> {
        let args: Vec<KernelArg<'_>> = args
            .iter()
            .map(|arg| match *arg {
                Arg::P(ptr) => KernelArg::u64(ptr),
                Arg::I(value) => KernelArg::i32(value),
                Arg::L(value) => KernelArg::i64(value),
                Arg::F(value) => KernelArg::f32(value),
            })
            .collect();
        let config = LaunchConfig {
            grid: [grid.0.max(1), grid.1.max(1), grid.2.max(1)],
            block: [block, 1, 1],
            shared_mem_bytes: shared,
        };
        // SAFETY: forwarded caller contract.
        unsafe { self.raw.launch(function, config, &args) }
    }

    /// `C = alpha op(A) op(B) + beta C` for column-major `C` `(m, n)`.
    ///
    /// # Safety
    ///
    /// Operand addresses and leading dimensions must describe in-range device
    /// memory that stays valid until the scope synchronizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm(
        &self,
        op_a: Op,
        op_b: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: DevPtr,
        lda: usize,
        b: DevPtr,
        ldb: usize,
        beta: f32,
        c: DevPtr,
        ldc: usize,
    ) -> tenferro_tensor::Result<()> {
        if m == 0 || n == 0 {
            return Ok(());
        }
        // SAFETY: forwarded caller contract; alpha/beta are host scalars
        // (pointer mode host, the handle default).
        unsafe {
            cublas::sgemm(
                self.blas,
                op_a.sys(),
                op_b.sys(),
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a as *const f32,
                lda.max(1) as i32,
                b as *const f32,
                ldb.max(1) as i32,
                &beta,
                c as *mut f32,
                ldc.max(1) as i32,
            )
        }
        .map_err(|error| backend("tenferro-ext::raw_exec::gemm", error.to_string()))
    }

    /// Strided-batched [`Self::gemm`].
    ///
    /// # Safety
    ///
    /// As [`Self::gemm`], for every batch member.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_batched(
        &self,
        op_a: Op,
        op_b: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: Mat,
        b: Mat,
        beta: f32,
        c: Mat,
        batch: usize,
    ) -> tenferro_tensor::Result<()> {
        if m == 0 || n == 0 || batch == 0 {
            return Ok(());
        }
        // SAFETY: forwarded caller contract.
        unsafe {
            cublas::sgemm_strided_batched(
                self.blas,
                op_a.sys(),
                op_b.sys(),
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a.ptr as *const f32,
                a.ld.max(1) as i32,
                a.stride as i64,
                b.ptr as *const f32,
                b.ld.max(1) as i32,
                b.stride as i64,
                &beta,
                c.ptr as *mut f32,
                c.ld.max(1) as i32,
                c.stride as i64,
                batch as i32,
            )
        }
        .map_err(|error| backend("tenferro-ext::raw_exec::gemm_batched", error.to_string()))
    }

    /// Copy host data to `dst` in stream order. Pageable sources are staged
    /// by the driver before this returns, so `data` may be dropped after.
    ///
    /// # Safety
    ///
    /// `dst` must have room for `data`.
    pub unsafe fn upload<T: Copy>(&self, dst: DevPtr, data: &[T]) -> tenferro_tensor::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: forwarded caller contract.
        unsafe { driver::memcpy_htod_async(dst, data, self.stream as driver_sys::CUstream) }
            .map_err(|error| backend("tenferro-ext::raw_exec::upload", error.to_string()))
    }

    /// Copy `src` into `out` after all enqueued work and wait for it.
    ///
    /// # Safety
    ///
    /// `src` must hold at least `out.len()` elements.
    pub unsafe fn download<T: Copy>(
        &self,
        out: &mut [T],
        src: DevPtr,
    ) -> tenferro_tensor::Result<()> {
        if !out.is_empty() {
            // SAFETY: forwarded caller contract.
            unsafe { driver::memcpy_dtoh_async(out, src, self.stream as driver_sys::CUstream) }
                .map_err(|error| backend("tenferro-ext::raw_exec::download", error.to_string()))?;
        }
        self.synchronize()
    }

    /// Fill `floats` floats at `dst` with zero bytes, in stream order.
    ///
    /// # Safety
    ///
    /// `dst` must have room for `floats` floats.
    pub unsafe fn zero(&self, dst: DevPtr, floats: usize) -> tenferro_tensor::Result<()> {
        if floats == 0 {
            return Ok(());
        }
        // SAFETY: forwarded caller contract.
        unsafe { driver::memset_d8_async(dst, 0, 4 * floats, self.stream as driver_sys::CUstream) }
            .map_err(|error| backend("tenferro-ext::raw_exec::zero", error.to_string()))
    }

    /// Block until every enqueued operation has completed.
    pub fn synchronize(&self) -> tenferro_tensor::Result<()> {
        // SAFETY: the stream is the scope's captured tenferro stream.
        unsafe { driver::stream::synchronize(self.stream as driver_sys::CUstream) }
            .map_err(|error| backend("tenferro-ext::raw_exec::synchronize", error.to_string()))
    }
}

/// The NVRTC virtual architecture: `TENFERRO_CUDA_ARCH` when set, else the
/// session device's compute capability.
fn cuda_arch(session: &mut EagerSession<'_>) -> Result<String> {
    if let Ok(arch) = std::env::var("TENFERRO_CUDA_ARCH") {
        if !arch.is_empty() {
            return Ok(arch);
        }
    }
    thread_local! {
        static ARCH: RefCell<Vec<(CudaRuntimeIdentity, String)>> = const { RefCell::new(Vec::new()) };
    }
    let (device, identity) = with_cuda_exec_session(session.backend_session(), |cuda| {
        (
            cuda.runtime().device_id(),
            cuda.runtime().runtime_identity(),
        )
    })
    .ok_or_else(|| {
        to_ad(backend(
            "tenferro-ext::raw_exec",
            "CUDA execution session required",
        ))
    })?;
    if let Some(arch) = ARCH.with(|cache| {
        cache
            .borrow()
            .iter()
            .find(|(id, _)| *id == identity)
            .map(|(_, arch)| arch.clone())
    }) {
        return Ok(arch);
    }
    let devices = tenferro_gpu::cuda::cuda_devices().map_err(|error| {
        to_ad(backend(
            "tenferro-ext::raw_exec",
            format!("device discovery: {error}"),
        ))
    })?;
    let info = devices
        .iter()
        .find(|info| info.id() == device)
        .ok_or_else(|| {
            to_ad(backend(
                "tenferro-ext::raw_exec",
                "session device not discovered",
            ))
        })?;
    let capability = info.compute_capability();
    let arch = format!("compute_{}{}", capability.major, capability.minor);
    ARCH.with(|cache| cache.borrow_mut().push((identity, arch.clone())));
    Ok(arch)
}

fn blas_handle(raw: &Session<'_>) -> tenferro_tensor::Result<cublas_sys::cublasHandle_t> {
    let identity = raw.runtime_identity();
    if let Some(handle) = BLAS.with(|cache| {
        cache
            .borrow()
            .iter()
            .find(|(id, _)| *id == identity)
            .map(|(_, handle)| *handle)
    }) {
        return Ok(handle as cublas_sys::cublasHandle_t);
    }
    let handle = cublas::create_handle()
        .map_err(|error| backend("tenferro-ext::raw_exec", format!("cublasCreate: {error}")))?;
    BLAS.with(|cache| cache.borrow_mut().push((identity, handle as usize)));
    Ok(handle)
}

/// Run `f` in one raw scope on the session's CUDA stream (see the module
/// docs). Errors when the session is not a CUDA session.
///
/// `f` must leave no work in flight that references memory it does not own
/// past its return: end with [`RawExec::download`] or
/// [`RawExec::synchronize`] (this function synchronizes again on success and
/// on error, so retained buffers may be dropped by the caller afterwards).
pub fn with_raw_exec<R>(
    session: &mut EagerSession<'_>,
    label: &'static str,
    f: impl FnOnce(&mut RawExec<'_, '_>) -> tenferro_tensor::Result<R>,
) -> Result<R> {
    let arch = cuda_arch(session)?;
    let result = with_cuda_exec_session(session.backend_session(), |cuda| {
        cuda.with_raw(label, |raw| {
            let blas = blas_handle(raw)?;
            // SAFETY: the stream handle is used only inside this scope.
            let stream = unsafe { raw.stream().raw_handle() };
            // SAFETY: valid handle and the scope's captured stream.
            unsafe { cublas::set_stream(blas, stream as cublas_sys::cudaStream_t) }
                .map_err(|error| backend(label, format!("cublasSetStream: {error}")))?;
            let mut exec = RawExec {
                raw,
                stream,
                blas,
                arch: &arch,
            };
            let result = f(&mut exec);
            let synced = exec.synchronize();
            match result {
                Ok(value) => synced.map(|()| value),
                Err(error) => Err(error),
            }
        })
    })
    .ok_or_else(|| to_ad(backend(label, "CUDA execution session required")))?;
    result.map_err(to_ad)
}

/// Alignment of arena entries (floats; 256 bytes, enough for vector loads
/// and cuBLAS's preferred operand alignment).
pub const ARENA_ALIGN: usize = 64;

/// Round `floats` up to [`ARENA_ALIGN`].
#[inline]
pub fn align(floats: usize) -> usize {
    floats.div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

/// Host staging of one weight arena: tensors are appended at aligned
/// offsets, then uploaded as one device allocation ([`Self::upload`]).
#[derive(Default)]
pub struct ArenaBuilder {
    data: Vec<f32>,
}

impl ArenaBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `values` and return their float offset.
    pub fn push(&mut self, values: &[f32]) -> usize {
        let offset = self.data.len();
        self.data.extend_from_slice(values);
        self.data.resize(align(self.data.len()), 0.0);
        offset
    }

    /// Append the transpose of column-major `(rows, cols)` `values` (i.e.
    /// store it column-major as `(cols, rows)`) and return its offset.
    pub fn push_transposed(&mut self, values: &[f32], rows: usize, cols: usize) -> usize {
        debug_assert_eq!(values.len(), rows * cols);
        use rayon::prelude::*;
        let offset = self.data.len();
        self.data.resize(offset + rows * cols, 0.0);
        const BLOCK: usize = 64;
        // Each task fills BLOCK destination columns (source rows r0..).
        self.data[offset..]
            .par_chunks_mut(cols * BLOCK)
            .enumerate()
            .for_each(|(block, dst)| {
                let r0 = block * BLOCK;
                let r1 = (r0 + BLOCK).min(rows);
                for c0 in (0..cols).step_by(BLOCK) {
                    for c in c0..(c0 + BLOCK).min(cols) {
                        for r in r0..r1 {
                            dst[c + cols * (r - r0)] = values[r + rows * c];
                        }
                    }
                }
            });
        self.data.resize(align(self.data.len()), 0.0);
        offset
    }

    /// Append parts given row-major `(inp, out_k)` (i.e. column-major
    /// `(out_k, inp)`), stacked along the output axis into one column-major
    /// `(sum out_k, inp)` matrix, and return its offset.
    pub fn push_stacked(&mut self, parts: &[(&[f32], usize)], inp: usize) -> usize {
        let offset = self.data.len();
        let rows: usize = parts.iter().map(|(_, out)| out).sum();
        self.data.reserve(rows * inp);
        for i in 0..inp {
            for (values, out) in parts {
                debug_assert_eq!(values.len(), out * inp);
                self.data.extend_from_slice(&values[i * out..(i + 1) * out]);
            }
        }
        self.data.resize(align(self.data.len()), 0.0);
        offset
    }

    /// The staged floats (aligned padding included).
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Floats staged so far.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether nothing is staged.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Allocate a device buffer and copy the staged data into it.
    pub fn upload(self, exec: &RawExec<'_, '_>) -> tenferro_tensor::Result<DeviceBuffer> {
        let buffer = exec.alloc(self.data.len())?;
        let ptr = exec.ptr(&buffer)?;
        // SAFETY: the buffer holds exactly `data.len()` floats; pageable
        // uploads are staged before returning, so `data` may be dropped.
        unsafe { exec.upload(ptr, &self.data)? };
        exec.synchronize()?;
        Ok(buffer)
    }
}
