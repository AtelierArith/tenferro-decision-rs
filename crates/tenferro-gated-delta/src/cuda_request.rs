//! Scoped recurrent/chunked layer scheduling with one success-path request fence.
//! Production dispatch remains gated on hardware parity.

use std::rc::Rc;

use tenferro_ad::{EagerSession, EagerTensor, Result};
use tenferro_gpu::cuda::with_cuda_exec_session;

use crate::{
    GatedDeltaConfig, GatedDeltaTensorWeights,
    cuda::CudaKernels,
    cuda_chunked_layer::{CudaChunkedWorkspace, PendingChunkedLayer, enqueue_chunked_layer},
    cuda_layer::{
        ActivationLayout, CudaRecurrentWorkspace, PendingRecurrentLayer, enqueue_recurrent_layer,
    },
};

/// Completed layer workspaces in request order. Private buffers remain disjoint.
pub enum CudaLayerWorkspace {
    /// Raw recurrent stages and cached weights.
    Recurrent(Box<CudaRecurrentWorkspace>),
    /// Raw convolution and native chunked scan constants.
    Chunked(Box<CudaChunkedWorkspace>),
}

enum PendingLayer {
    Recurrent(Box<PendingRecurrentLayer>),
    Chunked(Box<PendingChunkedLayer>),
}

impl PendingLayer {
    fn mark_completed(&mut self) {
        match self {
            Self::Recurrent(layer) => layer.mark_completed(),
            Self::Chunked(layer) => layer.mark_completed(),
        }
    }

    fn complete_workspace(&mut self) -> CudaLayerWorkspace {
        match self {
            Self::Recurrent(layer) => {
                CudaLayerWorkspace::Recurrent(Box::new(layer.complete_workspace()))
            }
            Self::Chunked(layer) => {
                CudaLayerWorkspace::Chunked(Box::new(layer.complete_workspace()))
            }
        }
    }
}

/// Pending resources local to the callback in [`with_cuda_request`].
/// No public constructor or owned pending result lets this owner escape.
pub struct CudaRequest {
    runtime: tenferro_gpu::cuda::CudaRuntime,
    kernels: Option<Rc<CudaKernels>>,
    available: std::vec::IntoIter<CudaLayerWorkspace>,
    pending: Vec<PendingLayer>,
    outputs: Vec<EagerTensor>,
    failed: bool,
    completed: bool,
}

fn failed_request() -> tenferro_ad::Error {
    tenferro_tensor::Error::invalid_argument(
        "cuda_request",
        "state",
        "request contains a failed layer",
    )
    .into()
}

impl CudaRequest {
    /// Select recurrent for key widths up to 256, native chunked otherwise.
    /// This prototype's CUDA selection depends only on the configured width.
    pub fn layer(
        &mut self,
        session: &mut EagerSession<'_>,
        cfg: &GatedDeltaConfig,
        weights: &GatedDeltaTensorWeights,
        x: &EagerTensor,
        mask: &EagerTensor,
    ) -> Result<EagerTensor> {
        if cfg.key_dim <= 256 {
            self.recurrent_layer(session, cfg, weights, x, mask)
        } else {
            self.chunked_layer(session, cfg, weights, x, mask)
        }
    }

    /// [`Self::layer`] for a time-first `x [length, hidden]`, returning
    /// `[length, hidden]`, with stacked `(out, in)` projection weights from
    /// [`crate::cuda_layer::prepare_time_first_weights`]. Key widths above 256
    /// (the large-key chunked formulation) are rejected. The recurrent formulation consumes this layout
    /// directly (no transposes); the large-key chunked formulation transposes
    /// around its feature-first adapter.
    pub fn layer_time_first(
        &mut self,
        session: &mut EagerSession<'_>,
        cfg: &GatedDeltaConfig,
        weights: &GatedDeltaTensorWeights,
        x: &EagerTensor,
        mask: &EagerTensor,
    ) -> Result<EagerTensor> {
        if cfg.key_dim <= 256 {
            self.recurrent_layer_with_layout(
                session,
                cfg,
                weights,
                x,
                mask,
                ActivationLayout::TimeFirstStacked,
            )
        } else {
            Err(tenferro_tensor::Error::unsupported(
                "cuda_request",
                "time-first stacked layers support key widths up to 256; use the \
                 feature-first chunked layer for larger keys",
            )
            .into())
        }
    }

    /// Enqueue raw recurrent stages plus native projections/readout without a
    /// host fence. Workspaces are consumed in layer order; a changed formulation
    /// replaces that slot. Matching workspaces must have the current shape/runtime.
    pub fn recurrent_layer(
        &mut self,
        session: &mut EagerSession<'_>,
        cfg: &GatedDeltaConfig,
        weights: &GatedDeltaTensorWeights,
        x: &EagerTensor,
        mask: &EagerTensor,
    ) -> Result<EagerTensor> {
        self.recurrent_layer_with_layout(
            session,
            cfg,
            weights,
            x,
            mask,
            ActivationLayout::FeatureFirst,
        )
    }

    fn recurrent_layer_with_layout(
        &mut self,
        session: &mut EagerSession<'_>,
        cfg: &GatedDeltaConfig,
        weights: &GatedDeltaTensorWeights,
        x: &EagerTensor,
        mask: &EagerTensor,
        layout: ActivationLayout,
    ) -> Result<EagerTensor> {
        if self.failed {
            return Err(failed_request());
        }
        let workspace = match self.available.next() {
            Some(CudaLayerWorkspace::Recurrent(workspace)) => Some(*workspace),
            _ => None,
        };
        let result = enqueue_recurrent_layer(
            session,
            cfg,
            weights,
            x,
            mask,
            self.kernels.as_ref().expect("owned module").clone(),
            workspace,
            layout,
        )
        .map(|(pending, output)| (PendingLayer::Recurrent(Box::new(pending)), output));
        self.retain(result)
    }

    /// Enqueue raw convolution plus native chunked scan/readout without a host
    /// fence. Supports large keys and the same ordered workspace rules.
    pub fn chunked_layer(
        &mut self,
        session: &mut EagerSession<'_>,
        cfg: &GatedDeltaConfig,
        weights: &GatedDeltaTensorWeights,
        x: &EagerTensor,
        mask: &EagerTensor,
    ) -> Result<EagerTensor> {
        if self.failed {
            return Err(failed_request());
        }
        let workspace = match self.available.next() {
            Some(CudaLayerWorkspace::Chunked(workspace)) => Some(*workspace),
            _ => None,
        };
        let result = enqueue_chunked_layer(
            session,
            cfg,
            weights,
            x,
            mask,
            self.kernels.as_ref().expect("owned module").clone(),
            workspace,
        )
        .map(|(pending, output)| (PendingLayer::Chunked(Box::new(pending)), output));
        self.retain(result)
    }

    fn retain(&mut self, result: Result<(PendingLayer, EagerTensor)>) -> Result<EagerTensor> {
        match result {
            Ok((pending, output)) => {
                self.pending.push(pending);
                self.outputs.push(output.clone());
                Ok(output)
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    fn mark_completed(&mut self) {
        self.completed = true;
        for layer in &mut self.pending {
            layer.mark_completed();
        }
    }
}

impl Drop for CudaRequest {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if self.runtime.synchronize().is_ok() {
            self.mark_completed();
        } else {
            // Unknown completion: retain module, raw allocations and returned
            // eager outputs instead of freeing resources used by queued work.
            std::mem::forget(self.kernels.take());
            std::mem::forget(std::mem::take(&mut self.pending));
            std::mem::forget(std::mem::take(&mut self.outputs));
        }
    }
}

/// Run recurrent/chunked layers inside one admitted eager callback, fencing once after
/// the supplied callback succeeds. Native operations may consume each result
/// immediately on the same stream. Do not download outputs in the callback if
/// a single success-path fence is required.
///
/// The scoped owner is borrowed and naturally !Send. Callback/result bounds
/// prevent exporting it through a Send admission boundary. Completed modules
/// and ordered workspaces return for reuse within the enclosing admitted scope.
/// Errors/panics establish completion or retain resources when it is unknown.
/// This explicit prototype does not enable normal production CUDA dispatch.
pub fn with_cuda_request<R: Send>(
    session: &mut EagerSession<'_>,
    kernels: CudaKernels,
    workspaces: Vec<CudaLayerWorkspace>,
    callback: impl FnOnce(&mut CudaRequest, &mut EagerSession<'_>) -> Result<R> + Send,
) -> Result<(CudaKernels, Vec<CudaLayerWorkspace>, R)> {
    let runtime = with_cuda_exec_session(session.backend_session(), |cuda| cuda.runtime().clone())
        .ok_or_else(|| {
            tenferro_tensor::Error::unsupported("cuda_request", "CUDA execution session required")
        })?;
    let mut request = CudaRequest {
        runtime,
        kernels: Some(Rc::new(kernels)),
        available: workspaces.into_iter(),
        pending: Vec::new(),
        outputs: Vec::new(),
        failed: false,
        completed: false,
    };
    let mut result = callback(&mut request, session);
    if request.failed && result.is_ok() {
        result = Err(failed_request());
    }
    if let Err(error) = request.runtime.synchronize() {
        std::mem::forget(result);
        return Err(error.into());
    }
    request.mark_completed();
    let result = result?;
    let mut workspaces: Vec<_> = request
        .pending
        .iter_mut()
        .map(PendingLayer::complete_workspace)
        .collect();
    workspaces.extend(request.available.by_ref());
    let kernels = Rc::try_unwrap(request.kernels.take().expect("owned module"))
        .ok()
        .expect("private module ownership");
    Ok((kernels, workspaces, result))
}

/// Chunked-only facade preserving the explicit chunked request API.
/// The general scoped owner retains all pending resources.
pub struct CudaChunkedRequest<'a> {
    request: &'a mut CudaRequest,
}

impl CudaChunkedRequest<'_> {
    /// Enqueue one chunked layer without an intermediate host fence.
    pub fn layer(
        &mut self,
        session: &mut EagerSession<'_>,
        cfg: &GatedDeltaConfig,
        weights: &GatedDeltaTensorWeights,
        x: &EagerTensor,
        mask: &EagerTensor,
    ) -> Result<EagerTensor> {
        self.request.chunked_layer(session, cfg, weights, x, mask)
    }
}

/// Run a chunked-only request with the same ownership/completion rules as
/// [`with_cuda_request`]. Existing chunked workspace caches remain usable.
pub fn with_cuda_chunked_request<R: Send>(
    session: &mut EagerSession<'_>,
    kernels: CudaKernels,
    workspaces: Vec<CudaChunkedWorkspace>,
    callback: impl FnOnce(&mut CudaChunkedRequest<'_>, &mut EagerSession<'_>) -> Result<R> + Send,
) -> Result<(CudaKernels, Vec<CudaChunkedWorkspace>, R)> {
    let workspaces = workspaces
        .into_iter()
        .map(|workspace| CudaLayerWorkspace::Chunked(Box::new(workspace)))
        .collect();
    let (kernels, workspaces, result) =
        with_cuda_request(session, kernels, workspaces, |request, session| {
            callback(&mut CudaChunkedRequest { request }, session)
        })?;
    let workspaces = workspaces
        .into_iter()
        .map(|workspace| match workspace {
            CudaLayerWorkspace::Chunked(workspace) => *workspace,
            CudaLayerWorkspace::Recurrent(_) => {
                unreachable!("chunked facade creates only chunked workspaces")
            }
        })
        .collect();
    Ok((kernels, workspaces, result))
}

thread_local! {
    // Compiled modules are `!Send`, so they cannot live in an engine that
    // crosses threads or in a `Send` extension cache. CUDA eager sessions run
    // their callback on the calling thread, so a thread-local owner is the
    // narrowest cache that survives across requests. Entries are matched by
    // runtime identity; a module is only taken out while one request runs and
    // is returned after that request's completion fence.
    static KERNEL_CACHE: std::cell::RefCell<Vec<CudaKernels>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Retained modules per thread (distinct CUDA runtimes).
const MAX_CACHED_MODULES: usize = 4;

/// The NVRTC virtual architecture for this session's device:
/// `TENFERRO_CUDA_ARCH` when set, else the device's compute capability.
pub fn cuda_arch(session: &mut EagerSession<'_>) -> Result<String> {
    if let Ok(arch) = std::env::var("TENFERRO_CUDA_ARCH") {
        if !arch.is_empty() {
            return Ok(arch);
        }
    }
    let device =
        with_cuda_exec_session(session.backend_session(), |cuda| cuda.runtime().device_id())
            .ok_or_else(|| {
                tenferro_tensor::Error::unsupported("cuda_arch", "CUDA execution session required")
            })?;
    let devices = tenferro_gpu::cuda::cuda_devices().map_err(|error| {
        tenferro_tensor::Error::unsupported("cuda_arch", format!("device discovery: {error}"))
    })?;
    let info = devices
        .iter()
        .find(|info| info.id() == device)
        .ok_or_else(|| {
            tenferro_tensor::Error::unsupported("cuda_arch", "session device not discovered")
        })?;
    let capability = info.compute_capability();
    Ok(format!("compute_{}{}", capability.major, capability.minor))
}

/// [`with_cuda_request`] using a per-thread compiled module for this runtime.
///
/// The first request on a thread/runtime compiles the kernels with NVRTC
/// ([`cuda_arch`]); later requests reuse the module. A failed request drops its
/// module (after the request's own completion handling), so the next request
/// recompiles. Workspaces are returned for the caller to retain.
pub fn with_cuda_request_cached<R: Send>(
    session: &mut EagerSession<'_>,
    workspaces: Vec<CudaLayerWorkspace>,
    callback: impl FnOnce(&mut CudaRequest, &mut EagerSession<'_>) -> Result<R> + Send,
) -> Result<(Vec<CudaLayerWorkspace>, R)> {
    let identity = with_cuda_exec_session(session.backend_session(), |cuda| {
        cuda.runtime().runtime_identity()
    })
    .ok_or_else(|| {
        tenferro_tensor::Error::unsupported("cuda_request", "CUDA execution session required")
    })?;
    let cached = KERNEL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache
            .iter()
            .position(|kernels| *kernels.runtime_identity() == identity)
            .map(|index| cache.swap_remove(index))
    });
    let kernels = match cached {
        Some(kernels) => kernels,
        None => {
            let arch = cuda_arch(session)?;
            with_cuda_exec_session(session.backend_session(), |cuda| {
                cuda.with_raw("gated_delta_compile", |raw| {
                    CudaKernels::compile(raw, &arch)
                })
            })
            .expect("CUDA session checked above")?
        }
    };
    let (kernels, workspaces, result) = with_cuda_request(session, kernels, workspaces, callback)?;
    KERNEL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= MAX_CACHED_MODULES {
            cache.remove(0);
        }
        cache.push(kernels);
    });
    Ok((workspaces, result))
}
