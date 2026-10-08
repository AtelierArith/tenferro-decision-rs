//! Scoped recurrent/chunked layer scheduling with one success-path request fence.
//! Production dispatch remains gated on hardware parity.

use std::rc::Rc;

use tenferro_ad::{EagerSession, EagerTensor, Result};
use tenferro_gpu::cuda::with_cuda_exec_session;

use crate::{
    GatedDeltaConfig, GatedDeltaTensorWeights,
    cuda::CudaKernels,
    cuda_chunked_layer::{CudaChunkedWorkspace, PendingChunkedLayer, enqueue_chunked_layer},
    cuda_layer::{CudaRecurrentWorkspace, PendingRecurrentLayer, enqueue_recurrent_layer},
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
