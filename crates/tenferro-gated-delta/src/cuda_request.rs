//! Scoped chunked layer scheduling with one success-path request fence.
//! Production dispatch remains gated on hardware parity.

use std::rc::Rc;

use tenferro_ad::{EagerSession, EagerTensor, Result};
use tenferro_gpu::cuda::with_cuda_exec_session;

use crate::{
    GatedDeltaConfig, GatedDeltaTensorWeights,
    cuda::CudaKernels,
    cuda_chunked_layer::{CudaChunkedWorkspace, PendingChunkedLayer, enqueue_chunked_layer},
};

/// Pending resources local to the callback in [`with_cuda_chunked_request`].
/// No public constructor or owned pending result lets this owner escape.
pub struct CudaChunkedRequest {
    runtime: tenferro_gpu::cuda::CudaRuntime,
    kernels: Option<Rc<CudaKernels>>,
    available: std::vec::IntoIter<CudaChunkedWorkspace>,
    pending: Vec<PendingChunkedLayer>,
    outputs: Vec<EagerTensor>,
    failed: bool,
    completed: bool,
}

fn failed_request() -> tenferro_ad::Error {
    tenferro_tensor::Error::invalid_argument(
        "cuda_chunked_request",
        "state",
        "request contains a failed layer",
    )
    .into()
}

impl CudaChunkedRequest {
    /// Enqueue a complete layer and native readout without a host barrier.
    /// Calls consume cached workspaces in layer order, allocating missing ones.
    /// Supplied workspaces must match each layer's shape/runtime as validated
    /// by the raw adapter. Earlier results stay independent of workspace reuse.
    pub fn layer(
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
        let result = enqueue_chunked_layer(
            session,
            cfg,
            weights,
            x,
            mask,
            self.kernels.as_ref().expect("owned module").clone(),
            self.available.next(),
        );
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

impl Drop for CudaChunkedRequest {
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

/// Run chunked layers inside one admitted eager callback, fencing once after
/// the supplied callback succeeds. Native operations may consume each result
/// immediately on the same stream. Do not download outputs in the callback if
/// a single success-path fence is required.
///
/// The scoped owner is borrowed and naturally !Send. Callback/result bounds
/// prevent exporting it through a Send admission boundary. Completed modules
/// and ordered workspaces return for reuse within the enclosing admitted scope.
/// Errors/panics establish completion or retain resources when it is unknown.
/// This explicit prototype does not enable normal production CUDA dispatch.
pub fn with_cuda_chunked_request<R: Send>(
    session: &mut EagerSession<'_>,
    kernels: CudaKernels,
    workspaces: Vec<CudaChunkedWorkspace>,
    callback: impl FnOnce(&mut CudaChunkedRequest, &mut EagerSession<'_>) -> Result<R> + Send,
) -> Result<(CudaKernels, Vec<CudaChunkedWorkspace>, R)> {
    let runtime = with_cuda_exec_session(session.backend_session(), |cuda| cuda.runtime().clone())
        .ok_or_else(|| {
            tenferro_tensor::Error::unsupported(
                "cuda_chunked_request",
                "CUDA execution session required",
            )
        })?;
    let mut request = CudaChunkedRequest {
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
        .map(PendingChunkedLayer::complete_workspace)
        .collect();
    workspaces.extend(request.available.by_ref());
    let kernels = Rc::try_unwrap(request.kernels.take().expect("owned module"))
        .ok()
        .expect("private module ownership");
    Ok((kernels, workspaces, result))
}
