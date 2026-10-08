//! Explicit CUDA recurrent layer adapter; production plan dispatch stays gated.
//!
//! Projections use tenferro operations. Raw stages own their operands through
//! completion, then the output projection remains in the eager session.

use tenferro_ad::{DType, EagerSession, EagerTensor, Result};
use tenferro_gpu::cuda::with_cuda_exec_session;
use tenferro_tensor::{Tensor, TensorRead, TypedTensor};

use crate::{
    GatedDeltaConfig, GatedDeltaTensorWeights,
    cuda::{CudaKernels, OwnedStageInputs, RecurrentGeometry, StageBuffers},
    layer::invalid_weights,
    tensor_layer::linear_col,
};

/// Completed, exclusively owned CUDA scratch tensors for reuse at one shape.
///
/// Fields stay private so callers cannot alias mutable workspace buffers.
/// The cached entry point validates shape/runtime through the raw adapter.
pub struct CudaRecurrentWorkspace {
    convolved: TypedTensor<f32>,
    scanned: TypedTensor<f32>,
    output: Tensor,
}

// Local to this function's admitted eager scope: the current execution stream
// remains active until this guard is completed or dropped. No pending owner
// escapes the scope or crosses a Send boundary.
struct PendingLayer {
    runtime: tenferro_gpu::cuda::CudaRuntime,
    kernels: Option<CudaKernels>,
    inputs: Option<OwnedStageInputs>,
    workspace: Option<CudaRecurrentWorkspace>,
    completed: bool,
}

impl Drop for PendingLayer {
    fn drop(&mut self) {
        if !self.completed && self.runtime.synchronize().is_err() {
            std::mem::forget(self.kernels.take());
            std::mem::forget(self.inputs.take());
            std::mem::forget(self.workspace.take());
        }
    }
}

/// Run the f32 CUDA recurrent formulation with prepared device weights/mask.
///
/// `x` is `[hidden, length]`, `mask` is `[length]`. The returned kernel owner
/// can be reused within the admitted eager callback (it is naturally !Send).
/// No tensors are downloaded. This prototype makes device copies of raw-stage
/// operands and allocates workspaces per call; persistent workspace caching is
/// still required. It retains the raw operands through native readout and
/// synchronizes once before returning completed resources.
///
/// This explicit entry point does not enable CUDA in normal plan resolution;
/// hardware parity is still required. Unsupported backends/shapes/dtypes fail.
pub fn delta_layer_cuda_recurrent(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaTensorWeights,
    x: &EagerTensor,
    mask: &EagerTensor,
    kernels: CudaKernels,
) -> Result<(CudaKernels, EagerTensor)> {
    let (kernels, _, output) =
        delta_layer_cuda_recurrent_cached(session, cfg, weights, x, mask, kernels, None)?;
    Ok((kernels, output))
}

/// Run the recurrent adapter with a workspace returned by an earlier call.
///
/// Pass `None` to allocate once, then retain the returned workspace and module
/// within the admitted callback. The workspace must match shape/runtime. Raw
/// operands still use device copies; completion is synchronized once after
/// native readout. The output is copied on device before eager registration, so a
/// later workspace reuse cannot overwrite the returned tensor.
pub fn delta_layer_cuda_recurrent_cached(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaTensorWeights,
    x: &EagerTensor,
    mask: &EagerTensor,
    kernels: CudaKernels,
    workspace: Option<CudaRecurrentWorkspace>,
) -> Result<(CudaKernels, CudaRecurrentWorkspace, EagerTensor)> {
    let invalid = |message| {
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            "delta_layer_cuda_recurrent",
            "inputs",
            message,
        ))
    };
    cfg.validate().map_err(invalid_weights)?;
    let [length] = mask.shape() else {
        return Err(invalid("mask must have shape [length]"));
    };
    let geometry = RecurrentGeometry::new(
        *length,
        cfg.key_dim,
        cfg.value_dim,
        cfg.key_heads,
        cfg.value_heads,
        cfg.conv_taps,
    )
    .map_err(invalid_weights)?;
    if x.shape() != [cfg.hidden, *length] {
        return Err(invalid("x must have shape [hidden, length]"));
    }
    for tensor in [
        x,
        mask,
        &weights.qkv,
        &weights.z,
        &weights.a,
        &weights.b,
        &weights.conv,
        &weights.a_decay,
        &weights.dt_bias,
        &weights.norm,
        &weights.out_proj,
    ] {
        if tensor.dtype() != DType::F32 {
            return Err(tenferro_ad::Error::TensorRuntime(
                tenferro_tensor::Error::unsupported(
                    "delta_layer_cuda_recurrent",
                    "only f32 operands are supported",
                ),
            ));
        }
    }
    if with_cuda_exec_session(session.backend_session(), |_| ()).is_none() {
        return Err(tenferro_ad::Error::TensorRuntime(
            tenferro_tensor::Error::unsupported(
                "delta_layer_cuda_recurrent",
                "CUDA execution session required",
            ),
        ));
    }
    let masked = session.mul(x, mask)?;
    let mut project = |weight| -> Result<tenferro_tensor::TypedTensor<f32>> {
        let projected = linear_col(session, &masked, weight)?;
        let time_first = session.transpose(&projected, &[1, 0])?;
        Ok(session
            .duplicate_value(&time_first)?
            .into_typed::<f32>()
            .map_err(|failure| failure.into_parts().1)?)
    };
    let mixed = project(&weights.qkv)?;
    let z = project(&weights.z)?;
    let a = project(&weights.a)?;
    let b = project(&weights.b)?;
    let conv = session
        .duplicate_value(&weights.conv)?
        .into_typed::<f32>()
        .map_err(|failure| failure.into_parts().1)?;
    let decay = session
        .duplicate_value(&weights.a_decay)?
        .into_typed::<f32>()
        .map_err(|failure| failure.into_parts().1)?;
    let bias = session
        .duplicate_value(&weights.dt_bias)?
        .into_typed::<f32>()
        .map_err(|failure| failure.into_parts().1)?;
    let norm = session
        .duplicate_value(&weights.norm)?
        .into_typed::<f32>()
        .map_err(|failure| failure.into_parts().1)?;
    let channels = 2 * cfg.key_heads * cfg.key_dim + cfg.value_heads * cfg.value_dim;
    let width = cfg.value_heads * cfg.value_dim;
    let runtime = with_cuda_exec_session(session.backend_session(), |cuda| cuda.runtime().clone())
        .expect("CUDA session checked above");
    let mut pending = PendingLayer {
        runtime,
        kernels: Some(kernels),
        inputs: Some([mixed, conv, a, b, decay, bias, z, norm]),
        workspace,
        completed: false,
    };
    if pending.workspace.is_none() {
        pending.workspace = Some(
            with_cuda_exec_session(session.backend_session(), |cuda| {
                cuda.with_raw("cuda_recurrent_workspace", |raw| {
                    Ok(CudaRecurrentWorkspace {
                        convolved: raw.alloc_output::<f32>(&[*length, channels])?,
                        scanned: raw.alloc_output::<f32>(&[*length, width])?,
                        output: Tensor::from_typed(raw.alloc_output::<f32>(&[*length, width])?),
                    })
                })
            })
            .expect("CUDA session checked above")?,
        );
    }
    let launched = with_cuda_exec_session(session.backend_session(), |cuda| {
        cuda.with_raw("cuda_recurrent_layer", |raw| {
            let [mixed_input, conv_weight, a, b, a_decay, dt_bias, z, norm] =
                pending.inputs.as_ref().expect("owned inputs");
            let workspace = pending.workspace.as_mut().expect("owned workspace");
            // SAFETY: operand copies and private workspace buffers are distinct;
            // pending owns all buffers/module before enqueue and retains them
            // through the final native-readout barrier, including errors/unwind.
            unsafe {
                pending
                    .kernels
                    .as_ref()
                    .expect("owned module")
                    .enqueue_stages(
                        raw,
                        &geometry,
                        StageBuffers {
                            mixed_input,
                            conv_weight,
                            a,
                            b,
                            a_decay,
                            dt_bias,
                            z,
                            norm,
                            convolved: &mut workspace.convolved,
                            scanned: &mut workspace.scanned,
                            output: workspace
                                .output
                                .as_typed_mut::<f32>()
                                .expect("private f32 workspace"),
                        },
                        cfg.eps,
                    )
            }
        })
    })
    .expect("CUDA session checked above");
    // Capture every fallible operation after enqueue rather than returning
    // early. Original raw allocations remain owned even if import fails.
    let computed = (|| {
        launched?;
        let output = session
            .backend_session()
            .to_contiguous_read(TensorRead::from_tensor(
                &pending.workspace.as_ref().expect("owned workspace").output,
            ))?;
        let output = session.constant_from(output)?;
        let output = session.transpose(&output, &[1, 0])?;
        linear_col(session, &output, &weights.out_proj)
    })();
    let synchronized = with_cuda_exec_session(session.backend_session(), |cuda| {
        cuda.with_raw("cuda_recurrent_layer_finish", |raw| raw.synchronize())
    })
    .expect("CUDA session checked above");
    if let Err(error) = synchronized {
        // Completion is unknown for readout/copies too. Preserve their output
        // owner while Drop retries the barrier and retains raw resources.
        std::mem::forget(computed);
        return Err(error.into());
    }
    pending.completed = true;
    let output = computed?;
    Ok((
        pending.kernels.take().expect("owned module"),
        pending.workspace.take().expect("owned workspace"),
        output,
    ))
}
