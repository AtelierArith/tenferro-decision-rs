//! Raw CUDA convolution with native head-batched chunked scan and readout.
//! Normal production CUDA plan dispatch remains gated on hardware parity.

use std::rc::Rc;

use tenferro_ad::{DType, EagerSession, EagerTensor, Result};
use tenferro_gpu::cuda::with_cuda_exec_session;
use tenferro_tensor::{Tensor, TensorRead, TypedTensor};

use crate::{
    GatedDeltaConfig, GatedDeltaTensorWeights, ProjectedDeltaTensors,
    cuda::{ConvGeometry, CudaKernels},
    cuda_layer::same_raw_weights,
    delta_layer_from_projected_with_constants,
    layer::invalid_weights,
    prepare_native_constants,
    tensor_layer::linear_col,
};

/// Exclusive convolution workspace and prepared weight copy for one shape.
pub struct CudaChunkedWorkspace {
    convolved: Tensor,
    weight: TypedTensor<f32>,
    source: EagerTensor,
    constants: crate::NativeDeltaConstants,
}

// Never escapes this function's admitted eager scope while work is pending.
pub(crate) struct PendingChunkedLayer {
    runtime: tenferro_gpu::cuda::CudaRuntime,
    kernels: Option<Rc<CudaKernels>>,
    input: Option<TypedTensor<f32>>,
    workspace: Option<CudaChunkedWorkspace>,
    completed: bool,
}

impl Drop for PendingChunkedLayer {
    fn drop(&mut self) {
        if !self.completed && self.runtime.synchronize().is_err() {
            std::mem::forget(self.kernels.take());
            std::mem::forget(self.input.take());
            std::mem::forget(self.workspace.take());
        }
    }
}

/// Compose CUDA convolution with native normalization, chunked solve/readout.
///
/// Supports large keys through native triangular solve, without CPU fallback.
/// `x` is `[hidden, length]`, `mask` is `[length]`. Pass `None` initially, then
/// reuse the returned module and workspace within the admitted eager callback.
/// All intermediates remain on device; success synchronizes once after readout.
/// A device copy separates returned results from reusable convolution storage.
/// The scoped request API can defer chunked-layer fences; production integration
/// and actual hardware parity are still pending.
pub fn delta_layer_cuda_chunked_cached(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaTensorWeights,
    x: &EagerTensor,
    mask: &EagerTensor,
    kernels: CudaKernels,
    workspace: Option<CudaChunkedWorkspace>,
) -> Result<(CudaKernels, CudaChunkedWorkspace, EagerTensor)> {
    let kernels = Rc::new(kernels);
    let (mut pending, output) =
        enqueue_chunked_layer(session, cfg, weights, x, mask, kernels.clone(), workspace)?;
    let synchronized = with_cuda_exec_session(session.backend_session(), |cuda| {
        cuda.with_raw("cuda_chunked_layer_finish", |raw| raw.synchronize())
    })
    .expect("CUDA session checked by enqueue");
    if let Err(error) = synchronized {
        std::mem::forget(output);
        return Err(error.into());
    }
    let workspace = pending.complete_workspace();
    Ok((
        Rc::try_unwrap(kernels)
            .ok()
            .expect("private module ownership"),
        workspace,
        output,
    ))
}

impl PendingChunkedLayer {
    /// Caller establishes stream completion before recovering resources.
    pub(crate) fn complete_workspace(&mut self) -> CudaChunkedWorkspace {
        self.completed = true;
        self.kernels.take();
        self.input.take();
        self.workspace.take().expect("owned workspace")
    }

    pub(crate) fn mark_completed(&mut self) {
        self.completed = true;
    }
}

// Only scoped owners call this: success leaves raw work pending and returns
// its owner; any error/unwind fences via PendingChunkedLayer::drop.
pub(crate) fn enqueue_chunked_layer(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaTensorWeights,
    x: &EagerTensor,
    mask: &EagerTensor,
    kernels: Rc<CudaKernels>,
    workspace: Option<CudaChunkedWorkspace>,
) -> Result<(PendingChunkedLayer, EagerTensor)> {
    let invalid = |message| {
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            "delta_layer_cuda_chunked",
            "inputs",
            message,
        ))
    };
    cfg.validate().map_err(invalid_weights)?;
    let [length] = mask.shape() else {
        return Err(invalid("mask must have shape [length]"));
    };
    let length = *length;
    let channels = cfg
        .key_heads
        .checked_mul(cfg.key_dim)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| {
            cfg.value_heads
                .checked_mul(cfg.value_dim)
                .and_then(|v| n.checked_add(v))
        })
        .ok_or_else(|| invalid("configured channel widths overflow"))?;
    let geometry = ConvGeometry::new(length, channels, cfg.conv_taps).map_err(invalid_weights)?;
    if x.shape() != [cfg.hidden, length] {
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
                    "delta_layer_cuda_chunked",
                    "only f32 operands are supported",
                ),
            ));
        }
    }
    let runtime = with_cuda_exec_session(session.backend_session(), |cuda| cuda.runtime().clone())
        .ok_or_else(|| {
            tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::unsupported(
                "delta_layer_cuda_chunked",
                "CUDA execution session required",
            ))
        })?;
    let masked = session.mul(x, mask)?;
    let projected = linear_col(session, &masked, &weights.qkv)?;
    let time_first = session.transpose(&projected, &[1, 0])?;
    let input = session
        .duplicate_value(&time_first)?
        .into_typed::<f32>()
        .map_err(|failure| failure.into_parts().1)?;
    let z = linear_col(session, &masked, &weights.z)?;
    let a = linear_col(session, &masked, &weights.a)?;
    let b = linear_col(session, &masked, &weights.b)?;
    let mut pending = PendingChunkedLayer {
        runtime,
        kernels: Some(kernels),
        input: Some(input),
        workspace,
        completed: false,
    };
    let reuse_weight = if let Some(workspace) = &pending.workspace {
        same_raw_weights(std::slice::from_ref(&workspace.source), &[&weights.conv])?
    } else {
        false
    };
    if !reuse_weight {
        let weight = session
            .duplicate_value(&weights.conv)?
            .into_typed::<f32>()
            .map_err(|failure| failure.into_parts().1)?;
        if let Some(workspace) = &mut pending.workspace {
            workspace.weight = weight;
            workspace.source = weights.conv.clone();
        } else {
            let convolved = with_cuda_exec_session(session.backend_session(), |cuda| {
                cuda.with_raw("cuda_chunked_workspace", |raw| {
                    Ok(Tensor::from_typed(
                        raw.alloc_output::<f32>(&[length, channels])?,
                    ))
                })
            })
            .expect("CUDA session checked above")?;
            pending.workspace = Some(CudaChunkedWorkspace {
                convolved,
                weight,
                source: weights.conv.clone(),
                constants: prepare_native_constants(session, cfg, length)?,
            });
        }
    }
    // A completed workspace from another request length reallocates scratch.
    if pending
        .workspace
        .as_ref()
        .is_some_and(|workspace| workspace.convolved.shape() != [length, channels])
    {
        let convolved = with_cuda_exec_session(session.backend_session(), |cuda| {
            cuda.with_raw("cuda_chunked_workspace_resize", |raw| {
                Ok(Tensor::from_typed(
                    raw.alloc_output::<f32>(&[length, channels])?,
                ))
            })
        })
        .expect("CUDA session checked above")?;
        pending
            .workspace
            .as_mut()
            .expect("owned workspace")
            .convolved = convolved;
    }
    let workspace = pending.workspace.as_mut().expect("owned workspace");
    if !workspace.constants.matches(cfg, length, x) {
        workspace.constants = prepare_native_constants(session, cfg, length)?;
    }
    let launched = with_cuda_exec_session(session.backend_session(), |cuda| {
        cuda.with_raw("cuda_chunked_convolution", |raw| {
            let workspace = pending.workspace.as_mut().expect("owned workspace");
            // SAFETY: input/weight are owning device copies; private workspace
            // is disjoint. Pending owns all raw resources through completion.
            unsafe {
                pending
                    .kernels
                    .as_ref()
                    .expect("owned module")
                    .enqueue_conv(
                        raw,
                        &geometry,
                        pending.input.as_ref().expect("owned input"),
                        &workspace.weight,
                        workspace
                            .convolved
                            .as_typed_mut::<f32>()
                            .expect("private f32 workspace"),
                    )
            }
        })
    })
    .expect("CUDA session checked above");
    let computed = (|| {
        launched?;
        let mixed = session
            .backend_session()
            .to_contiguous_read(TensorRead::from_tensor(
                &pending
                    .workspace
                    .as_ref()
                    .expect("owned workspace")
                    .convolved,
            ))?;
        let mixed = session.constant_from(mixed)?;
        let mixed = session.transpose(&mixed, &[1, 0])?;
        delta_layer_from_projected_with_constants(
            session,
            cfg,
            weights,
            ProjectedDeltaTensors {
                mixed: &mixed,
                z: &z,
                a: &a,
                b: &b,
            },
            &pending
                .workspace
                .as_ref()
                .expect("owned workspace")
                .constants,
        )
    })();
    let output = computed?;
    Ok((pending, output))
}
