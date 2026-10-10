//! Explicit CUDA recurrent layer adapter; production plan dispatch stays gated.
//!
//! Projections use tenferro operations. Raw stages own their operands through
//! completion, then the output projection remains in the eager session.

use std::rc::Rc;

use tenferro_ad::{DType, EagerSession, EagerTensor, Result};
use tenferro_gpu::cuda::with_cuda_exec_session;
use tenferro_tensor::{Tensor, TensorRead, TensorView, TypedTensor};

use crate::{
    GatedDeltaConfig, GatedDeltaTensorWeights,
    cuda::{CudaKernels, RecurrentGeometry, StageBuffers},
    layer::invalid_weights,
    tensor_layer::linear_col,
};

/// Orientation of the layer's activation `x` and its output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActivationLayout {
    /// `x` and the output are `[hidden, length]` (the original adapter layout).
    #[default]
    FeatureFirst,
    /// `x` and the output are `[length, hidden]`, and the projection weights
    /// `qkv`/`z`/`a`/`b`/`out_proj` are `(out, in)` (see
    /// [`prepare_time_first_weights`]). Projections then produce the raw
    /// stages' time-first operands directly, without transposes or permutes.
    TimeFirst,
    /// [`Self::TimeFirst`] with one row-stacked `(qkv | z | a | b)` `(out,
    /// in)` projection in `weights.qkv` (one GEMM instead of four; `z`, `a`
    /// and `b` are ignored). Built by [`prepare_time_first_weights`].
    TimeFirstStacked,
}

impl ActivationLayout {
    fn time_first(self) -> bool {
        !matches!(self, Self::FeatureFirst)
    }
}

/// Tensor weights for [`ActivationLayout::TimeFirstStacked`]: `qkv` holds
/// the `(qkv | z | a | b)` projections row-stacked as one `(out, in)` matrix
/// (built once on the host from the row-major `(in, out)` buffers and cached
/// in `cache`); `z`/`a`/`b` alias it and are ignored by that layout. The
/// output projection is `(hidden, value_width)` = `out_projᵀ`, cached over the
/// raw buffer without a transpose.
pub fn prepare_time_first_weights(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &crate::GatedDeltaWeights,
    cache: &mut tenferro_infer::TensorCache,
) -> Result<GatedDeltaTensorWeights> {
    weights.validate(cfg).map_err(invalid_weights)?;
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let channels = 2 * key_width + value_width;
    let parts: [(&[f32], usize); 4] = [
        (&weights.qkv, channels),
        (&weights.z, value_width),
        (&weights.a, cfg.value_heads),
        (&weights.b, cfg.value_heads),
    ];
    let total: usize = parts.iter().map(|(_, out)| out).sum();
    let hidden = cfg.hidden;
    // Row-major (in, total) with each part's columns side by side; as a
    // column-major (total, in) tensor this is the stacked `(out, in)` matrix.
    let stacked = cache.prepared_host(&weights.qkv, &[total, hidden], || {
        let mut stacked = Vec::with_capacity(total * hidden);
        for row in 0..hidden {
            for (data, out) in parts {
                stacked.extend_from_slice(&data[row * out..(row + 1) * out]);
            }
        }
        Ok(stacked)
    })?;
    let projection = cache.col_major(session, vec![total, hidden], &stacked)?;
    Ok(GatedDeltaTensorWeights {
        z: projection.clone(),
        a: projection.clone(),
        b: projection.clone(),
        qkv: projection,
        conv: cache.col_major(session, vec![channels, cfg.conv_taps], &weights.conv)?,
        a_decay: cache.col_major(session, vec![cfg.value_heads], &weights.a_decay)?,
        dt_bias: cache.col_major(session, vec![cfg.value_heads], &weights.dt_bias)?,
        norm: cache.col_major(session, vec![cfg.value_dim], &weights.norm)?,
        out_proj: cache.col_major(session, vec![cfg.hidden, value_width], &weights.out_proj)?,
    })
}

/// `x [length, in] · wᵀ` for an `(out, in)` weight `w`, giving `[length, out]`.
pub(crate) fn linear_time_first(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    weight: &EagerTensor,
) -> Result<EagerTensor> {
    session.dot_general(
        x,
        weight,
        tenferro_ad::DotGeneralConfig {
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [1].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        },
    )
}

/// Mask an activation's token axis in either layout.
pub(crate) fn mask_tokens(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    mask: &EagerTensor,
    layout: ActivationLayout,
) -> Result<EagerTensor> {
    match layout {
        ActivationLayout::FeatureFirst => session.mul(x, mask),
        ActivationLayout::TimeFirst | ActivationLayout::TimeFirstStacked => {
            let mask = session.broadcast_in_dim(mask, x.shape(), &[0])?;
            session.mul(x, &mask)
        }
    }
}

/// Completed, exclusively owned CUDA scratch tensors for reuse at one shape.
///
/// Fields stay private so callers cannot alias mutable workspace buffers.
/// The cached entry point validates shape/runtime through the raw adapter.
pub struct CudaRecurrentWorkspace {
    convolved: TypedTensor<f32>,
    scanned: TypedTensor<f32>,
    output: Tensor,
    weights: RawWeights,
}

struct RawWeights {
    // Retain sources as well as copies: allocation identities cannot be recycled.
    sources: [EagerTensor; 4],
    tensors: [TypedTensor<f32>; 4],
}

pub(crate) fn same_raw_weights(previous: &[EagerTensor], current: &[&EagerTensor]) -> Result<bool> {
    if previous.len() != current.len() {
        return Ok(false);
    }
    for (previous, current) in previous.iter().zip(current) {
        if previous.ctx_id() != current.ctx_id() {
            return Ok(false);
        }
        let previous = previous.value()?;
        let current = current.value()?;
        let (TensorView::F32(previous), TensorView::F32(current)) =
            (previous.as_tensor_view(), current.as_tensor_view())
        else {
            return Ok(false);
        };
        let (Some(previous_buffer), Some(current_buffer)) =
            (previous.backend_buffer(), current.backend_buffer())
        else {
            return Ok(false);
        };
        if previous_buffer.allocation_id().is_none()
            || previous_buffer.allocation_id() != current_buffer.allocation_id()
            || previous_buffer.allocation_domain() != current_buffer.allocation_domain()
            || previous.shape() != current.shape()
            || previous.strides() != current.strides()
            || previous.offset() != current.offset()
        {
            return Ok(false);
        }
    }
    Ok(true)
}

// Local to this function's admitted eager scope: the current execution stream
// remains active until this guard is completed or dropped. No pending owner
// escapes the scope or crosses a Send boundary.
pub(crate) struct PendingRecurrentLayer {
    runtime: tenferro_gpu::cuda::CudaRuntime,
    kernels: Option<Rc<CudaKernels>>,
    inputs: Option<[TypedTensor<f32>; 4]>,
    workspace: Option<CudaRecurrentWorkspace>,
    completed: bool,
}

impl Drop for PendingRecurrentLayer {
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
/// projected operands and allocates a workspace per convenience call. The
/// cached variant retains raw weight copies and scratch buffers. It retains the raw operands through native readout and
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
/// projected operands still use device copies; raw weight copies are cached
/// using allocation identity and view metadata. Completion is synchronized once after
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
    let kernels = Rc::new(kernels);
    let (mut pending, output) = enqueue_recurrent_layer(
        session,
        cfg,
        weights,
        x,
        mask,
        kernels.clone(),
        workspace,
        ActivationLayout::FeatureFirst,
    )?;
    let synchronized = with_cuda_exec_session(session.backend_session(), |cuda| {
        cuda.with_raw("cuda_recurrent_layer_finish", |raw| raw.synchronize())
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

impl PendingRecurrentLayer {
    /// Caller establishes stream completion before recovering resources.
    pub(crate) fn complete_workspace(&mut self) -> CudaRecurrentWorkspace {
        self.completed = true;
        self.kernels.take();
        self.inputs.take();
        self.workspace.take().expect("owned workspace")
    }

    pub(crate) fn mark_completed(&mut self) {
        self.completed = true;
    }
}

// Success leaves raw work owned by the scoped request or single-layer caller.
// Any error/unwind fences through the private pending owner.
#[allow(clippy::too_many_arguments)]
pub(crate) fn enqueue_recurrent_layer(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaTensorWeights,
    x: &EagerTensor,
    mask: &EagerTensor,
    kernels: Rc<CudaKernels>,
    workspace: Option<CudaRecurrentWorkspace>,
    layout: ActivationLayout,
) -> Result<(PendingRecurrentLayer, EagerTensor)> {
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
    let expected = if layout.time_first() {
        [*length, cfg.hidden]
    } else {
        [cfg.hidden, *length]
    };
    if x.shape() != expected {
        return Err(invalid("x shape does not match the activation layout"));
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
    let masked = mask_tokens(session, x, mask, layout)?;
    let channels = 2 * cfg.key_heads * cfg.key_dim + cfg.value_heads * cfg.value_dim;
    let width = cfg.value_heads * cfg.value_dim;
    let owned =
        |session: &mut EagerSession<'_>, tensor: &EagerTensor| -> Result<TypedTensor<f32>> {
            Ok(session
                .duplicate_value(tensor)?
                .into_typed::<f32>()
                .map_err(|failure| failure.into_parts().1)?)
        };
    let [mixed, z, a, b] = if layout == ActivationLayout::TimeFirstStacked {
        let total = channels + width + 2 * cfg.value_heads;
        if weights.qkv.shape() != [total, cfg.hidden] {
            return Err(invalid("stacked projection must be (qkv|z|a|b, hidden)"));
        }
        let projected = linear_time_first(session, &masked, &weights.qkv)?;
        let mut parts = Vec::with_capacity(4);
        let mut start = 0;
        for size in [channels, width, cfg.value_heads, cfg.value_heads] {
            let part = session.slice(
                &projected,
                tenferro_ad::SliceConfig {
                    starts: vec![0, start],
                    limits: vec![*length, start + size],
                    strides: vec![1, 1],
                },
            )?;
            parts.push(owned(session, &part)?);
            start += size;
        }
        let [mixed, z, a, b]: [TypedTensor<f32>; 4] =
            parts.try_into().map_err(|_| invalid("projection split"))?;
        [mixed, z, a, b]
    } else {
        let mut project = |weight| -> Result<TypedTensor<f32>> {
            let time_first = if layout.time_first() {
                linear_time_first(session, &masked, weight)?
            } else {
                let projected = linear_col(session, &masked, weight)?;
                session.transpose(&projected, &[1, 0])?
            };
            owned(session, &time_first)
        };
        [
            project(&weights.qkv)?,
            project(&weights.z)?,
            project(&weights.a)?,
            project(&weights.b)?,
        ]
    };
    let sources = [
        &weights.conv,
        &weights.a_decay,
        &weights.dt_bias,
        &weights.norm,
    ];
    let reuse_weights = if let Some(workspace) = &workspace {
        same_raw_weights(&workspace.weights.sources, &sources)?
    } else {
        false
    };
    let mut new_weights = if reuse_weights {
        None
    } else {
        let mut copy = |tensor| -> Result<TypedTensor<f32>> {
            Ok(session
                .duplicate_value(tensor)?
                .into_typed::<f32>()
                .map_err(|failure| failure.into_parts().1)?)
        };
        Some(RawWeights {
            sources: sources.map(EagerTensor::clone),
            tensors: [
                copy(sources[0])?,
                copy(sources[1])?,
                copy(sources[2])?,
                copy(sources[3])?,
            ],
        })
    };
    let runtime = with_cuda_exec_session(session.backend_session(), |cuda| cuda.runtime().clone())
        .expect("CUDA session checked above");
    // A completed workspace from a different request length keeps its cached
    // raw weights but replaces its scratch buffers.
    let mut workspace = workspace;
    let resize = workspace
        .as_ref()
        .is_some_and(|previous| previous.convolved.shape() != [*length, channels]);
    if let (true, Some(previous)) = (resize, &mut workspace) {
        let buffers = with_cuda_exec_session(session.backend_session(), |cuda| {
            cuda.with_raw("cuda_recurrent_workspace_resize", |raw| {
                Ok((
                    raw.alloc_output::<f32>(&[*length, channels])?,
                    raw.alloc_output::<f32>(&[*length, width])?,
                    raw.alloc_output::<f32>(&[*length, width])?,
                ))
            })
        })
        .expect("CUDA session checked above")?;
        previous.convolved = buffers.0;
        previous.scanned = buffers.1;
        previous.output = Tensor::from_typed(buffers.2);
    }
    let mut pending = PendingRecurrentLayer {
        runtime,
        kernels: Some(kernels),
        inputs: Some([mixed, a, b, z]),
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
                        weights: new_weights.take().expect("prepared raw weights"),
                    })
                })
            })
            .expect("CUDA session checked above")?,
        );
    }
    if let Some(weights) = new_weights {
        pending.workspace.as_mut().expect("owned workspace").weights = weights;
    }
    let launched = with_cuda_exec_session(session.backend_session(), |cuda| {
        cuda.with_raw("cuda_recurrent_layer", |raw| {
            let [mixed_input, a, b, z] = pending.inputs.as_ref().expect("owned inputs");
            let workspace = pending.workspace.as_mut().expect("owned workspace");
            let [conv_weight, a_decay, dt_bias, norm] = &workspace.weights.tensors;
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
        if layout.time_first() {
            linear_time_first(session, &output, &weights.out_proj)
        } else {
            let output = session.transpose(&output, &[1, 0])?;
            linear_col(session, &output, &weights.out_proj)
        }
    })();
    let output = computed?;
    Ok((pending, output))
}
