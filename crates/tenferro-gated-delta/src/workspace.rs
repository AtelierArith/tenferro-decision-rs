//! Reusable execution buffers for the fused recurrent DeltaNet kernel.
//!
//! Design: `docs/agents/specs/docs/12_TENFERRO_GATED_DELTA.md` §9 and §11. A
//! workspace is owned by an inference context (not global), so concurrent
//! requests use separate workspaces. The layer-level buffers are shared, while
//! each value head owns a [`HeadScratch`] so the heads can run in parallel.
//! Buffers grow to the largest sequence length seen and are then reused.
//! Native execution also retains one immutable constants entry, replaced when
//! its configuration, sequence length or eager runtime changes.

use crate::config::GatedDeltaConfig;

/// Per-value-head scratch for the recurrent scan.
#[derive(Clone, Debug, Default)]
pub(crate) struct HeadScratch {
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) z: Vec<f32>,
    pub(crate) state: Vec<f32>,
    pub(crate) prediction: Vec<f32>,
    pub(crate) correction: Vec<f32>,
    pub(crate) result: Vec<f32>,
    pub(crate) output: Vec<f32>,
    pub(crate) scratch: Vec<f32>,
}

/// Scratch buffers sized for a `(hidden, length)` layer invocation.
#[derive(Clone, Debug, Default)]
pub struct GatedDeltaWorkspace {
    pub(crate) masked: Vec<f32>,
    pub(crate) qkv_proj: Vec<f32>,
    pub(crate) mixed: Vec<f32>,
    pub(crate) z_proj: Vec<f32>,
    pub(crate) a_proj: Vec<f32>,
    pub(crate) b_proj: Vec<f32>,
    pub(crate) beta: Vec<f32>,
    pub(crate) decay: Vec<f32>,
    pub(crate) out: Vec<f32>,
    pub(crate) output: Vec<f32>,
    pub(crate) heads: Vec<HeadScratch>,
    pub(crate) native_constants: Option<crate::tensor_layer::NativeDeltaConstants>,
}

impl GatedDeltaWorkspace {
    /// An empty workspace. Buffers allocate lazily on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty workspace. Provided for symmetry with plan construction.
    pub fn for_plan(_config: &GatedDeltaConfig, _sequence_length: usize) -> Self {
        Self::default()
    }

    /// The total bytes retained by the host scratch buffers (excludes native constants).
    pub fn retained_bytes(&self) -> usize {
        let layer: usize = [
            &self.masked,
            &self.qkv_proj,
            &self.mixed,
            &self.z_proj,
            &self.a_proj,
            &self.b_proj,
            &self.beta,
            &self.decay,
            &self.out,
            &self.output,
        ]
        .iter()
        .map(|buffer| buffer.len() * 4)
        .sum();
        let heads: usize = self
            .heads
            .iter()
            .map(|head| {
                head.q.len()
                    + head.k.len()
                    + head.v.len()
                    + head.z.len()
                    + head.state.len()
                    + head.prediction.len()
                    + head.correction.len()
                    + head.result.len()
                    + head.output.len()
                    + head.scratch.len()
            })
            .sum::<usize>()
            * 4;
        layer + heads
    }
}
