//! Reusable execution buffers for the fused recurrent DeltaNet kernel.
//!
//! Design: `docs/agents/specs/docs/12_TENFERRO_GATED_DELTA.md` §9 and §11. A
//! workspace is owned by an inference context (not global), so concurrent
//! requests use separate workspaces. Buffers grow to the largest sequence length
//! seen and are then reused; the fused kernel writes the final result into
//! [`GatedDeltaWorkspace::output`] and returns a borrow of it.

use crate::config::GatedDeltaConfig;

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
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) z: Vec<f32>,
    pub(crate) state: Vec<f32>,
    pub(crate) prediction: Vec<f32>,
    pub(crate) correction: Vec<f32>,
    pub(crate) result: Vec<f32>,
    pub(crate) out: Vec<f32>,
    pub(crate) output: Vec<f32>,
    pub(crate) scratch: Vec<f32>,
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

    /// The total bytes retained by the workspace buffers.
    pub fn retained_bytes(&self) -> usize {
        let f32_buffers = [
            &self.masked,
            &self.qkv_proj,
            &self.mixed,
            &self.z_proj,
            &self.a_proj,
            &self.b_proj,
            &self.beta,
            &self.decay,
            &self.q,
            &self.k,
            &self.v,
            &self.z,
            &self.state,
            &self.prediction,
            &self.correction,
            &self.result,
            &self.out,
            &self.output,
            &self.scratch,
        ];
        f32_buffers.iter().map(|buffer| buffer.len() * 4).sum()
    }
}
