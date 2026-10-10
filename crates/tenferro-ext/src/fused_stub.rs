//! Uninhabited stand-in for the CUDA fused scope when the `cuda` feature is
//! off, so model code type-checks without feature gates. No value of
//! [`FusedScope`] can exist, so these methods are unreachable.

use tenferro_ad::{EagerSession, EagerTensor, Result};

/// Activation applied by [`FusedScope::bias_act_first`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedActivation {
    /// No activation (bias only).
    Identity,
    /// Exact erf-based GELU.
    GeluErf,
    /// ReLU.
    Relu,
}

/// Fused CUDA kernels (requires the `cuda` feature; uninhabited otherwise).
pub struct FusedScope {
    never: std::convert::Infallible,
}

#[allow(missing_docs, clippy::too_many_arguments)]
impl FusedScope {
    pub fn finish(self) -> Result<()> {
        match self.never {}
    }

    pub fn embedding_rows(
        &mut self,
        _session: &mut EagerSession<'_>,
        _table: &[f32],
        _hidden: usize,
        _vocab: usize,
        _ids: &[i64],
    ) -> Result<EagerTensor> {
        match self.never {}
    }

    pub fn embedding_columns(
        &mut self,
        _session: &mut EagerSession<'_>,
        _table: &[f32],
        _hidden: usize,
        _vocab: usize,
        _ids: &[i64],
    ) -> Result<EagerTensor> {
        match self.never {}
    }

    pub fn rms_norm_last(
        &mut self,
        _session: &mut EagerSession<'_>,
        _x: &EagerTensor,
        _weight: &EagerTensor,
        _centered: bool,
        _eps: f64,
    ) -> Result<EagerTensor> {
        match self.never {}
    }

    pub fn layer_norm_first(
        &mut self,
        _session: &mut EagerSession<'_>,
        _x: &EagerTensor,
        _weight: &EagerTensor,
        _bias: Option<&EagerTensor>,
        _eps: f64,
    ) -> Result<EagerTensor> {
        match self.never {}
    }

    pub fn gated_silu(
        &mut self,
        _session: &mut EagerSession<'_>,
        _gate: &EagerTensor,
        _up: &EagerTensor,
    ) -> Result<EagerTensor> {
        match self.never {}
    }

    pub fn geglu_first(
        &mut self,
        _session: &mut EagerSession<'_>,
        _u: &EagerTensor,
        _inter: usize,
    ) -> Result<EagerTensor> {
        match self.never {}
    }

    pub fn bias_act_first(
        &mut self,
        _session: &mut EagerSession<'_>,
        _x: &EagerTensor,
        _bias: Option<&EagerTensor>,
        _act: FusedActivation,
    ) -> Result<EagerTensor> {
        match self.never {}
    }

    pub fn jeff_attention(
        &mut self,
        _session: &mut EagerSession<'_>,
        _qkvg: &EagerTensor,
        _q_norm: &EagerTensor,
        _k_norm: &EagerTensor,
        _active: &EagerTensor,
        _heads: usize,
        _head_dim: usize,
        _rotary_dim: usize,
        _theta: f64,
        _eps: f64,
    ) -> Result<EagerTensor> {
        match self.never {}
    }

    pub fn gated_silu_stacked(
        &mut self,
        _session: &mut EagerSession<'_>,
        _gu: &EagerTensor,
    ) -> Result<EagerTensor> {
        match self.never {}
    }

    pub fn laya_attention(
        &mut self,
        _session: &mut EagerSession<'_>,
        _qkv: &EagerTensor,
        _bias: &EagerTensor,
        _hidden: usize,
        _heads: usize,
        _rope_base: f64,
    ) -> Result<EagerTensor> {
        match self.never {}
    }
}
