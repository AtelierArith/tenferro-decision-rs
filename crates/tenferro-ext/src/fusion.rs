//! Backend-selected fused kernels for the engines' forwards.
//!
//! [`Fusion`] is always available so model code needs no feature gates: on a
//! CUDA session (with the `cuda` feature) it owns a
//! [`FusedScope`](crate::FusedScope) whose single-kernel norms, activations
//! and attention replace long chains of eager ops; elsewhere it is inactive
//! and callers use their native compositions. Set
//! `TENFERRO_DECISION_FUSED=0` to disable fusion (for A/B measurements).

use tenferro_ad::{EagerSession, Result};

pub use crate::fused_scope::{FusedActivation, FusedScope};

/// An optional fused-kernel scope for one forward; see the module docs.
pub struct Fusion {
    scope: Option<FusedScope>,
}

impl Fusion {
    /// Begin fusion when `session` is a CUDA session and fusion is enabled.
    pub fn begin(session: &mut EagerSession<'_>) -> Result<Self> {
        if crate::cpu_extensions_supported(session) || !fusion_enabled() {
            return Ok(Self::inactive());
        }
        Ok(Self {
            scope: begin_scope(session)?,
        })
    }

    /// A fusion that never fuses.
    pub fn inactive() -> Self {
        Self { scope: None }
    }

    /// The active scope, if any.
    pub fn scope(&mut self) -> Option<&mut FusedScope> {
        self.scope.as_mut()
    }

    /// Whether fused kernels are in use.
    pub fn is_active(&self) -> bool {
        self.scope.is_some()
    }

    /// Complete the scope (synchronizing its stream) and release operands.
    pub fn finish(self) -> Result<()> {
        match self.scope {
            Some(scope) => scope.finish(),
            None => Ok(()),
        }
    }
}

fn fusion_enabled() -> bool {
    !matches!(
        std::env::var("TENFERRO_DECISION_FUSED").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

#[cfg(feature = "cuda")]
fn begin_scope(session: &mut EagerSession<'_>) -> Result<Option<FusedScope>> {
    let cuda =
        tenferro_gpu::cuda::with_cuda_exec_session(session.backend_session(), |_| ()).is_some();
    if cuda {
        FusedScope::begin(session).map(Some)
    } else {
        Ok(None)
    }
}

#[cfg(not(feature = "cuda"))]
fn begin_scope(_session: &mut EagerSession<'_>) -> Result<Option<FusedScope>> {
    Ok(None)
}
