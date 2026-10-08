//! Shared inference primitives used by `laya-infer` and `jeff-infer`.
//!
//! This crate is **not** a general neural-network framework. It exposes exactly
//! the operations the two engines need, on top of tenferro's eager runtime
//! (`tenferro-ad`), each as a reference form first and optimized/fused forms
//! added only where profiling justifies them
//! (`docs/agents/specs/docs/15_TENFERRO_INFER_DESIGN.md`).
//!
//! The primitives take a borrowed [`tenferro_ad::EagerSession`] and
//! [`tenferro_ad::EagerTensor`] values, so a model forward pass is one session
//! entry. (`15_TENFERRO_INFER_DESIGN.md` sketched `&mut dyn BackendSession`;
//! the eager session is the concrete tenferro surface that exposes the full op
//! set, including `dot_general`, reductions, gather, slice, and concatenate.)
//!
//! ## Known gap
//!
//! tenferro has no `erf` op; the exact erf-based GELU used by Laya comes from
//! the self-hosted `tenferro-ext` extension op (`EagerSessionErfExt`).

pub mod activation;
pub mod attention;
pub mod cache;
pub mod embedding;
pub mod linear;
pub mod norm;
pub mod output;
pub mod rope;
pub mod softmax;
pub(crate) mod util;

/// Reusable weight-tensor cache for the eager session.
pub use cache::TensorCache;
/// Re-exported tenferro error and result types used by every primitive.
pub use tenferro_ad::{Error, Result};
