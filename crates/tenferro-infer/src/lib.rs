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
//! entry that borrows the caller-held CPU engine. tenferro 0.11 stores no
//! engine in the runtime, so [`cpu_engine`] builds the process-sized engine the
//! engines keep for their lifetime.
//!
//! ## `erf`
//!
//! tenferro 0.11 ships a standard `erf`. The exact erf-based GELU used by Laya
//! keeps the MLX-compatible `f32` kernel from the self-hosted `tenferro-ext`
//! extension op (`EagerSessionErfExt::erf_ext` / `gelu_erf`) for parity with the
//! Julia reference.

pub mod activation;
pub mod attention;
pub mod cache;
pub mod embedding;
pub mod engine;
pub mod input;
pub mod linear;
pub mod norm;
pub mod output;
pub mod rope;
pub mod softmax;
pub(crate) mod util;

/// Reusable weight-tensor cache for the eager session.
pub use cache::TensorCache;
/// A CPU engine sized from `RAYON_NUM_THREADS` or the process parallelism.
pub use engine::cpu_engine;
/// Re-exported tenferro error and result types used by every primitive.
pub use tenferro_ad::{Error, Result};
