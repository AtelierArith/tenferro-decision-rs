//! Shared inference primitives used by `laya-infer` and `jeff-infer`.
//!
//! This crate is **not** a general neural-network framework. It exposes exactly
//! the operations the two engines need, on top of tenferro-rs, each with a
//! reference form first and optimized/fused forms added only where profiling
//! justifies them (`docs/agents/specs/docs/15_TENFERRO_INFER_DESIGN.md`).
//!
//! Planned modules (Phase 1):
//!
//! - `linear` — prepared linear / GEMM
//! - `embedding` — gather
//! - `norm` — LayerNorm and RMSNorm (centered and non-centered)
//! - `activation` — erf-GELU, GeGLU, SiLU, gated SiLU
//! - `softmax` — stable and masked softmax
//! - `rope` — ModernBERT and Qwen partial RoPE
//! - `attention` — reference multi-head attention
//! - `layout` — head split/merge and packing helpers
//!
//! Phase 0 deliberately ships only the dependency wiring and a smoke test; the
//! primitives are implemented in Phase 1.
