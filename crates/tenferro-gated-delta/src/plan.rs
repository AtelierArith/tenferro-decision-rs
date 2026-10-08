//! Deterministic algorithm selection and prepared execution plans.
//!
//! Design: `docs/agents/specs/docs/12_TENFERRO_GATED_DELTA.md` §4 and §11. A
//! [`GatedDeltaPlan`] freezes the resolved [`Algorithm`] and the chunk size for
//! a concrete config/backend/sequence length. Resolution depends only on the
//! config, the declared backend capability, and the requested choice — never on
//! runtime load or input values — so runs are reproducible.

use crate::config::{Algorithm, GatedDeltaConfig};
use decision_core::{DecisionError, Result};

/// Declared backend capability used by [`resolve_algorithm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendCaps {
    /// A CPU provider is available.
    pub cpu: bool,
    /// A CUDA provider is available.
    pub cuda: bool,
    /// A vectorized CPU recurrent path is available.
    pub simd: bool,
}

impl BackendCaps {
    /// The portable CPU capability (chunked, no SIMD channel).
    pub const fn cpu() -> Self {
        Self {
            cpu: true,
            cuda: false,
            simd: false,
        }
    }

    /// A CPU capability with the recurrent SIMD channel enabled.
    pub const fn cpu_simd() -> Self {
        Self {
            cpu: true,
            cuda: false,
            simd: true,
        }
    }
}

/// Requested formulation. `Auto` defers to [`resolve_algorithm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgorithmChoice {
    /// Pick the best formulation for the backend.
    Auto,
    /// Force the host recurrent reference.
    Reference,
    /// Force the fused host recurrent kernel.
    Recurrent,
    /// Force the tenferro chunked formulation.
    Chunked,
}

impl AlgorithmChoice {
    /// Use the config's explicit [`Algorithm`] (no auto-selection).
    pub fn from_config(config: &GatedDeltaConfig) -> Self {
        match config.algorithm {
            Algorithm::Reference => AlgorithmChoice::Reference,
            Algorithm::Recurrent => AlgorithmChoice::Recurrent,
            Algorithm::Chunked => AlgorithmChoice::Chunked,
        }
    }
}

/// Resolve a formulation deterministically from config, capability, and choice.
///
/// The CPU portable default is the chunked formulation; the recurrent kernel is
/// selected when the backend declares SIMD support and `key_dim` is small
/// (matching the design table). CUDA execution is not implemented, so declaring
/// CUDA returns a typed error, even when CPU capability is also present.
pub fn resolve_algorithm(
    config: &GatedDeltaConfig,
    choice: AlgorithmChoice,
    caps: BackendCaps,
) -> Result<Algorithm> {
    if caps.cuda {
        return Err(DecisionError::unsupported(
            "GatedDelta CUDA execution is not implemented; CPU fallback is disabled",
        ));
    }
    if !caps.cpu {
        return Err(DecisionError::unsupported(
            "GatedDelta requires an available execution backend",
        ));
    }
    Ok(match choice {
        AlgorithmChoice::Reference => Algorithm::Reference,
        AlgorithmChoice::Recurrent => Algorithm::Recurrent,
        AlgorithmChoice::Chunked => Algorithm::Chunked,
        AlgorithmChoice::Auto => {
            if caps.simd && config.key_dim <= 256 {
                Algorithm::Recurrent
            } else {
                Algorithm::Chunked
            }
        }
    })
}

/// Frozen execution plan: resolved algorithm, chunk size, and the sequence
/// length the caller intends to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatedDeltaPlan {
    algorithm: Algorithm,
    chunk_size: usize,
    sequence_length: usize,
}

impl GatedDeltaPlan {
    /// Resolve a plan from config, choice, capability, and sequence length.
    pub fn resolve(
        config: &GatedDeltaConfig,
        choice: AlgorithmChoice,
        caps: BackendCaps,
        sequence_length: usize,
    ) -> Result<Self> {
        Ok(Self {
            algorithm: resolve_algorithm(config, choice, caps)?,
            chunk_size: config.chunk_size,
            sequence_length,
        })
    }

    /// Resolve a plan from the config's explicit algorithm.
    pub fn from_config(config: &GatedDeltaConfig, sequence_length: usize) -> Self {
        Self::resolve(
            config,
            AlgorithmChoice::from_config(config),
            BackendCaps::cpu(),
            sequence_length,
        )
        .expect("the explicitly selected CPU backend supports every CPU formulation")
    }

    /// The resolved formulation.
    pub const fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// The chunk size used by the chunked formulation.
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// The sequence length this plan was resolved for.
    pub const fn sequence_length(&self) -> usize {
        self.sequence_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GatedDeltaConfig;

    fn config(key_dim: usize) -> GatedDeltaConfig {
        GatedDeltaConfig {
            hidden: 8,
            key_heads: 1,
            value_heads: 2,
            key_dim,
            value_dim: 4,
            conv_taps: 3,
            eps: 1e-5,
            chunk_size: 4,
            algorithm: Algorithm::Chunked,
        }
    }

    #[test]
    fn auto_prefers_recurrent_for_small_key_dim_with_simd() {
        assert_eq!(
            resolve_algorithm(&config(64), AlgorithmChoice::Auto, BackendCaps::cpu_simd()).unwrap(),
            Algorithm::Recurrent
        );
    }

    #[test]
    fn auto_is_chunked_without_simd() {
        assert_eq!(
            resolve_algorithm(&config(64), AlgorithmChoice::Auto, BackendCaps::cpu()).unwrap(),
            Algorithm::Chunked
        );
    }

    #[test]
    fn auto_is_chunked_for_large_key_dim() {
        assert_eq!(
            resolve_algorithm(&config(512), AlgorithmChoice::Auto, BackendCaps::cpu_simd())
                .unwrap(),
            Algorithm::Chunked
        );
    }

    #[test]
    fn explicit_choice_wins() {
        assert_eq!(
            resolve_algorithm(
                &config(64),
                AlgorithmChoice::Reference,
                BackendCaps::cpu_simd()
            )
            .unwrap(),
            Algorithm::Reference
        );
    }

    #[test]
    fn resolution_is_deterministic() {
        let cfg = config(64);
        let first =
            GatedDeltaPlan::resolve(&cfg, AlgorithmChoice::Auto, BackendCaps::cpu_simd(), 128)
                .unwrap();
        let second =
            GatedDeltaPlan::resolve(&cfg, AlgorithmChoice::Auto, BackendCaps::cpu_simd(), 128)
                .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.sequence_length(), 128);
    }

    #[test]
    fn unsupported_backends_never_select_a_cpu_plan() {
        for caps in [
            BackendCaps {
                cpu: false,
                cuda: true,
                simd: false,
            },
            BackendCaps {
                cpu: true,
                cuda: true,
                simd: true,
            },
            BackendCaps {
                cpu: false,
                cuda: false,
                simd: false,
            },
        ] {
            for choice in [
                AlgorithmChoice::Auto,
                AlgorithmChoice::Reference,
                AlgorithmChoice::Recurrent,
                AlgorithmChoice::Chunked,
            ] {
                assert!(matches!(
                    GatedDeltaPlan::resolve(&config(64), choice, caps, 65),
                    Err(DecisionError::UnsupportedConfig { .. })
                ));
            }
        }
    }
}
