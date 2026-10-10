//! Owned Jeff CPU projections through tenferro's packed portable extension.
use std::sync::Arc;

use cpu_kernels::{PortableProjection, RowMajorProjection};
use decision_core::{DecisionError, Result};
use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_ext::{EagerSessionPackedGemmExt, PackedGemm};
use tenferro_infer::TensorCache;

use crate::host_opt::{HostOptWorkspace, forward_host_opt_with_projection};
use crate::model::{JeffConfig, JeffWeights};

/// Immutable checkpoint ownership and reusable packed CPU projections.
/// Clones share checkpoint storage and prepared weights. Source addresses stay
/// valid because this owner keeps the immutable checkpoint alive.
#[derive(Clone, Debug)]
pub struct PreparedCpuModel {
    config: JeffConfig,
    weights: Arc<JeffWeights>,
    runtime: Arc<EagerRuntime>,
    cache: TensorCache,
}

impl PreparedCpuModel {
    /// Own a model for the prepared CPU path.
    pub fn new(config: JeffConfig, weights: JeffWeights) -> Result<Self> {
        Self::from_shared(config, Arc::new(weights))
    }

    /// Share immutable checkpoint ownership with an engine or benchmark.
    pub fn from_shared(config: JeffConfig, weights: Arc<JeffWeights>) -> Result<Self> {
        weights.validate(&config)?;
        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).map_err(backend_error)?;
        Ok(Self {
            config,
            weights,
            runtime,
            cache: TensorCache::new(),
        })
    }

    /// Reuse packed tenferro extension projections and the existing CPU scan.
    /// Preparation is lazy and excluded from warmed inference measurements.
    pub fn forward(
        &mut self,
        workspace: &mut HostOptWorkspace,
        ids: &[i64],
        mask: &[f32],
    ) -> Result<Vec<f32>> {
        let mut projection = SessionProjections {
            runtime: &self.runtime,
            cache: &mut self.cache,
        };
        forward_host_opt_with_projection(
            workspace,
            &self.config,
            &self.weights,
            ids,
            mask,
            &mut projection,
        )
    }
}

fn backend_error(error: impl std::fmt::Display) -> DecisionError {
    DecisionError::Backend {
        message: error.to_string(),
        source: None,
    }
}

struct SessionProjections<'a> {
    runtime: &'a Arc<EagerRuntime>,
    cache: &'a mut TensorCache,
}

// Layout conversion only. Arithmetic stays in the tenferro extension/session.
fn transpose_into(input: &[f32], features: usize, length: usize, output: &mut [f32]) {
    for feature_begin in (0..features).step_by(16) {
        for token_begin in (0..length).step_by(4) {
            for feature in feature_begin..(feature_begin + 16).min(features) {
                for token in token_begin..(token_begin + 4).min(length) {
                    output[token * features + feature] = input[feature * length + token];
                }
            }
        }
    }
}
fn token_major(input: &[f32], features: usize, length: usize) -> Vec<f32> {
    let mut output = vec![0.; input.len()];
    transpose_into(input, features, length, &mut output);
    output
}

impl RowMajorProjection for SessionProjections<'_> {
    #[allow(clippy::too_many_arguments)]
    fn project(
        &mut self,
        weight: &[f32],
        input: usize,
        output: usize,
        x: &[f32],
        length: usize,
        y: &mut [f32],
        accumulate: bool,
    ) -> std::result::Result<(), String> {
        if output < 32 || length == 0 || length > 16 {
            return PortableProjection.project(weight, input, output, x, length, y, accumulate);
        }
        let prepared = self
            .cache
            .prepared_host(weight, &[input, output], || {
                let transposed = token_major(weight, input, output);
                PackedGemm::new(&transposed, input, output)
            })
            .map_err(|error| error.to_string())?;
        self.runtime
            .with_eager_session(|session| -> tenferro_ad::Result<()> {
                let source = session.constant_from_host(Tensor::from_vec_col_major(
                    vec![input, length],
                    token_major(x, input, length),
                )?)?;
                let mut projected = session.packed_gemm(&source, &prepared, None, false)?;
                if accumulate {
                    let previous = session.constant_from_host(Tensor::from_vec_col_major(
                        vec![output, length],
                        token_major(y, output, length),
                    )?)?;
                    projected = session.add(&projected, &previous)?;
                }
                let host = session.duplicate_value(&projected)?;
                let values = host.as_slice::<f32>()?;
                transpose_into(values, length, output, y);
                Ok(())
            })
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())
    }
}
