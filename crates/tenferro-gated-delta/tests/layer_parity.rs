//! End-to-end layer parity: the tenferro-backed layer must match the host
//! reference layer with identical weights.

use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::{
    delta_layer_reference, delta_layer_tenferro, Algorithm, GatedDeltaConfig, GatedDeltaWeights,
};

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    fn fill(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| self.range(lo, hi)).collect()
    }
}

fn config() -> GatedDeltaConfig {
    GatedDeltaConfig {
        hidden: 8,
        key_heads: 1,
        value_heads: 2,
        key_dim: 4,
        value_dim: 4,
        conv_taps: 3,
        eps: 1e-5,
        chunk_size: 4,
        algorithm: Algorithm::Chunked,
    }
}

fn weights(cfg: &GatedDeltaConfig, seed: u64) -> GatedDeltaWeights {
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;
    let mut rng = Lcg(seed);
    GatedDeltaWeights {
        qkv: rng.fill(cfg.hidden * conv_channels, -0.3, 0.3),
        z: rng.fill(cfg.hidden * value_width, -0.3, 0.3),
        a: rng.fill(cfg.hidden * cfg.value_heads, -0.3, 0.3),
        b: rng.fill(cfg.hidden * cfg.value_heads, -0.3, 0.3),
        conv: rng.fill(cfg.conv_taps * conv_channels, -0.5, 0.5),
        a_decay: rng.fill(cfg.value_heads, -1.0, -0.05),
        dt_bias: rng.fill(cfg.value_heads, -0.5, 0.5),
        norm: rng.fill(cfg.value_dim, 0.5, 1.5),
        out_proj: rng.fill(value_width * cfg.hidden, -0.3, 0.3),
    }
}

#[test]
fn layer_tenferro_matches_reference() {
    let cfg = config();
    let w = weights(&cfg, 7);
    let length = 6;
    let mut rng = Lcg(99);
    let x = rng.fill(cfg.hidden * length, -1.0, 1.0);
    // A mask hole in the middle.
    let mask = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 1.0];

    let reference = delta_layer_reference(&cfg, &w, &x, &mask).unwrap();

    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let tenferro = runtime
        .with_eager_session(|session| delta_layer_tenferro(session, &cfg, &w, &x, &mask))
        .unwrap()
        .unwrap();

    assert_eq!(reference.len(), cfg.hidden * length);
    let mut max_diff = 0.0f32;
    for (a, b) in reference.iter().zip(&tenferro) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff <= 5e-3,
        "layer max diff {max_diff} exceeds tolerance"
    );
}

#[test]
fn weights_validation_catches_shape_errors() {
    let cfg = config();
    let mut w = weights(&cfg, 1);
    w.norm.pop();
    assert!(w.validate(&cfg).is_err());
}
