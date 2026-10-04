//! Cross-implementation parity for the Jeff layer stack: the tenferro forward
//! must match the host reference with identical weights, for a mixed stack
//! containing both a Gated DeltaNet layer and a full-attention layer.

use jeff_infer::model::{
    AttentionWeights, FullAttentionWeights, JeffConfig, JeffWeights, LayerWeights, MlpWeights,
    forward_reference, forward_reference_with, forward_tenferro, forward_tenferro_with,
};
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::{Algorithm, GatedDeltaConfig, GatedDeltaWeights, GatedDeltaWorkspace};

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }

    fn fill(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| lo + (hi - lo) * self.next_f32()).collect()
    }
}

fn config() -> JeffConfig {
    JeffConfig {
        hidden: 8,
        heads: 2,
        head_dim: 4,
        intermediate: 16,
        eps: 1e-5,
    }
}

fn build_weights(cfg: &JeffConfig, vocab: usize, options: usize, seed: u64) -> JeffWeights {
    let mut rng = Lcg(seed);
    let width = cfg.heads * cfg.head_dim;

    let delta_cfg = GatedDeltaConfig {
        hidden: cfg.hidden,
        key_heads: 1,
        value_heads: 2,
        key_dim: 4,
        value_dim: 4,
        conv_taps: 3,
        eps: cfg.eps,
        chunk_size: 4,
        algorithm: Algorithm::Chunked,
    };
    let key_width = delta_cfg.key_dim * delta_cfg.key_heads;
    let value_width = delta_cfg.value_dim * delta_cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;
    let delta = GatedDeltaWeights {
        qkv: rng.fill(cfg.hidden * conv_channels, -0.3, 0.3),
        z: rng.fill(cfg.hidden * value_width, -0.3, 0.3),
        a: rng.fill(cfg.hidden * delta_cfg.value_heads, -0.3, 0.3),
        b: rng.fill(cfg.hidden * delta_cfg.value_heads, -0.3, 0.3),
        conv: rng.fill(delta_cfg.conv_taps * conv_channels, -0.5, 0.5),
        a_decay: rng.fill(delta_cfg.value_heads, -1.0, -0.05),
        dt_bias: rng.fill(delta_cfg.value_heads, -0.5, 0.5),
        norm: rng.fill(delta_cfg.value_dim, 0.5, 1.5),
        out_proj: rng.fill(value_width * cfg.hidden, -0.3, 0.3),
    };

    let full = FullAttentionWeights {
        q: rng.fill(cfg.hidden * width, -0.3, 0.3),
        gate: rng.fill(cfg.hidden * width, -0.3, 0.3),
        k: rng.fill(cfg.hidden * width, -0.3, 0.3),
        v: rng.fill(cfg.hidden * width, -0.3, 0.3),
        o: rng.fill(width * cfg.hidden, -0.3, 0.3),
        q_norm: rng.fill(cfg.head_dim, 0.5, 1.5),
        k_norm: rng.fill(cfg.head_dim, 0.5, 1.5),
        rope_theta: 1_000_000.0,
        rotary_dim: 2,
    };

    let layer = |rng: &mut Lcg, attention: AttentionWeights| LayerWeights {
        input_norm: rng.fill(cfg.hidden, 0.5, 1.5),
        post_norm: rng.fill(cfg.hidden, 0.5, 1.5),
        attention,
        mlp: MlpWeights {
            gate: rng.fill(cfg.hidden * cfg.intermediate, -0.3, 0.3),
            up: rng.fill(cfg.hidden * cfg.intermediate, -0.3, 0.3),
            down: rng.fill(cfg.intermediate * cfg.hidden, -0.3, 0.3),
        },
    };

    let layers = vec![
        layer(
            &mut rng,
            AttentionWeights::Delta {
                weights: delta,
                config: delta_cfg,
            },
        ),
        layer(&mut rng, AttentionWeights::Full(full)),
    ];

    JeffWeights {
        embedding: rng.fill(cfg.hidden * vocab, -1.0, 1.0),
        vocab,
        layers,
        final_norm: rng.fill(cfg.hidden, 0.5, 1.5),
        readout: rng.fill(cfg.hidden * options, -0.3, 0.3),
        options,
    }
}

#[test]
fn tenferro_matches_reference_for_mixed_stack() {
    let cfg = config();
    let vocab = 12;
    let options = 3;
    let weights = build_weights(&cfg, vocab, options, 11);
    let ids = vec![3_i64, 1, 4, 1, 5, 2];
    let mask = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 1.0];
    compare(&cfg, &weights, &ids, &mask, 2e-2);
}

#[test]
fn delta_only_stack_matches() {
    let cfg = config();
    let mut weights = build_weights(&cfg, 12, 3, 11);
    weights.layers.truncate(1); // layer 0 is the DeltaNet layer
    let ids = vec![3_i64, 1, 4, 1, 5, 2];
    let mask = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 1.0];
    compare(&cfg, &weights, &ids, &mask, 2e-2);
}

#[test]
fn full_only_stack_matches() {
    let cfg = config();
    let mut weights = build_weights(&cfg, 12, 3, 11);
    weights.layers.remove(0); // keep the full-attention layer
    let ids = vec![3_i64, 1, 4, 1, 5, 2];
    let mask = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 1.0];
    compare(&cfg, &weights, &ids, &mask, 2e-2);
}

#[test]
fn full_only_single_token() {
    let cfg = config();
    let mut weights = build_weights(&cfg, 12, 3, 11);
    weights.layers.remove(0);
    let ids = vec![4_i64];
    let mask = vec![1.0f32];
    compare(&cfg, &weights, &ids, &mask, 2e-2);
}

#[test]
fn full_only_two_tokens_all_active() {
    let cfg = config();
    let mut weights = build_weights(&cfg, 12, 3, 11);
    weights.layers.remove(0);
    let ids = vec![4_i64, 1];
    let mask = vec![1.0f32, 1.0];
    compare(&cfg, &weights, &ids, &mask, 2e-2);
}

fn compare(cfg: &JeffConfig, weights: &JeffWeights, ids: &[i64], mask: &[f32], tol: f32) {
    let reference = forward_reference(cfg, weights, ids, mask).unwrap();
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let tenferro = runtime
        .with_eager_session(|session| forward_tenferro(session, cfg, weights, ids, mask))
        .unwrap()
        .unwrap();
    let mut max_diff = 0.0f32;
    for (a, b) in reference.iter().zip(&tenferro) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff <= tol,
        "max diff {max_diff} exceeds {tol}; ref={reference:?} tenferro={tenferro:?}"
    );
}

#[test]
fn reference_is_deterministic() {
    let cfg = config();
    let weights = build_weights(&cfg, 12, 3, 5);
    let ids = vec![2_i64, 0, 1, 2];
    let mask = vec![1.0f32; 4];
    let a = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
    let b = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
    assert_eq!(a, b);
}

#[test]
fn host_workspace_reuse_matches_fresh_forward() {
    let cfg = config();
    let weights = build_weights(&cfg, 12, 3, 11);
    let ids = vec![3_i64, 1, 4, 1, 5, 2];
    let mask = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 1.0];
    let expected = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
    let mut workspace = GatedDeltaWorkspace::new();
    for _ in 0..3 {
        let got = forward_reference_with(&mut workspace, &cfg, &weights, &ids, &mask).unwrap();
        assert_eq!(got, expected);
    }
    assert!(workspace.retained_bytes() > 0);
}

#[test]
fn tenferro_workspace_reuse_matches_reference() {
    let cfg = config();
    let weights = build_weights(&cfg, 12, 3, 11);
    let ids = vec![3_i64, 1, 4, 1, 5, 2];
    let mask = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 1.0];
    let reference = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let mut workspace = GatedDeltaWorkspace::new();
    for _ in 0..2 {
        let tenferro = runtime
            .with_eager_session(|session| {
                forward_tenferro_with(&mut workspace, session, &cfg, &weights, &ids, &mask)
            })
            .unwrap()
            .unwrap();
        let max_diff = reference
            .iter()
            .zip(&tenferro)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff <= 2e-2, "workspace reuse diff {max_diff}");
    }
}
