//! Synthetic-weight parity tests for the Laya encoder and decision model.
//!
//! The host reference and the tenferro forward are compared with small random
//! weights, so no checkpoint is needed. GELU is the tanh approximation on both
//! sides (see the `model` module note); [`exact_gelu`] is checked separately.

use laya_infer::config::{AgentConfig, EncoderConfig, LayerKind};
use laya_infer::model::{
    erf, exact_gelu, forward_encoder_reference, forward_encoder_tenferro, forward_reference,
    forward_tenferro, forward_tenferro_cached, EncoderLayerWeights, HeadLayerWeights, LayaWeights,
    LayerNormWeights, LinearWeights, ModernBertWeights, TensorCache,
};
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;

/// Small deterministic PRNG so the tests do not pull in a dependency.
struct Lcg(u64);

impl Lcg {
    fn fill(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let x = ((self.0 >> 40) as f32) / (1u64 << 24) as f32;
                lo + (hi - lo) * x
            })
            .collect()
    }
}

fn norm(rng: &mut Lcg, d: usize, with_bias: bool) -> LayerNormWeights {
    LayerNormWeights {
        weight: rng.fill(d, 0.5, 1.5),
        bias: with_bias.then(|| rng.fill(d, -0.1, 0.1)),
    }
}

fn linear(rng: &mut Lcg, input: usize, output: usize, with_bias: bool) -> LinearWeights {
    LinearWeights {
        weight: rng.fill(input * output, -0.3, 0.3),
        bias: with_bias.then(|| rng.fill(output, -0.1, 0.1)),
    }
}

fn encoder_config() -> EncoderConfig {
    EncoderConfig {
        vocab_size: 5,
        hidden_size: 8,
        intermediate_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        model_type: "modernbert".to_string(),
        norm_eps: 1e-5,
        norm_bias: true,
        attention_bias: false,
        mlp_bias: false,
        hidden_activation: "gelu".to_string(),
        local_attention: 2,
        global_attn_every_n_layers: 3,
        global_rope_theta: 160000.0,
        local_rope_theta: 10000.0,
        max_position_embeddings: 64,
        layer_types: vec![LayerKind::FullAttention, LayerKind::SlidingAttention],
    }
}

fn encoder_weights(cfg: &EncoderConfig, rng: &mut Lcg) -> ModernBertWeights {
    let d = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let mut layers = Vec::new();
    for (index, kind) in cfg.layer_types.iter().enumerate() {
        layers.push(EncoderLayerWeights {
            kind: *kind,
            attn_norm: (index > 0).then(|| norm(rng, d, cfg.norm_bias)),
            wqkv: linear(rng, d, 3 * d, cfg.attention_bias),
            wo: linear(rng, d, d, cfg.attention_bias),
            num_heads: cfg.num_attention_heads,
            rope_base: cfg.rope_base(*kind),
            mlp_norm: norm(rng, d, cfg.norm_bias),
            wi: linear(rng, d, 2 * i, cfg.mlp_bias),
            wo_mlp: linear(rng, i, d, cfg.mlp_bias),
        });
    }
    ModernBertWeights {
        tok_embeddings: rng.fill(d * cfg.vocab_size, -0.5, 0.5),
        embed_norm: norm(rng, d, cfg.norm_bias),
        layers,
        final_norm: norm(rng, d, cfg.norm_bias),
    }
}

fn agent_config() -> AgentConfig {
    AgentConfig {
        head_layers: 1,
        max_len: 32,
        head_max_len: 8,
        action_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    }
}

fn laya_weights(cfg: &EncoderConfig, agent: &AgentConfig, rng: &mut Lcg) -> LayaWeights {
    let d = cfg.hidden_size;
    let head_heads = (d / 64).max(1);
    let action_hidden = 5;
    LayaWeights {
        encoder: encoder_weights(cfg, rng),
        head: (0..agent.head_layers)
            .map(|_| HeadLayerWeights {
                num_heads: head_heads,
                norm1: norm(rng, d, cfg.norm_bias),
                in_proj: linear(rng, d, 3 * d, cfg.attention_bias),
                out_proj: linear(rng, d, d, cfg.attention_bias),
                norm2: norm(rng, d, cfg.norm_bias),
                linear1: linear(rng, d, 2 * d, cfg.mlp_bias),
                linear2: linear(rng, 2 * d, d, cfg.mlp_bias),
            })
            .collect(),
        type_emb: rng.fill(d * 3, -0.2, 0.2),
        scorer_norm: norm(rng, d, cfg.norm_bias),
        scorer1: linear(rng, d, d, cfg.mlp_bias),
        scorer2: linear(rng, d, 1, cfg.mlp_bias),
        act1: linear(rng, d + 4, action_hidden, cfg.mlp_bias),
        act2: linear(rng, action_hidden, agent.action_count(), cfg.mlp_bias),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn exact_gelu_matches_reference_values() {
    assert!((erf(0.0)).abs() < 1e-6);
    assert!((erf(1.0) - 0.842_700_8).abs() < 1e-5);
    assert!((erf(-1.0) + 0.842_700_8).abs() < 1e-5);
    assert!(exact_gelu(0.0).abs() < 1e-6);
    assert!((exact_gelu(1.0) - 0.841_344_8).abs() < 1e-5);
    assert!((exact_gelu(-1.0) + 0.158_655_3).abs() < 1e-5);
    assert!((exact_gelu(2.0) - 1.954_499_7).abs() < 1e-5);
}

#[test]
fn gelu_tanh_is_close_to_exact() {
    for x in [-3.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0] {
        let tanh = laya_infer::model::gelu_tanh(x);
        assert!(
            (tanh - exact_gelu(x)).abs() < 3e-3,
            "gelu mismatch at {x}: {tanh} vs {}",
            exact_gelu(x)
        );
    }
}

fn encoder_batch(length: usize, batch: usize) -> (Vec<i64>, Vec<bool>) {
    let mut ids = Vec::with_capacity(length * batch);
    let mut mask = Vec::with_capacity(length * batch);
    for b in 0..batch {
        for l in 0..length {
            ids.push(((l + b) % 5) as i64);
            let valid = if length == 1 {
                true
            } else if batch == 1 || b == 0 {
                // An interior key hole.
                l != 1
            } else {
                // Padded (query) positions at the end.
                l < length - 1
            };
            mask.push(valid);
        }
    }
    (ids, mask)
}

#[test]
fn encoder_parity() {
    let cfg = encoder_config();
    let mut rng = Lcg(11);
    let weights = encoder_weights(&cfg, &mut rng);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    for length in [1usize, 4] {
        for batch in [1usize, 2] {
            let (ids, mask) = encoder_batch(length, batch);
            let reference = forward_encoder_reference(&cfg, &weights, &ids, &mask, batch).unwrap();
            let tenferro = runtime
                .with_eager_session(|session| {
                    forward_encoder_tenferro(session, &cfg, &weights, &ids, &mask, batch)
                })
                .unwrap()
                .unwrap();
            let diff = max_abs_diff(&reference, &tenferro);
            assert!(diff <= 2e-2, "L={length} B={batch} encoder max diff {diff}");
            assert_eq!(reference.len(), cfg.hidden_size * length * batch);
        }
    }
}

#[test]
fn decision_model_parity() {
    let cfg = encoder_config();
    let agent = agent_config();
    let mut rng = Lcg(29);
    let weights = laya_weights(&cfg, &agent, &mut rng);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    let length = 4;
    let batch = 2;
    let (ids, mask) = encoder_batch(length, batch);
    // Two used markers and one padded slot per sequence.
    let marker_pos = vec![0i64, 3, 1, 2, 3, 0];
    let marker_mask = vec![true, true, false, true, false, true];
    let qtype = vec![0i64, 2];

    let (ref_logits, ref_action) = forward_reference(
        &cfg,
        &agent,
        &weights,
        &ids,
        &mask,
        &marker_pos,
        &marker_mask,
        &qtype,
    )
    .unwrap();
    let (ten_logits, ten_action) = runtime
        .with_eager_session(|session| {
            forward_tenferro(
                session,
                &cfg,
                &agent,
                &weights,
                &ids,
                &mask,
                &marker_pos,
                &marker_mask,
                &qtype,
            )
        })
        .unwrap()
        .unwrap();

    assert_eq!(ref_logits.len(), 3 * batch);
    assert_eq!(ref_action.len(), agent.action_count() * batch);
    let logits_diff = max_abs_diff(&ref_logits, &ten_logits);
    let action_diff = max_abs_diff(&ref_action, &ten_action);
    assert!(logits_diff <= 2e-2, "logits max diff {logits_diff}");
    assert!(action_diff <= 2e-2, "action max diff {action_diff}");
}

#[test]
fn decision_model_cached_reuse_parity() {
    let cfg = encoder_config();
    let agent = agent_config();
    let mut rng = Lcg(29);
    let weights = laya_weights(&cfg, &agent, &mut rng);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    let length = 4;
    let batch = 2;
    let (ids, mask) = encoder_batch(length, batch);
    let marker_pos = vec![0i64, 3, 1, 2, 3, 0];
    let marker_mask = vec![true, true, false, true, false, true];
    let qtype = vec![0i64, 2];

    let (ref_logits, ref_action) = forward_reference(
        &cfg,
        &agent,
        &weights,
        &ids,
        &mask,
        &marker_pos,
        &marker_mask,
        &qtype,
    )
    .unwrap();

    // Reuse one cache across calls: the second call must match the first.
    let mut cache = TensorCache::new();
    for _ in 0..2 {
        let (logits, action) = runtime
            .with_eager_session(|session| {
                forward_tenferro_cached(
                    session,
                    &mut cache,
                    &cfg,
                    &agent,
                    &weights,
                    &ids,
                    &mask,
                    &marker_pos,
                    &marker_mask,
                    &qtype,
                )
            })
            .unwrap()
            .unwrap();
        assert!(max_abs_diff(&ref_logits, &logits) <= 2e-2);
        assert!(max_abs_diff(&ref_action, &action) <= 2e-2);
    }
    assert!(!cache.is_empty());
}

#[test]
fn padded_markers_are_masked() {
    let cfg = encoder_config();
    let agent = agent_config();
    let mut rng = Lcg(5);
    let weights = laya_weights(&cfg, &agent, &mut rng);

    let (ids, mask) = encoder_batch(4, 1);
    let marker_pos = vec![0i64, 1, 2];
    let marker_mask = vec![true, true, false];
    let qtype = vec![0i64];
    let (logits, _) = forward_reference(
        &cfg,
        &agent,
        &weights,
        &ids,
        &mask,
        &marker_pos,
        &marker_mask,
        &qtype,
    )
    .unwrap();

    let k = marker_pos.len();
    assert_eq!(logits[k - 1], -1.0e4);
    assert!(logits[0] > -1.0e4 && logits[1] > -1.0e4);
}
