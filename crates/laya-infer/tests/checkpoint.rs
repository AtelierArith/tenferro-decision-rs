//! Checkpoint loading: a synthetic safetensors checkpoint is written byte by
//! byte, loaded, and compared against a manually built reference. Missing,
//! unexpected and wrongly shaped tensors must fail with clear errors.
//!
//! Both the upstream PyTorch parameter spellings and the already-converted MLX
//! `layers.` forms are exercised, since [`load_checkpoint`] understands both.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use laya_infer::checkpoint::load_checkpoint;
use laya_infer::config::{AgentConfig, EncoderConfig, LayerKind};
use laya_infer::model::{
    EncoderLayerWeights, HeadLayerWeights, LayaWeights, LayerNormWeights, LinearWeights,
    ModernBertWeights,
};

const HIDDEN: usize = 8;
const HEADS: usize = 2;
const INTERMEDIATE: usize = 12;
const ENC_LAYERS: usize = 2;
const HEAD_LAYERS: usize = 1;
const VOCAB: usize = 20;
const ACTION_COUNT: usize = 2;
const ACTION_HIDDEN: usize = 256;
/// Decision-head MLP width is fixed at `4 * hidden` by `weights.jl`.
const HEAD_FF: usize = 4 * HIDDEN;
/// Decision-head attention heads: `max(1, hidden / 64)`.
const HEAD_HEADS: usize = 1;

static COUNTER: AtomicU64 = AtomicU64::new(0);

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

fn norm(rng: &mut Lcg, dim: usize, with_bias: bool) -> LayerNormWeights {
    LayerNormWeights {
        weight: rng.fill(dim, 0.5, 1.5),
        bias: with_bias.then(|| rng.fill(dim, -0.1, 0.1)),
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
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        num_hidden_layers: ENC_LAYERS,
        num_attention_heads: HEADS,
        model_type: "modernbert".to_string(),
        norm_eps: 1e-5,
        norm_bias: true,
        attention_bias: true,
        mlp_bias: true,
        hidden_activation: "gelu".to_string(),
        local_attention: 2,
        global_attn_every_n_layers: 3,
        global_rope_theta: 160000.0,
        local_rope_theta: 10000.0,
        max_position_embeddings: 64,
        layer_types: vec![LayerKind::FullAttention, LayerKind::SlidingAttention],
    }
}

fn agent_config() -> AgentConfig {
    AgentConfig {
        head_layers: HEAD_LAYERS,
        max_len: 32,
        head_max_len: 8,
        action_names: vec!["escalate".to_string(), "notify".to_string()],
    }
}

/// Build the reference weights the loader must reproduce.
fn build_reference(cfg: &EncoderConfig, seed: u64) -> LayaWeights {
    let mut rng = Lcg(seed);
    let d = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;

    let mut layers = Vec::new();
    for (index, kind) in cfg.layer_types.iter().enumerate() {
        layers.push(EncoderLayerWeights {
            kind: *kind,
            attn_norm: (index > 0).then(|| norm(&mut rng, d, cfg.norm_bias)),
            wqkv: linear(&mut rng, d, 3 * d, cfg.attention_bias),
            wo: linear(&mut rng, d, d, cfg.attention_bias),
            num_heads: cfg.num_attention_heads,
            rope_base: cfg.rope_base(*kind),
            mlp_norm: norm(&mut rng, d, cfg.norm_bias),
            wi: linear(&mut rng, d, 2 * intermediate, cfg.mlp_bias),
            wo_mlp: linear(&mut rng, intermediate, d, cfg.mlp_bias),
        });
    }

    let encoder = ModernBertWeights {
        tok_embeddings: rng.fill(d * cfg.vocab_size, -0.5, 0.5),
        embed_norm: norm(&mut rng, d, cfg.norm_bias),
        layers,
        final_norm: norm(&mut rng, d, cfg.norm_bias),
    };

    let head = (0..HEAD_LAYERS)
        .map(|_| HeadLayerWeights {
            num_heads: HEAD_HEADS,
            norm1: norm(&mut rng, d, true),
            in_proj: linear(&mut rng, d, 3 * d, true),
            out_proj: linear(&mut rng, d, d, true),
            norm2: norm(&mut rng, d, true),
            linear1: linear(&mut rng, d, HEAD_FF, true),
            linear2: linear(&mut rng, HEAD_FF, d, true),
        })
        .collect();

    LayaWeights {
        encoder,
        head,
        type_emb: rng.fill(d * 3, -0.2, 0.2),
        scorer_norm: norm(&mut rng, d, true),
        scorer1: linear(&mut rng, d, d, true),
        scorer2: linear(&mut rng, d, 1, true),
        act1: linear(&mut rng, d + 4, ACTION_HIDDEN, true),
        act2: linear(&mut rng, ACTION_HIDDEN, ACTION_COUNT, true),
    }
}

/// Which set of parameter names to emit.
#[derive(Clone, Copy)]
enum Naming {
    /// Upstream PyTorch: `.in_proj_weight`, top-level `scorer.<x>` / `act_head.<x>`.
    Upstream,
    /// MLX-converted: `.in_proj.weight`, `scorer.layers.<x>` / `act_head.layers.<x>`.
    Mlx,
}

fn push_linear(
    tensors: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
    weight_name: String,
    bias_name: String,
    linear: &LinearWeights,
    out_dim: usize,
    in_dim: usize,
) {
    tensors.push((weight_name, vec![out_dim, in_dim], linear.weight.clone()));
    if let Some(bias) = &linear.bias {
        tensors.push((bias_name, vec![out_dim], bias.clone()));
    }
}

fn push_norm(
    tensors: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
    weight_name: String,
    bias_name: String,
    norm: &LayerNormWeights,
    dim: usize,
) {
    tensors.push((weight_name, vec![dim], norm.weight.clone()));
    if let Some(bias) = &norm.bias {
        tensors.push((bias_name, vec![dim], bias.clone()));
    }
}

/// Encode the reference weights in the file's row-major `(out, in)` layout.
///
/// A host `(in, out)` weight and the file's `(out, in)` weight share the same
/// flat buffer (see the checkpoint module note), so no transpose is needed.
fn model_tensors(weights: &LayaWeights, naming: Naming) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let d = HIDDEN;
    let mut tensors = Vec::new();

    tensors.push((
        "encoder.embeddings.tok_embeddings.weight".to_string(),
        vec![VOCAB, d],
        weights.encoder.tok_embeddings.clone(),
    ));
    push_norm(
        &mut tensors,
        "encoder.embeddings.norm.weight".to_string(),
        "encoder.embeddings.norm.bias".to_string(),
        &weights.encoder.embed_norm,
        d,
    );

    for (index, layer) in weights.encoder.layers.iter().enumerate() {
        let prefix = format!("encoder.layers.{index}");
        if let Some(attn_norm) = &layer.attn_norm {
            push_norm(
                &mut tensors,
                format!("{prefix}.attn_norm.weight"),
                format!("{prefix}.attn_norm.bias"),
                attn_norm,
                d,
            );
        }
        push_linear(
            &mut tensors,
            format!("{prefix}.attn.Wqkv.weight"),
            format!("{prefix}.attn.Wqkv.bias"),
            &layer.wqkv,
            3 * d,
            d,
        );
        push_linear(
            &mut tensors,
            format!("{prefix}.attn.Wo.weight"),
            format!("{prefix}.attn.Wo.bias"),
            &layer.wo,
            d,
            d,
        );
        push_norm(
            &mut tensors,
            format!("{prefix}.mlp_norm.weight"),
            format!("{prefix}.mlp_norm.bias"),
            &layer.mlp_norm,
            d,
        );
        push_linear(
            &mut tensors,
            format!("{prefix}.mlp.Wi.weight"),
            format!("{prefix}.mlp.Wi.bias"),
            &layer.wi,
            2 * INTERMEDIATE,
            d,
        );
        push_linear(
            &mut tensors,
            format!("{prefix}.mlp.Wo.weight"),
            format!("{prefix}.mlp.Wo.bias"),
            &layer.wo_mlp,
            d,
            INTERMEDIATE,
        );
    }

    push_norm(
        &mut tensors,
        "encoder.final_norm.weight".to_string(),
        "encoder.final_norm.bias".to_string(),
        &weights.encoder.final_norm,
        d,
    );

    for (index, layer) in weights.head.iter().enumerate() {
        let prefix = format!("head.layers.{index}");
        push_norm(
            &mut tensors,
            format!("{prefix}.norm1.weight"),
            format!("{prefix}.norm1.bias"),
            &layer.norm1,
            d,
        );
        let (in_weight, in_bias) = match naming {
            Naming::Upstream => (
                format!("{prefix}.self_attn.in_proj_weight"),
                format!("{prefix}.self_attn.in_proj_bias"),
            ),
            Naming::Mlx => (
                format!("{prefix}.self_attn.in_proj.weight"),
                format!("{prefix}.self_attn.in_proj.bias"),
            ),
        };
        push_linear(&mut tensors, in_weight, in_bias, &layer.in_proj, 3 * d, d);
        push_linear(
            &mut tensors,
            format!("{prefix}.self_attn.out_proj.weight"),
            format!("{prefix}.self_attn.out_proj.bias"),
            &layer.out_proj,
            d,
            d,
        );
        push_norm(
            &mut tensors,
            format!("{prefix}.norm2.weight"),
            format!("{prefix}.norm2.bias"),
            &layer.norm2,
            d,
        );
        push_linear(
            &mut tensors,
            format!("{prefix}.linear1.weight"),
            format!("{prefix}.linear1.bias"),
            &layer.linear1,
            HEAD_FF,
            d,
        );
        push_linear(
            &mut tensors,
            format!("{prefix}.linear2.weight"),
            format!("{prefix}.linear2.bias"),
            &layer.linear2,
            d,
            HEAD_FF,
        );
    }

    tensors.push((
        "type_emb.weight".to_string(),
        vec![3, d],
        weights.type_emb.clone(),
    ));

    let scorer = |index: usize| match naming {
        Naming::Upstream => format!("scorer.{index}"),
        Naming::Mlx => format!("scorer.layers.{index}"),
    };
    push_norm(
        &mut tensors,
        format!("{}.weight", scorer(0)),
        format!("{}.bias", scorer(0)),
        &weights.scorer_norm,
        d,
    );
    push_linear(
        &mut tensors,
        format!("{}.weight", scorer(1)),
        format!("{}.bias", scorer(1)),
        &weights.scorer1,
        d,
        d,
    );
    push_linear(
        &mut tensors,
        format!("{}.weight", scorer(3)),
        format!("{}.bias", scorer(3)),
        &weights.scorer2,
        1,
        d,
    );

    let act = |index: usize| match naming {
        Naming::Upstream => format!("act_head.{index}"),
        Naming::Mlx => format!("act_head.layers.{index}"),
    };
    push_linear(
        &mut tensors,
        format!("{}.weight", act(0)),
        format!("{}.bias", act(0)),
        &weights.act1,
        ACTION_HIDDEN,
        d + 4,
    );
    push_linear(
        &mut tensors,
        format!("{}.weight", act(2)),
        format!("{}.bias", act(2)),
        &weights.act2,
        ACTION_COUNT,
        ACTION_HIDDEN,
    );

    tensors
}

fn encoder_json(norm_bias: bool, attention_bias: bool, mlp_bias: bool) -> String {
    format!(
        r#"{{
            "model_type": "modernbert",
            "vocab_size": {VOCAB},
            "hidden_size": {HIDDEN},
            "intermediate_size": {INTERMEDIATE},
            "num_hidden_layers": {ENC_LAYERS},
            "num_attention_heads": {HEADS},
            "norm_bias": {norm_bias},
            "attention_bias": {attention_bias},
            "mlp_bias": {mlp_bias},
            "local_attention": 2,
            "max_position_embeddings": 64,
            "layer_types": ["full_attention", "sliding_attention"]
        }}"#
    )
}

fn agent_json() -> String {
    format!(
        r#"{{"head_layers": {HEAD_LAYERS}, "max_len": 32, "head_max_len": 8,
            "act_costs": {{"escalate": 0.5, "notify": 0.0}}}}"#
    )
}

/// Serialize one tensor into the running safetensors output.
fn push_tensor(
    header: &mut serde_json::Map<String, serde_json::Value>,
    data: &mut Vec<u8>,
    name: &str,
    shape: &[usize],
    values: &[f32],
) {
    let start = data.len();
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    let end = data.len();
    header.insert(
        name.to_string(),
        serde_json::json!({
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [start, end],
        }),
    );
}

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        push_tensor(&mut header, &mut data, name, shape, values);
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data);
    fs::write(path, out).unwrap();
}

fn temp_dir(tag: &str) -> PathBuf {
    let index = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "laya-checkpoint-{tag}-{}-{index}",
        std::process::id()
    ));
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_configs(dir: &Path, cfg: &EncoderConfig) {
    let encoder_dir = dir.join("encoder");
    fs::create_dir_all(&encoder_dir).unwrap();
    fs::write(
        encoder_dir.join("config.json"),
        encoder_json(cfg.norm_bias, cfg.attention_bias, cfg.mlp_bias),
    )
    .unwrap();
    fs::write(dir.join("rl_agent_config.json"), agent_json()).unwrap();
}

fn write_checkpoint(
    dir: &Path,
    cfg: &EncoderConfig,
    weights: &LayaWeights,
    naming: Naming,
    with_temperature: bool,
) {
    write_configs(dir, cfg);
    let mut tensors = model_tensors(weights, naming);
    if with_temperature {
        tensors.push(("temperature".to_string(), vec![3], vec![1.0, 1.0, 1.0]));
    }
    write_safetensors(&dir.join("model.safetensors"), &tensors);
}

fn linear_eq(actual: &LinearWeights, expected: &LinearWeights) -> bool {
    actual.weight == expected.weight && actual.bias == expected.bias
}

fn norm_eq(actual: &LayerNormWeights, expected: &LayerNormWeights) -> bool {
    actual.weight == expected.weight && actual.bias == expected.bias
}

fn assert_weights_eq(actual: &LayaWeights, expected: &LayaWeights) {
    let ae = &actual.encoder;
    let ee = &expected.encoder;
    assert_eq!(ae.tok_embeddings, ee.tok_embeddings);
    assert!(norm_eq(&ae.embed_norm, &ee.embed_norm));
    assert!(norm_eq(&ae.final_norm, &ee.final_norm));
    assert_eq!(ae.layers.len(), ee.layers.len());
    for (a, e) in ae.layers.iter().zip(&ee.layers) {
        assert_eq!(a.kind, e.kind);
        assert_eq!(a.num_heads, e.num_heads);
        assert_eq!(a.rope_base, e.rope_base);
        match (&a.attn_norm, &e.attn_norm) {
            (Some(a), Some(e)) => assert!(norm_eq(a, e)),
            (None, None) => {}
            _ => panic!("attn_norm presence mismatch"),
        }
        assert!(linear_eq(&a.wqkv, &e.wqkv));
        assert!(linear_eq(&a.wo, &e.wo));
        assert!(norm_eq(&a.mlp_norm, &e.mlp_norm));
        assert!(linear_eq(&a.wi, &e.wi));
        assert!(linear_eq(&a.wo_mlp, &e.wo_mlp));
    }

    assert_eq!(actual.head.len(), expected.head.len());
    for (a, e) in actual.head.iter().zip(&expected.head) {
        assert_eq!(a.num_heads, e.num_heads);
        assert!(norm_eq(&a.norm1, &e.norm1));
        assert!(linear_eq(&a.in_proj, &e.in_proj));
        assert!(linear_eq(&a.out_proj, &e.out_proj));
        assert!(norm_eq(&a.norm2, &e.norm2));
        assert!(linear_eq(&a.linear1, &e.linear1));
        assert!(linear_eq(&a.linear2, &e.linear2));
    }

    assert_eq!(actual.type_emb, expected.type_emb);
    assert!(norm_eq(&actual.scorer_norm, &expected.scorer_norm));
    assert!(linear_eq(&actual.scorer1, &expected.scorer1));
    assert!(linear_eq(&actual.scorer2, &expected.scorer2));
    assert!(linear_eq(&actual.act1, &expected.act1));
    assert!(linear_eq(&actual.act2, &expected.act2));
}

#[test]
fn loads_upstream_checkpoint_matching_reference() {
    let dir = temp_dir("upstream");
    let cfg = encoder_config();
    let reference = build_reference(&cfg, 20);
    write_checkpoint(&dir, &cfg, &reference, Naming::Upstream, true);

    let checkpoint = load_checkpoint(&dir).expect("checkpoint should load");
    assert_eq!(checkpoint.encoder, cfg);
    assert_eq!(checkpoint.agent, agent_config());
    assert_weights_eq(&checkpoint.weights, &reference);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn mlx_names_are_accepted() {
    let dir = temp_dir("mlx");
    let cfg = encoder_config();
    let reference = build_reference(&cfg, 31);
    write_checkpoint(&dir, &cfg, &reference, Naming::Mlx, false);

    let checkpoint = load_checkpoint(&dir).expect("MLX names should load");
    assert_weights_eq(&checkpoint.weights, &reference);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn disabled_biases_are_accepted() {
    let dir = temp_dir("no-bias");
    let mut cfg = encoder_config();
    cfg.norm_bias = false;
    cfg.attention_bias = false;
    cfg.mlp_bias = false;
    let reference = build_reference(&cfg, 37);
    write_checkpoint(&dir, &cfg, &reference, Naming::Upstream, false);

    let checkpoint = load_checkpoint(&dir).expect("bias-free checkpoint should load");
    assert!(
        checkpoint.weights.encoder.embed_norm.bias.is_none(),
        "encoder norm bias must be absent"
    );
    assert_weights_eq(&checkpoint.weights, &reference);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn missing_tensor_is_rejected() {
    let dir = temp_dir("missing");
    let cfg = encoder_config();
    let reference = build_reference(&cfg, 41);
    write_configs(&dir, &cfg);
    let mut tensors = model_tensors(&reference, Naming::Upstream);
    tensors.retain(|(name, _, _)| name != "encoder.layers.1.mlp.Wi.weight");
    write_safetensors(&dir.join("model.safetensors"), &tensors);

    let error = load_checkpoint(&dir).expect_err("missing tensor must fail");
    assert!(
        error.to_string().contains("mlp.Wi.weight"),
        "unexpected error message: {error}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn unexpected_tensor_is_rejected() {
    let dir = temp_dir("unexpected");
    let cfg = encoder_config();
    let reference = build_reference(&cfg, 43);
    write_configs(&dir, &cfg);
    let mut tensors = model_tensors(&reference, Naming::Upstream);
    tensors.push(("encoder.mystery.weight".to_string(), vec![1], vec![0.0]));
    write_safetensors(&dir.join("model.safetensors"), &tensors);

    let error = load_checkpoint(&dir).expect_err("unexpected tensor must fail");
    assert!(
        error.to_string().contains("mystery"),
        "unexpected error message: {error}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn wrong_shape_is_rejected() {
    let dir = temp_dir("shape");
    let cfg = encoder_config();
    let reference = build_reference(&cfg, 47);
    write_configs(&dir, &cfg);
    let mut tensors = model_tensors(&reference, Naming::Upstream);
    for (name, shape, values) in tensors.iter_mut() {
        if name == "head.layers.0.self_attn.in_proj_weight" {
            *shape = vec![HIDDEN, HIDDEN];
            values.truncate(HIDDEN * HIDDEN);
        }
    }
    write_safetensors(&dir.join("model.safetensors"), &tensors);

    assert!(load_checkpoint(&dir).is_err());

    let _ = fs::remove_dir_all(&dir);
}
