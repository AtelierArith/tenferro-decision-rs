//! Checkpoint loading: a synthetic safetensors checkpoint is written byte by
//! byte, loaded, and compared against a manually built reference. Malformed and
//! incomplete checkpoints must fail with clear errors.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use jeff_infer::checkpoint::{DEFAULT_DELTA_CHUNK_SIZE, load_checkpoint};
use jeff_infer::model::{
    AttentionWeights, FullAttentionWeights, JeffConfig, JeffWeights, LayerWeights, MlpWeights,
    forward_reference,
};
use tenferro_gated_delta::{Algorithm, GatedDeltaConfig, GatedDeltaWeights};

const HIDDEN: usize = 8;
const HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const KV_HEADS: usize = 1;
const KEY_HEADS: usize = 1;
const VALUE_HEADS: usize = 2;
const KEY_DIM: usize = 2;
const VALUE_DIM: usize = 2;
const INTERMEDIATE: usize = 16;
const VOCAB: usize = 12;
const OPTIONS: usize = 3;
const TAPS: usize = 4;
const WIDTH: usize = HEADS * HEAD_DIM;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const KEY_WIDTH: usize = KEY_DIM * KEY_HEADS;
const VALUE_WIDTH: usize = VALUE_DIM * VALUE_HEADS;
const CONV_CHANNELS: usize = 2 * KEY_WIDTH + VALUE_WIDTH;

const EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;
const ROTARY_DIM: usize = 2; // head_dim * partial_rotary_factor(0.5)

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

fn config() -> JeffConfig {
    JeffConfig {
        hidden: HIDDEN,
        heads: HEADS,
        head_dim: HEAD_DIM,
        intermediate: INTERMEDIATE,
        eps: EPS,
    }
}

/// Row-major transpose: `out[col * rows + row] = data[row * cols + col]`.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = data[row * cols + col];
        }
    }
    out
}

fn build_reference(seed: u64) -> (JeffWeights, Vec<f32>) {
    let mut rng = Lcg(seed);

    let delta_config = GatedDeltaConfig {
        hidden: HIDDEN,
        key_heads: KEY_HEADS,
        value_heads: VALUE_HEADS,
        key_dim: KEY_DIM,
        value_dim: VALUE_DIM,
        conv_taps: TAPS,
        eps: EPS,
        chunk_size: DEFAULT_DELTA_CHUNK_SIZE,
        algorithm: Algorithm::Chunked,
    };
    let a_log = rng.fill(VALUE_HEADS, -0.7, -0.1);
    let delta = GatedDeltaWeights {
        qkv: rng.fill(HIDDEN * CONV_CHANNELS, -0.3, 0.3),
        z: rng.fill(HIDDEN * VALUE_WIDTH, -0.3, 0.3),
        a: rng.fill(HIDDEN * VALUE_HEADS, -0.3, 0.3),
        b: rng.fill(HIDDEN * VALUE_HEADS, -0.3, 0.3),
        conv: rng.fill(TAPS * CONV_CHANNELS, -0.5, 0.5),
        a_decay: a_log.iter().map(|value| -value.exp()).collect(),
        dt_bias: rng.fill(VALUE_HEADS, -0.5, 0.5),
        norm: rng.fill(VALUE_DIM, 0.5, 1.5),
        out_proj: rng.fill(VALUE_WIDTH * HIDDEN, -0.3, 0.3),
    };

    // One KV head expanded to both query heads: build a single block and repeat.
    let k_block = rng.fill(HIDDEN * HEAD_DIM, -0.3, 0.3);
    let v_block = rng.fill(HIDDEN * HEAD_DIM, -0.3, 0.3);
    let mut k = vec![0.0f32; HIDDEN * WIDTH];
    let mut v = vec![0.0f32; HIDDEN * WIDTH];
    for head in 0..HEADS {
        for d in 0..HEAD_DIM {
            for input in 0..HIDDEN {
                k[input * WIDTH + head * HEAD_DIM + d] = k_block[input * HEAD_DIM + d];
                v[input * WIDTH + head * HEAD_DIM + d] = v_block[input * HEAD_DIM + d];
            }
        }
    }
    let full = FullAttentionWeights {
        q: rng.fill(HIDDEN * WIDTH, -0.3, 0.3),
        gate: rng.fill(HIDDEN * WIDTH, -0.3, 0.3),
        k,
        v,
        o: rng.fill(WIDTH * HIDDEN, -0.3, 0.3),
        q_norm: rng.fill(HEAD_DIM, 0.5, 1.5),
        k_norm: rng.fill(HEAD_DIM, 0.5, 1.5),
        rope_theta: ROPE_THETA,
        rotary_dim: ROTARY_DIM,
    };

    let layer = |rng: &mut Lcg, attention: AttentionWeights| LayerWeights {
        input_norm: rng.fill(HIDDEN, 0.5, 1.5),
        post_norm: rng.fill(HIDDEN, 0.5, 1.5),
        attention,
        mlp: MlpWeights {
            gate: rng.fill(HIDDEN * INTERMEDIATE, -0.3, 0.3),
            up: rng.fill(HIDDEN * INTERMEDIATE, -0.3, 0.3),
            down: rng.fill(INTERMEDIATE * HIDDEN, -0.3, 0.3),
        },
    };

    let layers = vec![
        layer(
            &mut rng,
            AttentionWeights::Delta {
                weights: delta,
                config: delta_config,
            },
        ),
        layer(&mut rng, AttentionWeights::Full(full)),
    ];

    let weights = JeffWeights {
        embedding: rng.fill(HIDDEN * VOCAB, -1.0, 1.0),
        vocab: VOCAB,
        layers,
        final_norm: rng.fill(HIDDEN, 0.5, 1.5),
        readout: rng.fill(HIDDEN * OPTIONS, -0.3, 0.3),
        options: OPTIONS,
    };
    (weights, a_log)
}

fn config_json() -> String {
    format!(
        r#"{{
            "model_type": "qwen3_5",
            "attention_bias": false,
            "hidden_act": "silu",
            "text_config": {{
                "hidden_size": {HIDDEN},
                "head_dim": {HEAD_DIM},
                "num_attention_heads": {HEADS},
                "num_key_value_heads": {KV_HEADS},
                "num_hidden_layers": 2,
                "intermediate_size": {INTERMEDIATE},
                "linear_num_key_heads": {KEY_HEADS},
                "linear_num_value_heads": {VALUE_HEADS},
                "linear_key_head_dim": {KEY_DIM},
                "linear_value_head_dim": {VALUE_DIM},
                "rms_norm_eps": {EPS},
                "rope_parameters": {{
                    "rope_type": "default",
                    "rope_theta": {ROPE_THETA},
                    "partial_rotary_factor": 0.5
                }},
                "layer_types": ["linear_attention", "full_attention"]
            }}
        }}"#
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

/// Encode the reference weights in PyTorch `(out, in)` row-major layout.
fn model_tensors(
    weights: &JeffWeights,
    a_log: &[f32],
) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let mut tensors = Vec::new();

    tensors.push((
        "language_model.embed_tokens.weight",
        vec![VOCAB, HIDDEN],
        transpose(&weights.embedding, HIDDEN, VOCAB),
    ));

    // Layer 0: Gated DeltaNet.
    let delta = match &weights.layers[0].attention {
        AttentionWeights::Delta { weights, .. } => weights,
        AttentionWeights::Full(_) => panic!("layer 0 must be a Delta layer"),
    };
    tensors.push((
        "language_model.layers.0.input_layernorm.weight",
        vec![HIDDEN],
        weights.layers[0].input_norm.clone(),
    ));
    tensors.push((
        "language_model.layers.0.post_attention_layernorm.weight",
        vec![HIDDEN],
        weights.layers[0].post_norm.clone(),
    ));
    tensors.push((
        "language_model.layers.0.linear_attn.in_proj_qkv.weight",
        vec![CONV_CHANNELS, HIDDEN],
        transpose(&delta.qkv, HIDDEN, CONV_CHANNELS),
    ));
    tensors.push((
        "language_model.layers.0.linear_attn.in_proj_z.weight",
        vec![VALUE_WIDTH, HIDDEN],
        transpose(&delta.z, HIDDEN, VALUE_WIDTH),
    ));
    tensors.push((
        "language_model.layers.0.linear_attn.in_proj_a.weight",
        vec![VALUE_HEADS, HIDDEN],
        transpose(&delta.a, HIDDEN, VALUE_HEADS),
    ));
    tensors.push((
        "language_model.layers.0.linear_attn.in_proj_b.weight",
        vec![VALUE_HEADS, HIDDEN],
        transpose(&delta.b, HIDDEN, VALUE_HEADS),
    ));
    // conv host `(taps, channels)` back to `(channels, 1, taps)`.
    let mut conv_file = vec![0.0f32; CONV_CHANNELS * TAPS];
    for tap in 0..TAPS {
        for channel in 0..CONV_CHANNELS {
            conv_file[channel * TAPS + tap] = delta.conv[tap * CONV_CHANNELS + channel];
        }
    }
    tensors.push((
        "language_model.layers.0.linear_attn.conv1d.weight",
        vec![CONV_CHANNELS, 1, TAPS],
        conv_file,
    ));
    // Fixed inference weights: the loader recomputes `a_decay = -exp(A_log)`.
    tensors.push((
        "language_model.layers.0.linear_attn.A_log",
        vec![VALUE_HEADS],
        a_log.to_vec(),
    ));
    tensors.push((
        "language_model.layers.0.linear_attn.dt_bias",
        vec![VALUE_HEADS],
        delta.dt_bias.clone(),
    ));
    tensors.push((
        "language_model.layers.0.linear_attn.norm.weight",
        vec![VALUE_DIM],
        delta.norm.clone(),
    ));
    tensors.push((
        "language_model.layers.0.linear_attn.out_proj.weight",
        vec![HIDDEN, VALUE_WIDTH],
        transpose(&delta.out_proj, VALUE_WIDTH, HIDDEN),
    ));
    push_mlp(&mut tensors, 0, &weights.layers[0].mlp);

    // Layer 1: full attention.
    let full = match &weights.layers[1].attention {
        AttentionWeights::Full(weights) => weights,
        AttentionWeights::Delta { .. } => panic!("layer 1 must be a full layer"),
    };
    tensors.push((
        "language_model.layers.1.input_layernorm.weight",
        vec![HIDDEN],
        weights.layers[1].input_norm.clone(),
    ));
    tensors.push((
        "language_model.layers.1.post_attention_layernorm.weight",
        vec![HIDDEN],
        weights.layers[1].post_norm.clone(),
    ));
    // Fused q_proj, per head `[query(head_dim), gate(head_dim)]`.
    let mut q_fused = vec![0.0f32; 2 * WIDTH * HIDDEN];
    for head in 0..HEADS {
        for d in 0..HEAD_DIM {
            let query_row = head * 2 * HEAD_DIM + d;
            let gate_row = query_row + HEAD_DIM;
            for input in 0..HIDDEN {
                let column = head * HEAD_DIM + d;
                q_fused[query_row * HIDDEN + input] = full.q[input * WIDTH + column];
                q_fused[gate_row * HIDDEN + input] = full.gate[input * WIDTH + column];
            }
        }
    }
    tensors.push((
        "language_model.layers.1.self_attn.q_proj.weight",
        vec![2 * WIDTH, HIDDEN],
        q_fused,
    ));
    // Single KV head: store query head 0's block.
    let mut k_file = vec![0.0f32; KV_WIDTH * HIDDEN];
    let mut v_file = vec![0.0f32; KV_WIDTH * HIDDEN];
    for d in 0..HEAD_DIM {
        for input in 0..HIDDEN {
            k_file[d * HIDDEN + input] = full.k[input * WIDTH + d];
            v_file[d * HIDDEN + input] = full.v[input * WIDTH + d];
        }
    }
    tensors.push((
        "language_model.layers.1.self_attn.k_proj.weight",
        vec![KV_WIDTH, HIDDEN],
        k_file,
    ));
    tensors.push((
        "language_model.layers.1.self_attn.v_proj.weight",
        vec![KV_WIDTH, HIDDEN],
        v_file,
    ));
    tensors.push((
        "language_model.layers.1.self_attn.o_proj.weight",
        vec![HIDDEN, WIDTH],
        transpose(&full.o, WIDTH, HIDDEN),
    ));
    tensors.push((
        "language_model.layers.1.self_attn.q_norm.weight",
        vec![HEAD_DIM],
        full.q_norm.clone(),
    ));
    tensors.push((
        "language_model.layers.1.self_attn.k_norm.weight",
        vec![HEAD_DIM],
        full.k_norm.clone(),
    ));
    push_mlp(&mut tensors, 1, &weights.layers[1].mlp);

    tensors.push((
        "language_model.norm.weight",
        vec![HIDDEN],
        weights.final_norm.clone(),
    ));

    tensors
}

fn push_mlp(
    tensors: &mut Vec<(&'static str, Vec<usize>, Vec<f32>)>,
    layer: usize,
    mlp: &MlpWeights,
) {
    tensors.push((
        match layer {
            0 => "language_model.layers.0.mlp.gate_proj.weight",
            _ => "language_model.layers.1.mlp.gate_proj.weight",
        },
        vec![INTERMEDIATE, HIDDEN],
        transpose(&mlp.gate, HIDDEN, INTERMEDIATE),
    ));
    tensors.push((
        match layer {
            0 => "language_model.layers.0.mlp.up_proj.weight",
            _ => "language_model.layers.1.mlp.up_proj.weight",
        },
        vec![INTERMEDIATE, HIDDEN],
        transpose(&mlp.up, HIDDEN, INTERMEDIATE),
    ));
    tensors.push((
        match layer {
            0 => "language_model.layers.0.mlp.down_proj.weight",
            _ => "language_model.layers.1.mlp.down_proj.weight",
        },
        vec![HIDDEN, INTERMEDIATE],
        transpose(&mlp.down, INTERMEDIATE, HIDDEN),
    ));
}

fn write_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
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
        "jeff-checkpoint-{tag}-{}-{index}",
        std::process::id()
    ));
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_configs(dir: &Path) {
    fs::write(dir.join("config.json"), config_json()).unwrap();
    fs::write(
        dir.join("decision_config.json"),
        format!(r#"{{"format_version": 1, "temperature": 1.25, "max_options": {OPTIONS}}}"#),
    )
    .unwrap();
}

fn write_readout(dir: &Path, weights: &JeffWeights) {
    write_safetensors(
        &dir.join("readout.safetensors"),
        &[(
            "weight",
            vec![OPTIONS, HIDDEN],
            transpose(&weights.readout, HIDDEN, OPTIONS),
        )],
    );
}

fn write_full_checkpoint(dir: &Path, weights: &JeffWeights, a_log: &[f32]) {
    write_configs(dir);
    write_safetensors(
        &dir.join("model.safetensors"),
        &model_tensors(weights, a_log),
    );
    write_readout(dir, weights);
}

#[allow(clippy::too_many_lines)]
fn assert_weights_eq(loaded: &JeffWeights, reference: &JeffWeights) {
    assert_eq!(loaded.vocab, reference.vocab);
    assert_eq!(loaded.options, reference.options);
    assert_eq!(loaded.embedding, reference.embedding);
    assert_eq!(loaded.final_norm, reference.final_norm);
    assert_eq!(loaded.readout, reference.readout);
    assert_eq!(loaded.layers.len(), reference.layers.len());
    for (actual, expected) in loaded.layers.iter().zip(&reference.layers) {
        assert_eq!(actual.input_norm, expected.input_norm);
        assert_eq!(actual.post_norm, expected.post_norm);
        assert_eq!(actual.mlp.gate, expected.mlp.gate);
        assert_eq!(actual.mlp.up, expected.mlp.up);
        assert_eq!(actual.mlp.down, expected.mlp.down);
        match (&actual.attention, &expected.attention) {
            (AttentionWeights::Full(actual), AttentionWeights::Full(expected)) => {
                assert_eq!(actual.q, expected.q);
                assert_eq!(actual.gate, expected.gate);
                assert_eq!(actual.k, expected.k);
                assert_eq!(actual.v, expected.v);
                assert_eq!(actual.o, expected.o);
                assert_eq!(actual.q_norm, expected.q_norm);
                assert_eq!(actual.k_norm, expected.k_norm);
                assert_eq!(actual.rope_theta, expected.rope_theta);
                assert_eq!(actual.rotary_dim, expected.rotary_dim);
            }
            (
                AttentionWeights::Delta {
                    weights: actual,
                    config: actual_config,
                },
                AttentionWeights::Delta {
                    weights: expected,
                    config: expected_config,
                },
            ) => {
                assert_eq!(actual_config, expected_config);
                assert_eq!(actual.qkv, expected.qkv);
                assert_eq!(actual.z, expected.z);
                assert_eq!(actual.a, expected.a);
                assert_eq!(actual.b, expected.b);
                assert_eq!(actual.conv, expected.conv);
                assert_eq!(actual.a_decay, expected.a_decay);
                assert_eq!(actual.dt_bias, expected.dt_bias);
                assert_eq!(actual.norm, expected.norm);
                assert_eq!(actual.out_proj, expected.out_proj);
            }
            _ => panic!("attention kind mismatch"),
        }
    }
}

#[test]
fn loads_synthetic_checkpoint_matching_reference() {
    let dir = temp_dir("roundtrip");
    let (reference, a_log) = build_reference(20);
    write_full_checkpoint(&dir, &reference, &a_log);

    let checkpoint = load_checkpoint(&dir).expect("checkpoint should load");
    assert_eq!(checkpoint.config, config());
    assert_eq!(checkpoint.text.rotary_dim(), ROTARY_DIM);
    assert_eq!(checkpoint.weights.vocab, VOCAB);
    assert_eq!(checkpoint.weights.options, OPTIONS);
    assert_weights_eq(&checkpoint.weights, &reference);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn loaded_weights_run_the_reference_forward() {
    let dir = temp_dir("forward");
    let (reference, a_log) = build_reference(31);
    write_full_checkpoint(&dir, &reference, &a_log);

    let checkpoint = load_checkpoint(&dir).expect("checkpoint should load");
    let ids = vec![3_i64, 1, 4, 1, 5, 2];
    let mask = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 1.0];
    let logits = forward_reference(&checkpoint.config, &checkpoint.weights, &ids, &mask)
        .expect("forward should run");
    assert_eq!(logits.len(), OPTIONS);
    assert!(logits.iter().all(|value| value.is_finite()));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn missing_tensor_is_rejected() {
    let dir = temp_dir("missing");
    let (reference, a_log) = build_reference(41);
    write_configs(&dir);
    let mut tensors = model_tensors(&reference, &a_log);
    tensors.retain(|(name, _, _)| *name != "language_model.layers.1.mlp.up_proj.weight");
    write_safetensors(&dir.join("model.safetensors"), &tensors);
    write_readout(&dir, &reference);

    let error = load_checkpoint(&dir).expect_err("missing tensor must fail");
    assert!(
        error.to_string().contains("up_proj"),
        "unexpected error message: {error}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn unexpected_tensor_is_rejected() {
    let dir = temp_dir("unexpected");
    let (reference, a_log) = build_reference(43);
    write_configs(&dir);
    let mut tensors = model_tensors(&reference, &a_log);
    tensors.push(("language_model.mystery.weight", vec![1], vec![0.0]));
    write_safetensors(&dir.join("model.safetensors"), &tensors);
    write_readout(&dir, &reference);

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
    let (reference, a_log) = build_reference(47);
    write_configs(&dir);
    let mut tensors = model_tensors(&reference, &a_log);
    for (name, shape, values) in tensors.iter_mut() {
        if *name == "language_model.layers.1.self_attn.q_proj.weight" {
            *shape = vec![WIDTH, HIDDEN];
            values.truncate(WIDTH * HIDDEN);
        }
    }
    write_safetensors(&dir.join("model.safetensors"), &tensors);
    write_readout(&dir, &reference);

    assert!(load_checkpoint(&dir).is_err());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn malformed_header_is_rejected() {
    let dir = temp_dir("malformed");
    let (reference, _) = build_reference(53);
    write_configs(&dir);
    // Header length larger than the remaining file.
    let mut bytes = vec![0u8; 8];
    bytes[0..8].copy_from_slice(&(1024u64).to_le_bytes());
    fs::write(dir.join("model.safetensors"), bytes).unwrap();
    write_readout(&dir, &reference);

    assert!(load_checkpoint(&dir).is_err());

    let _ = fs::remove_dir_all(&dir);
}
