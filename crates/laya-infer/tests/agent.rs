//! End-to-end tests for the text [`LayaEngine`].
//!
//! Small synthetic weights and a WordLevel tokenizer keep the suite hermetic.

use decision_core::{
    Answer, ChoiceQuestion, Content, DecisionEngine, NoulCriteria, NoulQuestion, PreparedState,
    Question, QuestionSet, ScoreQuestion, State,
};
use laya_infer::agent::LayaEngine;
use laya_infer::calibration::Calibration;
use laya_infer::config::{AgentConfig, EncoderConfig, LayerKind};
use laya_infer::model::{
    EncoderLayerWeights, HeadLayerWeights, LayaWeights, LayerNormWeights, LinearWeights,
    ModernBertWeights,
};
use laya_infer::tokenizer::BpeTokenizer;

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

fn norm(rng: &mut Lcg, d: usize) -> LayerNormWeights {
    LayerNormWeights {
        weight: rng.fill(d, 0.5, 1.5),
        bias: Some(rng.fill(d, -0.1, 0.1)),
    }
}

fn linear(rng: &mut Lcg, input: usize, output: usize) -> LinearWeights {
    LinearWeights {
        weight: rng.fill(input * output, -0.3, 0.3),
        bias: Some(rng.fill(output, -0.1, 0.1)),
    }
}

fn encoder_config() -> EncoderConfig {
    EncoderConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 8,
        num_hidden_layers: 1,
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
        layer_types: vec![LayerKind::FullAttention],
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

fn weights(cfg: &EncoderConfig, agent: &AgentConfig, rng: &mut Lcg) -> LayaWeights {
    let d = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let encoder = ModernBertWeights {
        tok_embeddings: rng.fill(d * cfg.vocab_size, -0.5, 0.5),
        embed_norm: norm(rng, d),
        layers: vec![EncoderLayerWeights {
            kind: LayerKind::FullAttention,
            attn_norm: None,
            wqkv: linear(rng, d, 3 * d),
            wo: linear(rng, d, d),
            num_heads: cfg.num_attention_heads,
            rope_base: cfg.rope_base(LayerKind::FullAttention),
            mlp_norm: norm(rng, d),
            wi: linear(rng, d, 2 * i),
            wo_mlp: linear(rng, i, d),
        }],
        final_norm: norm(rng, d),
    };
    let action_hidden = 5;
    LayaWeights {
        encoder,
        head: vec![HeadLayerWeights {
            num_heads: 1,
            norm1: norm(rng, d),
            in_proj: linear(rng, d, 3 * d),
            out_proj: linear(rng, d, d),
            norm2: norm(rng, d),
            linear1: linear(rng, d, 2 * d),
            linear2: linear(rng, 2 * d, d),
        }],
        type_emb: rng.fill(d * 3, -0.2, 0.2),
        scorer_norm: norm(rng, d),
        scorer1: linear(rng, d, d),
        scorer2: linear(rng, d, 1),
        act1: linear(rng, d + 4, action_hidden),
        act2: linear(rng, action_hidden, agent.action_count()),
    }
}

/// A tiny WordLevel tokenizer: specials plus a few known words; every other
/// word encodes to `[UNK]`. The engine only needs consistent ids, not fidelity.
fn word_tokenizer(tag: &str) -> (std::path::PathBuf, BpeTokenizer) {
    let words = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "choice", "question",
        "pick", "one", "first", "second", "third", "rate", "low", "mid", "high", "yes", "no", "is",
        "it", "so",
    ];
    let vocab: serde_json::Value = words
        .iter()
        .enumerate()
        .map(|(index, word)| (word.to_string(), serde_json::json!(index)))
        .collect::<serde_json::Map<_, _>>()
        .into();
    let added: Vec<serde_json::Value> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        .iter()
        .enumerate()
        .map(|(index, content)| {
            serde_json::json!({
                "id": index,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();
    let tokenizer = serde_json::json!({
        "added_tokens": added,
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    });
    let dir = std::env::temp_dir().join(format!("laya-agent-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    let tokenizer = BpeTokenizer::from_directory(&dir).unwrap();
    (dir, tokenizer)
}

fn engine(tag: &str) -> (std::path::PathBuf, LayaEngine) {
    let (dir, tokenizer) = word_tokenizer(tag);
    let cfg = encoder_config();
    let agent = agent_config();
    let mut rng = Lcg(7);
    let weights = weights(&cfg, &agent, &mut rng);
    let engine = LayaEngine::new(cfg, agent, weights, tokenizer, Calibration::default()).unwrap();
    (dir, engine)
}

fn question_set() -> QuestionSet {
    let mut set = QuestionSet::new();
    set.push(
        "choice",
        Question::Choice(
            ChoiceQuestion::new(
                Content::string("pick one"),
                vec![
                    ("a".into(), Content::string("first")),
                    ("b".into(), Content::string("second")),
                    ("c".into(), Content::string("third")),
                ],
            )
            .unwrap(),
        ),
    )
    .unwrap();
    set.push(
        "score",
        Question::Score(
            ScoreQuestion::new(
                Content::string("rate it"),
                vec!["low".into(), "mid".into(), "high".into()],
            )
            .unwrap(),
        ),
    )
    .unwrap();
    set.push(
        "noul",
        Question::Noul(
            NoulQuestion::new(
                Content::string("is it so?"),
                NoulCriteria {
                    truthy: Some(Content::string("yes")),
                    falsy: Some(Content::string("no")),
                },
            )
            .unwrap(),
        ),
    )
    .unwrap();
    set
}

#[test]
fn answers_choice_score_and_noul_from_text() {
    let (_dir, mut engine) = engine("text");
    let state = State::Text("hello world".to_string());
    let set = question_set();
    let answers = engine.system_one(&state, &set).unwrap();
    assert_eq!(answers.len(), 3);

    match &answers[0] {
        Answer::Choice(answer) => {
            assert_eq!(answer.probabilities.len(), 3);
            let sum: f64 = answer.probabilities.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 2e-3, "probabilities sum to {sum}");
            assert!((0.0..=1.0).contains(&answer.confidence));
        }
        other => panic!("expected choice, found {other:?}"),
    }
    match &answers[1] {
        Answer::Score(answer) => {
            assert_eq!(answer.legend, vec!["low", "mid", "high"]);
            assert!((0.0..=2.0).contains(&answer.score));
        }
        other => panic!("expected score, found {other:?}"),
    }
    match &answers[2] {
        Answer::Noul(answer) => assert!((0.0..=1.0).contains(&answer.noul)),
        other => panic!("expected noul, found {other:?}"),
    }
}

#[test]
fn decide_reports_action_probability_and_json_states_work() {
    let (_dir, mut engine) = engine("json");
    let state = State::Json(Content::object([("body", Content::string("hello world"))]));
    let decisions = engine.decide(&state, &question_set()).unwrap();
    assert_eq!(decisions.len(), 3);
    for decision in &decisions {
        assert!((0.0..=1.0).contains(&decision.action_probability));
    }
}

#[test]
fn rejects_prepared_states() {
    let (_dir, mut engine) = engine("prepared");
    let set = question_set();
    let state = State::Prepared(PreparedState {
        input_ids: vec![vec![1, 2, 3]],
        attention_mask: vec![vec![true; 3]],
    });
    assert!(engine.system_one(&state, &set).is_err());
}
