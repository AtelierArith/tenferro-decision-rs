use std::collections::BTreeMap;

use decision_core::Answer;
use laya_infer::calibration::{
    Calibration, QType, action_probability, choice_answer, clamp_temperature,
    confidence_from_probs, noul_answer, score_answer, softmax, temp_bucket,
};
use laya_infer::config::{AgentConfig, EncoderConfig, LayerKind};

const ENCODER_JSON: &str = r#"{
    "vocab_size": 50368,
    "hidden_size": 768,
    "intermediate_size": 1152,
    "num_hidden_layers": 22,
    "num_attention_heads": 12,
    "model_type": "modernbert",
    "norm_eps": 1e-5,
    "hidden_activation": "gelu",
    "local_attention": 128,
    "global_attn_every_n_layers": 3,
    "global_rope_theta": 160000.0,
    "local_rope_theta": 10000.0,
    "max_position_embeddings": 8192
}"#;

#[test]
fn parses_encoder_config_and_derives_layer_types() {
    let config = EncoderConfig::from_json_str(ENCODER_JSON).unwrap();
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.head_dim(), 64);
    assert_eq!(config.layer_types.len(), 22);
    // Every third layer (indices 0, 3, 6, ...) is full attention.
    for (index, kind) in config.layer_types.iter().enumerate() {
        let expected = if index % 3 == 0 {
            LayerKind::FullAttention
        } else {
            LayerKind::SlidingAttention
        };
        assert_eq!(*kind, expected, "layer {index}");
    }
    assert_eq!(config.rope_base(LayerKind::FullAttention), 160000.0);
    assert_eq!(config.rope_base(LayerKind::SlidingAttention), 10000.0);
}

#[test]
fn explicit_layer_types_are_parsed() {
    let json = r#"{
        "vocab_size": 10,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "layer_types": ["sliding_attention", "full_attention"]
    }"#;
    let config = EncoderConfig::from_json_str(json).unwrap();
    assert_eq!(
        config.layer_types,
        vec![LayerKind::SlidingAttention, LayerKind::FullAttention]
    );
}

#[test]
fn rejects_non_modernbert_or_odd_head_dim() {
    let mut value: serde_json::Value = serde_json::from_str(ENCODER_JSON).unwrap();
    value["model_type"] = serde_json::json!("bert");
    assert!(EncoderConfig::from_json(&value).is_err());

    let json = r#"{
        "vocab_size": 10, "hidden_size": 10, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2
    }"#;
    // head_dim = 5 is odd.
    assert!(EncoderConfig::from_json_str(json).is_err());
}

#[test]
fn parses_agent_config() {
    let json = r#"{
        "head_layers": 2,
        "max_len": 512,
        "head_max_len": 192,
        "act_costs": {"notify": 0.0, "escalate": 0.5}
    }"#;
    let agent = AgentConfig::from_json_str(json).unwrap();
    assert_eq!(agent.head_layers, 2);
    assert_eq!(agent.action_count(), 3);
    assert_eq!(agent.action_names, vec!["escalate", "notify"]); // BTreeMap order? value is serde Map (preserve order off) -> sorted
}

#[test]
fn action_probability_is_softmax_first() {
    // Softmax of [0, 0] is [0.5, 0.5].
    assert!((action_probability(&[0.0, 0.0]) - 0.5).abs() < 1e-12);
    // Dominant first action.
    let p = action_probability(&[10.0, 0.0]);
    assert!(p > 0.999);
}

#[test]
fn temperature_clamping_and_buckets() {
    assert_eq!(clamp_temperature(0.1), 0.5);
    assert_eq!(clamp_temperature(10.0), 5.0);
    assert_eq!(clamp_temperature(f64::NAN), 1.0);
    assert_eq!(temp_bucket(QType::Choice, 2), "choice:2");
    assert_eq!(temp_bucket(QType::Score, 4), "score:3-5");
    assert_eq!(temp_bucket(QType::Noul, 8), "noul:6-10");
    assert_eq!(temp_bucket(QType::Choice, 11), "choice:11+");
}

#[test]
fn confidence_extremes() {
    assert!((confidence_from_probs(&[0.5, 0.5], 2) - 0.0).abs() < 1e-9);
    assert!((confidence_from_probs(&[1.0, 0.0], 2) - 1.0).abs() < 1e-6);
    assert_eq!(confidence_from_probs(&[1.0], 1), 1.0);
}

#[test]
fn calibration_uses_bucket_then_base_temperature() {
    let mut by_options = BTreeMap::new();
    by_options.insert("choice:2".to_string(), 2.0);
    let calibration = Calibration::new([1.5, 1.0, 2.0], by_options).unwrap();

    assert_eq!(calibration.scale(QType::Choice, 2), 2.0); // bucket override
    assert_eq!(calibration.scale(QType::Choice, 3), 1.5); // base
    assert_eq!(calibration.scale(QType::Noul, 2), 2.0); // base index 2
}

#[test]
fn calibration_clamps_shipped_temperatures() {
    let calibration = Calibration::new([0.1, 1.0, 9.0], BTreeMap::new()).unwrap();
    assert_eq!(calibration.scale(QType::Choice, 2), 0.5);
    assert_eq!(calibration.scale(QType::Noul, 2), 5.0);
}

#[test]
fn choice_answer_picks_argmax_and_rounds() {
    let labels: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let probabilities = [0.1, 0.7, 0.2];
    let answer = choice_answer(&labels, &probabilities);
    assert_eq!(answer.choice, "b");
    assert_eq!(
        answer.probabilities,
        vec![
            ("a".to_string(), 0.1),
            ("b".to_string(), 0.7),
            ("c".to_string(), 0.2),
        ]
    );
    assert!(answer.confidence >= 0.0 && answer.confidence <= 1.0);
}

#[test]
fn score_answer_computes_expected_value() {
    let levels: Vec<String> = ["low", "mid", "high"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let probabilities = [0.2, 0.5, 0.3];
    let answer = score_answer(&levels, &probabilities);
    assert!((answer.score - 1.1).abs() < 1e-9);
    assert_eq!(answer.legend, levels);
    assert_eq!(answer.probabilities, vec![0.2, 0.5, 0.3]);
}

#[test]
fn noul_answer_is_affirmative_probability() {
    let answer = noul_answer(&[0.3, 0.7]);
    assert!((answer.noul - 0.7).abs() < 1e-9);
}

#[test]
fn softmax_matches_manual() {
    let p = softmax(&[1.0, 1.0]);
    assert!((p[0] - 0.5).abs() < 1e-12);
    let p = softmax(&[-1.0, 1.0]);
    assert!(p[0] < p[1]);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn answer_enum_wraps_expected_variant() {
    let answer = Answer::Noul(noul_answer(&[0.4, 0.6]));
    assert!(matches!(answer, Answer::Noul(_)));
}
