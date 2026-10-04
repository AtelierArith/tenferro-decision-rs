use jeff_infer::config::{DecisionConfig, LayerKind, TextConfig};
use jeff_infer::readout::{choice_answer, noul_answer, probabilities, score_answer};

const CONFIG_JSON: &str = r#"{
    "model_type": "qwen3_5",
    "attention_bias": false,
    "hidden_act": "silu",
    "text_config": {
        "hidden_size": 512,
        "head_dim": 64,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_hidden_layers": 4,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default", "partial_rotary_factor": 0.5},
        "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"]
    }
}"#;

#[test]
fn parses_text_config() {
    let config = TextConfig::from_json_str(CONFIG_JSON).unwrap();
    assert_eq!(config.hidden_size, 512);
    assert_eq!(config.rotary_dim(), 32);
    assert_eq!(
        config.layer_types,
        vec![
            LayerKind::LinearAttention,
            LayerKind::FullAttention,
            LayerKind::LinearAttention,
            LayerKind::FullAttention
        ]
    );
}

#[test]
fn rejects_wrong_model_type_and_unsupported_attention_bias() {
    let mut value: serde_json::Value = serde_json::from_str(CONFIG_JSON).unwrap();
    value["model_type"] = serde_json::json!("qwen3");
    assert!(TextConfig::from_json(&value).is_err());

    let mut value: serde_json::Value = serde_json::from_str(CONFIG_JSON).unwrap();
    value["attention_bias"] = serde_json::json!(true);
    assert!(TextConfig::from_json(&value).is_err());
}

#[test]
fn rejects_incompatible_head_counts() {
    let json = r#"{
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 512, "head_dim": 64, "num_attention_heads": 8,
            "num_key_value_heads": 3, "num_hidden_layers": 1,
            "linear_num_key_heads": 4, "linear_num_value_heads": 8,
            "linear_key_head_dim": 32, "linear_value_head_dim": 32
        }
    }"#;
    assert!(TextConfig::from_json_str(json).is_err());
}

#[test]
fn parses_decision_config() {
    let json = r#"{"format_version": 1, "temperature": 0.75, "max_options": 255}"#;
    let config = DecisionConfig::from_json_str(json).unwrap();
    assert_eq!(config.temperature, 0.75);
    assert_eq!(config.max_options, 255);

    assert!(DecisionConfig::from_json_str(
        r#"{"format_version": 2, "temperature": 1.0, "max_options": 10}"#
    )
    .is_err());
    assert!(DecisionConfig::from_json_str(
        r#"{"format_version": 1, "temperature": 0.0, "max_options": 10}"#
    )
    .is_err());
    assert!(DecisionConfig::from_json_str(
        r#"{"format_version": 1, "temperature": 1.0, "max_options": 0}"#
    )
    .is_err());
}

#[test]
fn probabilities_are_temperature_scaled() {
    let p = probabilities(&[0.0, 0.0], 1.0).unwrap();
    assert!((p[0] - 0.5).abs() < 1e-12);

    // A lower temperature sharpens the distribution.
    let sharp = probabilities(&[1.0, 0.0], 0.5).unwrap();
    let soft = probabilities(&[1.0, 0.0], 2.0).unwrap();
    assert!(sharp[0] > soft[0]);

    assert!(probabilities(&[f64::NAN, 0.0], 1.0).is_err());
    assert!(probabilities(&[1.0, 0.0], 0.0).is_err());
}

#[test]
fn choice_confidence_follows_jeff_formula() {
    let labels: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
    let answer = choice_answer(&labels, &[1.0, 0.0], 1.0).unwrap();
    assert_eq!(answer.choice, "a");
    // n = 2: confidence = clamp(2*p_best - 1, 0, 1).
    let expected = (2.0 * answer.probabilities[0].1 - 1.0).clamp(0.0, 1.0);
    assert!((answer.confidence - expected).abs() < 1e-12);

    // Four equal options -> zero confidence.
    let labels: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
    let answer = choice_answer(&labels, &[0.0; 4], 1.0).unwrap();
    assert!(answer.confidence.abs() < 1e-12);
}

#[test]
fn noul_answer_is_true_probability() {
    let answer = noul_answer(&[0.0, 2.0], 1.0).unwrap();
    assert!(answer.noul > 0.5);
    assert!(noul_answer(&[0.0], 1.0).is_err());
}

#[test]
fn score_answer_expected_value_and_confidence() {
    let levels: Vec<String> = ["low", "mid", "high"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let answer = score_answer(&levels, &[10.0, 0.0, 0.0], 1.0).unwrap();
    assert!(answer.score < 0.01);
    assert!(answer.confidence > 0.99);
    assert_eq!(answer.legend, levels);

    let uniform = score_answer(&levels, &[0.0, 0.0, 0.0], 1.0).unwrap();
    assert!(uniform.confidence.abs() < 1e-12);
}
