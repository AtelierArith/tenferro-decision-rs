use decision_core::{
    ChoiceQuestion, Content, NoulCriteria, NoulQuestion, Question, ScoreQuestion, State,
};
use laya_infer::prompt::{
    Tokenizer, build_prefix, build_sequence, py_float, py_json_content, render_options,
    serialize_state,
};

/// A deterministic fake: one id per ASCII byte, plus fixed special tokens.
struct FakeTokenizer;

impl Tokenizer for FakeTokenizer {
    fn encode(&self, text: &str) -> Vec<i64> {
        text.bytes().map(|b| b as i64 + 100).collect()
    }
    fn mask_token(&self) -> &str {
        "[MASK]"
    }
    fn mask_token_id(&self) -> i64 {
        1
    }
    fn cls_token_id(&self) -> i64 {
        2
    }
    fn sep_token_id(&self) -> i64 {
        3
    }
    fn pad_token_id(&self) -> i64 {
        0
    }
}

#[test]
fn py_float_matches_python_repr() {
    assert_eq!(py_float(0.0), "0.0");
    assert_eq!(py_float(-0.0), "-0.0");
    assert_eq!(py_float(1.0), "1.0");
    assert_eq!(py_float(-1.0), "-1.0");
    assert_eq!(py_float(0.5), "0.5");
    assert_eq!(py_float(123.45), "123.45");
    assert_eq!(py_float(12.0), "12.0");
    assert_eq!(py_float(100.0), "100.0");
    assert_eq!(py_float(1e-4), "0.0001");
    assert_eq!(py_float(1e-5), "1e-05");
    assert_eq!(py_float(1e15), "1000000000000000.0");
    assert_eq!(py_float(1e16), "1e+16");
    assert_eq!(py_float(1.5e-7), "1.5e-07");
    assert_eq!(py_float(f64::NAN), "NaN");
    assert_eq!(py_float(f64::INFINITY), "Infinity");
}

#[test]
fn py_json_content_uses_python_separators_and_order() {
    let value = Content::Object(vec![
        ("a".into(), Content::Int(1)),
        (
            "b".into(),
            Content::Array(vec![Content::Bool(true), Content::Null]),
        ),
        ("c".into(), Content::String("hi".into())),
    ]);
    assert_eq!(
        py_json_content(&value, false),
        r#"{"a": 1, "b": [true, null], "c": "hi"}"#
    );
}

#[test]
fn py_json_ascii_escapes_non_ascii_with_surrogate_pairs() {
    let value = Content::String("é✓𝄞".into());
    let ascii = py_json_content(&value, true);
    assert_eq!(ascii, r#""\u00e9\u2713\ud834\udd1e""#);
    let unicode = py_json_content(&value, false);
    assert_eq!(unicode, "\"é✓𝄞\"");
}

#[test]
fn serialize_state_passes_text_through() {
    assert_eq!(
        serialize_state(&State::Text("hello".into())).unwrap(),
        "hello"
    );
    let json = State::Json(Content::Object(vec![("k".into(), Content::Int(2))]));
    assert_eq!(serialize_state(&json).unwrap(), r#"{"k": 2}"#);
}

#[test]
fn render_options_covers_all_question_types() {
    let choice = Question::Choice(
        ChoiceQuestion::new(
            Content::string("pick"),
            vec![
                ("a".into(), Content::string("first")),
                ("b".into(), Content::Null),
            ],
        )
        .unwrap(),
    );
    assert_eq!(render_options(&choice), vec!["a: first", "b"]);

    let score = Question::Score(
        ScoreQuestion::new(
            Content::string("rate"),
            vec!["bad".into(), "ok".into(), "good".into()],
        )
        .unwrap(),
    );
    assert_eq!(
        render_options(&score),
        vec!["level 0: bad", "level 1: ok", "level 2: good"]
    );

    let noul = Question::Noul(
        NoulQuestion::new(
            Content::string("agree?"),
            NoulCriteria {
                truthy: Some(Content::string("yes")),
                falsy: None,
            },
        )
        .unwrap(),
    );
    assert_eq!(
        render_options(&noul),
        vec!["false: no, the statement does not hold", "true: yes"]
    );
}

#[test]
fn build_prefix_marks_each_option() {
    let tokenizer = FakeTokenizer;
    let question = Question::Choice(
        ChoiceQuestion::new(
            Content::string("pick one"),
            vec![
                ("a".into(), Content::string("alpha")),
                ("b".into(), Content::string("beta")),
            ],
        )
        .unwrap(),
    );
    let (ids, markers) = build_prefix(&tokenizer, &question, 192);
    assert_eq!(ids[0], tokenizer.cls_token_id());
    assert_eq!(*ids.last().unwrap(), tokenizer.sep_token_id());
    assert_eq!(markers.len(), 2);
    for position in &markers {
        assert_eq!(ids[*position], tokenizer.mask_token_id());
    }
    assert!(markers[0] < markers[1]);
}

#[test]
fn build_prefix_truncates_options_under_tight_budget() {
    let tokenizer = FakeTokenizer;
    let question = Question::Choice(
        ChoiceQuestion::new(
            Content::string("pick one"),
            (0..5)
                .map(|i| (format!("opt{i}"), Content::string("x".repeat(60))))
                .collect(),
        )
        .unwrap(),
    );
    // A very small head budget forces per-option truncation, but the option
    // masks must survive.
    let (ids, markers) = build_prefix(&tokenizer, &question, 40);
    assert_eq!(markers.len(), 5);
    for position in &markers {
        assert_eq!(ids[*position], tokenizer.mask_token_id());
    }
}

#[test]
fn build_sequence_inserts_state_and_truncates() {
    let tokenizer = FakeTokenizer;
    let question = Question::Noul(
        NoulQuestion::new(
            Content::string("agree?"),
            NoulCriteria {
                truthy: Some(Content::string("yes")),
                falsy: None,
            },
        )
        .unwrap(),
    );
    let state = State::Text("the state".into());
    let (ids, markers) = build_sequence(&tokenizer, &state, &question, 32, 24).unwrap();
    assert!(ids.len() <= 32);
    // Sequence ends with a separator.
    assert_eq!(*ids.last().unwrap(), tokenizer.sep_token_id());
    for position in &markers {
        assert!(*position < 32);
        assert_eq!(ids[*position], tokenizer.mask_token_id());
    }
}
