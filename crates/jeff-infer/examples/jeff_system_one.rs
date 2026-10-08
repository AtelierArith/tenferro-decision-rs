//! Answer a `QuestionSet` with prepared tokens or a natural-language state.
//!
//!     cargo run --release -p jeff-infer --example jeff_system_one -- <CHECKPOINT_DIR>
//!
//! `<CHECKPOINT_DIR>` is the snapshot directory printed by `hf-fetch jeff`.
//! Prepared states are `(input_ids, attention_mask)` rows; `row i` answers
//! question `i`, and leading padding is trimmed by the engine. Pass `--tenferro`
//! to run the tenferro-native forward instead of the default optimized host one.
//! Pass `--text "your state"` to use the checkpoint tokenizer and chat template.

use decision_core::{
    ChoiceQuestion, Content, DecisionEngine, PreparedState, Question, QuestionSet, State,
};
use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::engine::{JeffBackend, JeffEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: jeff_system_one <checkpoint_dir> [--tenferro] [--text STATE]");
    let backend = if args.iter().any(|arg| arg == "--tenferro") {
        JeffBackend::Tenferro
    } else {
        JeffBackend::Auto
    };

    let checkpoint = load_checkpoint(dir)?;
    let mut engine = JeffEngine::with_backend(
        checkpoint.config,
        checkpoint.decision,
        checkpoint.weights,
        backend,
    )?;

    // One prepared row per question; `row 0` answers `questions[0]`.
    let state = if let Some(index) = args.iter().position(|arg| arg == "--text") {
        engine =
            engine.with_tokenizer(jeff_infer::tokenizer::JeffTokenizer::from_directory(dir)?)?;
        State::Text(
            args.get(index + 1)
                .ok_or("--text requires a state")?
                .clone(),
        )
    } else {
        State::Prepared(PreparedState {
            input_ids: vec![vec![1, 2, 3, 4]],
            attention_mask: vec![vec![true; 4]],
        })
    };

    let mut questions = QuestionSet::new();
    questions.push(
        "next_action",
        Question::Choice(ChoiceQuestion::new(
            Content::string("Pick the next action."),
            vec![
                ("a".into(), Content::string("first")),
                ("b".into(), Content::string("second")),
                ("c".into(), Content::string("third")),
            ],
        )?),
    )?;

    for answer in engine.system_one(&state, &questions)? {
        println!("{answer:?}");
    }
    Ok(())
}
