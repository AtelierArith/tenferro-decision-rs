//! Answer a `QuestionSet` with the natural-language Laya engine.
//!
//!     cargo run --release -p laya-infer --example laya_system_one -- <CHECKPOINT_DIR> [TEXT]
//!
//! `<CHECKPOINT_DIR>` is the snapshot directory printed by `hf-fetch laya`.
//! When `TEXT` is omitted a short example state is used.

use decision_core::{
    ChoiceQuestion, Content, DecisionEngine, NoulCriteria, NoulQuestion, Question, QuestionSet,
    ScoreQuestion, State,
};
use laya_infer::agent::LayaEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: laya_system_one <checkpoint_dir> [text]");
    let text = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "A traveler knocks on the door.".to_string());

    let mut engine = LayaEngine::load(dir)?;

    let mut questions = QuestionSet::new();
    questions.push(
        "next_action",
        Question::Choice(ChoiceQuestion::new(
            Content::string("What should the character do next?"),
            vec![
                ("greet".into(), Content::string("greet the visitor")),
                ("wait".into(), Content::string("keep waiting")),
                ("leave".into(), Content::string("walk away")),
            ],
        )?),
    )?;
    questions.push(
        "risk",
        Question::Score(ScoreQuestion::new(
            Content::string("How risky is this plan?"),
            vec!["low".into(), "medium".into(), "high".into()],
        )?),
    )?;
    questions.push(
        "agrees",
        Question::Noul(NoulQuestion::new(
            Content::string("Does the character agree?"),
            NoulCriteria {
                truthy: Some(Content::string("yes")),
                falsy: Some(Content::string("no")),
            },
        )?),
    )?;

    let state = State::Text(text);
    let answers = engine.system_one(&state, &questions)?;
    for ((id, _), answer) in questions.questions().iter().zip(&answers) {
        println!("{id}: {answer:?}");
    }
    Ok(())
}
