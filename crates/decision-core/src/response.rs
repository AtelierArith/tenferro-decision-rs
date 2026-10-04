use std::ops::Index;

use crate::Answer;

/// Token accounting reported by an engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Input tokens consumed.
    pub input_tokens: u64,
    /// Output tokens produced.
    pub output_tokens: u64,
}

/// A completed decision run.
///
/// Answers are ordered by the originating [`QuestionSet`](crate::QuestionSet)
/// and identified by question id.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemOneResponse {
    /// The answering model identifier.
    pub model: String,
    /// Question id → answer, in question order.
    pub answers: Vec<(String, Answer)>,
    /// Token usage, when reported.
    pub usage: Usage,
    /// Remote request identifier, when present.
    pub request_id: Option<String>,
}

impl SystemOneResponse {
    /// Look up an answer by question id.
    pub fn answer(&self, id: &str) -> Option<&Answer> {
        self.answers
            .iter()
            .find(|(candidate, _)| candidate == id)
            .map(|(_, answer)| answer)
    }
}

impl Index<&str> for SystemOneResponse {
    type Output = Answer;

    /// Panics when the id is unknown; use [`SystemOneResponse::answer`] for a
    /// fallible lookup.
    fn index(&self, id: &str) -> &Self::Output {
        self.answer(id)
            .unwrap_or_else(|| panic!("no answer for question id `{id}`"))
    }
}
