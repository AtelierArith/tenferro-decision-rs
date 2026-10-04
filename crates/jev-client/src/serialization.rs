//! The single write path from [`State`] and [`QuestionSet`] to wire JSON.
//!
//! The request is serialized directly from the typed model in insertion order —
//! never through a generic `serde_json::Value` — so object key order (which can
//! affect results) is preserved and arbitrary structs cannot be serialized
//! (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §7–§8).

use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

use decision_core::{Content, NoulCriteria, Question, QuestionSet, ScoreQuestion, State};

use crate::errors::{JevError, Result};
use crate::limits::ResourceLimits;
use crate::models::ModelRef;

/// Serialize a System One request body, enforcing local limits before writing.
pub fn serialize_request(
    state: &State,
    model: &ModelRef,
    questions: &QuestionSet,
    limits: &ResourceLimits,
) -> Result<Vec<u8>> {
    limits.validate()?;
    state.validate()?;
    questions.validate()?;

    let model_id = model.as_str();
    if model_id.len() > limits.max_model_id_bytes {
        return Err(JevError::validation_field("model", "model id is too long"));
    }
    if model_id.len() > limits.max_string_bytes {
        return Err(JevError::validation_field("model", "model id is too long"));
    }
    if questions.len() > limits.max_questions {
        return Err(JevError::validation_field(
            "questions",
            "question set exceeds the configured limit",
        ));
    }

    // Reject top-level state shapes the wire contract does not allow.
    match state {
        State::Text(text) => {
            if text.len() > limits.max_string_bytes {
                return Err(JevError::validation_field(
                    "state",
                    "state exceeds the configured string limit",
                ));
            }
        }
        State::Json(content) => {
            if !matches!(
                content,
                Content::String(_) | Content::Object(_) | Content::Array(_)
            ) {
                return Err(JevError::validation_field(
                    "state",
                    "state must be a string, object, or array",
                ));
            }
            check_content(content, limits, "state", 0)?;
        }
        State::Prepared(_) => {
            return Err(JevError::configuration(
                "the System One API does not accept prepared token states",
            ));
        }
    }

    for (id, question) in questions.questions() {
        let id = id.as_str();
        if id.chars().count() > limits.max_question_id_chars || id.len() > limits.max_string_bytes {
            return Err(JevError::validation_field(
                "questions",
                "question id exceeds the configured limit",
            ));
        }
        check_question(question, limits, id)?;
    }

    let wire = WireRequest {
        state,
        model: model_id,
        questions,
    };
    let bytes = serde_json::to_vec(&wire)
        .map_err(|_| JevError::validation("request could not be serialized"))?;
    if bytes.len() > limits.max_request_bytes {
        return Err(JevError::validation_field(
            "request",
            "request exceeds the configured byte limit",
        ));
    }
    Ok(bytes)
}

fn check_question(question: &Question, limits: &ResourceLimits, id: &str) -> Result<()> {
    let path = format!("questions.{id}");
    match question {
        Question::Choice(choice) => {
            check_content(&choice.instructions, limits, &path, 0)?;
            if choice.criteria.len() > limits.max_container_items {
                return Err(JevError::validation_field(
                    path,
                    "choice criteria exceeds the configured container limit",
                ));
            }
            for (candidate, description) in &choice.criteria {
                if candidate.len() > limits.max_string_bytes {
                    return Err(JevError::validation_field(
                        path,
                        "candidate id exceeds the configured string limit",
                    ));
                }
                check_content(description, limits, &path, 0)?;
            }
        }
        Question::Noul(noul) => {
            check_content(&noul.instructions, limits, &path, 0)?;
            check_noul_criteria(&noul.criteria, limits, &path)?;
        }
        Question::Score(score) => {
            check_content(&score.instructions, limits, &path, 0)?;
            check_score_criteria(score, limits, &path)?;
        }
    }
    Ok(())
}

fn check_noul_criteria(criteria: &NoulCriteria, limits: &ResourceLimits, path: &str) -> Result<()> {
    for value in [&criteria.truthy, &criteria.falsy].into_iter().flatten() {
        check_content(value, limits, path, 0)?;
    }
    Ok(())
}

fn check_score_criteria(score: &ScoreQuestion, limits: &ResourceLimits, path: &str) -> Result<()> {
    if score.criteria.len() > limits.max_container_items {
        return Err(JevError::validation_field(
            path,
            "score criteria exceeds the configured container limit",
        ));
    }
    for label in &score.criteria {
        if label.len() > limits.max_string_bytes {
            return Err(JevError::validation_field(
                path,
                "score label exceeds the configured string limit",
            ));
        }
    }
    Ok(())
}

fn check_content(
    content: &Content,
    limits: &ResourceLimits,
    path: &str,
    depth: usize,
) -> Result<()> {
    if depth > limits.max_json_depth {
        return Err(JevError::validation_field(
            path,
            "value exceeds the configured JSON depth",
        ));
    }
    match content {
        Content::String(value) => {
            if value.len() > limits.max_string_bytes {
                return Err(JevError::validation_field(
                    path,
                    "string exceeds the configured byte limit",
                ));
            }
        }
        Content::Array(items) => {
            if items.len() > limits.max_container_items {
                return Err(JevError::validation_field(
                    path,
                    "array exceeds the configured container limit",
                ));
            }
            for item in items {
                check_content(item, limits, path, depth + 1)?;
            }
        }
        Content::Object(pairs) => {
            if pairs.len() > limits.max_container_items {
                return Err(JevError::validation_field(
                    path,
                    "object exceeds the configured container limit",
                ));
            }
            for (key, value) in pairs {
                if key.len() > limits.max_string_bytes {
                    return Err(JevError::validation_field(
                        path,
                        "object key exceeds the configured string limit",
                    ));
                }
                check_content(value, limits, path, depth + 1)?;
            }
        }
        Content::Null | Content::Bool(_) | Content::Int(_) | Content::Float(_) => {}
    }
    Ok(())
}

struct WireRequest<'a> {
    state: &'a State,
    model: &'a str,
    questions: &'a QuestionSet,
}

impl Serialize for WireRequest<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("state", &StateWire(self.state))?;
        map.serialize_entry("model", self.model)?;
        map.serialize_entry("questions", &QuestionsWire(self.questions))?;
        map.end()
    }
}

struct StateWire<'a>(&'a State);

impl Serialize for StateWire<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self.0 {
            State::Text(text) => serializer.serialize_str(text),
            State::Json(content) => OrderedContent(content).serialize(serializer),
            State::Prepared(_) => Err(serde::ser::Error::custom(
                "prepared token states are not serializable",
            )),
        }
    }
}

struct QuestionsWire<'a>(&'a QuestionSet);

impl Serialize for QuestionsWire<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let questions = self.0.questions();
        let mut map = serializer.serialize_map(Some(questions.len()))?;
        for (id, question) in questions {
            map.serialize_entry(id.as_str(), &QuestionWire(question))?;
        }
        map.end()
    }
}

struct QuestionWire<'a>(&'a Question);

impl Serialize for QuestionWire<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        match self.0 {
            Question::Noul(question) => {
                map.serialize_entry("type", "noul")?;
                map.serialize_entry("instructions", &OrderedContent(&question.instructions))?;
                map.serialize_entry("criteria", &NoulCriteriaWire(&question.criteria))?;
            }
            Question::Choice(question) => {
                map.serialize_entry("type", "choice")?;
                map.serialize_entry("instructions", &OrderedContent(&question.instructions))?;
                map.serialize_entry("criteria", &ChoiceCriteriaWire(&question.criteria))?;
            }
            Question::Score(question) => {
                map.serialize_entry("type", "score")?;
                map.serialize_entry("instructions", &OrderedContent(&question.instructions))?;
                map.serialize_entry("criteria", &ScoreCriteriaWire(&question.criteria))?;
            }
        }
        map.end()
    }
}

struct ChoiceCriteriaWire<'a>(&'a [(String, Content)]);

impl Serialize for ChoiceCriteriaWire<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (candidate, description) in self.0 {
            map.serialize_entry(candidate, &OrderedContent(description))?;
        }
        map.end()
    }
}

struct NoulCriteriaWire<'a>(&'a NoulCriteria);

impl Serialize for NoulCriteriaWire<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let count = usize::from(self.0.truthy.is_some()) + usize::from(self.0.falsy.is_some());
        let mut map = serializer.serialize_map(Some(count))?;
        if let Some(value) = &self.0.truthy {
            map.serialize_entry("true", &OrderedContent(value))?;
        }
        if let Some(value) = &self.0.falsy {
            map.serialize_entry("false", &OrderedContent(value))?;
        }
        map.end()
    }
}

struct ScoreCriteriaWire<'a>(&'a [String]);

impl Serialize for ScoreCriteriaWire<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for label in self.0 {
            seq.serialize_element(label)?;
        }
        seq.end()
    }
}

/// Serialize [`Content`] in insertion order without going through a map type
/// that could reorder or drop keys.
struct OrderedContent<'a>(&'a Content);

impl Serialize for OrderedContent<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self.0 {
            Content::Null => serializer.serialize_unit(),
            Content::Bool(value) => serializer.serialize_bool(*value),
            Content::Int(value) => serializer.serialize_i64(*value),
            Content::Float(value) => serializer.serialize_f64(*value),
            Content::String(value) => serializer.serialize_str(value),
            Content::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(&OrderedContent(item))?;
                }
                seq.end()
            }
            Content::Object(pairs) => {
                let mut map = serializer.serialize_map(Some(pairs.len()))?;
                for (key, value) in pairs {
                    map.serialize_entry(key, &OrderedContent(value))?;
                }
                map.end()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use decision_core::{ChoiceQuestion, NoulQuestion, QuestionSet};

    fn model() -> ModelRef {
        ModelRef::Pinned(crate::models::PinnedModel::new("jev-1.13.0").unwrap())
    }

    #[test]
    fn serializes_wire_shape_in_order() {
        let mut questions = QuestionSet::new();
        questions
            .push(
                "is_urgent",
                Question::Noul(
                    NoulQuestion::new(
                        Content::string("Does this convey urgency?"),
                        NoulCriteria {
                            truthy: Some(Content::string("time-sensitive")),
                            falsy: Some(Content::string("not urgent")),
                        },
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
        questions
            .push(
                "department",
                Question::Choice(
                    ChoiceQuestion::new(
                        Content::string("Which team?"),
                        vec![
                            ("billing".into(), Content::string("Payments")),
                            ("sales".into(), Content::string("Pricing")),
                        ],
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
        questions
            .push(
                "frustration",
                Question::Score(
                    ScoreQuestion::new(
                        Content::string("How frustrated?"),
                        vec!["Calm".into(), "Angry".into()],
                    )
                    .unwrap(),
                ),
            )
            .unwrap();

        let state = State::Json(Content::object([
            ("z", Content::Int(1)),
            ("a", Content::Int(2)),
        ]));
        let bytes =
            serialize_request(&state, &model(), &questions, &ResourceLimits::default()).unwrap();
        let text = String::from_utf8(bytes).unwrap();

        // Top-level and nested keys preserve insertion order.
        assert!(text.starts_with(r#"{"state":{"z":1,"a":2},"model":"jev-1.13.0","questions":{"is_urgent":{"type":"noul","instructions":"Does this convey urgency?","criteria":{"true":"time-sensitive","false":"not urgent"}},"department":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments","sales":"Pricing"}},"frustration":{"type":"score","instructions":"How frustrated?","criteria":["Calm","Angry"]}}}"#));
    }

    #[test]
    fn rejects_null_state_top_level() {
        let questions = {
            let mut set = QuestionSet::new();
            set.push(
                "q",
                Question::Noul(
                    NoulQuestion::new(
                        Content::string("x"),
                        NoulCriteria {
                            truthy: Some(Content::Bool(true)),
                            falsy: None,
                        },
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
            set
        };
        let state = State::Json(Content::Int(3));
        let error = serialize_request(&state, &model(), &questions, &ResourceLimits::default())
            .unwrap_err();
        assert!(matches!(error, JevError::Validation { .. }));
    }
}
