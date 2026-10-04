//! Strict response parsing and cross-field validation.
//!
//! Responses are untrusted. Before materializing JSON we scan for depth,
//! duplicate keys, container-size, and UTF-8 violations, then validate every
//! answer against the question that produced it
//! (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §9). No raw body is
//! retained or exposed.

use std::collections::HashSet;

use serde_json::{Map, Value};

use decision_core::{
    Answer, ChoiceAnswer, NoulAnswer, Question, QuestionSet, ScoreAnswer, SystemOneResponse, Usage,
};

use crate::errors::{JevError, Result};
use crate::limits::ResourceLimits;
use crate::models::{ModelInfo, ModelList, validate_iso_date};

/// Probability sums must be within this tolerance of one.
const SUM_TOLERANCE: f64 = 1e-4;
/// The chosen candidate's probability must be within this tolerance of the max.
const ARGMAX_TOLERANCE: f64 = 1e-6;

/// Validate bounded, duplicate-free JSON before parsing.
pub(crate) fn scan_json(bytes: &[u8], limits: &ResourceLimits) -> Result<()> {
    if bytes.len() > limits.max_response_bytes {
        return Err(JevError::ResponseTooLarge { status: None });
    }
    std::str::from_utf8(bytes).map_err(|_| malformed("response is not valid UTF-8"))?;
    let mut scanner = Scanner {
        bytes,
        pos: 0,
        limits,
    };
    scanner.scan_value(0)?;
    scanner.skip_whitespace();
    if scanner.pos != bytes.len() {
        return Err(malformed("trailing JSON data"));
    }
    Ok(())
}

struct Scanner<'a> {
    bytes: &'a [u8],
    pos: usize,
    limits: &'a ResourceLimits,
}

impl Scanner<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(byte) = self.peek() {
            if matches!(byte, 0x20 | 0x09 | 0x0a | 0x0d) {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn scan_value(&mut self, depth: usize) -> Result<()> {
        if depth > self.limits.max_json_depth {
            return Err(malformed("JSON value exceeds maximum depth"));
        }
        self.skip_whitespace();
        let byte = self
            .peek()
            .ok_or_else(|| malformed("unexpected end of JSON"))?;
        match byte {
            b'{' => self.scan_object(depth),
            b'[' => self.scan_array(depth),
            b'"' => {
                self.scan_string()?;
                Ok(())
            }
            b't' => self.scan_literal(b"true"),
            b'f' => self.scan_literal(b"false"),
            b'n' => self.scan_literal(b"null"),
            b'-' | b'0'..=b'9' => self.scan_number(),
            _ => Err(malformed("unexpected character in JSON")),
        }
    }

    fn scan_object(&mut self, depth: usize) -> Result<()> {
        self.pos += 1; // consume '{'
        self.skip_whitespace();
        let mut keys: HashSet<String> = HashSet::new();
        let mut count = 0usize;
        if self.peek() != Some(b'}') {
            loop {
                count += 1;
                if count > self.limits.max_container_items {
                    return Err(malformed("JSON object is too large"));
                }
                self.skip_whitespace();
                let key = self.scan_string()?;
                if !keys.insert(key) {
                    return Err(malformed("duplicate JSON object key"));
                }
                self.skip_whitespace();
                if self.peek() != Some(b':') {
                    return Err(malformed("expected `:` in JSON object"));
                }
                self.pos += 1;
                self.scan_value(depth + 1)?;
                self.skip_whitespace();
                match self.peek() {
                    Some(b'}') => break,
                    Some(b',') => {
                        self.pos += 1;
                        self.skip_whitespace();
                    }
                    _ => return Err(malformed("expected `,` or `}` in JSON object")),
                }
            }
        }
        if self.peek() != Some(b'}') {
            return Err(malformed("unterminated JSON object"));
        }
        self.pos += 1;
        Ok(())
    }

    fn scan_array(&mut self, depth: usize) -> Result<()> {
        self.pos += 1; // consume '['
        self.skip_whitespace();
        let mut count = 0usize;
        if self.peek() != Some(b']') {
            loop {
                count += 1;
                if count > self.limits.max_container_items {
                    return Err(malformed("JSON array is too large"));
                }
                self.scan_value(depth + 1)?;
                self.skip_whitespace();
                match self.peek() {
                    Some(b']') => break,
                    Some(b',') => {
                        self.pos += 1;
                        self.skip_whitespace();
                    }
                    _ => return Err(malformed("expected `,` or `]` in JSON array")),
                }
            }
        }
        if self.peek() != Some(b']') {
            return Err(malformed("unterminated JSON array"));
        }
        self.pos += 1;
        Ok(())
    }

    fn scan_string(&mut self) -> Result<String> {
        if self.peek() != Some(b'"') {
            return Err(malformed("expected a JSON string"));
        }
        let start = self.pos;
        self.pos += 1;
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| malformed("unterminated JSON string"))?;
            match byte {
                b'"' => {
                    self.pos += 1;
                    break;
                }
                b'\\' => {
                    self.pos += 1;
                    let escaped = self
                        .peek()
                        .ok_or_else(|| malformed("invalid JSON escape"))?;
                    match escaped {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => self.pos += 1,
                        b'u' => {
                            self.pos += 1;
                            for _ in 0..4 {
                                let hex = self
                                    .peek()
                                    .ok_or_else(|| malformed("invalid unicode escape"))?;
                                if !hex.is_ascii_hexdigit() {
                                    return Err(malformed("invalid unicode escape"));
                                }
                                self.pos += 1;
                            }
                        }
                        _ => return Err(malformed("invalid JSON escape")),
                    }
                }
                0x00..=0x1f => return Err(malformed("control character in JSON string")),
                _ => self.pos += 1,
            }
        }
        serde_json::from_slice::<String>(&self.bytes[start..self.pos])
            .map_err(|_| malformed("invalid JSON string"))
    }

    fn scan_literal(&mut self, literal: &[u8]) -> Result<()> {
        let end = self.pos + literal.len();
        if end > self.bytes.len() || &self.bytes[self.pos..end] != literal {
            return Err(malformed("invalid JSON literal"));
        }
        self.pos = end;
        Ok(())
    }

    fn scan_number(&mut self) -> Result<()> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(malformed("invalid JSON number"));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(malformed("invalid JSON number")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(malformed("invalid JSON number"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(malformed("invalid JSON number"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if let Some(byte) = self.peek() {
            if !matches!(byte, 0x20 | 0x09 | 0x0a | 0x0d | b',' | b']' | b'}') {
                return Err(malformed("invalid JSON number"));
            }
        }
        if self.pos == start {
            return Err(malformed("empty JSON number"));
        }
        Ok(())
    }
}

fn malformed(message: impl Into<String>) -> JevError {
    JevError::MalformedJson {
        message: message.into(),
    }
}

fn parse_value(bytes: &[u8], limits: &ResourceLimits) -> Result<Value> {
    scan_json(bytes, limits)?;
    serde_json::from_slice(bytes).map_err(|_| malformed("response contains malformed JSON"))
}

/// Parse and validate a System One response against the questions sent.
pub(crate) fn parse_system_one_response(
    bytes: &[u8],
    questions: &QuestionSet,
    limits: &ResourceLimits,
    request_id: Option<String>,
) -> Result<SystemOneResponse> {
    let value = parse_value(bytes, limits)?;
    let object = value
        .as_object()
        .ok_or_else(|| rv("response top-level must be an object", "response"))?;

    let model = nonempty_string(object.get("model"), "model")?;
    let usage = parse_usage(object.get("usage"))?;
    let answers = object
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| rv("answers must be an object", "answers"))?;

    let expected: Vec<&str> = questions
        .questions()
        .iter()
        .map(|(id, _)| id.as_str())
        .collect();
    let actual: HashSet<&str> = answers.keys().map(String::as_str).collect();
    let expected_set: HashSet<&str> = expected.iter().copied().collect();
    if actual != expected_set {
        return Err(rv(
            "answer ids do not match the sent question ids",
            "answers",
        ));
    }

    let mut collected = Vec::with_capacity(questions.len());
    for (id, question) in questions.questions() {
        let answer_value = answers
            .get(id.as_str())
            .expect("key set equality checked above");
        let answer = parse_answer(id.as_str(), question, answer_value)?;
        collected.push((id.as_str().to_string(), answer));
    }

    Ok(SystemOneResponse {
        model,
        answers: collected,
        usage,
        request_id,
    })
}

/// Parse and validate the model list response.
pub(crate) fn parse_model_list(bytes: &[u8], limits: &ResourceLimits) -> Result<ModelList> {
    let value = parse_value(bytes, limits)?;
    let object = value
        .as_object()
        .ok_or_else(|| rv("model list must be an object", "models"))?;
    let models = object
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| rv("models must be an array", "models"))?;
    if models.len() > limits.max_container_items {
        return Err(rv("models array is too large", "models"));
    }
    let mut parsed = Vec::with_capacity(models.len());
    for (index, entry) in models.iter().enumerate() {
        let path = format!("models[{index}]");
        let entry = entry
            .as_object()
            .ok_or_else(|| rv("model entry must be an object", &path))?;
        let name = nonempty_string(entry.get("name"), &format!("{path}.name"))?;
        let description = entry
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let release_date =
            nonempty_string(entry.get("release_date"), &format!("{path}.release_date"))?;
        validate_iso_date(&release_date)?;
        parsed.push(ModelInfo {
            name,
            description,
            release_date,
        });
    }
    Ok(ModelList { models: parsed })
}

fn parse_usage(value: Option<&Value>) -> Result<Usage> {
    let usage = value
        .and_then(Value::as_object)
        .ok_or_else(|| rv("usage must be an object", "usage"))?;
    Ok(Usage {
        input_tokens: nonnegative_int(usage.get("input_tokens"), "usage.input_tokens")?,
        output_tokens: nonnegative_int(usage.get("output_tokens"), "usage.output_tokens")?,
    })
}

fn parse_answer(id: &str, question: &Question, value: &Value) -> Result<Answer> {
    let path = format!("answers.{id}");
    let object = value
        .as_object()
        .ok_or_else(|| rv("answer must be an object", &path))?;
    let answer_type = nonempty_string(object.get("type"), &format!("{path}.type"))?;
    match question {
        Question::Noul(_) => {
            expect_type(&answer_type, "noul", &path)?;
            let noul = finite_number(object.get("noul"), &format!("{path}.noul"))?;
            if !(0.0..=1.0).contains(&noul) {
                return Err(rv("noul probability is outside [0, 1]", &path));
            }
            Ok(Answer::Noul(NoulAnswer { noul }))
        }
        Question::Choice(choice) => {
            expect_type(&answer_type, "choice", &path)?;
            let candidate_ids: Vec<&str> =
                choice.criteria.iter().map(|(id, _)| id.as_str()).collect();
            let candidate_set: HashSet<&str> = candidate_ids.iter().copied().collect();

            let selected = nonempty_string(object.get("choice"), &format!("{path}.choice"))?;
            if !candidate_set.contains(selected.as_str()) {
                return Err(rv("choice is not a known candidate", &path));
            }

            let probabilities = object
                .get("probabilities")
                .and_then(Value::as_object)
                .ok_or_else(|| rv("probabilities must be an object", &path))?;
            let probability_keys: HashSet<&str> =
                probabilities.keys().map(String::as_str).collect();
            if probability_keys != candidate_set {
                return Err(rv("probability keys do not match the candidates", &path));
            }

            let mut ordered = Vec::with_capacity(candidate_ids.len());
            let mut total = 0.0_f64;
            let mut max_probability = 0.0_f64;
            for candidate in &candidate_ids {
                let probability = finite_number(
                    probabilities.get(*candidate),
                    &format!("{path}.probabilities.{candidate}"),
                )?;
                if !(0.0..=1.0).contains(&probability) {
                    return Err(rv("probability is outside [0, 1]", &path));
                }
                total += probability;
                max_probability = max_probability.max(probability);
                ordered.push(((*candidate).to_string(), probability));
            }
            if (total - 1.0).abs() > SUM_TOLERANCE {
                return Err(rv("probabilities do not sum to one", &path));
            }
            let chosen = ordered
                .iter()
                .find(|(candidate, _)| candidate == &selected)
                .map(|(_, probability)| *probability)
                .expect("selected candidate is known");
            if (chosen - max_probability).abs() > ARGMAX_TOLERANCE {
                return Err(rv("choice is not the maximum-probability candidate", &path));
            }
            let confidence = bounded_confidence(object.get("confidence"), &path)?;
            Ok(Answer::Choice(ChoiceAnswer {
                choice: selected,
                probabilities: ordered,
                confidence,
            }))
        }
        Question::Score(score) => {
            expect_type(&answer_type, "score", &path)?;
            let count = score.criteria.len();
            let legend = score_field(object.get("legend"), count, &path)?
                .into_iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|text| !text.is_empty())
                        .map(str::to_string)
                        .ok_or_else(|| rv("legend entry must be a non-empty string", &path))
                })
                .collect::<Result<Vec<String>>>()?;
            let probabilities = score_field(object.get("probabilities"), count, &path)?
                .into_iter()
                .map(|value| finite_number(Some(&value), &path))
                .collect::<Result<Vec<f64>>>()?;
            if legend.len() != count || probabilities.len() != count {
                return Err(rv("score legend/probabilities length is invalid", &path));
            }
            let mut total = 0.0;
            for probability in &probabilities {
                if !(0.0..=1.0).contains(probability) {
                    return Err(rv("score probability is outside [0, 1]", &path));
                }
                total += probability;
            }
            if (total - 1.0).abs() > SUM_TOLERANCE {
                return Err(rv("score probabilities do not sum to one", &path));
            }
            let value = finite_number(object.get("score"), &format!("{path}.score"))?;
            let upper = (count - 1) as f64;
            if !(0.0..=upper).contains(&value) {
                return Err(rv("score is outside the level range", &path));
            }
            let expected: f64 = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum();
            if (value - expected).abs() > SUM_TOLERANCE {
                return Err(rv("score is inconsistent with probabilities", &path));
            }
            let confidence = bounded_confidence(object.get("confidence"), &path)?;
            Ok(Answer::Score(ScoreAnswer {
                score: value,
                legend,
                probabilities,
                confidence,
            }))
        }
    }
}

fn expect_type(actual: &str, expected: &str, path: &str) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(rv(
            "answer type does not match the question type",
            &format!("{path}.type"),
        ))
    }
}

fn bounded_confidence(value: Option<&Value>, path: &str) -> Result<f64> {
    let confidence = finite_number(value, &format!("{path}.confidence"))?;
    if !(0.0..=1.0).contains(&confidence) {
        return Err(rv("confidence is outside [0, 1]", path));
    }
    Ok(confidence)
}

/// Extract a score field given as an ordered array or a canonical 0-origin
/// object keyed by `"0".."n-1"`.
fn score_field(value: Option<&Value>, count: usize, path: &str) -> Result<Vec<Value>> {
    let value = value.ok_or_else(|| rv("score field is missing", path))?;
    match value {
        Value::Array(items) => {
            if items.len() != count {
                return Err(rv("score field length is invalid", path));
            }
            Ok(items.clone())
        }
        Value::Object(map) => object_score_field(map, count, path),
        _ => Err(rv("score field must be an array or object", path)),
    }
}

fn object_score_field(map: &Map<String, Value>, count: usize, path: &str) -> Result<Vec<Value>> {
    let mut ordered: Vec<Option<Value>> = vec![None; count];
    for (key, value) in map {
        let index: usize = key
            .parse()
            .map_err(|_| rv("score index is not an integer", &format!("{path}.{key}")))?;
        if key != &index.to_string() {
            return Err(rv("score index is not canonical", &format!("{path}.{key}")));
        }
        if index >= count {
            return Err(rv(
                "score index is outside the level range",
                &format!("{path}.{key}"),
            ));
        }
        if ordered[index].is_some() {
            return Err(rv("duplicate score index", &format!("{path}.{key}")));
        }
        ordered[index] = Some(value.clone());
    }
    ordered
        .into_iter()
        .map(|value| value.ok_or_else(|| rv("score index is missing", path)))
        .collect()
}

fn nonempty_string(value: Option<&Value>, path: &str) -> Result<String> {
    match value.and_then(Value::as_str) {
        Some(text) if !text.is_empty() => Ok(text.to_string()),
        _ => Err(rv("field must be a non-empty string", path)),
    }
}

fn finite_number(value: Option<&Value>, path: &str) -> Result<f64> {
    match value.and_then(Value::as_f64) {
        Some(number) if number.is_finite() => Ok(number),
        _ => Err(rv("field must be a finite number", path)),
    }
}

fn nonnegative_int(value: Option<&Value>, path: &str) -> Result<u64> {
    value
        .and_then(Value::as_u64)
        .ok_or_else(|| rv("field must be a non-negative integer", path))
}

fn rv(message: impl Into<String>, field: &str) -> JevError {
    JevError::ResponseValidation {
        message: message.into(),
        field: Some(field.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use decision_core::{ChoiceQuestion, Content, NoulCriteria, NoulQuestion, ScoreQuestion};

    fn noul_questions() -> QuestionSet {
        let mut set = QuestionSet::new();
        set.push(
            "q",
            Question::Noul(
                NoulQuestion::new(
                    Content::string("x"),
                    NoulCriteria {
                        truthy: Some(Content::string("yes")),
                        falsy: None,
                    },
                )
                .unwrap(),
            ),
        )
        .unwrap();
        set
    }

    #[test]
    fn rejects_duplicate_keys() {
        let bytes = br#"{"model":"m","model":"m2","usage":{},"answers":{}}"#;
        let error = scan_json(bytes, &ResourceLimits::default()).unwrap_err();
        assert!(matches!(error, JevError::MalformedJson { .. }));
    }

    #[test]
    fn rejects_excessive_depth() {
        let mut json = String::new();
        for _ in 0..40 {
            json.push('[');
        }
        for _ in 0..40 {
            json.push(']');
        }
        let limits = ResourceLimits {
            max_json_depth: 10,
            ..ResourceLimits::default()
        };
        assert!(scan_json(json.as_bytes(), &limits).is_err());
    }

    #[test]
    fn rejects_trailing_data() {
        assert!(scan_json(b"{} trailing", &ResourceLimits::default()).is_err());
    }

    #[test]
    fn rejects_answer_key_mismatch() {
        let body = br#"{"model":"m","usage":{"input_tokens":1,"output_tokens":2},"answers":{"other":{"type":"noul","noul":0.5}}}"#;
        let error =
            parse_system_one_response(body, &noul_questions(), &ResourceLimits::default(), None)
                .unwrap_err();
        assert!(matches!(error, JevError::ResponseValidation { .. }));
    }

    #[test]
    fn validates_choice_cross_fields() {
        let mut set = QuestionSet::new();
        set.push(
            "c",
            Question::Choice(
                ChoiceQuestion::new(
                    Content::string("pick"),
                    vec![
                        ("a".into(), Content::string("A")),
                        ("b".into(), Content::string("B")),
                    ],
                )
                .unwrap(),
            ),
        )
        .unwrap();

        let good = br#"{"model":"m","usage":{"input_tokens":1,"output_tokens":1},"answers":{"c":{"type":"choice","choice":"a","probabilities":{"a":0.7,"b":0.3},"confidence":0.7}}}"#;
        parse_system_one_response(good, &set, &ResourceLimits::default(), None).unwrap();

        let missing = br#"{"model":"m","usage":{"input_tokens":1,"output_tokens":1},"answers":{"c":{"type":"choice","choice":"a","probabilities":{"a":1.0},"confidence":0.7}}}"#;
        assert!(
            parse_system_one_response(missing, &set, &ResourceLimits::default(), None).is_err()
        );

        let wrong_argmax = br#"{"model":"m","usage":{"input_tokens":1,"output_tokens":1},"answers":{"c":{"type":"choice","choice":"a","probabilities":{"a":0.3,"b":0.7},"confidence":0.7}}}"#;
        assert!(
            parse_system_one_response(wrong_argmax, &set, &ResourceLimits::default(), None)
                .is_err()
        );
    }

    #[test]
    fn validates_score_cross_fields() {
        let mut set = QuestionSet::new();
        set.push(
            "s",
            Question::Score(
                ScoreQuestion::new(Content::string("rate"), vec!["low".into(), "high".into()])
                    .unwrap(),
            ),
        )
        .unwrap();

        let good = br#"{"model":"m","usage":{"input_tokens":1,"output_tokens":1},"answers":{"s":{"type":"score","score":0.3,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.7,"1":0.3},"confidence":0.7}}}"#;
        parse_system_one_response(good, &set, &ResourceLimits::default(), None).unwrap();

        let bad_score = br#"{"model":"m","usage":{"input_tokens":1,"output_tokens":1},"answers":{"s":{"type":"score","score":0.9,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.7,"1":0.3},"confidence":0.7}}}"#;
        assert!(
            parse_system_one_response(bad_score, &set, &ResourceLimits::default(), None).is_err()
        );
    }

    #[test]
    fn rejects_negative_usage() {
        // `-1` is a valid JSON number but not a non-negative integer.
        let body = br#"{"model":"m","usage":{"input_tokens":-1,"output_tokens":0},"answers":{"q":{"type":"noul","noul":0.5}}}"#;
        assert!(
            parse_system_one_response(body, &noul_questions(), &ResourceLimits::default(), None)
                .is_err()
        );
    }
}
