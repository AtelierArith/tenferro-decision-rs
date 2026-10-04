use crate::{Content, DecisionError, Result};

/// Maximum number of questions accepted in a single set.
pub const MAX_QUESTIONS: usize = 1024;
/// Maximum length of a question identifier, in Unicode scalar values.
pub const MAX_QUESTION_ID_CHARS: usize = 128;
/// Minimum number of candidates in a choice question.
pub const MIN_CHOICE_CANDIDATES: usize = 2;
/// Maximum number of candidates in a choice question.
pub const MAX_CHOICE_CANDIDATES: usize = 255;
/// Minimum number of levels in a score question.
pub const MIN_SCORE_LEVELS: usize = 2;
/// Maximum number of levels in a score question.
pub const MAX_SCORE_LEVELS: usize = 10;

/// A validated question identifier.
///
/// Mirrors the client rules: non-empty, at most
/// [`MAX_QUESTION_ID_CHARS`], no surrounding whitespace, and no control
/// characters.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuestionId(String);

impl QuestionId {
    /// Validate and wrap an identifier.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_identifier("question id", &value)?;
        Ok(Self(value))
    }

    /// The identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for QuestionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for QuestionId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// A `choice` question: pick one of several candidates.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceQuestion {
    /// The question text or structured prompt; must not be empty.
    pub instructions: Content,
    /// Ordered candidate id → description.
    pub criteria: Vec<(String, Content)>,
}

impl ChoiceQuestion {
    /// Validate the question.
    pub fn validate(&self) -> Result<()> {
        validate_instructions("choice.instructions", &self.instructions)?;
        validate_count(
            "choice.criteria",
            self.criteria.len(),
            MIN_CHOICE_CANDIDATES,
            MAX_CHOICE_CANDIDATES,
        )?;
        validate_candidate_ids("choice.criteria", &self.criteria)?;
        for (index, (_, description)) in self.criteria.iter().enumerate() {
            description
                .validate()
                .map_err(|e| annotate(e, &format!("choice.criteria[{index}].description")))?;
        }
        Ok(())
    }

    /// Construct and validate a choice question.
    pub fn new(instructions: Content, criteria: Vec<(String, Content)>) -> Result<Self> {
        let question = Self {
            instructions,
            criteria,
        };
        question.validate()?;
        Ok(question)
    }
}

/// The allowed true/false descriptions of a `noul` question.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NoulCriteria {
    /// Description of the affirmative outcome.
    pub truthy: Option<Content>,
    /// Description of the negative outcome.
    pub falsy: Option<Content>,
}

/// A `noul` question: a binary yes/no-style decision.
#[derive(Clone, Debug, PartialEq)]
pub struct NoulQuestion {
    /// The question text or structured prompt; must not be empty.
    pub instructions: Content,
    /// The true/false descriptions; at least one must be present.
    pub criteria: NoulCriteria,
}

impl NoulQuestion {
    /// Validate the question.
    pub fn validate(&self) -> Result<()> {
        validate_instructions("noul.instructions", &self.instructions)?;
        if self.criteria.truthy.is_none() && self.criteria.falsy.is_none() {
            return Err(DecisionError::invalid_field(
                "noul.criteria",
                "at least one of `true` or `false` must be provided",
            ));
        }
        for (name, value) in [
            ("noul.criteria.true", &self.criteria.truthy),
            ("noul.criteria.false", &self.criteria.falsy),
        ] {
            if let Some(value) = value {
                value.validate().map_err(|e| annotate(e, name))?;
            }
        }
        Ok(())
    }

    /// Construct and validate a `noul` question.
    pub fn new(instructions: Content, criteria: NoulCriteria) -> Result<Self> {
        let question = Self {
            instructions,
            criteria,
        };
        question.validate()?;
        Ok(question)
    }
}

/// A `score` question: rate along ordered levels.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreQuestion {
    /// The question text or structured prompt; must not be empty.
    pub instructions: Content,
    /// Ordered, non-empty level labels. The zero-based index is the wire value.
    pub criteria: Vec<String>,
}

impl ScoreQuestion {
    /// Validate the question.
    pub fn validate(&self) -> Result<()> {
        validate_instructions("score.instructions", &self.instructions)?;
        validate_count(
            "score.criteria",
            self.criteria.len(),
            MIN_SCORE_LEVELS,
            MAX_SCORE_LEVELS,
        )?;
        for (index, label) in self.criteria.iter().enumerate() {
            if label.is_empty() {
                return Err(DecisionError::invalid_field(
                    format!("score.criteria[{index}]"),
                    "level label must not be empty",
                ));
            }
        }
        Ok(())
    }

    /// Construct and validate a score question.
    pub fn new(instructions: Content, criteria: Vec<String>) -> Result<Self> {
        let question = Self {
            instructions,
            criteria,
        };
        question.validate()?;
        Ok(question)
    }
}

/// A typed decision question.
#[derive(Clone, Debug, PartialEq)]
pub enum Question {
    /// Choose one candidate.
    Choice(ChoiceQuestion),
    /// Affirmative / negative decision.
    Noul(NoulQuestion),
    /// Ordered-level score.
    Score(ScoreQuestion),
}

impl Question {
    /// Validate the question.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Choice(q) => q.validate(),
            Self::Noul(q) => q.validate(),
            Self::Score(q) => q.validate(),
        }
    }
}

/// An ordered set of uniquely identified questions.
///
/// Insertion order is preserved because evaluation order can affect results
/// (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §7).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QuestionSet {
    questions: Vec<(QuestionId, Question)>,
}

impl QuestionSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of questions.
    pub fn len(&self) -> usize {
        self.questions.len()
    }

    /// `true` when no questions are present.
    pub fn is_empty(&self) -> bool {
        self.questions.is_empty()
    }

    /// The questions in insertion order.
    pub fn questions(&self) -> &[(QuestionId, Question)] {
        &self.questions
    }

    /// Look up a question by identifier.
    pub fn get(&self, id: &str) -> Option<&Question> {
        self.questions
            .iter()
            .find(|(candidate, _)| candidate.as_str() == id)
            .map(|(_, question)| question)
    }

    /// Add a question, validating its identifier and body and rejecting
    /// duplicates.
    pub fn push(&mut self, id: impl Into<String>, question: Question) -> Result<()> {
        let id = QuestionId::new(id)?;
        if self.questions.iter().any(|(existing, _)| existing == &id) {
            return Err(DecisionError::invalid_field(
                "questions",
                format!("duplicate question id `{id}`"),
            ));
        }
        question.validate()?;
        if self.questions.len() >= MAX_QUESTIONS {
            return Err(DecisionError::invalid_field(
                "questions",
                format!("at most {MAX_QUESTIONS} questions are allowed"),
            ));
        }
        self.questions.push((id, question));
        Ok(())
    }

    /// Validate the whole set: at least one question, at most
    /// [`MAX_QUESTIONS`], unique ids, and each body valid.
    pub fn validate(&self) -> Result<()> {
        if self.questions.is_empty() {
            return Err(DecisionError::invalid_field(
                "questions",
                "at least one question is required",
            ));
        }
        if self.questions.len() > MAX_QUESTIONS {
            return Err(DecisionError::invalid_field(
                "questions",
                format!("at most {MAX_QUESTIONS} questions are allowed"),
            ));
        }
        for (id, question) in &self.questions {
            question
                .validate()
                .map_err(|e| annotate(e, &format!("questions.{id}")))?;
        }
        Ok(())
    }
}

impl FromIterator<(QuestionId, Question)> for QuestionSet {
    fn from_iter<T: IntoIterator<Item = (QuestionId, Question)>>(iter: T) -> Self {
        Self {
            questions: iter.into_iter().collect(),
        }
    }
}

fn validate_instructions(field: &str, value: &Content) -> Result<()> {
    value.validate().map_err(|e| annotate(e, field))?;
    if value.is_empty() {
        return Err(DecisionError::invalid_field(
            field,
            "instructions must not be empty",
        ));
    }
    Ok(())
}

fn validate_identifier(field: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(DecisionError::invalid_field(
            field,
            "identifier must not be empty",
        ));
    }
    if value.chars().count() > MAX_QUESTION_ID_CHARS {
        return Err(DecisionError::invalid_field(
            field,
            format!("identifier must be at most {MAX_QUESTION_ID_CHARS} characters"),
        ));
    }
    if value.trim() != value {
        return Err(DecisionError::invalid_field(
            field,
            "identifier must not have surrounding whitespace",
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(DecisionError::invalid_field(
            field,
            "identifier must not contain control characters",
        ));
    }
    Ok(())
}

fn validate_candidate_ids(field: &str, criteria: &[(String, Content)]) -> Result<()> {
    for (index, (id, _)) in criteria.iter().enumerate() {
        if id.is_empty() {
            return Err(DecisionError::invalid_field(
                format!("{field}[{index}]"),
                "candidate id must not be empty",
            ));
        }
        if criteria[..index].iter().any(|(seen, _)| seen == id) {
            return Err(DecisionError::invalid_field(
                format!("{field}[{index}]"),
                format!("duplicate candidate id `{id}`"),
            ));
        }
    }
    Ok(())
}

fn validate_count(field: &str, count: usize, min: usize, max: usize) -> Result<()> {
    if count < min || count > max {
        return Err(DecisionError::invalid_field(
            field,
            format!("expected between {min} and {max} entries, found {count}"),
        ));
    }
    Ok(())
}

/// Prefix an existing error's field with an outer path.
fn annotate(error: DecisionError, outer: &str) -> DecisionError {
    match error {
        DecisionError::InvalidInput { field, message } => DecisionError::InvalidInput {
            field: Some(match field {
                Some(inner) => format!("{outer}.{inner}"),
                None => outer.to_string(),
            }),
            message,
        },
        other => other,
    }
}
