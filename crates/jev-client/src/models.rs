//! Typed model references and the model-list response.
//!
//! [`PinnedModel`] is recommended for reproducible production. [`MovingAlias`]
//! is explicitly not reproducible and is for experiments only
//! (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §11).

use crate::errors::{JevError, Result};

const MOVING_ALIASES: [&str; 4] = ["latest", "preview", "jev-latest", "jev-preview"];

fn validate_model_id(id: &str, limits_bytes: usize) -> Result<()> {
    if id.is_empty() {
        return Err(JevError::validation_field(
            "model",
            "model id must not be empty",
        ));
    }
    if !id.is_ascii() {
        return Err(JevError::validation_field(
            "model",
            "model id must be ASCII",
        ));
    }
    if id.len() > limits_bytes {
        return Err(JevError::validation_field("model", "model id is too long"));
    }
    if id.bytes().any(|byte| byte < 0x20 || byte == 0x7f) {
        return Err(JevError::validation_field(
            "model",
            "model id must not contain control characters",
        ));
    }
    if id.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(JevError::validation_field(
            "model",
            "model id must not contain whitespace",
        ));
    }
    Ok(())
}

/// A fixed, versioned model id such as `jev-1.13.0`.
///
/// Known moving aliases are rejected; use [`MovingAlias`] instead.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PinnedModel(String);

impl PinnedModel {
    /// Validate and wrap a pinned model id.
    pub fn new(id: impl Into<String>) -> Result<Self> {
        let id = id.into();
        validate_model_id(&id, 128)?;
        if MOVING_ALIASES.contains(&id.as_str()) {
            return Err(JevError::validation_field(
                "model",
                "moving model aliases require `MovingAlias`",
            ));
        }
        Ok(Self(id))
    }

    /// The model id.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A moving alias such as `jev-latest`.
///
/// The model behind the alias can change, so results are not reproducible. Only
/// the known aliases are accepted. The client warns once on first use.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MovingAlias(String);

impl MovingAlias {
    /// Validate and wrap a known moving alias.
    pub fn new(id: impl Into<String>) -> Result<Self> {
        let id = id.into();
        validate_model_id(&id, 128)?;
        if !MOVING_ALIASES.contains(&id.as_str()) {
            return Err(JevError::validation_field(
                "model",
                "unknown moving model alias",
            ));
        }
        Ok(Self(id))
    }

    /// The alias id.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A model reference: pinned or moving.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModelRef {
    /// A versioned, reproducible model.
    Pinned(PinnedModel),
    /// A moving alias; not reproducible.
    MovingAlias(MovingAlias),
}

impl ModelRef {
    /// The model id sent on the wire.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Pinned(model) => model.as_str(),
            Self::MovingAlias(alias) => alias.as_str(),
        }
    }

    /// `true` when this is a moving alias.
    pub fn is_moving_alias(&self) -> bool {
        matches!(self, Self::MovingAlias(_))
    }
}

impl From<PinnedModel> for ModelRef {
    fn from(model: PinnedModel) -> Self {
        Self::Pinned(model)
    }
}

impl From<MovingAlias> for ModelRef {
    fn from(alias: MovingAlias) -> Self {
        Self::MovingAlias(alias)
    }
}

/// Metadata for one model from [`crate::Client::list_models`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    /// Model name.
    pub name: String,
    /// Human-readable description; may be empty.
    pub description: String,
    /// Release date as a validated `YYYY-MM-DD` string.
    pub release_date: String,
}

/// A list of available models.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelList {
    /// The models, in response order.
    pub models: Vec<ModelInfo>,
}

/// Validate an ISO `YYYY-MM-DD` date, returning it unchanged.
pub(crate) fn validate_iso_date(value: &str) -> Result<()> {
    let mut parts = value.split('-');
    let (year, month, day) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(year), Some(month), Some(day), None) => (year, month, day),
        _ => return Err(date_error()),
    };
    if year.len() != 4 || !year.bytes().all(|b| b.is_ascii_digit()) {
        return Err(date_error());
    }
    let parse = |part: &str| -> Option<u32> {
        if part.is_empty() || part.len() > 2 || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        part.parse().ok()
    };
    let month = parse(month).ok_or_else(date_error)?;
    let day = parse(day).ok_or_else(date_error)?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(date_error());
    }
    Ok(())
}

fn date_error() -> JevError {
    JevError::response_validation_field("models.release_date", "release_date is not an ISO date")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_rejects_aliases() {
        assert!(PinnedModel::new("jev-1.13.0").is_ok());
        assert!(PinnedModel::new("jev-latest").is_err());
        assert!(PinnedModel::new("").is_err());
        assert!(PinnedModel::new("a b").is_err());
        assert!(PinnedModel::new("caf\u{e9}").is_err());
    }

    #[test]
    fn moving_alias_requires_known_alias() {
        assert!(MovingAlias::new("jev-latest").is_ok());
        assert!(MovingAlias::new("jev-1.13.0").is_err());
    }

    #[test]
    fn iso_dates() {
        assert!(validate_iso_date("2026-10-04").is_ok());
        assert!(validate_iso_date("2026-13-04").is_err());
        assert!(validate_iso_date("2026-10").is_err());
        assert!(validate_iso_date("not-a-date").is_err());
    }
}
