use crate::{DecisionError, Result};

/// Maximum nesting depth accepted for [`Content`], matching the client wire
/// limit (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §6).
pub const MAX_CONTENT_DEPTH: usize = 32;

/// The value model accepted for `state`, `instructions`, and option
/// descriptions.
///
/// It intentionally covers only explicitly allowed JSON-compatible values so
/// that arbitrary Rust structs are never auto-serialized into a request
/// (`docs/agents/specs/docs/16_DECISION_CORE_DESIGN.md` §3.1). Differences from
/// a generic JSON value:
///
/// - floats must be finite (no `NaN`/`Infinity`)
/// - object keys must be unique
/// - nesting depth is bounded
///
/// Object key order is preserved because it can carry semantic meaning.
#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    /// JSON `null`.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// JSON integer.
    Int(i64),
    /// JSON floating-point number; always finite.
    Float(f64),
    /// JSON string.
    String(String),
    /// JSON array.
    Array(Vec<Content>),
    /// JSON object, in insertion order.
    Object(Vec<(String, Content)>),
}

impl Content {
    /// Build a string value.
    pub fn string(value: impl Into<String>) -> Self {
        Self::String(value.into())
    }

    /// Build an object from ordered pairs.
    pub fn object(pairs: impl IntoIterator<Item = (impl Into<String>, Content)>) -> Self {
        Self::Object(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// `true` when the value is absent or empty (`null`, empty string, array, or
    /// object). Used to reject empty `instructions`.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Null => true,
            Self::String(s) => s.is_empty(),
            Self::Array(items) => items.is_empty(),
            Self::Object(pairs) => pairs.is_empty(),
            Self::Bool(_) | Self::Int(_) | Self::Float(_) => false,
        }
    }

    /// Validate finiteness, unique keys, and depth.
    pub fn validate(&self) -> Result<()> {
        self.validate_at("$", 0)
    }

    /// Construct a float value, rejecting non-finite inputs.
    pub fn float(value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(DecisionError::invalid_field(
                "$",
                "floating-point values must be finite",
            ));
        }
        Ok(Self::Float(value))
    }

    fn validate_at(&self, path: &str, depth: usize) -> Result<()> {
        if depth >= MAX_CONTENT_DEPTH {
            return Err(DecisionError::invalid_field(
                path,
                format!("value nesting exceeds the maximum depth of {MAX_CONTENT_DEPTH}"),
            ));
        }
        match self {
            Self::Float(value) if !value.is_finite() => Err(DecisionError::invalid_field(
                path,
                "floating-point values must be finite",
            )),
            Self::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    item.validate_at(&format!("{path}[{index}]"), depth + 1)?;
                }
                Ok(())
            }
            Self::Object(pairs) => {
                for (index, (key, value)) in pairs.iter().enumerate() {
                    if pairs[..index].iter().any(|(seen, _)| seen == key) {
                        return Err(DecisionError::invalid_field(
                            path,
                            format!("duplicate object key `{key}`"),
                        ));
                    }
                    value.validate_at(&format!("{path}.{key}"), depth + 1)?;
                }
                Ok(())
            }
            Self::Null | Self::Bool(_) | Self::Int(_) | Self::Float(_) | Self::String(_) => Ok(()),
        }
    }
}

#[cfg(feature = "serde")]
impl Content {
    /// Convert to a generic JSON value (all values are JSON-compatible by
    /// construction).
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            Self::Null => serde_json::Value::Null,
            Self::Bool(b) => serde_json::Value::Bool(*b),
            Self::Int(i) => serde_json::Value::Number((*i).into()),
            Self::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Self::String(s) => serde_json::Value::String(s.clone()),
            Self::Array(items) => {
                serde_json::Value::Array(items.iter().map(Self::to_json_value).collect())
            }
            Self::Object(pairs) => serde_json::Value::Object(
                pairs
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_json_value()))
                    .collect(),
            ),
        }
    }

    /// Build a value from generic JSON, enforcing the `Content` rules.
    pub fn from_json_value(value: serde_json::Value) -> Result<Self> {
        use serde_json::Value;
        let content = match value {
            Value::Null => Self::Null,
            Value::Bool(b) => Self::Bool(b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Self::Int(i)
                } else if let Some(u) = n.as_u64() {
                    Self::Int(i64::try_from(u).map_err(|_| {
                        DecisionError::invalid_field("$", "integer does not fit in i64")
                    })?)
                } else {
                    Self::float(n.as_f64().ok_or_else(|| {
                        DecisionError::invalid_field("$", "unsupported numeric value")
                    })?)?
                }
            }
            Value::String(s) => Self::String(s),
            Value::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(Self::from_json_value(item)?);
                }
                Self::Array(out)
            }
            Value::Object(map) => {
                let mut out = Vec::with_capacity(map.len());
                for (k, v) in map {
                    out.push((k, Self::from_json_value(v)?));
                }
                Self::Object(out)
            }
        };
        content.validate()?;
        Ok(content)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Content {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.to_json_value().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Content {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        Content::from_json_value(value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_finite() {
        assert!(Content::Float(f64::NAN).validate().is_err());
        assert!(Content::Float(f64::INFINITY).validate().is_err());
        assert!(Content::float(1.5).is_ok());
        assert!(Content::float(f64::NAN).is_err());
    }

    #[test]
    fn rejects_duplicate_keys() {
        let value = Content::Object(vec![
            ("a".into(), Content::Int(1)),
            ("a".into(), Content::Int(2)),
        ]);
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_excessive_depth() {
        let mut value = Content::Null;
        for _ in 0..(MAX_CONTENT_DEPTH + 1) {
            value = Content::Array(vec![value]);
        }
        assert!(value.validate().is_err());
    }

    #[test]
    fn emptiness() {
        assert!(Content::Null.is_empty());
        assert!(Content::string("").is_empty());
        assert!(Content::Array(vec![]).is_empty());
        assert!(!Content::string("x").is_empty());
        assert!(!Content::Int(0).is_empty());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip() {
        let value = Content::object([
            ("a", Content::Int(1)),
            (
                "b",
                Content::Array(vec![Content::Bool(true), Content::float(2.5).unwrap()]),
            ),
        ]);
        let json = serde_json::to_string(&value).unwrap();
        let back: Content = serde_json::from_str(&json).unwrap();
        assert_eq!(value, back);
    }
}
