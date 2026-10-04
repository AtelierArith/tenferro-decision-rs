//! Reference fixture format and loader.
//!
//! Engine tests compare optimized Rust results against fixture tensors captured
//! from the reference implementations (Laya.jl / JeffClient.jl). This crate
//! defines one small, dependency-light JSON format and a strict loader; it is a
//! test-support crate and never a runtime dependency of an engine.
//!
//! See `fixtures/README.md` for the on-disk schema and
//! `docs/agents/specs/docs/05_TESTING_BENCHMARKS.md` for how fixtures are used.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;
use thiserror::Error;

/// Errors raised while loading a fixture.
#[derive(Debug, Error)]
pub enum FixtureError {
    /// The file could not be read.
    #[error("failed to read fixture `{path}`: {source}")]
    Io {
        /// Path that failed.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The file was not valid JSON or did not match the schema.
    #[error("invalid fixture: {0}")]
    Format(String),
}

/// Element type of a fixture tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 64-bit signed integer.
    I64,
    /// Boolean.
    Bool,
}

impl Dtype {
    fn parse(value: &str) -> Result<Self, FixtureError> {
        match value {
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "i64" => Ok(Self::I64),
            "bool" => Ok(Self::Bool),
            other => Err(FixtureError::Format(format!(
                "unknown dtype `{other}` (expected f32, f64, i64, or bool)"
            ))),
        }
    }
}

/// Storage order of the flat `data` array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorOrder {
    /// First axis varies fastest (the engine's native layout).
    ColMajor,
    /// Last axis varies fastest (NumPy / Julia-reversed reference order).
    RowMajor,
}

impl TensorOrder {
    fn parse(value: &str) -> Result<Self, FixtureError> {
        match value {
            "col-major" => Ok(Self::ColMajor),
            "row-major" => Ok(Self::RowMajor),
            other => Err(FixtureError::Format(format!(
                "unknown order `{other}` (expected col-major or row-major)"
            ))),
        }
    }
}

/// Typed tensor payload.
#[derive(Clone, Debug, PartialEq)]
pub enum Storage {
    /// `f32` values.
    F32(Vec<f32>),
    /// `f64` values.
    F64(Vec<f64>),
    /// `i64` values.
    I64(Vec<i64>),
    /// `bool` values.
    Bool(Vec<bool>),
}

impl Storage {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }

    /// `true` when the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The element type.
    pub fn dtype(&self) -> Dtype {
        match self {
            Self::F32(_) => Dtype::F32,
            Self::F64(_) => Dtype::F64,
            Self::I64(_) => Dtype::I64,
            Self::Bool(_) => Dtype::Bool,
        }
    }
}

/// A named tensor with a shape and a flat payload.
#[derive(Clone, Debug, PartialEq)]
pub struct FixtureTensor {
    /// Logical shape, first axis fastest when `order` is `ColMajor`.
    pub shape: Vec<usize>,
    /// Storage order of `storage`.
    pub order: TensorOrder,
    /// Flat element data.
    pub storage: Storage,
}

impl FixtureTensor {
    /// Number of elements implied by the shape.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Borrow the payload as `f32`, when the dtype matches.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.storage {
            Storage::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow the payload as `f64`, when the dtype matches.
    pub fn as_f64(&self) -> Option<&[f64]> {
        match &self.storage {
            Storage::F64(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow the payload as `i64`, when the dtype matches.
    pub fn as_i64(&self) -> Option<&[i64]> {
        match &self.storage {
            Storage::I64(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow the payload as `bool`, when the dtype matches.
    pub fn as_bool(&self) -> Option<&[bool]> {
        match &self.storage {
            Storage::Bool(v) => Some(v),
            _ => None,
        }
    }
}

/// Free-form fixture provenance.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Metadata {
    /// Reference implementation or revision the data came from.
    pub source: Option<String>,
    /// Human-readable description.
    pub notes: Option<String>,
    /// Captured dtype label, when recorded.
    pub dtype: Option<String>,
    /// Random seed used to produce the data, when recorded.
    pub seed: Option<i64>,
    /// Any additional JSON-valued fields.
    pub extra: BTreeMap<String, Value>,
}

/// A loaded fixture: named tensors plus metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Fixture {
    /// Fixture name.
    pub name: String,
    /// Provenance metadata.
    pub metadata: Metadata,
    /// Tensors in file order.
    pub tensors: Vec<(String, FixtureTensor)>,
}

impl Fixture {
    /// Look up a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&FixtureTensor> {
        self.tensors
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, tensor)| tensor)
    }

    /// Load and validate a fixture from a JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, FixtureError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| FixtureError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_json(&text)
    }

    /// Parse and validate a fixture from a JSON string.
    pub fn from_json(text: &str) -> Result<Self, FixtureError> {
        let root: Value =
            serde_json::from_str(text).map_err(|e| FixtureError::Format(e.to_string()))?;
        let object = root
            .as_object()
            .ok_or_else(|| FixtureError::Format("top level must be an object".into()))?;

        let name = object
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| FixtureError::Format("missing string field `name`".into()))?
            .to_string();

        let metadata = parse_metadata(object.get("metadata"))?;

        let tensors_object = object
            .get("tensors")
            .and_then(Value::as_object)
            .ok_or_else(|| FixtureError::Format("missing object field `tensors`".into()))?;

        let mut tensors = Vec::with_capacity(tensors_object.len());
        for (tensor_name, value) in tensors_object {
            let tensor = parse_tensor(tensor_name, value)?;
            tensors.push((tensor_name.clone(), tensor));
        }

        Ok(Self {
            name,
            metadata,
            tensors,
        })
    }
}

fn parse_metadata(value: Option<&Value>) -> Result<Metadata, FixtureError> {
    let Some(value) = value else {
        return Ok(Metadata::default());
    };
    let object = value
        .as_object()
        .ok_or_else(|| FixtureError::Format("`metadata` must be an object".into()))?;
    let mut metadata = Metadata::default();
    for (key, item) in object {
        match key.as_str() {
            "source" => metadata.source = item.as_str().map(str::to_string),
            "notes" => metadata.notes = item.as_str().map(str::to_string),
            "dtype" => metadata.dtype = item.as_str().map(str::to_string),
            "seed" => metadata.seed = item.as_i64(),
            _ => {
                metadata.extra.insert(key.clone(), item.clone());
            }
        }
    }
    Ok(metadata)
}

fn parse_tensor(name: &str, value: &Value) -> Result<FixtureTensor, FixtureError> {
    let object = value
        .as_object()
        .ok_or_else(|| FixtureError::Format(format!("tensor `{name}` must be an object")))?;

    let dtype = object
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| FixtureError::Format(format!("tensor `{name}` is missing `dtype`")))
        .and_then(Dtype::parse)?;

    let order = match object.get("order").and_then(Value::as_str) {
        Some(value) => TensorOrder::parse(value)?,
        None => TensorOrder::ColMajor,
    };

    let shape_values = object
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| FixtureError::Format(format!("tensor `{name}` is missing `shape`")))?;
    let mut shape = Vec::with_capacity(shape_values.len());
    for axis in shape_values {
        let dim = axis.as_u64().ok_or_else(|| {
            FixtureError::Format(format!(
                "tensor `{name}` shape must be non-negative integers"
            ))
        })?;
        shape.push(dim as usize);
    }

    let data = object
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| FixtureError::Format(format!("tensor `{name}` is missing `data`")))?;

    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(FixtureError::Format(format!(
            "tensor `{name}` has {} values but shape {shape:?} implies {expected}",
            data.len()
        )));
    }

    let storage = parse_storage(name, dtype, data)?;
    Ok(FixtureTensor {
        shape,
        order,
        storage,
    })
}

fn parse_storage(name: &str, dtype: Dtype, data: &[Value]) -> Result<Storage, FixtureError> {
    match dtype {
        Dtype::F32 => Ok(Storage::F32(
            data.iter()
                .map(|v| number_as_f64(name, v).map(|x| x as f32))
                .collect::<Result<_, _>>()?,
        )),
        Dtype::F64 => Ok(Storage::F64(
            data.iter()
                .map(|v| number_as_f64(name, v))
                .collect::<Result<_, _>>()?,
        )),
        Dtype::I64 => Ok(Storage::I64(
            data.iter()
                .map(|v| {
                    v.as_i64().ok_or_else(|| {
                        FixtureError::Format(format!("tensor `{name}` expects i64 values"))
                    })
                })
                .collect::<Result<_, _>>()?,
        )),
        Dtype::Bool => Ok(Storage::Bool(
            data.iter()
                .map(|v| {
                    v.as_bool().ok_or_else(|| {
                        FixtureError::Format(format!("tensor `{name}` expects bool values"))
                    })
                })
                .collect::<Result<_, _>>()?,
        )),
    }
}

fn number_as_f64(name: &str, value: &Value) -> Result<f64, FixtureError> {
    value
        .as_f64()
        .ok_or_else(|| FixtureError::Format(format!("tensor `{name}` expects numeric values")))
}
