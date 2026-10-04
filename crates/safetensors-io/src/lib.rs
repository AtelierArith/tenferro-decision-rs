//! A minimal, dependency-light safetensors reader.
//!
//! Only `std` plus `serde_json` are used. `F32`, `F64`, `F16`, and `BF16`
//! tensors are converted to `f32`. Safetensors stores tensors row-major in
//! PyTorch layout `(out, in)`; consumers are responsible for any transpose.
//!
//! This crate is shared by `laya-infer` and `jeff-infer`; it deliberately has
//! no tenferro dependency.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use decision_core::{DecisionError, Result};
use serde_json::Value;

/// A tensor dtype supported by the reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dtype {
    /// 32-bit IEEE-754 float.
    F32,
    /// 64-bit IEEE-754 float.
    F64,
    /// 16-bit IEEE-754 half float.
    F16,
    /// Brain float 16.
    Bf16,
}

impl Dtype {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "F32" => Some(Self::F32),
            "F64" => Some(Self::F64),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            _ => None,
        }
    }

    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

/// One tensor's header entry.
#[derive(Clone, Debug)]
struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    /// `[start, end)` into the data buffer (after the header).
    offsets: (usize, usize),
}

/// A parsed safetensors file held in memory.
///
/// The entire file is read once; tensor access borrows from the retained data
/// buffer. Only little-endian tensors are supported, matching the format.
#[derive(Clone, Debug)]
pub struct SafetensorsFile {
    path: PathBuf,
    /// Offset of the data buffer within `bytes`.
    data_start: usize,
    bytes: Vec<u8>,
    entries: Vec<(String, Entry)>,
}

impl SafetensorsFile {
    /// Open and parse a safetensors file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let bytes = fs::read(&path).map_err(|error| io_error(&path, error))?;
        if bytes.len() < 8 {
            return Err(invalid_file(
                &path,
                "file is shorter than the 8-byte header length",
            ));
        }
        let header_len = u64::from_le_bytes(
            bytes[0..8]
                .try_into()
                .map_err(|_| invalid_file(&path, "malformed header length"))?,
        ) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| invalid_file(&path, "header length overflows the file size"))?;
        if header_len == 0 || header_end > bytes.len() {
            return Err(invalid_file(&path, "header length is outside the file"));
        }
        let header: Value = serde_json::from_slice(&bytes[8..header_end])
            .map_err(|error| invalid_file(&path, format!("invalid header JSON: {error}")))?;
        let object = header
            .as_object()
            .ok_or_else(|| invalid_file(&path, "safetensors header must be a JSON object"))?;

        let data_start = header_end;
        let data_len = bytes.len() - data_start;
        let mut entries = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for (name, spec) in object {
            if name == "__metadata__" {
                continue;
            }
            if !seen.insert(name.clone()) {
                return Err(invalid_file(
                    &path,
                    format!("duplicate tensor `{name}` in header"),
                ));
            }
            let spec = spec.as_object().ok_or_else(|| {
                invalid_file(&path, format!("tensor `{name}` entry must be an object"))
            })?;
            let dtype_name = spec.get("dtype").and_then(Value::as_str).ok_or_else(|| {
                invalid_file(&path, format!("tensor `{name}` is missing its dtype"))
            })?;
            let dtype = Dtype::parse(dtype_name).ok_or_else(|| {
                DecisionError::unsupported(format!(
                    "safetensors tensor `{name}` has unsupported dtype `{dtype_name}`"
                ))
            })?;
            let shape = spec
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    invalid_file(&path, format!("tensor `{name}` is missing its shape"))
                })?
                .iter()
                .map(|dim| {
                    dim.as_u64().map(|v| v as usize).ok_or_else(|| {
                        invalid_file(&path, format!("tensor `{name}` has a non-integer shape"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let offsets = spec
                .get("data_offsets")
                .and_then(Value::as_array)
                .filter(|values| values.len() == 2)
                .ok_or_else(|| {
                    invalid_file(&path, format!("tensor `{name}` is missing data_offsets"))
                })?;
            let start = offsets[0].as_u64().map(|v| v as usize).ok_or_else(|| {
                invalid_file(&path, format!("tensor `{name}` has a bad start offset"))
            })?;
            let end = offsets[1].as_u64().map(|v| v as usize).ok_or_else(|| {
                invalid_file(&path, format!("tensor `{name}` has a bad end offset"))
            })?;
            if start > end || end > data_len {
                return Err(invalid_file(
                    &path,
                    format!("tensor `{name}` offsets [{start}, {end}) exceed the data buffer"),
                ));
            }
            let element_count = shape
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim));
            let expected = element_count.and_then(|count| count.checked_mul(dtype.size()));
            if expected != Some(end - start) {
                return Err(invalid_file(
                    &path,
                    format!(
                        "tensor `{name}` shape {shape:?} does not match its {} byte span",
                        end - start
                    ),
                ));
            }
            entries.push((
                name.clone(),
                Entry {
                    dtype,
                    shape,
                    offsets: (start, end),
                },
            ));
        }

        Ok(Self {
            path,
            data_start,
            bytes,
            entries,
        })
    }

    /// The file path this reader was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Tensor names in header order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(name, _)| name.as_str())
    }

    /// Whether a tensor with `name` exists.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|(candidate, _)| candidate == name)
    }

    /// The recorded shape of a tensor, if present.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, entry)| entry.shape.as_slice())
    }

    /// Load a tensor as `(shape, values)` where `values` is row-major.
    ///
    /// The shape is exactly as recorded in the file (PyTorch order).
    pub fn tensor(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let entry = self
            .entries
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, entry)| entry)
            .ok_or_else(|| {
                DecisionError::invalid_field(
                    "safetensors.tensor",
                    format!("tensor `{name}` is missing from {}", self.path.display()),
                )
            })?;
        let start = self.data_start + entry.offsets.0;
        let end = self.data_start + entry.offsets.1;
        let raw = &self.bytes[start..end];
        let values = decode(entry.dtype, raw, self.path.as_path(), name)?;
        Ok((entry.shape.clone(), values))
    }
}

fn decode(dtype: Dtype, raw: &[u8], path: &Path, name: &str) -> Result<Vec<f32>> {
    let values: Vec<f32> = match dtype {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        Dtype::F64 => raw
            .chunks_exact(8)
            .map(|chunk| {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32
            })
            .collect(),
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect(),
        Dtype::Bf16 => raw
            .chunks_exact(2)
            .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16))
            .collect(),
    };
    if values.is_empty() && !raw.is_empty() {
        return Err(invalid_file(
            path,
            format!("tensor `{name}` has a byte length that is not a whole number of elements"),
        ));
    }
    Ok(values)
}

/// Convert an IEEE-754 half float to `f32` without new dependencies.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x3ff);
    let magnitude = if exponent == 0 {
        if fraction == 0 {
            0.0
        } else {
            (fraction as f32) * 2f32.powi(-24)
        }
    } else if exponent == 0x1f {
        if fraction == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + fraction as f32 / 1024.0) * 2f32.powi(exponent as i32 - 15)
    };
    if sign == 1 {
        -magnitude
    } else {
        magnitude
    }
}

fn invalid_file(path: &Path, message: impl Into<String>) -> DecisionError {
    DecisionError::invalid_field(
        "safetensors",
        format!("{}: {}", path.display(), message.into()),
    )
}

fn io_error(path: &Path, error: std::io::Error) -> DecisionError {
    DecisionError::Backend {
        message: format!("failed to read `{}`: {error}", path.display()),
        source: Some(Box::new(error)),
    }
}
