//! Reusable cache of weight tensors for the eager session.
//!
//! An [`EagerTensor`] keeps an `Arc<EagerRuntime>`, so a tensor created for a
//! weight can be reused by any later session that shares the same runtime. The
//! engines cache their weight tensors here so the tenferro path does not
//! re-transpose and re-create every weight on each forward (its dominant
//! overhead). Entries are keyed by the host storage identity, so callers must
//! keep the weight slices alive while the cache is in use.

use std::collections::HashMap;

use tenferro_ad::{EagerSession, EagerTensor, Result};
use tenferro_tensor::Tensor;

/// A cache of column-major weight tensors keyed by host storage identity.
#[derive(Clone, Debug, Default)]
pub struct TensorCache {
    weights: HashMap<WeightKey, EagerTensor>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WeightKey {
    address: usize,
    length: usize,
    shape: Vec<usize>,
    transpose: bool,
}

impl TensorCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of cached tensors.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// A column-major `[rows, cols]` tensor for the **row-major** `row_major`
    /// data, cached across calls by the identity of `row_major`.
    pub fn col(
        &mut self,
        session: &mut EagerSession<'_>,
        shape: Vec<usize>,
        row_major: &[f32],
    ) -> Result<EagerTensor> {
        let key = WeightKey {
            address: row_major.as_ptr() as usize,
            length: row_major.len(),
            shape: shape.clone(),
            transpose: true,
        };
        if let Some(tensor) = self.weights.get(&key) {
            return Ok(tensor.clone());
        }
        let rows = shape.first().copied().unwrap_or(1);
        let cols = shape.get(1).copied().unwrap_or(1);
        let mut column_major = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                column_major[row + col * rows] = row_major[row * cols + col];
            }
        }
        let tensor =
            session.constant_from_host(Tensor::from_vec_col_major(shape, column_major)?)?;
        self.weights.insert(key, tensor.clone());
        Ok(tensor)
    }

    /// A column-major `shape` tensor for `data` that is **already** column-major,
    /// cached by the identity of `data`.
    pub fn col_major(
        &mut self,
        session: &mut EagerSession<'_>,
        shape: Vec<usize>,
        data: &[f32],
    ) -> Result<EagerTensor> {
        let key = WeightKey {
            address: data.as_ptr() as usize,
            length: data.len(),
            shape: shape.clone(),
            transpose: false,
        };
        if let Some(tensor) = self.weights.get(&key) {
            return Ok(tensor.clone());
        }
        let tensor =
            session.constant_from_host(Tensor::from_vec_col_major(shape, data.to_vec())?)?;
        self.weights.insert(key, tensor.clone());
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_storage_keeps_shapes_and_layouts_distinct() {
        let runtime = tenferro_ad::EagerRuntime::new().unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut cache = TensorCache::new();
        runtime
            .with_eager_session(|session| {
                let transposed = cache.col(session, vec![2, 3], &data)?;
                let direct = cache.col_major(session, vec![2, 3], &data)?;
                let reshaped = cache.col_major(session, vec![3, 2], &data)?;
                assert_eq!(
                    transposed.value()?.as_slice::<f32>()?,
                    &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
                );
                assert_eq!(direct.value()?.as_slice::<f32>()?, data.as_slice());
                assert_eq!(reshaped.shape(), &[3, 2]);
                assert_eq!(cache.len(), 3);
                cache.col(session, vec![2, 3], &data)?;
                assert_eq!(cache.len(), 3);
                Ok::<_, tenferro_ad::Error>(())
            })
            .unwrap()
            .unwrap();
    }
}
