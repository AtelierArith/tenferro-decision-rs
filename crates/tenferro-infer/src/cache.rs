//! Reusable cache of weight tensors for the eager session.
//!
//! An [`EagerTensor`] keeps an `Arc<EagerRuntime>`, so a tensor created for a
//! weight can be reused by any later session that shares the same runtime. The
//! engines cache their weight tensors here so the tenferro path does not
//! re-transpose and re-create every weight on each forward (its dominant
//! overhead). Entries are keyed by the host storage identity, so callers must
//! keep the weight slices alive while the cache is in use.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use tenferro_ad::{ContextId, EagerSession, EagerTensor, Result};
use tenferro_tensor::{DType, Tensor};

/// Reusable column-major weights and runtime-local floating scalar constants.
#[derive(Clone, Debug, Default)]
pub struct TensorCache {
    weights: HashMap<WeightKey, EagerTensor>,
    scalars: HashMap<ScalarKey, EagerTensor>,
    scalar_runtime: Option<ContextId>,
    prepared: HashMap<(TypeId, usize, usize, Vec<usize>), PreparedValue>,
}

#[derive(Clone)]
struct PreparedValue(Arc<dyn Any + Send + Sync>);
impl std::fmt::Debug for PreparedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PreparedValue")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ScalarKey {
    F32(u32),
    F64(u64),
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
        self.weights.len() + self.scalars.len() + self.prepared.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty() && self.scalars.is_empty() && self.prepared.is_empty()
    }

    /// Reuse an owned host-weight preparation, keyed by type, source storage
    /// identity and logical shape. As with weight tensors, keep the source
    /// slice alive and immutable while this cache is in use. The prepared
    /// value must own its resources and must not retain an eager runtime.
    pub fn prepared_host<T: Any + Send + Sync>(
        &mut self,
        source: &[f32],
        shape: &[usize],
        create: impl FnOnce() -> Result<T>,
    ) -> Result<Arc<T>> {
        let key = (
            TypeId::of::<T>(),
            source.as_ptr() as usize,
            source.len(),
            shape.to_vec(),
        );
        if let Some(value) = self.prepared.get(&key) {
            return Arc::clone(&value.0).downcast::<T>().map_err(|_| {
                tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
                    "TensorCache::prepared_host",
                    "type",
                    "prepared cache type mismatch",
                ))
            });
        }
        let value = Arc::new(create()?);
        self.prepared.insert(key, PreparedValue(value.clone()));
        Ok(value)
    }

    /// Cache a rank-zero floating constant in `like`'s dtype and runtime.
    ///
    /// Values are keyed by their effective floating bits (including signed
    /// zero). The scalar cache retains only one runtime's entries; switching
    /// runtime clears those entries. Weight entries keep their existing
    /// same-runtime/living-host-storage contract. `like` must belong to this
    /// admitted session, as for every native operation using the result.
    pub fn scalar_like(
        &mut self,
        session: &mut EagerSession<'_>,
        like: &EagerTensor,
        value: f64,
    ) -> Result<EagerTensor> {
        let key = match like.dtype() {
            DType::F32 => ScalarKey::F32((value as f32).to_bits()),
            DType::F64 => ScalarKey::F64(value.to_bits()),
            dtype => return Err(crate::util::unsupported(dtype, "floating point")),
        };
        if self.scalar_runtime == Some(like.ctx_id()) {
            if let Some(tensor) = self.scalars.get(&key) {
                return Ok(tensor.clone());
            }
        }
        let tensor = crate::util::scalar_like(session, like, value)?;
        if tensor.ctx_id() != like.ctx_id() {
            return Err(tenferro_ad::Error::TensorRuntime(
                tenferro_tensor::Error::invalid_argument(
                    "TensorCache::scalar_like",
                    "runtime",
                    "like belongs to another runtime",
                ),
            ));
        }
        if self.scalar_runtime != Some(like.ctx_id()) {
            self.scalars.clear();
            self.scalar_runtime = Some(like.ctx_id());
        }
        self.scalars.insert(key, tensor.clone());
        Ok(tensor)
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
    fn scalar_values_keep_bits_dtype_and_runtime_distinct() {
        let first = tenferro_ad::EagerRuntime::new().unwrap();
        let second = tenferro_ad::EagerRuntime::new().unwrap();
        let mut cache = TensorCache::new();
        let retained = first
            .with_eager_session(|session| {
                let x = session
                    .constant_from_host(Tensor::from_vec_col_major(vec![1], vec![1.0f32])?)?;
                let a = cache.scalar_like(session, &x, 0.25)?;
                let pointer = a.value()?.as_slice::<f32>()?.as_ptr();
                let same = cache.scalar_like(session, &x, 0.25 + 1e-10)?;
                assert_eq!(same.value()?.as_slice::<f32>()?.as_ptr(), pointer);
                assert_eq!(cache.len(), 1);
                let plus = cache.scalar_like(session, &x, 0.0)?;
                let minus = cache.scalar_like(session, &x, -0.0)?;
                assert_eq!(
                    plus.value()?.as_slice::<f32>()?[0].to_bits(),
                    0.0f32.to_bits()
                );
                assert_eq!(
                    minus.value()?.as_slice::<f32>()?[0].to_bits(),
                    (-0.0f32).to_bits()
                );
                let double = session
                    .constant_from_host(Tensor::from_vec_col_major(vec![1], vec![1.0f64])?)?;
                let d = cache.scalar_like(session, &double, 0.25)?;
                assert_eq!(d.dtype(), DType::F64);
                assert_eq!(cache.len(), 4);
                Ok::<_, tenferro_ad::Error>(a)
            })
            .unwrap()
            .unwrap();
        second
            .with_eager_session(|session| {
                let x = session
                    .constant_from_host(Tensor::from_vec_col_major(vec![1], vec![2.0f32])?)?;
                let a = cache.scalar_like(session, &x, 0.25)?;
                assert_eq!(a.ctx_id(), x.ctx_id());
                assert_ne!(a.ctx_id(), retained.ctx_id());
                assert_eq!(cache.len(), 1);
                assert_eq!(retained.value()?.as_slice::<f32>()?, &[0.25]);
                Ok::<_, tenferro_ad::Error>(())
            })
            .unwrap()
            .unwrap();
    }

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

#[cfg(test)]
mod prepared_tests {
    use super::*;
    #[test]
    fn owned_preparations_are_keyed_by_type_storage_shape_and_shared_by_clones() {
        let source = vec![1., 2., 3., 4.];
        let other = source.clone();
        let mut cache = TensorCache::new();
        let first = cache
            .prepared_host(&source, &[2, 2], || Ok(17usize))
            .unwrap();
        let again = cache
            .prepared_host(&source, &[2, 2], || panic!("prepared twice"))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        assert!(Arc::ptr_eq(
            &first,
            &cache
                .clone()
                .prepared_host(&source, &[2, 2], || panic!("clone prepared twice"))
                .unwrap()
        ));
        let different = cache
            .prepared_host(&other, &[2, 2], || Ok(19usize))
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &different));
        let reshaped = cache.prepared_host(&source, &[4], || Ok(23usize)).unwrap();
        assert_eq!(*reshaped, 23);
        let typed = cache
            .prepared_host(&source, &[2, 2], || Ok(String::from("owned")))
            .unwrap();
        assert_eq!(&*typed, "owned");
        assert_eq!(cache.len(), 4);
        drop(cache);
        assert_eq!(*first, 17);
    }
    #[test]
    fn preparation_errors_do_not_populate_the_cache() {
        let source = [1.];
        let mut cache = TensorCache::new();
        let result = cache.prepared_host::<usize>(&source, &[1], || {
            Err(tenferro_ad::Error::TensorRuntime(
                tenferro_tensor::Error::invalid_argument("test", "prepare", "failed"),
            ))
        });
        assert!(result.is_err());
        assert!(cache.is_empty());
        assert_eq!(
            *cache.prepared_host(&source, &[1], || Ok(7usize)).unwrap(),
            7
        );
    }
}
