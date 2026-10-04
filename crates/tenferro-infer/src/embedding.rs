//! Token embedding lookup.

use tenferro_ad::{EagerSession, EagerTensor, GatherConfig, Result};

/// Look up rows of `table` (shape `(vocab, dim)`) by integer `indices`.
///
/// `indices` is an `i64` tensor of any shape; the output is
/// `indices.shape() ++ [dim]`. Implemented as a single `gather`.
pub fn embedding(
    session: &mut EagerSession<'_>,
    table: &EagerTensor,
    indices: &EagerTensor,
) -> Result<EagerTensor> {
    let dim = table.shape()[1];
    let config = GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: indices.shape().len(),
        slice_sizes: vec![1, dim],
    };
    session.gather(table, indices, config)
}
