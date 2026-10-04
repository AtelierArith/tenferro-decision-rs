//! Chunked Gated DeltaNet building blocks for the Jeff engine.
//!
//! Design: `docs/agents/specs/docs/12_TENFERRO_GATED_DELTA.md`. This crate is
//! Phase 6 work in progress.
//!
//! Implemented now (host reference, no tenferro dependency yet): the chunked
//! effective linear system from `docs/agents/specs/docs/09_JEFFCLIENT_ANALYSIS.md`
//! §7.3,
//!
//! ```text
//! M = I + L,   L[i, j] = beta_i * (k_i . k_j) * exp(c_i - c_j)   for i > j
//! ```
//!
//! solved with a unit-diagonal forward substitution. The full layer (causal
//! convolution, Q/K normalization, decay/beta preparation, and the
//! tenferro-backed chunked/recurrent forms) follows in Phase 6.

use decision_core::{DecisionError, Result};

/// Build the `n x n` row-major effective matrix `M = I + L`.
///
/// `beta[i]` is the write strength, `keys[i]` the (already L2-normalized) key
/// vector of row `i`, and `cum_decay[i]` the cumulative log-decay `c_i`.
pub fn build_effective_matrix(
    beta: &[f64],
    keys: &[Vec<f64>],
    cum_decay: &[f64],
) -> Result<Vec<Vec<f64>>> {
    let n = beta.len();
    if keys.len() != n || cum_decay.len() != n {
        return Err(DecisionError::invalid_field(
            "delta",
            "beta, keys, and cum_decay must have the same length",
        ));
    }
    let key_dim = keys.first().map(Vec::len).unwrap_or(0);
    if keys.iter().any(|k| k.len() != key_dim) {
        return Err(DecisionError::invalid_field(
            "delta.keys",
            "all key vectors must have the same length",
        ));
    }

    let mut m = vec![vec![0.0; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for i in 0..n {
        for j in 0..i {
            let dot: f64 = keys[i].iter().zip(&keys[j]).map(|(a, b)| a * b).sum();
            m[i][j] = beta[i] * dot * (cum_decay[i] - cum_decay[j]).exp();
        }
    }
    Ok(m)
}

/// Solve `M x = rhs` for a unit-diagonal lower-triangular row-major `M`.
///
/// `rhs` is `n x dv`, row-major. The system matrix must have ones on the
/// diagonal (as produced by [`build_effective_matrix`]).
pub fn forward_substitute(m: &[Vec<f64>], rhs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = m.len();
    if rhs.len() != n {
        return Err(DecisionError::invalid_field(
            "delta",
            "matrix and right-hand side must agree on the chunk length",
        ));
    }
    let dv = rhs.first().map(Vec::len).unwrap_or(0);
    if rhs.iter().any(|row| row.len() != dv) {
        return Err(DecisionError::invalid_field(
            "delta.rhs",
            "all right-hand-side rows must have the same length",
        ));
    }
    for (i, row) in m.iter().enumerate() {
        if row.len() != n {
            return Err(DecisionError::invalid_field(
                "delta.matrix",
                "matrix must be square",
            ));
        }
        if row[i] != 1.0 {
            return Err(DecisionError::invalid_field(
                "delta.matrix",
                "the diagonal must be unit",
            ));
        }
    }

    let mut out = vec![vec![0.0; dv]; n];
    for i in 0..n {
        for d in 0..dv {
            let mut value = rhs[i][d];
            for j in 0..i {
                value -= m[i][j] * out[j][d];
            }
            out[i][d] = value;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_no_coupling() {
        // Orthogonal keys -> zero off-diagonal coupling.
        let beta = [1.0, 1.0];
        let keys = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cum_decay = [0.0, 0.0];
        let m = build_effective_matrix(&beta, &keys, &cum_decay).unwrap();
        assert_eq!(m, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        let rhs = vec![vec![2.0], vec![3.0]];
        let x = forward_substitute(&m, &rhs).unwrap();
        assert_eq!(x, rhs);
    }

    #[test]
    fn solves_lower_triangular_system() {
        let m = vec![vec![1.0, 0.0], vec![0.5, 1.0]];
        let rhs = vec![vec![1.0], vec![0.0]];
        let x = forward_substitute(&m, &rhs).unwrap();
        // x0 = 1; x1 = 0 - 0.5 * 1 = -0.5.
        assert_eq!(x, vec![vec![1.0], vec![-0.5]]);
    }

    #[test]
    fn rejects_shape_mismatch() {
        assert!(build_effective_matrix(&[1.0], &[], &[0.0]).is_err());
        assert!(forward_substitute(&[vec![1.0, 0.0]], &[vec![1.0]]).is_err());
    }
}
