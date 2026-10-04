//! Activation functions shared by the engines.
//!
//! tenferro has no `erf`, `sigmoid`, `silu`, or `gelu` op, so these are
//! composed from the elementwise surface. `gelu` uses the tanh approximation;
//! the exact erf-based form that Laya's MLX kernels use requires a dedicated
//! extension op (documented in `15_TENFERRO_INFER_DESIGN.md` and tracked as a
//! Phase 2 gap).

use std::f64::consts::PI;

use tenferro_ad::{EagerSession, EagerTensor, Result};

use crate::util::scalar_like;

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
pub fn sigmoid(session: &mut EagerSession<'_>, x: &EagerTensor) -> Result<EagerTensor> {
    let one = scalar_like(session, x, 1.0)?;
    let neg = session.neg(x)?;
    let exp = session.exp(&neg)?;
    let denom = session.add(&exp, &one)?;
    session.div(&one, &denom)
}

/// SiLU / swish: `x * sigmoid(x)`.
pub fn silu(session: &mut EagerSession<'_>, x: &EagerTensor) -> Result<EagerTensor> {
    let gate = sigmoid(session, x)?;
    session.mul(x, &gate)
}

/// Gated SiLU used by the Jeff MLP: `silu(gate) * up`.
pub fn gated_silu(
    session: &mut EagerSession<'_>,
    gate: &EagerTensor,
    up: &EagerTensor,
) -> Result<EagerTensor> {
    let activated = silu(session, gate)?;
    session.mul(&activated, up)
}

/// GELU, tanh approximation.
///
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
///
/// This is *not* bit-identical to the erf form used by the Laya reference; see
/// the module note.
pub fn gelu(session: &mut EagerSession<'_>, x: &EagerTensor) -> Result<EagerTensor> {
    let x2 = session.mul(x, x)?;
    let x3 = session.mul(&x2, x)?;
    let cubic = session.scale_real(&x3, 0.044715)?;
    let inner = session.add(x, &cubic)?;
    let root_two_over_pi = (2.0 / PI).sqrt();
    let inner = session.scale_real(&inner, root_two_over_pi)?;
    let tanh = session.tanh(&inner)?;
    let one = scalar_like(session, x, 1.0)?;
    let gate = session.add(&tanh, &one)?;
    let half = session.scale_real(x, 0.5)?;
    session.mul(&half, &gate)
}

/// GeGLU in Laya's order: `gelu(value) * gate`, where `value` is the first half
/// of the up-projection output and `gate` the second.
pub fn geglu(
    session: &mut EagerSession<'_>,
    value: &EagerTensor,
    gate: &EagerTensor,
) -> Result<EagerTensor> {
    let activated = gelu(session, value)?;
    session.mul(&activated, gate)
}
