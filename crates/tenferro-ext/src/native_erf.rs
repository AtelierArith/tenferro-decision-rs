//! Backend-native evaluation of erf using the CPU extension's coefficients.

use tenferro_ad::{CompareDir, EagerSession, EagerTensor, Result};
use tenferro_tensor::DType;

use crate::scalar_like;

/// Erf-based GELU evaluated entirely through native session operations.
pub fn gelu_erf_tensor_native(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
) -> Result<EagerTensor> {
    let scaled = session.scale_real(x, std::f64::consts::FRAC_1_SQRT_2)?;
    let erf = erf_tensor_native(session, &scaled)?;
    let one = scalar_like(session, x, 1.0)?;
    let gate = session.add(&erf, &one)?;
    let half = session.scale_real(x, 0.5)?;
    session.mul(&half, &gate)
}

/// Evaluate erf through admitted tenferro elementwise operations.
///
/// F32 uses the MLX piecewise polynomial coefficients of the CPU extension,
/// with native `expm1` and separate multiply/add operations. Results are
/// numerically equivalent, not bit-identical to its fused arithmetic. F64
/// uses the same Abramowitz–Stegun approximation as the CPU extension.
/// No tensor values are downloaded. Unsupported backend primitives return
/// their typed errors.
pub fn erf_tensor_native(session: &mut EagerSession<'_>, x: &EagerTensor) -> Result<EagerTensor> {
    let zero = scalar_like(session, x, 0.0)?;
    let magnitude = match x.dtype() {
        DType::F32 => {
            let abs = session.abs(x)?;
            // Beyond four, erf rounds to one in F32. Bounding the polynomial
            // also prevents overflow for very large finite values/infinities.
            let limit = scalar_like(session, x, 4.0)?;
            let t = session.minimum(&abs, &limit)?;
            let s = session.mul(&t, &t)?;
            let r = polynomial(session, x, &t, &[-1.728_534_7e-5, 3.831_971_3e-4])?;
            let u = polynomial(session, x, &t, &[-3.883_964_4e-3, 2.425_462_2e-2])?;
            let rs = session.mul(&r, &s)?;
            let mut r = session.add(&rs, &u)?;
            for coefficient in [-1.067_778_8e-1, -6.348_466_9e-1, -1.287_175_1e-1] {
                r = multiply_add_scalar(session, x, &r, &t, coefficient)?;
            }
            let rt = session.mul(&r, &t)?;
            let r = session.sub(&rt, &t)?;
            let e = session.expm1(&r)?;
            let large = session.neg(&e)?;
            let negative = session.compare(x, &zero, CompareDir::Lt)?;
            let neg_large = session.neg(&large)?;
            let large = session.where_select(&negative, &neg_large, &large)?;
            let small = polynomial(
                session,
                x,
                &s,
                &[
                    -5.967_617e-4,
                    4.991_194_2e-3,
                    -2.676_813_5e-2,
                    1.128_199_2e-1,
                    -3.761_253_4e-1,
                    1.283_791_7e-1,
                ],
            )?;
            let small = session.mul(&small, x)?;
            let small = session.add(&small, x)?;
            let boundary = scalar_like(session, x, 0.927_734_4)?;
            let use_large = session.compare(&abs, &boundary, CompareDir::Gt)?;
            session.where_select(&use_large, &large, &small)?
        }
        DType::F64 => {
            let abs = session.abs(x)?;
            let one = scalar_like(session, x, 1.0)?;
            let scaled = session.scale_real(&abs, 0.327_591_1)?;
            let denom = session.add(&one, &scaled)?;
            let t = session.div(&one, &denom)?;
            let poly = polynomial(
                session,
                x,
                &t,
                &[
                    1.061_405_429,
                    -1.453_152_027,
                    1.421_413_741,
                    -0.284_496_736,
                    0.254_829_592,
                ],
            )?;
            let poly = session.mul(&poly, &t)?;
            let square = session.mul(&abs, &abs)?;
            let neg = session.neg(&square)?;
            let exp = session.exp(&neg)?;
            let tail = session.mul(&poly, &exp)?;
            let positive = session.sub(&one, &tail)?;
            let negative = session.compare(x, &zero, CompareDir::Lt)?;
            let neg_positive = session.neg(&positive)?;
            session.where_select(&negative, &neg_positive, &positive)?
        }
        _ => unreachable!("scalar_like validated the floating dtype"),
    };
    // Preserve both signed zeros and NaNs independently of min/select backend
    // behavior. At zero the F64 approximation has a small constant residual.
    let is_zero = session.compare(x, &zero, CompareDir::Eq)?;
    let magnitude = session.where_select(&is_zero, x, &magnitude)?;
    let is_number = session.compare(x, x, CompareDir::Eq)?;
    session.where_select(&is_number, &magnitude, x)
}

fn multiply_add_scalar(
    session: &mut EagerSession<'_>,
    like: &EagerTensor,
    r: &EagerTensor,
    t: &EagerTensor,
    coefficient: f64,
) -> Result<EagerTensor> {
    let product = session.mul(r, t)?;
    let constant = scalar_like(session, like, coefficient)?;
    session.add(&product, &constant)
}

fn polynomial(
    session: &mut EagerSession<'_>,
    like: &EagerTensor,
    t: &EagerTensor,
    coefficients: &[f64],
) -> Result<EagerTensor> {
    let mut r = scalar_like(session, like, coefficients[0])?;
    for &coefficient in &coefficients[1..] {
        r = multiply_add_scalar(session, like, &r, t, coefficient)?;
    }
    Ok(r)
}
