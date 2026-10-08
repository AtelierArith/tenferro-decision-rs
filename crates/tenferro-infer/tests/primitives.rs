//! Numerical tests for the Phase 1 primitives, compared against small
//! independent Rust references.

use tenferro_ad::{EagerRuntime, EagerSession, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_infer as ti;

fn run<R: Send>(f: impl FnOnce(&mut EagerSession<'_>) -> ti::Result<R> + Send) -> R {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    ctx.with_eager_session(f).unwrap().unwrap()
}

fn vec1(values: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![values.len()], values.to_vec()).unwrap()
}

fn mat(rows: usize, cols: usize, values: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![rows, cols], values.to_vec()).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {index}: actual {a} expected {e}"
        );
    }
}

// ------------------------------------------------------------------ LayerNorm

#[test]
fn layer_norm_matches_reference() {
    let rows = 4;
    let cols = 3;
    let x = mat(
        rows,
        cols,
        &[1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.5, -1.5, 4.0],
    );
    let weight = vec1(&[0.5, 1.0, 2.0]);
    let bias = vec1(&[0.1, -0.2, 0.3]);
    let eps = 1e-5;

    let out = run(|s| {
        let x = s.constant_from(x)?;
        let w = s.constant_from(weight)?;
        let b = s.constant_from(bias)?;
        let y = ti::norm::layer_norm(s, &x, &w, Some(&b), eps)?;
        s.duplicate_value(&y)
    });

    let xv: [f64; 12] = [1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.5, -1.5, 4.0];
    let mut expected = vec![0.0; rows * cols];
    for i in 0..rows {
        let mean = (0..cols).map(|j| xv[i + j * rows]).sum::<f64>() / cols as f64;
        let var = (0..cols)
            .map(|j| (xv[i + j * rows] - mean).powi(2))
            .sum::<f64>()
            / cols as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..cols {
            let w = [0.5, 1.0, 2.0][j];
            let b = [0.1, -0.2, 0.3][j];
            expected[i + j * rows] = (xv[i + j * rows] - mean) * inv * w + b;
        }
    }
    assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-9);
}

// -------------------------------------------------------------------- RMSNorm

#[test]
fn rms_norm_matches_reference() {
    let rows = 3;
    let cols = 4;
    let xv: [f64; 12] = [1.0, -2.0, 0.5, 3.0, 0.0, 1.0, -1.0, 2.0, 4.0, 4.0, 4.0, 4.0];
    let wv: [f64; 4] = [1.0, 0.5, -0.25, 2.0];
    let eps = 1e-6;

    for centered in [false, true] {
        let out = run(|s| {
            let x = s.constant_from(Tensor::from_vec_col_major(vec![rows, cols], xv.to_vec())?)?;
            let w = s.constant_from(Tensor::from_vec_col_major(vec![cols], wv.to_vec())?)?;
            let y = ti::norm::rms_norm(s, &x, &w, centered, eps)?;
            let prepared_eps = s.constant_from(Tensor::from_vec_col_major(vec![], vec![eps])?)?;
            let prepared = ti::norm::rms_norm_with_epsilon(s, &x, &w, centered, &prepared_eps)?;
            let doubled = s.scale_real(&x, 2.0)?;
            // Reuse one epsilon for different inputs and retain the earlier
            // result while evaluating the second normalization.
            let reused = ti::norm::rms_norm_with_epsilon(s, &doubled, &w, centered, &prepared_eps)?;
            let expected_reused = ti::norm::rms_norm(s, &doubled, &w, centered, eps)?;
            assert_close(
                s.duplicate_value(&reused)?.as_slice::<f64>()?,
                s.duplicate_value(&expected_reused)?.as_slice::<f64>()?,
                1e-9,
            );
            assert_close(
                s.duplicate_value(&prepared)?.as_slice::<f64>()?,
                s.duplicate_value(&y)?.as_slice::<f64>()?,
                1e-9,
            );
            let vector_eps = s.constant_from(Tensor::from_vec_col_major(vec![1], vec![eps])?)?;
            assert!(ti::norm::rms_norm_with_epsilon(s, &x, &w, centered, &vector_eps).is_err());
            s.duplicate_value(&y)
        });

        let mut expected = vec![0.0; rows * cols];
        for i in 0..rows {
            let mean_sq = (0..cols).map(|j| xv[i + j * rows].powi(2)).sum::<f64>() / cols as f64;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for j in 0..cols {
                let scale = if centered { 1.0 + wv[j] } else { wv[j] };
                expected[i + j * rows] = xv[i + j * rows] * inv * scale;
            }
        }
        assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-9);
    }
}

// -------------------------------------------------------------------- Softmax

#[test]
fn softmax_matches_reference() {
    let rows = 3;
    let cols = 4;
    let x = mat(
        rows,
        cols,
        &[
            1.0, 2.0, 3.0, 0.0, -1.0, -2.0, 0.5, 0.5, 5.0, -5.0, 0.0, 1.0,
        ],
    );

    let out = run(|s| {
        let x = s.constant_from(x)?;
        let y = ti::softmax::softmax(s, &x, 1)?;
        s.duplicate_value(&y)
    });

    let xv = [
        1.0, 2.0, 3.0, 0.0, -1.0, -2.0, 0.5, 0.5, 5.0, -5.0, 0.0, 1.0,
    ];
    let mut expected = vec![0.0; rows * cols];
    for i in 0..rows {
        let max = (0..cols).map(|j| xv[i + j * rows]).fold(f64::MIN, f64::max);
        let exps: Vec<f64> = (0..cols).map(|j| (xv[i + j * rows] - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for j in 0..cols {
            expected[i + j * rows] = exps[j] / sum;
        }
    }
    assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-12);

    // Rows sum to one.
    let values = out.as_slice::<f64>().unwrap();
    for i in 0..rows {
        let sum: f64 = (0..cols).map(|j| values[i + j * rows]).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
}

#[test]
fn rms_norm_rank3_matches_reference() {
    let a = 2;
    let b = 3;
    let c = 4;
    let wv: [f64; 4] = [1.0, 0.5, -0.25, 2.0];
    let eps = 1e-6;
    let xv: Vec<f64> = (0..a * b * c).map(|i| (i as f64) * 0.37 - 2.0).collect();
    for centered in [false, true] {
        let out = run(|s| {
            let x = s.constant_from(Tensor::from_vec_col_major(vec![a, b, c], xv.clone())?)?;
            let w = s.constant_from(Tensor::from_vec_col_major(vec![c], wv.to_vec())?)?;
            let y = ti::norm::rms_norm(s, &x, &w, centered, eps)?;
            s.duplicate_value(&y)
        });
        // Column-major: index = i + a*j + a*b*k.
        let mut expected = vec![0.0; a * b * c];
        for i in 0..a {
            for j in 0..b {
                let base = i + a * j;
                let mean_sq: f64 =
                    (0..c).map(|k| xv[base + a * b * k].powi(2)).sum::<f64>() / c as f64;
                let inv = 1.0 / (mean_sq + eps).sqrt();
                for k in 0..c {
                    let scale = if centered { 1.0 + wv[k] } else { wv[k] };
                    expected[base + a * b * k] = xv[base + a * b * k] * inv * scale;
                }
            }
        }
        assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-9);
    }
}

// ------------------------------------------------------- Activations

#[test]
fn sigmoid_and_silu_match_reference() {
    let input = vec1(&[-2.0, -0.5, 0.0, 0.5, 2.0]);

    let (sigmoid, silu) = run(|s| {
        let x = s.constant_from(input)?;
        let sig = ti::activation::sigmoid(s, &x)?;
        let si = ti::activation::silu(s, &x)?;
        let sig = s.duplicate_value(&sig)?;
        let si = s.duplicate_value(&si)?;
        Ok((sig, si))
    });

    let xv: [f64; 5] = [-2.0, -0.5, 0.0, 0.5, 2.0];
    let expected_sig: Vec<f64> = xv
        .iter()
        .copied()
        .map(|x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    let expected_silu: Vec<f64> = xv.iter().copied().map(|x| x / (1.0 + (-x).exp())).collect();
    assert_close(sigmoid.as_slice::<f64>().unwrap(), &expected_sig, 1e-12);
    assert_close(silu.as_slice::<f64>().unwrap(), &expected_silu, 1e-12);
}

#[test]
fn gelu_tanh_matches_reference() {
    let input = vec1(&[-3.0, -0.5, 0.0, 0.5, 3.0]);
    let out = run(|s| {
        let x = s.constant_from(input)?;
        let y = ti::activation::gelu(s, &x)?;
        s.duplicate_value(&y)
    });

    let c = (2.0 / std::f64::consts::PI).sqrt();
    let expected: Vec<f64> = [-3.0f64, -0.5, 0.0, 0.5, 3.0]
        .iter()
        .copied()
        .map(|x| 0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh()))
        .collect();
    assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-12);
}

// --------------------------------------------------------------------- Linear

#[test]
fn linear_matches_reference() {
    let x = mat(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // (2,3)
    let w = mat(3, 2, &[1.0, 0.0, -1.0, 2.0, 0.5, 1.0]); // (3,2)

    let out = run(|s| {
        let x = s.constant_from(x)?;
        let w = s.constant_from(w)?;
        let y = ti::linear::linear(s, &x, &w)?;
        s.duplicate_value(&y)
    });

    // Column-major (in, out): w[i + j*3].
    let xv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let wv = [1.0, 0.0, -1.0, 2.0, 0.5, 1.0];
    let mut expected = vec![0.0; 2 * 2];
    for row in 0..2 {
        for j in 0..2 {
            let mut acc = 0.0;
            for i in 0..3 {
                acc += xv[row + i * 2] * wv[i + j * 3];
            }
            expected[row + j * 2] = acc;
        }
    }
    assert_eq!(out.shape(), &[2, 2]);
    assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-12);
}

// ------------------------------------------------------------------ Embedding

#[test]
fn embedding_matches_reference() {
    let table = mat(
        5,
        3,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
        ],
    );
    let indices = Tensor::from_vec_col_major(vec![3], vec![4_i64, 0, 2]).unwrap();

    let out = run(|s| {
        let table = s.constant_from(table)?;
        let indices = s.constant_from(indices)?;
        let y = ti::embedding::embedding(s, &table, &indices)?;
        s.duplicate_value(&y)
    });

    assert_eq!(out.shape(), &[3, 3]);
    let expected = [4.0, 0.0, 2.0, 9.0, 5.0, 7.0, 14.0, 10.0, 12.0];
    assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-12);
}

// ------------------------------------------------------------------------ RoPE

#[test]
fn modernbert_rope_matches_reference() {
    let seq = 3;
    let head = 4;
    let base = 10000.0;
    let x = mat(
        seq,
        head,
        &[1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 0.5, -1.0, 2.0, 2.0],
    );

    let out = run(|s| {
        let x = s.constant_from(x)?;
        let y = ti::rope::rope_modernbert(s, &x, base)?;
        s.duplicate_value(&y)
    });

    let xv = [1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 0.5, -1.0, 2.0, 2.0];
    let half = head / 2;
    let mut expected = vec![0.0; seq * head];
    for p in 0..seq {
        for i in 0..half {
            let theta = (p as f64) * base.powf(-2.0 * i as f64 / head as f64);
            let first = xv[p + i * seq];
            let second = xv[p + (i + half) * seq];
            expected[p + i * seq] = first * theta.cos() - second * theta.sin();
            expected[p + (i + half) * seq] = first * theta.sin() + second * theta.cos();
        }
    }
    assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-12);

    // Position 0 is unchanged.
    let values = out.as_slice::<f64>().unwrap();
    for k in 0..head {
        assert!((values[k * seq] - xv[k * seq]).abs() < 1e-12);
    }
}

#[test]
fn qwen_partial_rope_matches_reference() {
    let heads = 2;
    let seq = 3;
    let head = 4;
    let rotary = 2;
    let base = 10000.0;
    // Layout (heads, seq, head): flat = h + heads*t + heads*seq*d.
    let data: Vec<f64> = (0..heads * seq * head)
        .map(|i| (i as f64) * 0.1 - 0.3)
        .collect();
    let x = Tensor::from_vec_col_major(vec![heads, seq, head], data.clone()).unwrap();

    let out = run(|s| {
        let x = s.constant_from(x)?;
        let y = ti::rope::rope_qwen_partial(s, &x, base, rotary)?;
        s.duplicate_value(&y)
    });

    let half = rotary / 2;
    let mut expected = data.clone();
    for h in 0..heads {
        for t in 0..seq {
            let base_idx = h + heads * t;
            for i in 0..half {
                let angle = (t as f64) * base.powf(-2.0 * i as f64 / rotary as f64);
                let first = data[base_idx + heads * seq * i];
                let second = data[base_idx + heads * seq * (i + half)];
                expected[base_idx + heads * seq * i] = first * angle.cos() - second * angle.sin();
                expected[base_idx + heads * seq * (i + half)] =
                    first * angle.sin() + second * angle.cos();
            }
        }
    }
    assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-12);
}

#[test]
fn attention_masked_multi_head_matches_reference() {
    let b = 1;
    let h = 2;
    let l = 2;
    let hd = 2;
    // Column-major (B,H,L,hd): flat = b + B*h + B*H*l + B*H*L*d.
    let q = Tensor::from_vec_col_major(
        vec![b, h, l, hd],
        vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.2, 0.4],
    )
    .unwrap();
    let k = Tensor::from_vec_col_major(
        vec![b, h, l, hd],
        vec![1.0, 0.0, 0.0, 1.0, 0.3, 0.7, 0.1, 0.9],
    )
    .unwrap();
    let v = Tensor::from_vec_col_major(
        vec![b, h, l, hd],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    // Causal mask: keep[r,c] = c <= r.
    let mask = Tensor::from_vec_col_major(vec![l, l], vec![true, true, false, true]).unwrap();

    let out = run(|s| {
        let q = s.constant_from(q)?;
        let k = s.constant_from(k)?;
        let v = s.constant_from(v)?;
        let mask = s.constant_from(mask)?;
        let y = ti::attention::attention(s, &q, &k, &v, Some(&mask), None)?;
        s.duplicate_value(&y)
    });

    let qv = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.2, 0.4];
    let kv = [1.0, 0.0, 0.0, 1.0, 0.3, 0.7, 0.1, 0.9];
    let vv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let idx = |hh: usize, ll: usize, d: usize| hh + ll * h + d * h * l;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut expected = vec![0.0; b * h * l * hd];
    for hh in 0..h {
        for qq in 0..l {
            let mut scores = vec![0.0; l];
            for kk in 0..l {
                let mut acc = 0.0;
                for d in 0..hd {
                    acc += qv[idx(hh, qq, d)] * kv[idx(hh, kk, d)];
                }
                scores[kk] = if kk <= qq {
                    acc * scale
                } else {
                    f64::NEG_INFINITY
                };
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores
                .iter()
                .map(|s| if s.is_finite() { (s - max).exp() } else { 0.0 })
                .collect();
            let sum: f64 = exps.iter().sum();
            for d in 0..hd {
                let mut acc = 0.0;
                for kk in 0..l {
                    acc += vv[idx(hh, kk, d)] * (exps[kk] / sum);
                }
                expected[idx(hh, qq, d)] = acc;
            }
        }
    }
    assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-12);
}

// ------------------------------------------------------------------- Attention

#[test]
fn attention_matches_reference() {
    let q = Tensor::from_vec_col_major(vec![1, 1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let k = Tensor::from_vec_col_major(vec![1, 1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let v = Tensor::from_vec_col_major(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let out = run(|s| {
        let q = s.constant_from(q)?;
        let k = s.constant_from(k)?;
        let v = s.constant_from(v)?;
        let y = ti::attention::attention(s, &q, &k, &v, None, None)?;
        s.duplicate_value(&y)
    });

    // Reference: scores = q k^T / sqrt(2); rows softmax; out = P v.
    let scale = 1.0 / 2.0_f64.sqrt();
    let scores = [[1.0 * scale, 0.0], [0.0, 1.0 * scale]];
    let mut probs = [[0.0; 2]; 2];
    for l in 0..2 {
        let max = scores[l][0].max(scores[l][1]);
        let e0 = (scores[l][0] - max).exp();
        let e1 = (scores[l][1] - max).exp();
        probs[l][0] = e0 / (e0 + e1);
        probs[l][1] = e1 / (e0 + e1);
    }
    let vv = [[1.0, 3.0], [2.0, 4.0]]; // v[m][d]: column-major (Lk, hd)
    let mut expected = [0.0; 4];
    for l in 0..2 {
        for d in 0..2 {
            expected[l + d * 2] = probs[l][0] * vv[0][d] + probs[l][1] * vv[1][d];
        }
    }

    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    assert_close(out.as_slice::<f64>().unwrap(), &expected, 1e-12);
}
