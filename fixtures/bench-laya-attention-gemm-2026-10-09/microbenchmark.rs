use rayon::prelude::*;
use std::time::Instant;
fn scalar_attention(
    qkv: &[f32],
    keep: &[bool],
    d: usize,
    heads: usize,
    length: usize,
    batch: usize,
    rope_base: f32,
    out: &mut [f32],
) {
    let n = d * length * batch;
    debug_assert_eq!(qkv.len(), 3 * n);
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(keep.len(), length * length * batch);
    if n == 0 || heads == 0 || d % heads != 0 {
        return;
    }
    let hd = d / heads;
    let half = hd / 2;
    let scale = 1.0 / (hd as f32).sqrt();

    // RoPE tables `(length, half)`, built once and shared across heads.
    let use_rope = rope_base != 0.0;
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    if use_rope {
        cos = vec![0.0f32; length * half];
        sin = vec![0.0f32; length * half];
        for l in 0..length {
            for i in 0..half {
                let theta = (l as f32) * rope_base.powf(-2.0 * (i as f32) / (hd as f32));
                cos[l * half + i] = theta.cos();
                sin[l * half + i] = theta.sin();
            }
        }
    }

    let qkv_addr = qkv.as_ptr() as usize;
    let keep_addr = keep.as_ptr() as usize;
    let out_addr = out.as_mut_ptr() as usize;
    let cos_addr = cos.as_ptr() as usize;
    let sin_addr = sin.as_ptr() as usize;
    (0..batch * heads).into_par_iter().for_each(|bh| {
        // SAFETY: each task owns one `(batch, head)` pair — a disjoint feature
        // band of `out` — and only reads `qkv`/`keep`/`cos`/`sin`.
        let b = bh / heads;
        let head = bh % heads;
        let feat = head * hd;
        unsafe {
            let qkv = qkv_addr as *const f32;
            let keep = keep_addr as *const bool;
            let out = out_addr as *mut f32;
            let mut q = vec![0.0f32; length * hd];
            let mut k = vec![0.0f32; length * hd];
            let mut v = vec![0.0f32; length * hd];
            // split qkv -> q/k/v head band
            for l in 0..length {
                let src = 3 * d * (l + length * b) + feat;
                let dst = l * hd;
                for i in 0..hd {
                    q[dst + i] = *qkv.add(src + i);
                    k[dst + i] = *qkv.add(src + d + i);
                    v[dst + i] = *qkv.add(src + 2 * d + i);
                }
            }
            if use_rope {
                let cos = cos_addr as *const f32;
                let sin = sin_addr as *const f32;
                for l in 0..length {
                    let row = l * hd;
                    let table = l * half;
                    for i in 0..half {
                        let c = *cos.add(table + i);
                        let s = *sin.add(table + i);
                        let a = q[row + i];
                        let bb = q[row + i + half];
                        q[row + i] = a * c - bb * s;
                        q[row + i + half] = a * s + bb * c;
                        let a = k[row + i];
                        let bb = k[row + i + half];
                        k[row + i] = a * c - bb * s;
                        k[row + i + half] = a * s + bb * c;
                    }
                }
            }
            // masked scaled dot-product attention for this head
            let mut probs = vec![0.0f32; length];
            for query in 0..length {
                let qo = query * hd;
                let mut max = f32::NEG_INFINITY;
                for (key, prob) in probs.iter_mut().enumerate() {
                    if !*keep.add(query + length * (key + length * b)) {
                        *prob = f32::NEG_INFINITY;
                        continue;
                    }
                    let ko = key * hd;
                    let mut acc = 0.0f32;
                    for i in 0..hd {
                        acc += q[qo + i] * k[ko + i];
                    }
                    let score = acc * scale;
                    *prob = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for value in probs.iter_mut() {
                    *value = if value.is_finite() {
                        (*value - max).exp()
                    } else {
                        0.0
                    };
                    sum += *value;
                }
                let o_base = feat + d * (query + length * b);
                for i in 0..hd {
                    *out.add(o_base + i) = 0.0;
                }
                for (key, &raw) in probs.iter().enumerate() {
                    let prob = raw / sum;
                    if prob == 0.0 {
                        continue;
                    }
                    let vo = key * hd;
                    for i in 0..hd {
                        *out.add(o_base + i) += v[vo + i] * prob;
                    }
                }
            }
        }
    });
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}
fn main() {
    println!(
        "{{\"rayon_threads\":{},\"warmup\":5,\"iterations\":15,\"rows\":[",
        rayon::current_num_threads()
    );
    let mut first = true;
    for (length, batch) in [
        (8usize, 1usize),
        (16, 1),
        (31, 1),
        (32, 1),
        (33, 1),
        (63, 1),
        (64, 1),
        (65, 1),
        (128, 1),
        (129, 1),
        (256, 1),
        (512, 1),
        (8, 8),
    ] {
        for window in [None, Some(1usize), Some(64)] {
            let sliding = window.is_some();
            let window_json = window
                .map(|width| width.to_string())
                .unwrap_or_else(|| "null".into());
            let d = 1024;
            let heads = 16;
            let qkv: Vec<f32> = (0..3 * d * length * batch)
                .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.002)
                .collect();
            let keep: Vec<bool> = (0..length * length * batch)
                .map(|i| {
                    let q = i % length;
                    let k = i / length % length;
                    (k % 13 != 7) && window.is_none_or(|width| q.abs_diff(k) <= width)
                })
                .collect();
            let mut old = vec![0.0f32; d * length * batch];
            let mut new = old.clone();
            let mut a = Vec::new();
            let mut b = Vec::new();
            for sample in 0..20 {
                for candidate in if sample % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let begin = Instant::now();
                    if candidate {
                        cpu_kernels::laya_attention_block_into(
                            &qkv, &keep, d, heads, length, batch, 160000.0, &mut new,
                        );
                    } else {
                        scalar_attention(&qkv, &keep, d, heads, length, batch, 160000.0, &mut old);
                    }
                    let ms = begin.elapsed().as_secs_f64() * 1000.0;
                    if sample >= 5 {
                        if candidate {
                            b.push(ms);
                        } else {
                            a.push(ms);
                        }
                    }
                }
            }
            let error = old
                .iter()
                .zip(&new)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(old.iter().chain(&new).all(|v| v.is_finite()));
            assert!(error <= 1e-5);
            if !first {
                println!(",");
            }
            first = false;
            println!(
                "{{\"length\":{length},\"batch\":{batch},\"sliding\":{sliding},\"window\":{window_json},\"scalar_ms\":{},\"gemm_ms\":{},\"max_abs_error\":{error},\"scalar_samples_ms\":{a:?},\"gemm_samples_ms\":{b:?}}}",
                median(&a),
                median(&b)
            );
        }
    }
    println!("]}}");
}
