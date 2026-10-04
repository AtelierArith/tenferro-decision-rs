//! Cross-formulation parity: the tenferro chunked scan must match the host
//! recurrent reference.

use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::{chunked::delta_scan_chunked, delta_scan_reference, DeltaScanInputs};

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

struct Case {
    key_dim: usize,
    value_dim: usize,
    length: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    beta: Vec<f32>,
    decay: Vec<f32>,
    z: Vec<f32>,
    norm: Vec<f32>,
    eps: f32,
}

fn make_case(key_dim: usize, value_dim: usize, length: usize, seed: u64) -> Case {
    let mut rng = Lcg(seed);
    let mut q = (0..key_dim * length)
        .map(|_| rng.range(-1.0, 1.0))
        .collect::<Vec<_>>();
    let mut k = (0..key_dim * length)
        .map(|_| rng.range(-1.0, 1.0))
        .collect::<Vec<_>>();
    // L2-normalize each key/query column; scale q by 1/sqrt(key_dim), matching
    // the reference (`query ./= sqrt(sum + eps) .* sqrt(key_dim)`).
    for t in 0..length {
        let mut qcol: Vec<f32> = (0..key_dim).map(|d| q[d * length + t]).collect();
        tenferro_gated_delta::ops::l2_normalize(&mut qcol, 1e-6);
        for d in 0..key_dim {
            q[d * length + t] = qcol[d] / (key_dim as f32).sqrt();
        }
        let mut kcol: Vec<f32> = (0..key_dim).map(|d| k[d * length + t]).collect();
        tenferro_gated_delta::ops::l2_normalize(&mut kcol, 1e-6);
        for d in 0..key_dim {
            k[d * length + t] = kcol[d];
        }
    }
    let v = (0..value_dim * length)
        .map(|_| rng.range(-1.0, 1.0))
        .collect();
    let beta = (0..length).map(|_| rng.range(0.0, 1.0)).collect();
    let decay = (0..length).map(|_| rng.range(-0.5, -0.01)).collect();
    let z = (0..value_dim * length)
        .map(|_| rng.range(-1.0, 1.0))
        .collect();
    let norm = (0..value_dim).map(|_| rng.range(0.5, 1.5)).collect();
    Case {
        key_dim,
        value_dim,
        length,
        q,
        k,
        v,
        beta,
        decay,
        z,
        norm,
        eps: 1e-5,
    }
}

fn compare(case: &Case, chunk_size: usize, tolerance: f32) {
    let inputs = DeltaScanInputs {
        q: &case.q,
        k: &case.k,
        v: &case.v,
        beta: &case.beta,
        decay: &case.decay,
        z: &case.z,
        norm: &case.norm,
        eps: case.eps,
        key_dim: case.key_dim,
        value_dim: case.value_dim,
        length: case.length,
    };
    let reference = delta_scan_reference(&inputs);

    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let chunked = runtime
        .with_eager_session(|session| delta_scan_chunked(session, &inputs, chunk_size))
        .unwrap()
        .unwrap();

    assert_eq!(reference.len(), chunked.len());
    let mut max_diff = 0.0f32;
    for (a, b) in reference.iter().zip(&chunked) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff <= tolerance,
        "chunk_size {chunk_size}: max diff {max_diff} exceeds {tolerance}"
    );
}

#[test]
fn chunked_matches_reference_across_boundaries() {
    // Multiple chunks and a short final chunk.
    let case = make_case(4, 4, 10, 1);
    compare(&case, 4, 2e-3);
    // Single chunk covering the whole sequence.
    compare(&case, 64, 2e-3);
    // A short final chunk of length 1 (multiple of 3).
    let case = make_case(3, 6, 7, 2);
    compare(&case, 3, 2e-3);
}

#[test]
fn chunked_matches_reference_single_token() {
    let case = make_case(2, 4, 1, 3);
    compare(&case, 1, 1e-4);
    compare(&case, 64, 1e-4);
}

#[test]
fn chunked_matches_reference_with_large_decay_across_boundary() {
    // Strongly negative decay, as in the Qwen3.5 checkpoint. The chunk decay
    // sum underflows to zero over a 64-token chunk; the state update must still
    // use the host cumulative values (regression for the `exp(sum).ln()` path).
    let mut case = make_case(2, 2, 66, 11);
    for (index, value) in case.decay.iter_mut().enumerate() {
        *value = -1.5 - 8.0 * ((index % 3) as f32);
    }
    compare(&case, 64, 2e-3);
}

#[test]
fn chunked_matches_reference_with_groups() {
    // Value width different from key width.
    let case = make_case(5, 3, 9, 4);
    compare(&case, 4, 2e-3);
}

#[test]
fn causal_depthwise_convolution_is_causal() {
    // A single channel, 3 taps; the output at t only depends on taps <= t.
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let weight = [1.0f32, 0.0, 0.0]; // only the most recent tap (tap 0 => lag 2? see impl)
    let out = tenferro_gated_delta::causal_depthwise_silu(&input, 1, 4, &weight, 3);
    // tap 0 => lag = taps-1 = 2, so out[t] = silu(input[t-2]) for t >= 2.
    assert!((out[0] - 0.0f32).abs() < 1e-6);
    assert!((out[1] - 0.0f32).abs() < 1e-6);
    let expected2 = tenferro_gated_delta::ops::silu(1.0);
    assert!((out[2] - expected2).abs() < 1e-6);
}
