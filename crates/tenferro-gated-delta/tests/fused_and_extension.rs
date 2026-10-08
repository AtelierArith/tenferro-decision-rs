//! Fused recurrent kernel, prepared plans/workspaces, direct entry, and the
//! `GatedDelta` extension op — all cross-checked against the host reference.

use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::{
    Algorithm, AlgorithmChoice, BackendCaps, EagerSessionGatedDeltaExt, GatedDeltaConfig,
    GatedDeltaOp, GatedDeltaPlan, GatedDeltaWeights, GatedDeltaWorkspace, delta_layer_recurrent,
    delta_layer_reference, gated_delta,
};

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }

    fn fill(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| lo + (hi - lo) * self.next_f32()).collect()
    }
}

fn config() -> GatedDeltaConfig {
    GatedDeltaConfig {
        hidden: 8,
        key_heads: 1,
        value_heads: 2,
        key_dim: 4,
        value_dim: 4,
        conv_taps: 3,
        eps: 1e-5,
        chunk_size: 4,
        algorithm: Algorithm::Chunked,
    }
}

fn weights(cfg: &GatedDeltaConfig, seed: u64) -> GatedDeltaWeights {
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;
    let mut rng = Lcg(seed);
    GatedDeltaWeights {
        qkv: rng.fill(cfg.hidden * conv_channels, -0.3, 0.3),
        z: rng.fill(cfg.hidden * value_width, -0.3, 0.3),
        a: rng.fill(cfg.hidden * cfg.value_heads, -0.3, 0.3),
        b: rng.fill(cfg.hidden * cfg.value_heads, -0.3, 0.3),
        conv: rng.fill(cfg.conv_taps * conv_channels, -0.5, 0.5),
        a_decay: rng.fill(cfg.value_heads, -1.0, -0.05),
        dt_bias: rng.fill(cfg.value_heads, -0.5, 0.5),
        norm: rng.fill(cfg.value_dim, 0.5, 1.5),
        out_proj: rng.fill(value_width * cfg.hidden, -0.3, 0.3),
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn sample() -> (GatedDeltaConfig, GatedDeltaWeights, Vec<f32>, Vec<f32>) {
    let cfg = config();
    let w = weights(&cfg, 7);
    let length = 6;
    let mut rng = Lcg(99);
    let x = rng.fill(cfg.hidden * length, -1.0, 1.0);
    let mask = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 1.0];
    (cfg, w, x, mask)
}

#[test]
fn recurrent_matches_reference() {
    let (cfg, w, x, mask) = sample();
    let reference = delta_layer_reference(&cfg, &w, &x, &mask).unwrap();
    let mut ws = GatedDeltaWorkspace::new();
    let recurrent = delta_layer_recurrent(&cfg, &w, &x, &mask, &mut ws)
        .unwrap()
        .to_vec();
    assert!(max_diff(&reference, &recurrent) < 1e-6);
    assert!(ws.retained_bytes() > 0);
}

#[test]
fn direct_entry_dispatches_on_plan() {
    let (cfg, w, x, mask) = sample();
    let reference = delta_layer_reference(&cfg, &w, &x, &mask).unwrap();
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let mut ws = GatedDeltaWorkspace::new();

    for choice in [
        AlgorithmChoice::Reference,
        AlgorithmChoice::Recurrent,
        AlgorithmChoice::Chunked,
    ] {
        let plan = GatedDeltaPlan::resolve(&cfg, choice, BackendCaps::cpu(), mask.len()).unwrap();
        let out = runtime
            .with_eager_session(|session| gated_delta(session, &cfg, &w, &x, &mask, &plan, &mut ws))
            .unwrap()
            .unwrap();
        let tolerance = if choice == AlgorithmChoice::Chunked {
            5e-3
        } else {
            1e-6
        };
        assert!(
            max_diff(&reference, &out) <= tolerance,
            "{choice:?} diff {}",
            max_diff(&reference, &out)
        );
    }
}

#[test]
fn extension_op_matches_reference() {
    let (cfg, w, x, mask) = sample();
    let reference = delta_layer_reference(&cfg, &w, &x, &mask).unwrap();
    let length = mask.len();
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;

    let op = GatedDeltaOp::from_config(&cfg);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let out = runtime
        .with_eager_session(|session| {
            let x_t = session.constant_from(Tensor::from_vec_col_major(
                vec![length, cfg.hidden],
                x.clone(),
            )?)?;
            let mask_t =
                session.constant_from(Tensor::from_vec_col_major(vec![length], mask.clone())?)?;
            let qkv = session.constant_from(Tensor::from_vec_col_major(
                vec![conv_channels, cfg.hidden],
                w.qkv.clone(),
            )?)?;
            let z = session.constant_from(Tensor::from_vec_col_major(
                vec![value_width, cfg.hidden],
                w.z.clone(),
            )?)?;
            let a = session.constant_from(Tensor::from_vec_col_major(
                vec![cfg.value_heads, cfg.hidden],
                w.a.clone(),
            )?)?;
            let b = session.constant_from(Tensor::from_vec_col_major(
                vec![cfg.value_heads, cfg.hidden],
                w.b.clone(),
            )?)?;
            let conv = session.constant_from(Tensor::from_vec_col_major(
                vec![conv_channels, cfg.conv_taps],
                w.conv.clone(),
            )?)?;
            let a_decay = session.constant_from(Tensor::from_vec_col_major(
                vec![cfg.value_heads],
                w.a_decay.clone(),
            )?)?;
            let dt_bias = session.constant_from(Tensor::from_vec_col_major(
                vec![cfg.value_heads],
                w.dt_bias.clone(),
            )?)?;
            let norm = session.constant_from(Tensor::from_vec_col_major(
                vec![cfg.value_dim],
                w.norm.clone(),
            )?)?;
            let out_proj = session.constant_from(Tensor::from_vec_col_major(
                vec![cfg.hidden, value_width],
                w.out_proj.clone(),
            )?)?;
            session.gated_delta(
                op,
                &[
                    &x_t, &mask_t, &qkv, &z, &a, &b, &conv, &a_decay, &dt_bias, &norm, &out_proj,
                ],
            )
        })
        .unwrap()
        .unwrap();
    let values = out.value().unwrap();
    let col = values.as_slice::<f32>().unwrap();
    // The op returns `(length, hidden)`; the reference is row-major
    // `(hidden, length)`.
    let mut got = vec![0.0f32; cfg.hidden * length];
    for h in 0..cfg.hidden {
        for l in 0..length {
            got[h * length + l] = col[l + h * length];
        }
    }
    assert!(max_diff(&reference, &got) < 1e-6);
}

#[test]
fn op_descriptor_round_trips_and_compares() {
    let cfg = config();
    let op = GatedDeltaOp::from_config(&cfg);
    assert_eq!(op.config(), cfg);
    let other = GatedDeltaOp::from_config(&config());
    assert_eq!(op, other);
}
