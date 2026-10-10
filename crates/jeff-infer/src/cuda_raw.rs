//! Jeff's single-stream CUDA forward (`cuda` feature).
//!
//! The whole Qwen3.5 hybrid forward runs in one raw scope
//! ([`tenferro_ext::raw_exec`]): cuBLAS products for every projection, the
//! Gated DeltaNet core of [`tenferro_gated_delta::raw`] (recurrent or chunked
//! WY scan), and the NVRTC kernels of `cuda/jeff_raw.cu` (centered RMSNorm,
//! SiLU gate, per-head norm + partial RoPE, gated causal attention) are
//! enqueued on tenferro's stream. The token ids and mask are uploaded once and
//! the readout logits downloaded once; nothing else synchronizes.
//!
//! Following QwenDecisionCore.jl's CUDA path, projections are packed (DeltaNet
//! `[qkv | z | a | b]`, attention `[q | gate | k | v]` with the GQA key/value
//! heads de-duplicated, MLP `[gate | up]`), residual adds are folded into the
//! output GEMMs (`beta = 1`), and when the last layer is a full-attention
//! layer only the last position's query, output projection and MLP run (the
//! readout reads one column). Weights upload once into per-layer arenas
//! (the embedding table as its own buffer); activations reuse one scratch
//! buffer sized for the longest request seen.
//!
//! Layouts are feature-major `(rows, T)`. The host weights are row-major
//! `(in, out)`, i.e. column-major `(out, in)`, so every product is `N, N`.

use decision_core::DecisionError;
use tenferro_ad::EagerSession;
use tenferro_ext::raw_exec::{
    ArenaBuilder, Arg, DevPtr, DeviceBuffer, Op, RawExec, align, at, with_raw_exec,
};
use tenferro_gated_delta::raw::{
    DELTA_RAW_KERNELS, DELTA_RAW_SOURCE, DeltaRawConfig, DeltaRawOperands, DeltaRawScan,
    delta_raw_mix,
};

use crate::model::{AttentionWeights, FullAttentionWeights, JeffConfig, JeffWeights};

const SOURCE: &str = include_str!("cuda/jeff_raw.cu");
const KERNELS: &[&str] = &[
    "jeff_embed",
    "jeff_rms",
    "jeff_silu_mul",
    "jeff_attn_prep",
    "jeff_attention",
    "jeff_attention_tile",
];
/// Queries per block of `jeff_attention_tile`.
const QUERY_TILE: usize = 8;
/// Static shared memory budget per block (bytes).
const SHARED_LIMIT: usize = 48 * 1024;
const MAX_HIDDEN: usize = 32 * 32;
/// Longest sequence the attention kernel accepts (scores in shared memory).
pub const MAX_TOKENS: usize = 8192;

fn invalid(field: &'static str, message: &'static str) -> tenferro_ad::Error {
    let error = DecisionError::invalid_field(field, message);
    tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        "jeff-infer::cuda_raw",
        "input",
        error.to_string(),
    ))
}

fn unsupported(message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::unsupported("jeff-infer::cuda_raw", message.into())
}

#[derive(Clone, Copy, Debug)]
struct Loc {
    arena: usize,
    offset: usize,
}

/// A weight matrix `W` `(out, inp)` of `y = W x`, stored column-major as
/// `(out, inp)` (`N` products) or, with [`Weights::tn`], as `(inp, out)`
/// (`T` products, QwenDecisionCore's orientation).
#[derive(Clone, Copy, Debug)]
struct Wm {
    loc: Loc,
    out: usize,
    inp: usize,
}

#[derive(Debug)]
enum Mix {
    Delta {
        /// Packed `[qkv | z | a | b]` `(rows, hidden)`.
        proj: Wm,
        rows: usize,
        conv: Loc,
        a_decay: Loc,
        dt_bias: Loc,
        norm: Loc,
        /// Output projection `(hidden, value_width)`.
        out: Wm,
        cfg: DeltaRawConfig,
        eps: f32,
    },
    Full {
        /// Packed `[q | gate | k | v]` `(rows, hidden)`.
        proj: Wm,
        rows: usize,
        q_norm: Loc,
        k_norm: Loc,
        /// Output projection `(hidden, heads * hd)`.
        out: Wm,
        heads: usize,
        kv_heads: usize,
        hd: usize,
        rotary: usize,
        theta: f32,
    },
}

#[derive(Debug)]
struct Layer {
    input_norm: Loc,
    post_norm: Loc,
    mix: Mix,
    /// Packed `[gate | up]` `(2I, hidden)`.
    gate_up: Wm,
    /// Down projection `(hidden, I)`.
    down: Wm,
}

#[derive(Debug)]
struct Weights {
    embedding: DeviceBuffer,
    arenas: Vec<DeviceBuffer>,
    layers: Vec<Layer>,
    final_norm: Loc,
    readout: Wm,
    options: usize,
    vocab: usize,
    /// Weights stored `(inp, out)` for `T, N` products.
    tn: bool,
}

/// Device state of the raw Jeff forward: resident weights, RoPE tables and
/// the reused activation scratch. Create one per engine and runtime; a clone
/// starts empty (device buffers are not shared) and uploads on first use.
#[derive(Debug, Default)]
pub struct JeffCudaRaw {
    weights: Option<Weights>,
    rope: Option<(DeviceBuffer, usize)>,
    scratch: Option<DeviceBuffer>,
    scan: DeltaRawScan,
}

impl Clone for JeffCudaRaw {
    fn clone(&self) -> Self {
        Self {
            scan: self.scan,
            ..Self::default()
        }
    }
}

impl JeffCudaRaw {
    /// Empty state; weights upload on the first forward.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the DeltaNet scan (default [`DeltaRawScan::Auto`]).
    pub fn with_scan(mut self, scan: DeltaRawScan) -> Self {
        self.scan = scan;
        self
    }

    /// Whether the raw path supports this model.
    pub fn supports(cfg: &JeffConfig, weights: &JeffWeights) -> bool {
        cfg.hidden <= MAX_HIDDEN
            && weights.layers.iter().all(|layer| match &layer.attention {
                AttentionWeights::Delta { config, .. } => delta_config(config).supported(),
                AttentionWeights::Full(full) => {
                    cfg.head_dim % 32 == 0
                        && cfg.head_dim <= 1024
                        && full.rotary_dim % 2 == 0
                        && full.rotary_dim <= cfg.head_dim
                }
            })
    }
}

fn delta_config(config: &tenferro_gated_delta::GatedDeltaConfig) -> DeltaRawConfig {
    DeltaRawConfig {
        key_heads: config.key_heads,
        value_heads: config.value_heads,
        key_dim: config.key_dim,
        value_dim: config.value_dim,
        conv_taps: config.conv_taps,
    }
}

/// The distinct key/value heads of GQA weights expanded to `heads` heads:
/// returns `(kv_heads, k, v)` row-major `(hidden, kv_heads * hd)`.
fn dedupe_kv(
    full: &FullAttentionWeights,
    hidden: usize,
    heads: usize,
    hd: usize,
) -> (usize, Vec<f32>, Vec<f32>) {
    let width = heads * hd;
    let same = |w: &[f32], a: usize, b: usize| {
        (0..hidden).all(|i| {
            w[i * width + a * hd..i * width + (a + 1) * hd]
                == w[i * width + b * hd..i * width + (b + 1) * hd]
        })
    };
    let group = (1..=heads)
        .rev()
        .filter(|g| heads % g == 0)
        .find(|&g| {
            (0..heads).all(|h| same(&full.k, h, (h / g) * g) && same(&full.v, h, (h / g) * g))
        })
        .unwrap_or(1);
    let kv_heads = heads / group;
    let pick = |w: &[f32]| {
        let mut out = Vec::with_capacity(hidden * kv_heads * hd);
        for i in 0..hidden {
            for kv in 0..kv_heads {
                let h = kv * group;
                out.extend_from_slice(&w[i * width + h * hd..i * width + (h + 1) * hd]);
            }
        }
        out
    };
    (kv_heads, pick(&full.k), pick(&full.v))
}

/// Whether weights are stored `(inp, out)` for `T, N` products
/// (`TENFERRO_DECISION_JEFF_GEMM=tn`) instead of `(out, inp)` (`nn`).
fn transposed_layout() -> bool {
    !matches!(
        std::env::var("TENFERRO_DECISION_JEFF_GEMM").as_deref(),
        Ok("nn")
    )
}

/// Stage `parts` (each row-major `(inp, out_k)`) stacked along the output
/// axis as one [`Wm`] in the arena.
fn push_wm(
    arena: &mut ArenaBuilder,
    index: usize,
    parts: &[(&[f32], usize)],
    inp: usize,
    tn: bool,
) -> Wm {
    let out: usize = parts.iter().map(|(_, o)| o).sum();
    let offset = if !tn {
        arena.push_stacked(parts, inp)
    } else if parts.len() == 1 {
        arena.push_transposed(parts[0].0, out, inp)
    } else {
        let mut staged = ArenaBuilder::new();
        staged.push_stacked(parts, inp);
        let values = staged.into_vec();
        arena.push_transposed(&values[..out * inp], out, inp)
    };
    Wm {
        loc: Loc {
            arena: index,
            offset,
        },
        out,
        inp,
    }
}

fn upload_weights(
    exec: &RawExec<'_, '_>,
    cfg: &JeffConfig,
    weights: &JeffWeights,
) -> tenferro_tensor::Result<Weights> {
    let tn = transposed_layout();
    let hidden = cfg.hidden;
    let inter = cfg.intermediate;
    let embedding = exec.alloc(weights.embedding.len())?;
    // SAFETY: the buffer holds the table; pageable uploads are staged.
    unsafe { exec.upload(exec.ptr(&embedding)?, &weights.embedding)? };
    let mut arenas = Vec::new();
    let mut top = ArenaBuilder::new();
    let final_norm = Loc {
        arena: 0,
        offset: top.push(&weights.final_norm),
    };
    let readout = push_wm(
        &mut top,
        0,
        &[(&weights.readout[..], weights.options)],
        hidden,
        tn,
    );
    arenas.push(top.upload(exec)?);
    let mut layers = Vec::with_capacity(weights.layers.len());
    for layer in &weights.layers {
        let index = arenas.len();
        let mut arena = ArenaBuilder::new();
        let loc = |offset: usize| Loc {
            arena: index,
            offset,
        };
        let input_norm = loc(arena.push(&layer.input_norm));
        let post_norm = loc(arena.push(&layer.post_norm));
        let mix = match &layer.attention {
            AttentionWeights::Delta { weights: w, config } => {
                let dc = delta_config(config);
                let parts = [
                    (&w.qkv[..], dc.conv_channels()),
                    (&w.z[..], dc.value_width()),
                    (&w.a[..], dc.value_heads),
                    (&w.b[..], dc.value_heads),
                ];
                let rows = parts.iter().map(|(_, out)| out).sum();
                Mix::Delta {
                    proj: push_wm(&mut arena, index, &parts, hidden, tn),
                    rows,
                    conv: loc(arena.push(&w.conv)),
                    a_decay: loc(arena.push(&w.a_decay)),
                    dt_bias: loc(arena.push(&w.dt_bias)),
                    norm: loc(arena.push(&w.norm)),
                    out: push_wm(
                        &mut arena,
                        index,
                        &[(&w.out_proj[..], hidden)],
                        dc.value_width(),
                        tn,
                    ),
                    cfg: dc,
                    eps: config.eps,
                }
            }
            AttentionWeights::Full(full) => {
                let (heads, hd) = (cfg.heads, cfg.head_dim);
                let width = heads * hd;
                let (kv_heads, k, v) = dedupe_kv(full, hidden, heads, hd);
                let parts = [
                    (&full.q[..], width),
                    (&full.gate[..], width),
                    (&k[..], kv_heads * hd),
                    (&v[..], kv_heads * hd),
                ];
                Mix::Full {
                    proj: push_wm(&mut arena, index, &parts, hidden, tn),
                    rows: 2 * width + 2 * kv_heads * hd,
                    q_norm: loc(arena.push(&full.q_norm)),
                    k_norm: loc(arena.push(&full.k_norm)),
                    out: push_wm(&mut arena, index, &[(&full.o[..], hidden)], width, tn),
                    heads,
                    kv_heads,
                    hd,
                    rotary: full.rotary_dim,
                    theta: full.rope_theta,
                }
            }
        };
        let gate_up = push_wm(
            &mut arena,
            index,
            &[(&layer.mlp.gate[..], inter), (&layer.mlp.up[..], inter)],
            hidden,
            tn,
        );
        let down = push_wm(
            &mut arena,
            index,
            &[(&layer.mlp.down[..], hidden)],
            inter,
            tn,
        );
        arenas.push(arena.upload(exec)?);
        layers.push(Layer {
            input_norm,
            post_norm,
            mix,
            gate_up,
            down,
        });
    }
    Ok(Weights {
        embedding,
        arenas,
        layers,
        final_norm,
        readout,
        options: weights.options,
        vocab: weights.vocab,
        tn,
    })
}

/// The distinct (rotary, theta bits) of the full-attention layers, in model
/// order: the entries of [`rope_tables`].
fn rope_keys(layers: &[Layer]) -> Vec<(usize, u32)> {
    let mut keys: Vec<(usize, u32)> = Vec::new();
    for layer in layers {
        if let Mix::Full { rotary, theta, .. } = layer.mix {
            if !keys.contains(&(rotary, theta.to_bits())) {
                keys.push((rotary, theta.to_bits()));
            }
        }
    }
    keys
}

/// RoPE tables `(rotary / 2, positions)` for every [`rope_keys`] entry,
/// concatenated `[cos, sin]` per entry, computed in f64.
fn rope_tables(layers: &[Layer], positions: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for (rotary, theta) in rope_keys(layers) {
        let half = rotary / 2;
        let theta = f32::from_bits(theta) as f64;
        let mut cos = vec![0.0f32; half * positions];
        let mut sin = vec![0.0f32; half * positions];
        for pos in 0..positions {
            for i in 0..half {
                let angle = pos as f64 / theta.powf(2.0 * i as f64 / rotary as f64);
                cos[i + half * pos] = angle.cos() as f32;
                sin[i + half * pos] = angle.sin() as f32;
            }
        }
        out.extend_from_slice(&cos);
        out.extend_from_slice(&sin);
    }
    out
}

struct Layout {
    x: usize,
    n: usize,
    proj: usize,
    merged: usize,
    act: usize,
    qh: usize,
    kh: usize,
    delta: usize,
    logits: usize,
    ints: usize,
    total: usize,
}

impl Layout {
    fn new(cfg: &JeffConfig, weights: &Weights, tokens: usize) -> Self {
        let hidden = cfg.hidden;
        let mut proj_rows = 2 * cfg.intermediate;
        let mut merged_rows = 0;
        let mut q_heads = 0;
        let mut k_heads = 0;
        let mut hd = 0;
        let mut delta = 0;
        for layer in &weights.layers {
            match &layer.mix {
                Mix::Delta { rows, cfg: dc, .. } => {
                    proj_rows = proj_rows.max(*rows);
                    merged_rows = merged_rows.max(dc.value_width());
                    delta = delta.max(dc.scratch_floats(tokens));
                }
                Mix::Full {
                    rows,
                    heads,
                    kv_heads,
                    hd: head_dim,
                    ..
                } => {
                    proj_rows = proj_rows.max(*rows);
                    merged_rows = merged_rows.max(heads * head_dim);
                    q_heads = q_heads.max(*heads);
                    k_heads = k_heads.max(*kv_heads);
                    hd = hd.max(*head_dim);
                }
            }
        }
        let mut next = 0usize;
        let mut take = |floats: usize| {
            let at = next;
            next += align(floats);
            at
        };
        let x = take(hidden * tokens);
        let n = take(hidden * tokens);
        let proj = take(proj_rows * tokens);
        let merged = take(merged_rows * tokens);
        let act = take(cfg.intermediate * tokens);
        let qh = take(hd * q_heads * tokens);
        let kh = take(hd * k_heads * tokens);
        let delta = take(delta);
        let logits = take(weights.options);
        let ints = take(2 * tokens);
        Self {
            x,
            n,
            proj,
            merged,
            act,
            qh,
            kh,
            delta,
            logits,
            ints,
            total: next,
        }
    }
}

/// The raw Jeff forward: readout logits `(options,)` for one row of `ids`
/// with its `mask` (1 = active). Same contract as
/// [`crate::model::forward_tenferro_device`].
pub fn forward_cuda_raw(
    state: &mut JeffCudaRaw,
    session: &mut EagerSession<'_>,
    cfg: &JeffConfig,
    weights: &JeffWeights,
    ids: &[i64],
    mask: &[f32],
) -> tenferro_ad::Result<Vec<f32>> {
    weights.validate(cfg).map_err(|error| {
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            "jeff-infer::cuda_raw",
            "weights",
            error.to_string(),
        ))
    })?;
    let tokens = ids.len();
    if tokens == 0 {
        return Err(invalid("jeff.ids", "at least one token is required"));
    }
    if mask.len() != tokens {
        return Err(invalid(
            "jeff.mask",
            "mask length must match the token length",
        ));
    }
    if ids.iter().any(|&id| id < 0 || id as usize >= weights.vocab) {
        return Err(invalid("jeff.ids", "token id is outside the vocabulary"));
    }
    if !JeffCudaRaw::supports(cfg, weights) || tokens > MAX_TOKENS {
        return Err(tenferro_ad::Error::TensorRuntime(unsupported(
            "the raw CUDA forward does not support this model or length",
        )));
    }
    let mut words: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
    words.extend(mask.iter().map(|m| m.to_bits()));
    let mut logits = vec![0.0f32; weights.options];
    let scan = state.scan;
    with_raw_exec(session, "jeff.forward_cuda_raw", |exec| {
        if state.weights.is_none() {
            state.weights = Some(upload_weights(exec, cfg, weights)?);
        }
        let w = state.weights.as_ref().expect("uploaded above");
        if w.vocab != weights.vocab || w.options != weights.options {
            return Err(unsupported("weights changed since the first forward"));
        }
        if state
            .rope
            .as_ref()
            .is_none_or(|(_, positions)| *positions < tokens)
        {
            let positions = tokens.next_power_of_two().max(256);
            let table = rope_tables(&w.layers, positions);
            let buffer = exec.alloc(table.len())?;
            // SAFETY: the buffer holds the table.
            unsafe { exec.upload(exec.ptr(&buffer)?, &table)? };
            exec.synchronize()?;
            state.rope = Some((buffer, positions));
        }
        let layout = Layout::new(cfg, w, tokens);
        if state
            .scratch
            .as_ref()
            .is_none_or(|scratch| scratch.len() < layout.total)
        {
            state.scratch = None;
            state.scratch = Some(exec.alloc(layout.total)?);
        }
        let module = exec.module(SOURCE, KERNELS)?;
        let delta_module = exec.module(DELTA_RAW_SOURCE, DELTA_RAW_KERNELS)?;
        let bases = w
            .arenas
            .iter()
            .map(|arena| exec.ptr(arena))
            .collect::<tenferro_tensor::Result<Vec<DevPtr>>>()?;
        let embedding = exec.ptr(&w.embedding)?;
        let (rope_buffer, positions) = state.rope.as_ref().expect("allocated above");
        let positions = *positions;
        let rope = exec.ptr(rope_buffer)?;
        let rope_keys = rope_keys(&w.layers);
        let rope_for = |rotary: usize, theta: f32| {
            let mut offset = 0usize;
            for &(r, t) in &rope_keys {
                let size = (r / 2) * positions;
                if r == rotary && t == theta.to_bits() {
                    return (at(rope, offset), at(rope, offset + size));
                }
                offset += 2 * size;
            }
            unreachable!("RoPE table built for every full-attention layer")
        };
        let scratch = exec.ptr(state.scratch.as_ref().expect("allocated above"))?;
        let p = |loc: Loc| at(bases[loc.arena], loc.offset);
        let s = |offset: usize| at(scratch, offset);
        let f_embed = module.function("jeff_embed")?;
        let f_rms = module.function("jeff_rms")?;
        let f_silu = module.function("jeff_silu_mul")?;
        let f_prep = module.function("jeff_attn_prep")?;
        let f_attn = module.function("jeff_attention")?;
        let f_attn_tile = module.function("jeff_attention_tile")?;
        let hidden = cfg.hidden;
        let inter = cfg.intermediate;
        let x = s(layout.x);
        let n = s(layout.n);
        let proj = s(layout.proj);
        let merged = s(layout.merged);
        let act = s(layout.act);
        let qh = s(layout.qh);
        let kh = s(layout.kh);
        let delta_scratch = s(layout.delta);
        let logits_ptr = s(layout.logits);
        let ids_ptr = s(layout.ints);
        let mask_ptr = at(ids_ptr, tokens);
        let i = |v: usize| Arg::I(v as i32);
        let last = tokens - 1;

        // SAFETY (block): every address lies in a buffer sized above
        // (weights by construction, scratch by `layout`), the arguments match
        // the kernels, and the scope synchronizes before buffers can drop.
        unsafe {
            exec.upload(ids_ptr, &words)?;
            let rms = |y: DevPtr, xin: DevPtr, weight: Loc, mask: DevPtr, cols: usize| {
                exec.launch(
                    f_rms,
                    ((cols as u32).div_ceil(8), 1, 1),
                    256,
                    0,
                    &[
                        Arg::P(y),
                        Arg::P(xin),
                        Arg::P(p(weight)),
                        Arg::P(mask),
                        i(hidden),
                        i(cols),
                        Arg::F(cfg.eps),
                    ],
                )
            };
            // y (m, cols) = W[row0 .. row0 + m, :] xin (+ beta y).
            let lin = |wm: Wm,
                       row0: usize,
                       m: usize,
                       cols: usize,
                       xin: DevPtr,
                       ld_x: usize,
                       beta: f32,
                       y: DevPtr,
                       ld_y: usize| {
                if w.tn {
                    let a = at(p(wm.loc), row0 * wm.inp);
                    exec.gemm(
                        Op::T,
                        Op::N,
                        m,
                        cols,
                        wm.inp,
                        1.0,
                        a,
                        wm.inp,
                        xin,
                        ld_x,
                        beta,
                        y,
                        ld_y,
                    )
                } else {
                    let a = at(p(wm.loc), row0);
                    exec.gemm(
                        Op::N,
                        Op::N,
                        m,
                        cols,
                        wm.inp,
                        1.0,
                        a,
                        wm.out,
                        xin,
                        ld_x,
                        beta,
                        y,
                        ld_y,
                    )
                }
            };
            let mlp = |layer: &Layer, col0: usize, cols: usize| -> tenferro_tensor::Result<()> {
                let xc = at(x, hidden * col0);
                rms(n, xc, layer.post_norm, 0, cols)?;
                lin(
                    layer.gate_up,
                    0,
                    2 * inter,
                    cols,
                    n,
                    hidden,
                    0.0,
                    proj,
                    2 * inter,
                )?;
                exec.launch(
                    f_silu,
                    (((inter * cols) as u32).div_ceil(256), 1, 1),
                    256,
                    0,
                    &[Arg::P(act), Arg::P(proj), i(inter), i(inter * cols)],
                )?;
                lin(layer.down, 0, hidden, cols, act, inter, 1.0, xc, hidden)
            };

            exec.launch(
                f_embed,
                (((hidden * tokens) as u32).div_ceil(256), 1, 1),
                256,
                0,
                &[
                    Arg::P(x),
                    Arg::P(embedding),
                    Arg::P(ids_ptr),
                    i(hidden),
                    Arg::L(w.vocab as i64),
                    i(hidden * tokens),
                ],
            )?;
            let count = w.layers.len();
            for (index, layer) in w.layers.iter().enumerate() {
                let is_last = index + 1 == count;
                match &layer.mix {
                    Mix::Delta {
                        proj: wp,
                        rows,
                        conv,
                        a_decay,
                        dt_bias,
                        norm,
                        out,
                        cfg: dc,
                        eps,
                    } => {
                        rms(n, x, layer.input_norm, mask_ptr, tokens)?;
                        lin(*wp, 0, *rows, tokens, n, hidden, 0.0, proj, *rows)?;
                        delta_raw_mix(
                            exec,
                            &delta_module,
                            dc,
                            &DeltaRawOperands {
                                proj,
                                ld_proj: *rows,
                                conv: p(*conv),
                                a_decay: p(*a_decay),
                                dt_bias: p(*dt_bias),
                                norm: p(*norm),
                                eps: *eps,
                                gated: merged,
                                scratch: delta_scratch,
                            },
                            tokens,
                            scan,
                        )?;
                        let vw = dc.value_width();
                        lin(*out, 0, hidden, tokens, merged, vw, 1.0, x, hidden)?;
                        mlp(layer, 0, tokens)?;
                    }
                    Mix::Full {
                        proj: wp,
                        rows,
                        q_norm,
                        k_norm,
                        out,
                        heads,
                        kv_heads,
                        hd,
                        rotary,
                        theta,
                    } => {
                        let (heads, kv_heads, hd, rows) = (*heads, *kv_heads, *hd, *rows);
                        let width = heads * hd;
                        let k_row = 2 * width;
                        let v_row = k_row + kv_heads * hd;
                        let (cos, sin) = rope_for(*rotary, *theta);
                        // Queries: every position, or only the last one when
                        // nothing after this layer reads the others.
                        let (col0, nq) = if is_last { (last, 1) } else { (0, tokens) };
                        rms(n, x, layer.input_norm, 0, tokens)?;
                        if is_last {
                            lin(
                                *wp,
                                k_row,
                                2 * kv_heads * hd,
                                tokens,
                                n,
                                hidden,
                                0.0,
                                at(proj, k_row),
                                rows,
                            )?;
                            lin(
                                *wp,
                                0,
                                k_row,
                                1,
                                at(n, hidden * last),
                                hidden,
                                0.0,
                                at(proj, rows * last),
                                rows,
                            )?;
                        } else {
                            lin(*wp, 0, rows, tokens, n, hidden, 0.0, proj, rows)?;
                        }
                        let block = hd.next_multiple_of(32) as u32;
                        let shared = (4 * hd) as u32;
                        exec.launch(
                            f_prep,
                            (nq as u32, heads as u32, 1),
                            block,
                            shared,
                            &[
                                Arg::P(qh),
                                Arg::P(proj),
                                i(rows),
                                i(0),
                                Arg::P(p(*q_norm)),
                                Arg::P(cos),
                                Arg::P(sin),
                                i(hd),
                                i(*rotary),
                                i(col0),
                                Arg::F(cfg.eps),
                            ],
                        )?;
                        exec.launch(
                            f_prep,
                            (tokens as u32, kv_heads as u32, 1),
                            block,
                            shared,
                            &[
                                Arg::P(kh),
                                Arg::P(proj),
                                i(rows),
                                i(k_row),
                                Arg::P(p(*k_norm)),
                                Arg::P(cos),
                                Arg::P(sin),
                                i(hd),
                                i(*rotary),
                                i(0),
                                Arg::F(cfg.eps),
                            ],
                        )?;
                        let mut args = vec![
                            Arg::P(merged),
                            Arg::P(qh),
                            Arg::P(kh),
                            Arg::P(proj),
                            i(rows),
                            i(v_row),
                            i(width),
                            Arg::P(mask_ptr),
                            i(hd),
                            i(heads),
                            i(kv_heads),
                            i(tokens),
                            i(col0),
                        ];
                        let tile_shared = 4 * QUERY_TILE * (hd + tokens);
                        if nq > 1 && tile_shared <= SHARED_LIMIT - 256 {
                            // Several queries per block share each K/V read.
                            args.extend([i(nq), Arg::F(1.0 / (hd as f32).sqrt())]);
                            exec.launch(
                                f_attn_tile,
                                ((nq as u32).div_ceil(QUERY_TILE as u32), heads as u32, 1),
                                256,
                                tile_shared as u32,
                                &args,
                            )?;
                        } else {
                            args.push(Arg::F(1.0 / (hd as f32).sqrt()));
                            exec.launch(
                                f_attn,
                                (nq as u32, heads as u32, 1),
                                256,
                                (4 * (hd + tokens)) as u32,
                                &args,
                            )?;
                        }
                        lin(
                            *out,
                            0,
                            hidden,
                            nq,
                            merged,
                            width,
                            1.0,
                            at(x, hidden * col0),
                            hidden,
                        )?;
                        mlp(layer, col0, nq)?;
                    }
                }
            }
            // Final norm on the last position, then the readout.
            rms(n, at(x, hidden * last), w.final_norm, 0, 1)?;
            lin(
                w.readout, 0, w.options, 1, n, hidden, 0.0, logits_ptr, w.options,
            )?;
            exec.download(&mut logits, logits_ptr)?;
        }
        Ok(())
    })?;
    Ok(logits)
}
