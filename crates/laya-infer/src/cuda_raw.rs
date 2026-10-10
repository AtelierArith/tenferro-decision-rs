//! Laya's single-stream CUDA forward (`cuda` feature).
//!
//! The whole decision-model forward runs in one raw scope
//! ([`tenferro_ext::raw_exec`]): cuBLAS products for every linear layer and
//! the NVRTC kernels of `cuda/laya_raw.cu` (LayerNorm, GeGLU, fused
//! flash-style attention with RoPE and local-window/padding tile skipping,
//! marker pooling) are enqueued on tenferro's stream, with one upload of the
//! integer inputs and one download of the logits at the end. Weights are
//! uploaded once (first call) into per-layer arenas; activations live in a
//! reused scratch buffer sized for the largest batch seen, so a warm forward
//! allocates no device memory.
//!
//! Layouts follow the host reference: activations are feature-major `(d, T)`
//! with token column `t = l + L * b`; linear weights are stored transposed,
//! column-major `(out, in)`, so every product is a plain `N, N` GEMM.
//!
//! [`LayaCudaRaw::supports`] gates the path (head_dim 64, hidden <= 1024);
//! engines fall back to the tenferro-native device forward otherwise.

use decision_core::DecisionError;
use tenferro_ad::EagerSession;
use tenferro_ext::raw_exec::{
    ArenaBuilder, Arg, DevPtr, DeviceBuffer, Op, RawExec, align, at, with_raw_exec,
};

use crate::config::{AgentConfig, EncoderConfig, LayerKind};
use crate::model::{LayaWeights, LayerNormWeights, LinearWeights};

const SOURCE: &str = include_str!("cuda/laya_raw.cu");
const KERNELS: &[&str] = &[
    "laya_embed",
    "laya_layernorm",
    "laya_geglu",
    "laya_bias_act",
    "laya_add_type",
    "laya_gather_markers",
    "laya_pool",
    "laya_attention",
];
const HEAD_DIM: usize = 64;
const MAX_HIDDEN: usize = 32 * 32;

fn invalid(field: &'static str, message: &'static str) -> tenferro_ad::Error {
    let error = DecisionError::invalid_field(field, message);
    tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        "laya-infer::cuda_raw",
        "input",
        error.to_string(),
    ))
}

fn unsupported(message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::unsupported("laya-infer::cuda_raw", message.into())
}

/// A location in one of the weight arenas.
#[derive(Clone, Copy, Debug)]
struct Loc {
    arena: usize,
    offset: usize,
}

#[derive(Clone, Copy, Debug)]
struct Norm {
    w: Loc,
    b: Option<Loc>,
}

/// A linear layer stored column-major `(out, in)`.
#[derive(Clone, Copy, Debug)]
struct Lin {
    w: Loc,
    b: Option<Loc>,
    inp: usize,
    out: usize,
}

#[derive(Debug)]
struct EncLayer {
    attn_norm: Option<Norm>,
    wqkv: Lin,
    wo: Lin,
    mlp_norm: Norm,
    wi: Lin,
    wo_mlp: Lin,
    heads: usize,
    /// Local attention half-window, or -1 for global attention.
    window: i32,
    /// Index of the RoPE table pair (0: global base, 1: local base).
    rope: usize,
}

#[derive(Debug)]
struct HeadLayer {
    norm1: Norm,
    in_proj: Lin,
    out_proj: Lin,
    norm2: Norm,
    linear1: Lin,
    linear2: Lin,
    heads: usize,
}

#[derive(Debug)]
struct Weights {
    arenas: Vec<DeviceBuffer>,
    embed: Loc,
    vocab: usize,
    embed_norm: Norm,
    layers: Vec<EncLayer>,
    final_norm: Norm,
    head: Vec<HeadLayer>,
    type_emb: Loc,
    scorer_norm: Norm,
    scorer1: Lin,
    scorer2: Lin,
    act1: Lin,
    act2: Lin,
    rope_bases: [f64; 2],
}

/// Device state of the raw Laya forward: resident weights, RoPE tables and
/// the reused activation scratch. Create one per engine and runtime.
#[derive(Debug, Default)]
pub struct LayaCudaRaw {
    weights: Option<Weights>,
    rope: Option<(DeviceBuffer, usize)>,
    scratch: Option<DeviceBuffer>,
}

impl LayaCudaRaw {
    /// Empty state; weights upload on the first forward.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the raw path supports this model (head_dim 64 attention
    /// everywhere and hidden <= 1024).
    pub fn supports(encoder: &EncoderConfig, weights: &LayaWeights) -> bool {
        let d = encoder.hidden_size;
        d <= MAX_HIDDEN
            && encoder.head_dim() == HEAD_DIM
            && weights
                .head
                .iter()
                .all(|layer| layer.num_heads * HEAD_DIM == d)
    }
}

fn push_norm(arena: &mut ArenaBuilder, index: usize, ln: &LayerNormWeights) -> Norm {
    Norm {
        w: Loc {
            arena: index,
            offset: arena.push(&ln.weight),
        },
        b: ln.bias.as_ref().map(|bias| Loc {
            arena: index,
            offset: arena.push(bias),
        }),
    }
}

/// `linear.weight` is column-major `(inp, out)`; store `(out, inp)`.
fn push_lin(
    arena: &mut ArenaBuilder,
    index: usize,
    linear: &LinearWeights,
    inp: usize,
    out: usize,
) -> Lin {
    Lin {
        w: Loc {
            arena: index,
            offset: arena.push_transposed(&linear.weight, inp, out),
        },
        b: linear.bias.as_ref().map(|bias| Loc {
            arena: index,
            offset: arena.push(bias),
        }),
        inp,
        out,
    }
}

fn upload_weights(
    exec: &RawExec<'_, '_>,
    encoder: &EncoderConfig,
    weights: &LayaWeights,
) -> tenferro_tensor::Result<Weights> {
    let d = encoder.hidden_size;
    let inter = encoder.intermediate_size;
    let enc = &weights.encoder;
    let mut arenas = Vec::new();

    // Arena 0: embeddings and the small top-level tensors.
    let mut top = ArenaBuilder::new();
    let embed = Loc {
        arena: 0,
        offset: top.push(&enc.tok_embeddings),
    };
    let vocab = enc.tok_embeddings.len() / d;
    let embed_norm = push_norm(&mut top, 0, &enc.embed_norm);
    let final_norm = push_norm(&mut top, 0, &enc.final_norm);
    let type_emb = Loc {
        arena: 0,
        offset: top.push(&weights.type_emb),
    };
    let scorer_norm = push_norm(&mut top, 0, &weights.scorer_norm);
    let scorer1 = push_lin(&mut top, 0, &weights.scorer1, d, d);
    let scorer2 = push_lin(&mut top, 0, &weights.scorer2, d, 1);
    let action_hidden = weights.act1.weight.len() / (d + 4);
    let act1 = push_lin(&mut top, 0, &weights.act1, d + 4, action_hidden);
    let action_count = weights.act2.weight.len() / action_hidden;
    let act2 = push_lin(&mut top, 0, &weights.act2, action_hidden, action_count);
    arenas.push(top.upload(exec)?);

    let window = (encoder.local_attention / 2) as i32;
    let mut layers = Vec::with_capacity(enc.layers.len());
    for layer in &enc.layers {
        let index = arenas.len();
        let mut arena = ArenaBuilder::new();
        let attn_norm = layer
            .attn_norm
            .as_ref()
            .map(|ln| push_norm(&mut arena, index, ln));
        let wqkv = push_lin(&mut arena, index, &layer.wqkv, d, 3 * d);
        let wo = push_lin(&mut arena, index, &layer.wo, d, d);
        let mlp_norm = push_norm(&mut arena, index, &layer.mlp_norm);
        let wi = push_lin(&mut arena, index, &layer.wi, d, 2 * inter);
        let wo_mlp = push_lin(&mut arena, index, &layer.wo_mlp, inter, d);
        arenas.push(arena.upload(exec)?);
        let (window, rope) = match layer.kind {
            LayerKind::FullAttention => (-1, 0),
            LayerKind::SlidingAttention => (window, 1),
        };
        layers.push(EncLayer {
            attn_norm,
            wqkv,
            wo,
            mlp_norm,
            wi,
            wo_mlp,
            heads: layer.num_heads,
            window,
            rope,
        });
    }

    let mut head = Vec::with_capacity(weights.head.len());
    for layer in &weights.head {
        let index = arenas.len();
        let ff = layer.linear1.weight.len() / d;
        let mut arena = ArenaBuilder::new();
        let norm1 = push_norm(&mut arena, index, &layer.norm1);
        let in_proj = push_lin(&mut arena, index, &layer.in_proj, d, 3 * d);
        let out_proj = push_lin(&mut arena, index, &layer.out_proj, d, d);
        let norm2 = push_norm(&mut arena, index, &layer.norm2);
        let linear1 = push_lin(&mut arena, index, &layer.linear1, d, ff);
        let linear2 = push_lin(&mut arena, index, &layer.linear2, ff, d);
        arenas.push(arena.upload(exec)?);
        head.push(HeadLayer {
            norm1,
            in_proj,
            out_proj,
            norm2,
            linear1,
            linear2,
            heads: layer.num_heads,
        });
    }

    Ok(Weights {
        arenas,
        embed,
        vocab,
        embed_norm,
        layers,
        final_norm,
        head,
        type_emb,
        scorer_norm,
        scorer1,
        scorer2,
        act1,
        act2,
        rope_bases: [
            encoder.rope_base(LayerKind::FullAttention),
            encoder.rope_base(LayerKind::SlidingAttention),
        ],
    })
}

/// RoPE tables `[cos_g, sin_g, cos_l, sin_l]`, each `(32, positions)`,
/// computed exactly as `rope_host` does.
fn rope_tables(bases: [f64; 2], positions: usize) -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let mut out = vec![0.0f32; 4 * half * positions];
    for (which, base) in bases.iter().enumerate() {
        let log_base = (*base as f32).log2();
        let (cos, rest) = out[2 * which * half * positions..].split_at_mut(half * positions);
        let sin = &mut rest[..half * positions];
        for pos in 0..positions {
            for i in 0..half {
                let theta = pos as f32 * (-(i as f32) / (half as f32) * log_base).exp2();
                let (s, c) = theta.sin_cos();
                cos[i + half * pos] = c;
                sin[i + half * pos] = s;
            }
        }
    }
    out
}

/// Scratch layout (float offsets) for one forward.
struct Layout {
    x: usize,
    hb: usize,
    qkv: usize,
    att: usize,
    u: usize,
    g: usize,
    markers: usize,
    s1: usize,
    raw: usize,
    pooled: usize,
    a1: usize,
    out: usize,
    ints: usize,
    total: usize,
}

impl Layout {
    #[allow(clippy::too_many_arguments)]
    fn new(
        d: usize,
        inter: usize,
        ff: usize,
        tokens: usize,
        k_count: usize,
        batch: usize,
        action_hidden: usize,
        action_count: usize,
    ) -> Self {
        let mut next = 0usize;
        let mut take = |floats: usize| {
            let at = next;
            next += align(floats);
            at
        };
        let x = take(d * tokens);
        let hb = take(d * tokens);
        let qkv = take(3 * d * tokens);
        let att = take(d * tokens);
        let u = take((2 * inter).max(ff) * tokens);
        let g = take(inter * tokens);
        let markers = take(d * k_count * batch);
        let s1 = take(d * k_count * batch);
        let raw = take(k_count * batch);
        let pooled = take((d + 4) * batch);
        let a1 = take(action_hidden * batch);
        let out = take(k_count * batch + action_count * batch);
        let ints = take(2 * tokens + 2 * k_count * batch + batch);
        Self {
            x,
            hb,
            qkv,
            att,
            u,
            g,
            markers,
            s1,
            raw,
            pooled,
            a1,
            out,
            ints,
            total: next,
        }
    }
}

/// The raw Laya forward. Same contract and outputs as
/// [`crate::model::forward_tenferro_cached`]: masked marker logits `(K, B)`
/// and action logits `(n_actions, B)`, column-major.
#[allow(clippy::too_many_arguments)]
pub fn forward_cuda_raw(
    state: &mut LayaCudaRaw,
    session: &mut EagerSession<'_>,
    encoder: &EncoderConfig,
    agent: &AgentConfig,
    weights: &LayaWeights,
    ids: &[i64],
    mask: &[bool],
    marker_pos: &[i64],
    marker_mask: &[bool],
    qtype: &[i64],
) -> tenferro_ad::Result<(Vec<f32>, Vec<f32>)> {
    if qtype.is_empty() {
        return Err(invalid("laya.qtype", "the batch must be non-empty"));
    }
    let batch = qtype.len();
    if ids.len() % batch != 0 || ids.is_empty() {
        return Err(invalid("laya.ids", "ids must be a non-empty (L, B) batch"));
    }
    if mask.len() != ids.len() {
        return Err(invalid(
            "laya.mask",
            "mask length must match the token batch",
        ));
    }
    if marker_pos.len() % batch != 0 || marker_mask.len() != marker_pos.len() {
        return Err(invalid(
            "laya.marker_pos",
            "marker_pos and marker_mask must be (K, B) batches",
        ));
    }
    weights.validate(encoder, agent).map_err(|error| {
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            "laya-infer::cuda_raw",
            "weights",
            error.to_string(),
        ))
    })?;
    if !LayaCudaRaw::supports(encoder, weights) {
        return Err(tenferro_ad::Error::TensorRuntime(unsupported(
            "the raw CUDA forward needs head_dim 64 and hidden <= 1024",
        )));
    }
    let d = encoder.hidden_size;
    let length = ids.len() / batch;
    let k_count = marker_pos.len() / batch;
    if k_count < 2 {
        return Err(invalid(
            "laya.marker_pos",
            "at least two marker slots are required",
        ));
    }
    let vocab = weights.encoder.tok_embeddings.len() / d;
    if ids.iter().any(|&id| id < 0 || id as usize >= vocab) {
        return Err(invalid("laya.ids", "token id is outside the vocabulary"));
    }
    if qtype.iter().any(|q| !(0..3).contains(q)) {
        return Err(invalid(
            "laya.qtype",
            "qtype must be 0 (choice), 1 (score) or 2 (noul)",
        ));
    }
    let tokens = length * batch;
    let inter = encoder.intermediate_size;
    let ff = weights
        .head
        .iter()
        .map(|layer| layer.linear1.weight.len() / d)
        .max()
        .unwrap_or(0);
    let action_hidden = weights.act1.weight.len() / (d + 4);
    let action_count = agent.action_count();
    let layout = Layout::new(
        d,
        inter,
        ff,
        tokens,
        k_count,
        batch,
        action_hidden,
        action_count,
    );

    // Integer inputs, uploaded once: ids, valid, marker_pos, marker_mask, qtype.
    let mut ints: Vec<i32> = Vec::with_capacity(2 * tokens + 2 * k_count * batch + batch);
    ints.extend(ids.iter().map(|&id| id as i32));
    ints.extend(mask.iter().map(|&m| m as i32));
    ints.extend(
        marker_pos
            .iter()
            .map(|&p| p.clamp(i32::MIN as i64, i32::MAX as i64) as i32),
    );
    ints.extend(marker_mask.iter().map(|&m| m as i32));
    ints.extend(qtype.iter().map(|&q| q as i32));

    let mut out = vec![0.0f32; k_count * batch + action_count * batch];
    with_raw_exec(session, "laya.forward_cuda_raw", |exec| {
        if state.weights.is_none() {
            state.weights = Some(upload_weights(exec, encoder, weights)?);
        }
        let w = state.weights.as_ref().expect("uploaded above");
        if w.vocab != vocab {
            return Err(unsupported("weights changed since the first forward"));
        }
        if state
            .rope
            .as_ref()
            .is_none_or(|(_, positions)| *positions < length)
        {
            let positions = length.next_power_of_two().max(256);
            let table = rope_tables(w.rope_bases, positions);
            let buffer = exec.alloc(table.len())?;
            let ptr = exec.ptr(&buffer)?;
            // SAFETY: the buffer holds the table.
            unsafe { exec.upload(ptr, &table)? };
            exec.synchronize()?;
            state.rope = Some((buffer, positions));
        }
        if state
            .scratch
            .as_ref()
            .is_none_or(|scratch| scratch.len() < layout.total)
        {
            state.scratch = None;
            state.scratch = Some(exec.alloc(layout.total)?);
        }
        let module = exec.module(SOURCE, KERNELS)?;
        let bases = w
            .arenas
            .iter()
            .map(|arena| exec.ptr(arena))
            .collect::<tenferro_tensor::Result<Vec<DevPtr>>>()?;
        let (rope_buffer, positions) = state.rope.as_ref().expect("allocated above");
        let positions = *positions;
        let rope = exec.ptr(rope_buffer)?;
        let scratch = exec.ptr(state.scratch.as_ref().expect("allocated above"))?;
        let p = |loc: Loc| at(bases[loc.arena], loc.offset);
        let s = |offset: usize| at(scratch, offset);

        let f_embed = module.function("laya_embed")?;
        let f_ln = module.function("laya_layernorm")?;
        let f_geglu = module.function("laya_geglu")?;
        let f_bias = module.function("laya_bias_act")?;
        let f_type = module.function("laya_add_type")?;
        let f_markers = module.function("laya_gather_markers")?;
        let f_pool = module.function("laya_pool")?;
        let f_attn = module.function("laya_attention")?;

        let x = s(layout.x);
        let hb = s(layout.hb);
        let qkv = s(layout.qkv);
        let att = s(layout.att);
        let u = s(layout.u);
        let g = s(layout.g);
        let ints_ptr = s(layout.ints);
        let ids_ptr = ints_ptr;
        let valid_ptr = ints_ptr + 4 * tokens as u64;
        let pos_ptr = ints_ptr + 8 * tokens as u64;
        let mmask_ptr = pos_ptr + 4 * (k_count * batch) as u64;
        let qtype_ptr = mmask_ptr + 4 * (k_count * batch) as u64;
        let eps = encoder.norm_eps as f32;
        let half = (HEAD_DIM / 2) as u64;
        let cos_sin = |which: usize| {
            let cos = rope + 4 * (2 * which as u64) * half * positions as u64;
            (cos, cos + 4 * half * positions as u64)
        };

        // SAFETY (whole block): every address lies in a buffer sized above
        // (weights by construction, scratch by `layout`), the arguments match
        // the kernels in `laya_raw.cu`, and the scope synchronizes before any
        // buffer can be dropped.
        unsafe {
            exec.upload(ints_ptr, &ints)?;
            let elementwise = |f, total: usize, args: &[Arg]| -> tenferro_tensor::Result<()> {
                exec.launch(f, ((total as u32).div_ceil(256), 1, 1), 256, 0, args)
            };
            let layernorm = |y: DevPtr, xin: DevPtr, norm: Norm, cols: usize| {
                exec.launch(
                    f_ln,
                    ((cols as u32).div_ceil(8), 1, 1),
                    256,
                    0,
                    &[
                        Arg::P(y),
                        Arg::P(xin),
                        Arg::P(p(norm.w)),
                        Arg::P(norm.b.map_or(0, p)),
                        Arg::I(d as i32),
                        Arg::I(cols as i32),
                        Arg::F(eps),
                    ],
                )
            };
            // y (out, cols) = W xin (+ beta y), then bias/activation.
            let linear = |lin: Lin, xin: DevPtr, y: DevPtr, cols: usize, beta: f32, act: i32| {
                exec.gemm(
                    Op::N,
                    Op::N,
                    lin.out,
                    cols,
                    lin.inp,
                    1.0,
                    p(lin.w),
                    lin.out,
                    xin,
                    lin.inp,
                    beta,
                    y,
                    lin.out,
                )?;
                if lin.b.is_some() || act != 0 {
                    elementwise(
                        f_bias,
                        lin.out * cols,
                        &[
                            Arg::P(y),
                            Arg::P(lin.b.map_or(0, p)),
                            Arg::I(lin.out as i32),
                            Arg::I((lin.out * cols) as i32),
                            Arg::I(act),
                        ],
                    )?;
                }
                Ok::<(), tenferro_tensor::Error>(())
            };
            let attention = |heads: usize, window: i32, rope: Option<usize>| {
                let (cos, sin) = rope.map_or((0, 0), cos_sin);
                exec.launch(
                    f_attn,
                    ((length as u32).div_ceil(64), heads as u32, batch as u32),
                    128,
                    0,
                    &[
                        Arg::P(att),
                        Arg::P(qkv),
                        Arg::P(cos),
                        Arg::P(sin),
                        Arg::P(valid_ptr),
                        Arg::I(window),
                        Arg::I(heads as i32),
                        Arg::I(length as i32),
                        Arg::I(rope.is_some() as i32),
                        Arg::F(1.0 / (HEAD_DIM as f32).sqrt()),
                    ],
                )
            };

            // Encoder.
            elementwise(
                f_embed,
                d * tokens,
                &[
                    Arg::P(x),
                    Arg::P(p(w.embed)),
                    Arg::P(ids_ptr),
                    Arg::I(d as i32),
                    Arg::I((d * tokens) as i32),
                ],
            )?;
            layernorm(x, x, w.embed_norm, tokens)?;
            for layer in &w.layers {
                let input = match layer.attn_norm {
                    Some(norm) => {
                        layernorm(hb, x, norm, tokens)?;
                        hb
                    }
                    None => x,
                };
                linear(layer.wqkv, input, qkv, tokens, 0.0, 0)?;
                attention(layer.heads, layer.window, Some(layer.rope))?;
                linear(layer.wo, att, x, tokens, 1.0, 0)?;
                layernorm(hb, x, layer.mlp_norm, tokens)?;
                linear(layer.wi, hb, u, tokens, 0.0, 0)?;
                elementwise(
                    f_geglu,
                    inter * tokens,
                    &[
                        Arg::P(g),
                        Arg::P(u),
                        Arg::I(inter as i32),
                        Arg::I((inter * tokens) as i32),
                    ],
                )?;
                linear(layer.wo_mlp, g, x, tokens, 1.0, 0)?;
            }
            layernorm(x, x, w.final_norm, tokens)?;

            // Typed decision head.
            elementwise(
                f_type,
                d * tokens,
                &[
                    Arg::P(x),
                    Arg::P(p(w.type_emb)),
                    Arg::P(qtype_ptr),
                    Arg::I(d as i32),
                    Arg::I(length as i32),
                    Arg::I((d * tokens) as i32),
                ],
            )?;
            for layer in &w.head {
                layernorm(hb, x, layer.norm1, tokens)?;
                linear(layer.in_proj, hb, qkv, tokens, 0.0, 0)?;
                attention(layer.heads, -1, None)?;
                linear(layer.out_proj, att, x, tokens, 1.0, 0)?;
                layernorm(hb, x, layer.norm2, tokens)?;
                linear(layer.linear1, hb, u, tokens, 0.0, 2)?;
                linear(layer.linear2, u, x, tokens, 1.0, 0)?;
            }

            // Marker scorer, pooling and action head.
            let kb = k_count * batch;
            let markers = s(layout.markers);
            let s1 = s(layout.s1);
            let raw_logits = s(layout.raw);
            let pooled = s(layout.pooled);
            let a1 = s(layout.a1);
            let out_ptr = s(layout.out);
            elementwise(
                f_markers,
                d * kb,
                &[
                    Arg::P(markers),
                    Arg::P(x),
                    Arg::P(pos_ptr),
                    Arg::I(d as i32),
                    Arg::I(k_count as i32),
                    Arg::I(length as i32),
                    Arg::I((d * kb) as i32),
                ],
            )?;
            layernorm(markers, markers, w.scorer_norm, kb)?;
            linear(w.scorer1, markers, s1, kb, 0.0, 1)?;
            linear(w.scorer2, s1, raw_logits, kb, 0.0, 0)?;
            exec.launch(
                f_pool,
                (batch as u32, 1, 1),
                32,
                0,
                &[
                    Arg::P(out_ptr),
                    Arg::P(pooled),
                    Arg::P(raw_logits),
                    Arg::P(mmask_ptr),
                    Arg::P(x),
                    Arg::I(k_count as i32),
                    Arg::I(d as i32),
                    Arg::I(length as i32),
                ],
            )?;
            linear(w.act1, pooled, a1, batch, 0.0, 1)?;
            linear(w.act2, a1, at(out_ptr, kb), batch, 0.0, 0)?;
            exec.download(&mut out, out_ptr)?;
        }
        Ok(())
    })?;
    let action = out.split_off(k_count * batch);
    Ok((out, action))
}
