//! Gated DeltaNet mixer for the raw single-stream CUDA forward (`cuda`
//! feature).
//!
//! [`delta_raw_mix`] enqueues, on a [`tenferro_ext::raw_exec`] scope, the
//! whole DeltaNet core between the packed input projection and the output
//! projection: causal depthwise convolution + SiLU, Q/K normalization, the
//! beta/decay gates, the scan, and the output RMSNorm gated by `silu(z)`.
//! The scan is either the warp-per-row recurrence or the chunked WY form of
//! QwenDecisionCore.jl (`cuda/delta_raw.cu`), whose intra-chunk products are
//! strided-batched cuBLAS GEMMs. Inputs and outputs are feature-major
//! `(rows, tokens)`; the caller owns every buffer and the scope's
//! synchronization.

use tenferro_ext::raw_exec::{Arg, DevPtr, Mat, Op, RawExec, RawModule, align, at};

/// The CUDA source of the raw DeltaNet kernels.
pub const DELTA_RAW_SOURCE: &str = include_str!("cuda/delta_raw.cu");

/// Kernel names in [`DELTA_RAW_SOURCE`].
pub const DELTA_RAW_KERNELS: &[&str] = &[
    "gd_conv_silu",
    "gd_qk_normalize",
    "gd_gates",
    "gd_recurrent",
    "gd_chunk_prepare",
    "gd_chunk_inverse",
    "gd_state_scale",
    "gd_rms_gate",
];

/// Tokens per chunk of the chunked scan.
pub const DELTA_RAW_CHUNK: usize = 64;

/// Scan algorithm of [`delta_raw_mix`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeltaRawScan {
    /// Recurrent below [`DeltaRawScan::AUTO_CHUNKED_FROM`] tokens, chunked
    /// from there on (`TENFERRO_DECISION_DELTA_SCAN=recurrent|chunked`
    /// overrides).
    #[default]
    Auto,
    /// Warp-per-value-row recurrence (one launch, sequential in tokens).
    Recurrent,
    /// Chunked WY form (batched GEMMs, sequential only across chunks).
    Chunked,
}

impl DeltaRawScan {
    /// Token count from which [`Self::Auto`] uses the chunked scan.
    pub const AUTO_CHUNKED_FROM: usize = 32;

    fn resolve(self, tokens: usize) -> Self {
        let selected = match self {
            DeltaRawScan::Auto => match std::env::var("TENFERRO_DECISION_DELTA_SCAN").as_deref() {
                Ok("recurrent") => DeltaRawScan::Recurrent,
                Ok("chunked") => DeltaRawScan::Chunked,
                _ => DeltaRawScan::Auto,
            },
            other => other,
        };
        match selected {
            DeltaRawScan::Auto if tokens >= Self::AUTO_CHUNKED_FROM => DeltaRawScan::Chunked,
            DeltaRawScan::Auto => DeltaRawScan::Recurrent,
            other => other,
        }
    }
}

/// DeltaNet dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaRawConfig {
    /// Query/key heads.
    pub key_heads: usize,
    /// Value heads (a multiple of `key_heads`).
    pub value_heads: usize,
    /// Query/key head width (<= 256).
    pub key_dim: usize,
    /// Value head width.
    pub value_dim: usize,
    /// Convolution taps (<= 8).
    pub conv_taps: usize,
}

impl DeltaRawConfig {
    /// Whether the raw kernels support these dimensions.
    pub fn supported(&self) -> bool {
        self.key_dim > 0
            && self.key_dim <= 256
            && self.value_dim > 0
            && self.conv_taps >= 1
            && self.conv_taps <= 8
            && self.key_heads > 0
            && self.value_heads % self.key_heads == 0
    }

    /// `2 * key_width + value_width`: the convolved q/k/v rows.
    pub fn conv_channels(&self) -> usize {
        2 * self.key_dim * self.key_heads + self.value_dim * self.value_heads
    }

    /// `value_heads * value_dim`.
    pub fn value_width(&self) -> usize {
        self.value_dim * self.value_heads
    }

    /// Scratch floats [`delta_raw_mix`] needs for `tokens` tokens.
    pub fn scratch_floats(&self, tokens: usize) -> usize {
        DeltaRawLayout::new(self, tokens).total
    }
}

struct DeltaRawLayout {
    mixed: usize,
    query: usize,
    key: usize,
    beta: usize,
    decay: usize,
    output: usize,
    kq: usize,
    scaled: usize,
    dq: usize,
    ek: usize,
    cumulative: usize,
    gates: usize,
    growth: usize,
    products: usize,
    inverse: usize,
    intra: usize,
    solved: usize,
    state: usize,
    total: usize,
}

impl DeltaRawLayout {
    fn new(cfg: &DeltaRawConfig, tokens: usize) -> Self {
        let c = DELTA_RAW_CHUNK;
        let chunks = tokens.div_ceil(c);
        let padded = chunks * c;
        let batches = cfg.value_heads * chunks;
        let (dk, dv) = (cfg.key_dim, cfg.value_dim);
        let mut next = 0usize;
        let mut take = |floats: usize| {
            let at = next;
            next += align(floats);
            at
        };
        let mixed = take(cfg.conv_channels() * tokens);
        let query = take(dk * cfg.key_heads * tokens);
        let key = take(dk * cfg.key_heads * tokens);
        let beta = take(cfg.value_heads * tokens);
        let decay = take(cfg.value_heads * tokens);
        let output = take(dv * cfg.value_heads * padded);
        let kq = take(dk * 2 * c * batches);
        let scaled = take((dk + dv) * c * batches);
        let dq = take(dk * c * batches);
        let ek = take(dk * c * batches);
        let cumulative = take(c * batches);
        let gates = take(c * batches);
        let growth = take(batches);
        let products = take(c * 2 * c * batches);
        let inverse = take(c * c * batches);
        let intra = take(c * c * batches);
        let solved = take((dk + dv) * c * batches);
        let state = take(dv * dk * cfg.value_heads);
        Self {
            mixed,
            query,
            key,
            beta,
            decay,
            output,
            kq,
            scaled,
            dq,
            ek,
            cumulative,
            gates,
            growth,
            products,
            inverse,
            intra,
            solved,
            state,
            total: next,
        }
    }
}

/// Device operands of one [`delta_raw_mix`] call.
#[derive(Clone, Copy, Debug)]
pub struct DeltaRawOperands {
    /// Packed projection `(ld_proj, T)`: rows `[qkv | z | a | b]`.
    pub proj: DevPtr,
    /// Leading dimension of `proj` (>= conv_channels + value_width + 2 heads).
    pub ld_proj: usize,
    /// Convolution kernel `(taps, conv_channels)` row-major.
    pub conv: DevPtr,
    /// `-exp(A_log)` `(value_heads,)`.
    pub a_decay: DevPtr,
    /// Decay bias `(value_heads,)`.
    pub dt_bias: DevPtr,
    /// Output RMSNorm scale `(value_dim,)`.
    pub norm: DevPtr,
    /// Output RMSNorm epsilon.
    pub eps: f32,
    /// Output `(value_width, T)`, the out-projection input.
    pub gated: DevPtr,
    /// Scratch of [`DeltaRawConfig::scratch_floats`] floats.
    pub scratch: DevPtr,
}

/// Enqueue the DeltaNet core (see the module docs) for `tokens` tokens.
///
/// # Safety
///
/// Every operand must address device memory of the documented size that
/// stays valid until the scope synchronizes; `module` must be compiled from
/// [`DELTA_RAW_SOURCE`]; `cfg` must be [`DeltaRawConfig::supported`].
pub unsafe fn delta_raw_mix(
    exec: &RawExec<'_, '_>,
    module: &RawModule,
    cfg: &DeltaRawConfig,
    ops: &DeltaRawOperands,
    tokens: usize,
    scan: DeltaRawScan,
) -> tenferro_tensor::Result<()> {
    if tokens == 0 {
        return Ok(());
    }
    let layout = DeltaRawLayout::new(cfg, tokens);
    let s = |offset: usize| at(ops.scratch, offset);
    let (dk, dv, heads) = (cfg.key_dim, cfg.value_dim, cfg.value_heads);
    let groups = heads / cfg.key_heads;
    let channels = cfg.conv_channels();
    let value_start = 2 * dk * cfg.key_heads;
    let z_row = channels;
    let a_row = channels + cfg.value_width();
    let b_row = a_row + heads;
    let mixed = s(layout.mixed);
    let query = s(layout.query);
    let key = s(layout.key);
    let beta = s(layout.beta);
    let decay = s(layout.decay);
    let output = s(layout.output);
    let i = |v: usize| Arg::I(v as i32);

    // SAFETY (block): forwarded caller contract; scratch offsets come from
    // the layout of this exact (cfg, tokens).
    unsafe {
        exec.launch(
            module.function("gd_conv_silu")?,
            (
                (channels as u32).div_ceil(128),
                (tokens as u32).div_ceil(8),
                1,
            ),
            128,
            0,
            &[
                Arg::P(mixed),
                Arg::P(ops.proj),
                Arg::P(ops.conv),
                i(channels),
                i(tokens),
                i(ops.ld_proj),
                i(cfg.conv_taps),
            ],
        )?;
        exec.launch(
            module.function("gd_qk_normalize")?,
            ((cfg.key_heads * tokens) as u32, 1, 1),
            32,
            0,
            &[
                Arg::P(query),
                Arg::P(key),
                Arg::P(mixed),
                i(dk),
                i(cfg.key_heads),
                i(channels),
            ],
        )?;
        exec.launch(
            module.function("gd_gates")?,
            (((heads * tokens) as u32).div_ceil(256), 1, 1),
            256,
            0,
            &[
                Arg::P(beta),
                Arg::P(decay),
                Arg::P(ops.proj),
                i(ops.ld_proj),
                i(a_row),
                i(b_row),
                Arg::P(ops.a_decay),
                Arg::P(ops.dt_bias),
                i(heads),
                i(heads * tokens),
            ],
        )?;
        match scan.resolve(tokens) {
            DeltaRawScan::Chunked => chunked(
                exec, module, cfg, &layout, ops, tokens, query, key, mixed, beta, decay, output,
            )?,
            _ => exec.launch(
                module.function("gd_recurrent")?,
                ((dv as u32).div_ceil(4), heads as u32, 1),
                128,
                0,
                &[
                    Arg::P(output),
                    Arg::P(query),
                    Arg::P(key),
                    Arg::P(mixed),
                    i(channels),
                    Arg::P(beta),
                    Arg::P(decay),
                    i(dk),
                    i(dv),
                    i(value_start),
                    i(heads),
                    i(groups),
                    i(tokens),
                ],
            )?,
        }
        exec.launch(
            module.function("gd_rms_gate")?,
            ((heads * tokens) as u32, 1, 1),
            32,
            0,
            &[
                Arg::P(ops.gated),
                Arg::P(output),
                Arg::P(ops.norm),
                Arg::P(ops.proj),
                i(ops.ld_proj),
                i(z_row),
                i(dv),
                i(heads),
                Arg::F(ops.eps),
            ],
        )?;
    }
    Ok(())
}

/// The chunked WY scan (QwenDecisionCore.jl `delta_chunked!`), writing
/// `output` `(dv, heads, chunks * C)`.
#[allow(clippy::too_many_arguments)]
unsafe fn chunked(
    exec: &RawExec<'_, '_>,
    module: &RawModule,
    cfg: &DeltaRawConfig,
    layout: &DeltaRawLayout,
    ops: &DeltaRawOperands,
    tokens: usize,
    query: DevPtr,
    key: DevPtr,
    mixed: DevPtr,
    beta: DevPtr,
    decay: DevPtr,
    output: DevPtr,
) -> tenferro_tensor::Result<()> {
    let c = DELTA_RAW_CHUNK;
    let (dk, dv, heads) = (cfg.key_dim, cfg.value_dim, cfg.value_heads);
    let chunks = tokens.div_ceil(c);
    let batches = heads * chunks;
    let width = dk + dv;
    let s = |offset: usize| at(ops.scratch, offset);
    let kq = s(layout.kq);
    let scaled = s(layout.scaled);
    let dq = s(layout.dq);
    let ek = s(layout.ek);
    let cumulative = s(layout.cumulative);
    let gates = s(layout.gates);
    let growth = s(layout.growth);
    let products = s(layout.products);
    let inverse = s(layout.inverse);
    let intra = s(layout.intra);
    let solved = s(layout.solved);
    let state = s(layout.state);
    let i = |v: usize| Arg::I(v as i32);
    // SAFETY (block): forwarded caller contract.
    unsafe {
        exec.launch(
            module.function("gd_chunk_prepare")?,
            (batches as u32, 4, 1),
            128,
            0,
            &[
                Arg::P(kq),
                Arg::P(scaled),
                Arg::P(dq),
                Arg::P(ek),
                Arg::P(cumulative),
                Arg::P(gates),
                Arg::P(growth),
                Arg::P(query),
                Arg::P(key),
                Arg::P(mixed),
                i(cfg.conv_channels()),
                Arg::P(beta),
                Arg::P(decay),
                i(dk),
                i(dv),
                i(2 * dk * cfg.key_heads),
                i(heads),
                i(heads / cfg.key_heads),
                i(tokens),
            ],
        )?;
        // products[:, 0:C] = k'k, products[:, C:2C] = k'q
        exec.gemm_batched(
            Op::T,
            Op::N,
            c,
            2 * c,
            dk,
            1.0,
            Mat::new(kq, dk, dk * 2 * c),
            Mat::new(kq, dk, dk * 2 * c),
            0.0,
            Mat::new(products, c, c * 2 * c),
            batches,
        )?;
        exec.launch(
            module.function("gd_chunk_inverse")?,
            (batches as u32, 1, 1),
            (4 * c) as u32,
            0,
            &[
                Arg::P(inverse),
                Arg::P(intra),
                Arg::P(products),
                Arg::P(cumulative),
                Arg::P(gates),
            ],
        )?;
        // solved[0:dk] = W' (reading keys), solved[dk:] = U' (new values)
        exec.gemm_batched(
            Op::N,
            Op::T,
            width,
            c,
            c,
            1.0,
            Mat::new(scaled, width, width * c),
            Mat::new(inverse, c, c * c),
            0.0,
            Mat::new(solved, width, width * c),
            batches,
        )?;
        for chunk in 0..chunks {
            let base = chunk * heads;
            let values = at(solved, base * width * c + dk);
            let keys = at(solved, base * width * c);
            let out = at(output, chunk * c * dv * heads);
            if chunk > 0 {
                // corrections: U' -= S W'
                exec.gemm_batched(
                    Op::N,
                    Op::N,
                    dv,
                    c,
                    dk,
                    -1.0,
                    Mat::new(state, dv, dv * dk),
                    Mat::new(keys, width, width * c),
                    1.0,
                    Mat::new(values, width, width * c),
                    heads,
                )?;
                // out = S (exp(G) q)
                exec.gemm_batched(
                    Op::N,
                    Op::N,
                    dv,
                    c,
                    dk,
                    1.0,
                    Mat::new(state, dv, dv * dk),
                    Mat::new(at(dq, base * dk * c), dk, dk * c),
                    0.0,
                    Mat::new(out, dv * heads, dv),
                    heads,
                )?;
            }
            // out += U' intra
            exec.gemm_batched(
                Op::N,
                Op::N,
                dv,
                c,
                c,
                1.0,
                Mat::new(values, width, width * c),
                Mat::new(at(intra, base * c * c), c, c * c),
                if chunk > 0 { 1.0 } else { 0.0 },
                Mat::new(out, dv * heads, dv),
                heads,
            )?;
            if chunk + 1 == chunks {
                break;
            }
            if chunk > 0 {
                exec.launch(
                    module.function("gd_state_scale")?,
                    (((dv * dk * heads) as u32).div_ceil(256), 1, 1),
                    256,
                    0,
                    &[
                        Arg::P(state),
                        Arg::P(growth),
                        i(base),
                        i(dv * dk),
                        i(dv * dk * heads),
                    ],
                )?;
            }
            // S = growth S + U' (exp(G_C - G) k)'
            exec.gemm_batched(
                Op::N,
                Op::T,
                dv,
                dk,
                c,
                1.0,
                Mat::new(values, width, width * c),
                Mat::new(at(ek, base * dk * c), dk, dk * c),
                if chunk > 0 { 1.0 } else { 0.0 },
                Mat::new(state, dv, dv * dk),
                heads,
            )?;
        }
    }
    Ok(())
}
