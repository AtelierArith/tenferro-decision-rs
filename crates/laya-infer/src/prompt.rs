//! Laya prompt construction and Python-compatible JSON serialization.
//!
//! Ports `extern/Laya.jl/src/prompt.jl`, which mirrors
//! `extern/laya-mlx/laya_mlx/common.py`. Exact string equality with
//! `json.dumps` matters because the model was trained on those prompt strings.

use decision_core::{Content, DecisionError, Question, Result, State};

/// Minimal tokenizer surface the prompt builder needs.
///
/// The concrete checkpoint tokenizer (byte-level / Metaspace BPE) implements
/// this; tests use a deterministic fake.
pub trait Tokenizer {
    /// Encode text into token ids.
    fn encode(&self, text: &str) -> Vec<i64>;
    /// The mask token surface form (for example `[MASK]`).
    fn mask_token(&self) -> &str;
    /// Id of the mask token.
    fn mask_token_id(&self) -> i64;
    /// Id of the sequence-start token.
    fn cls_token_id(&self) -> i64;
    /// Id of the separator token.
    fn sep_token_id(&self) -> i64;
    /// Id of the padding token.
    fn pad_token_id(&self) -> i64;
}

/// Python `repr(float)`: shortest round-trip digits, scientific below `1e-4`
/// or from `1e16`.
pub fn py_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }

    let negative = value < 0.0;
    let scientific = format!("{:e}", value.abs()); // e.g. "1.2345e2"
    let (mantissa, exponent) = match scientific.split_once('e') {
        Some((m, e)) => (m.to_string(), e.parse::<i64>().unwrap_or(0)),
        None => (scientific.clone(), 0),
    };
    let integer_digits = mantissa.split('.').next().map(str::len).unwrap_or(0);
    let all_digits: String = mantissa.chars().filter(|c| *c != '.').collect();

    let mut point = integer_digits as i64 + exponent;
    let leading_zeros = all_digits.chars().take_while(|c| *c == '0').count() as i64;
    let digits = all_digits
        .trim_start_matches('0')
        .trim_end_matches('0')
        .to_string();
    point -= leading_zeros;

    let sign = if negative { "-" } else { "" };
    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    let exp10 = point - 1;
    if (-4..16).contains(&exp10) {
        if point <= 0 {
            format!("{sign}0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!(
                "{sign}{}{}.0",
                digits,
                "0".repeat(point as usize - digits.len())
            )
        } else {
            let (head, tail) = digits.split_at(point as usize);
            format!("{sign}{head}.{tail}")
        }
    } else {
        let mantissa = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[0..1], &digits[1..])
        };
        format!(
            "{sign}{mantissa}e{}{:02}",
            if exp10 < 0 { "-" } else { "+" },
            exp10.abs()
        )
    }
}

fn push_json_string(out: &mut String, text: &str, ascii: bool) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (ascii && (c as u32) > 0x7e) => {
                let mut u = c as u32;
                if u > 0xffff {
                    u -= 0x10000;
                    out.push_str(&format!("\\u{:04x}", 0xd800 + (u >> 10)));
                    out.push_str(&format!("\\u{:04x}", 0xdc00 + (u & 0x3ff)));
                } else {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.dumps(value, ensure_ascii=ascii)` with Python's default `(", ", ": ")`
/// separators, for a [`Content`] value.
pub fn py_json_content(content: &Content, ascii: bool) -> String {
    let mut out = String::new();
    write_content(&mut out, content, ascii);
    out
}

fn write_content(out: &mut String, content: &Content, ascii: bool) {
    match content {
        Content::Null => out.push_str("null"),
        Content::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Content::Int(i) => out.push_str(&i.to_string()),
        Content::Float(f) => out.push_str(&py_float(*f)),
        Content::String(s) => push_json_string(out, s, ascii),
        Content::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_content(out, item, ascii);
            }
            out.push(']');
        }
        Content::Object(pairs) => {
            out.push('{');
            for (index, (key, value)) in pairs.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                push_json_string(out, key, ascii);
                out.push_str(": ");
                write_content(out, value, ascii);
            }
            out.push('}');
        }
    }
}

/// `serialize_state`: a string passes through, anything else is JSON.
pub fn serialize_state(state: &State) -> Result<String> {
    match state {
        State::Text(text) => Ok(text.clone()),
        State::Json(content) => Ok(py_json_content(content, false)),
        State::Prepared(_) => Err(DecisionError::invalid_field(
            "state",
            "prepared tokens cannot be serialized as a Laya prompt",
        )),
    }
}

/// `render_criterion`: a string passes through, anything else is JSON.
pub fn render_criterion(value: &Content) -> String {
    match value {
        Content::String(text) => text.clone(),
        other => py_json_content(other, false),
    }
}

fn is_blank(value: &Content) -> bool {
    match value {
        Content::Null => true,
        Content::String(text) => text.is_empty(),
        _ => false,
    }
}

/// Option texts in label order. `noul` is always `[false, true]`.
pub fn render_options(question: &Question) -> Vec<String> {
    match question {
        Question::Choice(choice) => choice
            .criteria
            .iter()
            .map(|(label, description)| {
                if is_blank(description) {
                    label.clone()
                } else {
                    format!("{label}: {}", render_criterion(description))
                }
            })
            .collect(),
        Question::Score(score) => score
            .criteria
            .iter()
            .enumerate()
            .map(|(index, label)| {
                format!(
                    "level {index}: {}",
                    render_criterion(&Content::String(label.clone()))
                )
            })
            .collect(),
        Question::Noul(noul) => {
            let false_text = noul
                .criteria
                .falsy
                .as_ref()
                .filter(|value| !is_blank(value))
                .map(render_criterion)
                .unwrap_or_else(|| "no, the statement does not hold".to_string());
            let true_text = noul
                .criteria
                .truthy
                .as_ref()
                .filter(|value| !is_blank(value))
                .map(render_criterion)
                .unwrap_or_else(|| "yes, the statement holds".to_string());
            vec![format!("false: {false_text}"), format!("true: {true_text}")]
        }
    }
}

fn question_type(question: &Question) -> &'static str {
    match question {
        Question::Choice(_) => "choice",
        Question::Score(_) => "score",
        Question::Noul(_) => "noul",
    }
}

fn instruction_text(question: &Question) -> String {
    let instructions = match question {
        Question::Choice(q) => &q.instructions,
        Question::Score(q) => &q.instructions,
        Question::Noul(q) => &q.instructions,
    };
    match instructions {
        Content::String(text) => text.clone(),
        other => py_json_content(other, true),
    }
}

/// The question-only prefix, before state tokens and final truncation.
///
/// Returns `(ids, 0-based marker positions)`.
pub fn build_prefix<T: Tokenizer>(
    tokenizer: &T,
    question: &Question,
    head_max_len: usize,
) -> (Vec<i64>, Vec<usize>) {
    let mask = tokenizer.mask_token().to_string();
    let options = render_options(question);
    let instructions = instruction_text(question).replace(&mask, " ");
    let head_text = format!("{} question: {instructions}", question_type(question));
    let head_ids = tokenizer.encode(&head_text);

    let mut option_ids: Vec<Vec<i64>> = options
        .iter()
        .map(|option| {
            let text = format!(" {}", option.replace(&mask, " "));
            let mut ids = vec![tokenizer.mask_token_id()];
            ids.extend(tokenizer.encode(&text).into_iter().take(48));
            ids
        })
        .collect();

    let total: usize = option_ids.iter().map(Vec::len).sum();
    let mut budget = head_max_len as i64 - total as i64;
    if budget < 16 {
        let per = ((head_max_len as i64 - 16) / option_ids.len().max(1) as i64).max(4) as usize;
        for ids in &mut option_ids {
            ids.truncate(per);
        }
        let total: usize = option_ids.iter().map(Vec::len).sum();
        budget = head_max_len as i64 - total as i64;
    }

    let head_take = std::cmp::max(8, budget) as usize;
    let mut ids = vec![tokenizer.cls_token_id()];
    ids.extend(head_ids.into_iter().take(head_take));
    ids.push(tokenizer.sep_token_id());

    let mut markers = Vec::with_capacity(option_ids.len());
    for option in option_ids {
        markers.push(ids.len());
        ids.extend(option);
    }
    ids.push(tokenizer.sep_token_id());
    (ids, markers)
}

/// `[CLS] <type> instructions [SEP] [MASK] opt0 ... [SEP] state [SEP]`,
/// truncated to `max_len`.
pub fn build_sequence<T: Tokenizer>(
    tokenizer: &T,
    state: &State,
    question: &Question,
    max_len: usize,
    head_max_len: usize,
) -> Result<(Vec<i64>, Vec<usize>)> {
    let (mut ids, markers) = build_prefix(tokenizer, question, head_max_len);
    let room = max_len.saturating_sub(ids.len() + 1);
    let mask = tokenizer.mask_token().to_string();
    let state_text = serialize_state(state)?.replace(&mask, " ");
    let state_ids = tokenizer.encode(&state_text);

    ids.extend(state_ids.into_iter().take(room));
    ids.push(tokenizer.sep_token_id());
    ids.truncate(max_len);

    let markers = markers
        .into_iter()
        .filter(|position| *position < max_len)
        .collect();
    Ok((ids, markers))
}
