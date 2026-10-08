//! Python-compatible JSON spelling used by trained model prompts.

use crate::Content;

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
