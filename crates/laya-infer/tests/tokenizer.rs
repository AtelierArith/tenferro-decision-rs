//! Golden tests for [`laya_infer::tokenizer::BpeTokenizer`].
//!
//! The `tokenizer.json` payloads below are built exactly like
//! `extern/Laya.jl/src/precompile.jl`'s `write_tiny_checkpoint`: a byte-level
//! BPE (NFC + ByteLevel, merges `h e` / `l l`) and a Metaspace BPE with byte
//! fallback. The expected ids were produced by encoding the same strings with
//! `Laya.Tokenizer` from the Julia reference:
//!
//! ```text
//! cd extern/Laya.jl
//! julia --project=. -e '
//!   using Laya, JSON
//!   # ... write the tokenizer exactly as in precompile.jl's write_tiny_checkpoint
//!   tok = Laya.Tokenizer(dir)
//!   println(tok("hello"))'
//! ```
//!
//! The goldens are hard-coded so the Rust test needs no Julia installation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use laya_infer::prompt::Tokenizer as _;
use laya_infer::tokenizer::BpeTokenizer;
use serde_json::{json, Value};

/// GPT-2 byte-to-unicode table, matching `Laya.BYTE_TO_CHAR`.
fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for (byte, slot) in table.iter_mut().enumerate() {
        let keep = (0x21..=0x7e).contains(&byte)
            || (0xa1..=0xac).contains(&byte)
            || (0xae..=0xff).contains(&byte);
        *slot = char::from_u32(if keep { byte as u32 } else { 256 + extra }).unwrap();
        if !keep {
            extra += 1;
        }
    }
    table
}

/// Build the tiny `tokenizer.json` value; `metaspace` selects the multilingual
/// variant. Mirrors `write_tiny_checkpoint`'s tokenizer section.
fn tiny_tokenizer_json(metaspace: bool) -> Value {
    let specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
    let (normalizer, pre_tokenizer, extra, pieces) = if metaspace {
        let mut pieces = vec!["\u{2581}".to_string()];
        pieces.extend(
            "abcdefghijklmnopqrstuvwxyz{}\":,.?"
                .chars()
                .map(String::from),
        );
        for byte in 0..256u32 {
            pieces.push(format!("<0x{byte:02X}>"));
        }
        (
            json!({"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}),
            json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": true}),
            json!({"unk_token": "[UNK]", "byte_fallback": true, "fuse_unk": true}),
            pieces,
        )
    } else {
        (
            json!({"type": "NFC"}),
            json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}),
            json!({}),
            byte_to_char_table().iter().map(char::to_string).collect(),
        )
    };

    let mut vocab: HashMap<String, i64> = HashMap::new();
    let mut next_id = 0i64;
    let mut insert = |token: String, vocab: &mut HashMap<String, i64>| {
        vocab.insert(token, next_id);
        next_id += 1;
    };
    for special in specials {
        insert(special.to_string(), &mut vocab);
    }
    for piece in &pieces {
        insert(piece.clone(), &mut vocab);
    }
    for merged in ["he", "ll"] {
        insert(merged.to_string(), &mut vocab);
    }

    let added: Vec<Value> = specials
        .iter()
        .map(|special| {
            json!({
                "id": vocab[*special],
                "content": special,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();

    let mut model = json!({
        "type": "BPE",
        "vocab": vocab,
        "merges": ["h e", "l l"],
    });
    for (key, value) in extra.as_object().unwrap() {
        model[key] = value.clone();
    }

    json!({
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "model": model,
    })
}

/// Write a tiny checkpoint tokenizer directory and return its path.
fn write_tiny_tokenizer(tag: &str, metaspace: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("laya-tokenizer-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec(&tiny_tokenizer_json(metaspace)).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        br#"{"pad_token":"[PAD]","cls_token":"[CLS]","sep_token":"[SEP]","mask_token":"[MASK]"}"#,
    )
    .unwrap();
    dir
}

fn load(tag: &str, metaspace: bool) -> (PathBuf, BpeTokenizer) {
    let dir = write_tiny_tokenizer(tag, metaspace);
    let tokenizer = BpeTokenizer::from_directory(&dir).unwrap();
    (dir, tokenizer)
}

fn cleanup(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

const STRINGS: [&str; 18] = [
    "hello",
    "hello world",
    "it's a test",
    "don't  stop",
    "a  b",
    "trailing   ",
    "  ",
    "{\"x\": 1}",
    "hello, world!",
    "I'm  sure",
    "1 2 3",
    "***",
    "tab\tsep",
    "\u{fc}z",
    "\u{fc}n\u{ef}code",
    "hello {\"x\": 1} \u{fc}n\u{ef}code",
    "caf\u{e9}",
    "\u{65e5}\u{672c}\u{8a9e}",
];

/// Goldens from `Laya.Tokenizer` for the byte-level (NFC + ByteLevel) variant.
const BYTELEVEL_GOLDEN: [&[i64]; 18] = [
    &[261, 262, 116],
    &[261, 262, 116, 37, 124, 116, 119, 113, 105],
    &[110, 121, 44, 120, 37, 102, 37, 121, 106, 120, 121],
    &[105, 116, 115, 44, 121, 37, 37, 120, 121, 116, 117],
    &[102, 37, 37, 103],
    &[121, 119, 102, 110, 113, 110, 115, 108, 37, 37, 37],
    &[37, 37],
    &[128, 39, 125, 39, 63, 37, 54, 130],
    &[261, 262, 116, 49, 37, 124, 116, 119, 113, 105, 38],
    &[78, 44, 114, 37, 37, 120, 122, 119, 106],
    &[54, 37, 55, 37, 56],
    &[47, 47, 47],
    &[121, 102, 103, 14, 120, 106, 117],
    &[200, 193, 127],
    &[200, 193, 115, 200, 180, 104, 116, 105, 106],
    &[
        261, 262, 116, 37, 128, 39, 125, 39, 63, 37, 54, 130, 37, 200, 193, 115, 200, 180, 104,
        116, 105, 106,
    ],
    &[104, 102, 107, 200, 174],
    &[235, 156, 170, 235, 161, 177, 237, 175, 163],
];

/// Goldens from `Laya.Tokenizer` for the Metaspace + byte-fallback variant.
const METASPACE_GOLDEN: [&[i64]; 18] = [
    &[5, 295, 296, 20],
    &[5, 295, 296, 20, 5, 28, 20, 23, 17, 9],
    &[5, 14, 25, 78, 24, 5, 6, 5, 25, 10, 24, 25],
    &[5, 9, 20, 19, 78, 25, 5, 5, 24, 25, 20, 21],
    &[5, 6, 5, 5, 7],
    &[5, 25, 23, 6, 14, 17, 14, 19, 12, 5, 5, 5],
    &[5, 5],
    &[5, 32, 34, 29, 34, 35, 5, 88, 33],
    &[5, 295, 296, 20, 36, 5, 28, 20, 23, 17, 9, 72],
    &[5, 112, 78, 18, 5, 5, 24, 26, 23, 10],
    &[5, 88, 5, 89, 5, 90],
    &[5, 81, 81, 81],
    &[5, 25, 6, 7, 48, 24, 10, 21],
    &[5, 234, 227, 31],
    &[5, 234, 227, 19, 234, 214, 8, 20, 9, 10],
    &[
        5, 295, 296, 20, 5, 32, 34, 29, 34, 35, 5, 88, 33, 5, 234, 227, 19, 234, 214, 8, 20, 9, 10,
    ],
    &[5, 8, 6, 11, 234, 208],
    &[5, 269, 190, 204, 269, 195, 211, 271, 209, 197],
];

#[test]
fn bytelevel_bpe_matches_reference() {
    let (dir, tokenizer) = load("bytelevel", false);
    for (text, expected) in STRINGS.iter().zip(BYTELEVEL_GOLDEN) {
        assert_eq!(
            tokenizer.encode(text),
            expected,
            "byte-level encode mismatch for {text:?}"
        );
    }
    cleanup(&dir);
}

#[test]
fn metaspace_bpe_with_byte_fallback_matches_reference() {
    let (dir, tokenizer) = load("metaspace", true);
    for (text, expected) in STRINGS.iter().zip(METASPACE_GOLDEN) {
        assert_eq!(
            tokenizer.encode(text),
            expected,
            "Metaspace encode mismatch for {text:?}"
        );
    }
    cleanup(&dir);
}

#[test]
fn specials_come_from_the_config() {
    let (dir, tokenizer) = load("specials", false);
    assert_eq!(tokenizer.mask_token(), "[MASK]");
    assert_eq!(tokenizer.mask_token_id(), 4);
    assert_eq!(tokenizer.cls_token_id(), 2);
    assert_eq!(tokenizer.sep_token_id(), 3);
    assert_eq!(tokenizer.pad_token_id(), 0);
    cleanup(&dir);
}

#[test]
fn missing_config_falls_back_to_vocab_defaults() {
    let dir = write_tiny_tokenizer("noconfig", false);
    std::fs::remove_file(dir.join("tokenizer_config.json")).unwrap();
    let tokenizer = BpeTokenizer::from_directory(&dir).unwrap();
    assert_eq!(tokenizer.mask_token(), "[MASK]");
    assert_eq!(tokenizer.mask_token_id(), 4);
    assert_eq!(tokenizer.cls_token_id(), 2);
    cleanup(&dir);
}

#[test]
fn malformed_tokenizer_json_is_an_error() {
    let dir = std::env::temp_dir().join(format!("laya-tokenizer-bad-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{ not json").unwrap();
    assert!(BpeTokenizer::from_directory(&dir).is_err());
    cleanup(&dir);
}
