//! Concrete Laya tokenizer: a Hugging Face `tokenizer.json` subset.
//!
//! Ports `extern/Laya.jl/src/tokenizer.jl` faithfully, covering the pieces the
//! Laya checkpoints use: added-token extraction (leftmost-longest, with
//! `lstrip` / `rstrip` / `single_word`), normalizers (NFC/NFD/NFKC/NFKD,
//! `Lowercase`, `Replace`, `Prepend`, `Sequence`), pre-tokenizers (`ByteLevel`
//! with the GPT-2 regex, `Metaspace`, `Whitespace`, `Sequence`) and token
//! models (`BPE` with vocabulary, merges, `byte_fallback`, `fuse_unk`,
//! `ignore_merges` and `unk_token`; `WordLevel`). Only token ids are produced;
//! character offsets are not tracked.
//!
//! [`BpeTokenizer::encode`] follows the same pipeline as the Julia reference:
//! split on raw added tokens, normalize, split on normalized added tokens,
//! pre-tokenize each remaining plain piece and tokenize each word. Encoding is
//! infallible: unknown symbols fall back to `unk` / byte tokens or are skipped.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use decision_core::{DecisionError, Result};
use regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

/// Tokenizer error type, re-exported for callers of [`BpeTokenizer`].
///
/// It is an alias of [`decision_core::DecisionError`] so that
/// [`BpeTokenizer::from_directory`] can return the crate-wide
/// [`decision_core::Result`].
pub type Error = DecisionError;

// ---------------------------------------------------------------------------
// Added tokens
// ---------------------------------------------------------------------------

/// One entry of the `added_tokens` array.
#[derive(Clone, Debug)]
struct AddedToken {
    content: String,
    id: i64,
    lstrip: bool,
    rstrip: bool,
    single_word: bool,
}

/// Leftmost-longest matcher over added tokens, grouped by their first character.
///
/// This mirrors the reference's crate-style trie: for every starting position
/// the longest candidate that also satisfies `single_word` wins.
#[derive(Clone, Debug, Default)]
struct AddedMatcher {
    by_first: HashMap<char, Vec<AddedToken>>,
}

impl AddedMatcher {
    fn new(tokens: &[AddedToken]) -> Self {
        let mut by_first: HashMap<char, Vec<AddedToken>> = HashMap::new();
        for token in tokens {
            let Some(first) = token.content.chars().next() else {
                continue;
            };
            by_first.entry(first).or_default().push(token.clone());
        }
        for candidates in by_first.values_mut() {
            candidates.sort_by_key(|token| std::cmp::Reverse(token.content.len()));
        }
        Self { by_first }
    }

    fn is_empty(&self) -> bool {
        self.by_first.is_empty()
    }
}

/// One piece produced by [`split_added`]: literal text or an added-token id.
enum Piece {
    Text(String),
    Id(i64),
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn prev_char(text: &str, byte: usize) -> Option<(usize, char)> {
    text[..byte].char_indices().next_back()
}

/// Split `text` around added tokens.
///
/// Plain pieces stay as [`Piece::Text`]; matches become [`Piece::Id`].
/// `lstrip` / `rstrip` tokens also swallow adjacent whitespace, as in the
/// reference crate.
fn split_added(matcher: &AddedMatcher, text: &str) -> Vec<Piece> {
    let mut out = Vec::new();
    if matcher.is_empty() {
        out.push(Piece::Text(text.to_string()));
        return out;
    }

    let n = text.len();
    let mut last = 0usize;
    let mut i = 0usize;
    while i < n {
        let Some(current) = text[i..].chars().next() else {
            break;
        };
        let mut matched: Option<&AddedToken> = None;
        if let Some(candidates) = matcher.by_first.get(&current) {
            for token in candidates {
                if !text[i..].starts_with(&token.content) {
                    continue;
                }
                let end = i + token.content.len();
                if token.single_word {
                    let before = i > 0
                        && prev_char(text, i)
                            .map(|(_, c)| is_word_char(c))
                            .unwrap_or(false);
                    let after = end < n
                        && text[end..]
                            .chars()
                            .next()
                            .map(is_word_char)
                            .unwrap_or(false);
                    if before || after {
                        continue;
                    }
                }
                matched = Some(token);
                break;
            }
        }

        let Some(token) = matched else {
            i += current.len_utf8();
            continue;
        };

        let mut start = i;
        let mut end = i + token.content.len();
        if token.lstrip {
            while start > last {
                match prev_char(text, start) {
                    Some((previous, c)) if c.is_whitespace() => start = previous,
                    _ => break,
                }
            }
        }
        if token.rstrip {
            while end < n {
                match text[end..].chars().next() {
                    Some(c) if c.is_whitespace() => end += c.len_utf8(),
                    _ => break,
                }
            }
        }

        if start > last {
            out.push(Piece::Text(text[last..start].to_string()));
        }
        out.push(Piece::Id(token.id));
        last = end;
        i = end;
    }

    if last < n {
        out.push(Piece::Text(text[last..].to_string()));
    }
    out
}

// ---------------------------------------------------------------------------
// Normalizers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum NormForm {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
}

#[derive(Clone, Debug)]
enum Normalizer {
    None,
    Unicode(NormForm),
    Lowercase,
    Replace(String, String),
    Prepend(String),
    Sequence(Vec<Normalizer>),
}

impl Normalizer {
    fn from_spec(spec: Option<&Value>) -> Result<Self> {
        let Some(spec) = spec else {
            return Ok(Self::None);
        };
        let kind = spec
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| DecisionError::invalid("normalizer is missing a string `type`"))?;
        match kind {
            "NFC" => Ok(Self::Unicode(NormForm::Nfc)),
            "NFD" => Ok(Self::Unicode(NormForm::Nfd)),
            "NFKC" => Ok(Self::Unicode(NormForm::Nfkc)),
            "NFKD" => Ok(Self::Unicode(NormForm::Nfkd)),
            "Lowercase" => Ok(Self::Lowercase),
            "Prepend" => {
                let prepend = spec.get("prepend").and_then(Value::as_str).ok_or_else(|| {
                    DecisionError::invalid("`Prepend` normalizer is missing `prepend`")
                })?;
                Ok(Self::Prepend(prepend.to_string()))
            }
            "Replace" => {
                let pattern = spec
                    .get("pattern")
                    .and_then(|pattern| pattern.get("String"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        DecisionError::unsupported("only string `Replace` patterns are supported")
                    })?;
                let content = spec.get("content").and_then(Value::as_str).unwrap_or("");
                Ok(Self::Replace(pattern.to_string(), content.to_string()))
            }
            "Sequence" => {
                let items = spec
                    .get("normalizers")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        DecisionError::invalid("`Sequence` normalizer is missing `normalizers`")
                    })?;
                let mut sequence = Vec::with_capacity(items.len());
                for item in items {
                    sequence.push(Self::from_spec(Some(item))?);
                }
                Ok(Self::Sequence(sequence))
            }
            other => Err(DecisionError::unsupported(format!(
                "unsupported normalizer: {other}"
            ))),
        }
    }

    fn apply(&self, text: &str) -> String {
        match self {
            Self::None => text.to_string(),
            Self::Unicode(form) => match form {
                NormForm::Nfc => text.nfc().collect(),
                NormForm::Nfd => text.nfd().collect(),
                NormForm::Nfkc => text.nfkc().collect(),
                NormForm::Nfkd => text.nfkd().collect(),
            },
            Self::Lowercase => text.to_lowercase(),
            Self::Replace(pattern, content) => text.replace(pattern.as_str(), content.as_str()),
            Self::Prepend(prepend) => {
                if text.is_empty() {
                    text.to_string()
                } else {
                    format!("{prepend}{text}")
                }
            }
            Self::Sequence(normalizers) => normalizers
                .iter()
                .fold(text.to_string(), |acc, normalizer| normalizer.apply(&acc)),
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-tokenizers
// ---------------------------------------------------------------------------

/// `Metaspace` prepend behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrependScheme {
    Always,
    First,
    Never,
}

#[derive(Clone, Debug)]
enum PreTokenizer {
    None,
    ByteLevel {
        add_prefix_space: bool,
        use_regex: bool,
    },
    Metaspace {
        replacement: char,
        prepend: PrependScheme,
        split: bool,
    },
    Whitespace,
    Sequence(Vec<PreTokenizer>),
}

impl PreTokenizer {
    fn from_spec(spec: Option<&Value>) -> Result<Self> {
        let Some(spec) = spec else {
            return Ok(Self::None);
        };
        let kind = spec
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| DecisionError::invalid("pre-tokenizer is missing a string `type`"))?;
        match kind {
            "ByteLevel" => Ok(Self::ByteLevel {
                add_prefix_space: spec
                    .get("add_prefix_space")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                use_regex: spec
                    .get("use_regex")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            }),
            "Whitespace" => Ok(Self::Whitespace),
            "Metaspace" => {
                let replacement = match spec.get("replacement").and_then(Value::as_str) {
                    Some(text) => {
                        let mut chars = text.chars();
                        match (chars.next(), chars.next()) {
                            (Some(c), None) => c,
                            _ => {
                                return Err(DecisionError::invalid(
                                    "Metaspace `replacement` must be a single character",
                                ))
                            }
                        }
                    }
                    None => '\u{2581}',
                };
                let prepend = match spec.get("prepend_scheme").and_then(Value::as_str) {
                    Some("always") => PrependScheme::Always,
                    Some("first") => PrependScheme::First,
                    Some("never") => PrependScheme::Never,
                    Some(other) => {
                        return Err(DecisionError::unsupported(format!(
                            "unsupported Metaspace prepend_scheme: {other}"
                        )))
                    }
                    None => match spec.get("add_prefix_space").and_then(Value::as_bool) {
                        Some(false) => PrependScheme::Never,
                        _ => PrependScheme::Always,
                    },
                };
                let split = spec.get("split").and_then(Value::as_bool).unwrap_or(true);
                Ok(Self::Metaspace {
                    replacement,
                    prepend,
                    split,
                })
            }
            "Sequence" => {
                let items = spec
                    .get("pretokenizers")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        DecisionError::invalid(
                            "`Sequence` pre-tokenizer is missing `pretokenizers`",
                        )
                    })?;
                let mut sequence = Vec::with_capacity(items.len());
                for item in items {
                    sequence.push(Self::from_spec(Some(item))?);
                }
                Ok(Self::Sequence(sequence))
            }
            other => Err(DecisionError::unsupported(format!(
                "unsupported pre-tokenizer: {other}"
            ))),
        }
    }
}

/// Run `pre_tokenizer` on one plain piece; `first` marks the piece at text offset 0.
fn pretokenize(pre_tokenizer: &PreTokenizer, text: &str, first: bool) -> Vec<String> {
    match pre_tokenizer {
        PreTokenizer::None => vec![text.to_string()],
        PreTokenizer::ByteLevel {
            add_prefix_space,
            use_regex,
        } => {
            let spaced = if *add_prefix_space && !text.starts_with(' ') {
                format!(" {text}")
            } else {
                text.to_string()
            };
            let words: Vec<String> = if *use_regex {
                gpt2_scan(&spaced)
            } else {
                vec![spaced]
            };
            let table = byte_to_char_table();
            words
                .into_iter()
                .map(|word| word.bytes().map(|b| table[b as usize]).collect())
                .collect()
        }
        PreTokenizer::Metaspace {
            replacement,
            prepend,
            split,
        } => {
            let delimiter = replacement.to_string();
            let mut text = text.replace(' ', &delimiter);
            if (*prepend == PrependScheme::Always || (*prepend == PrependScheme::First && first))
                && !text.starts_with(*replacement)
            {
                text = format!("{delimiter}{text}");
            }
            if !*split {
                return vec![text];
            }
            // Merge the delimiter into the following word:
            // "▁a▁▁b" -> ["▁a", "▁", "▁b"].
            let mut words = Vec::new();
            let mut start = 0usize;
            for (index, c) in text.char_indices() {
                if c == *replacement && index > start {
                    words.push(text[start..index].to_string());
                    start = index;
                }
            }
            words.push(text[start..].to_string());
            words
        }
        PreTokenizer::Whitespace => whitespace_pretokenize(text),
        PreTokenizer::Sequence(pre_tokenizers) => {
            let mut words = vec![text.to_string()];
            for pre_tokenizer in pre_tokenizers {
                let mut next = Vec::new();
                for word in &words {
                    next.extend(pretokenize(pre_tokenizer, word, first));
                }
                words = next;
            }
            words
        }
    }
}

/// GPT-2 byte-to-unicode table: printable bytes map to themselves, the rest to
/// `U+0100`-based code points. This is the reference's `BYTE_TO_CHAR`.
fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for (byte, slot) in table.iter_mut().enumerate() {
        let keep = (0x21..=0x7e).contains(&byte)
            || (0xa1..=0xac).contains(&byte)
            || (0xae..=0xff).contains(&byte);
        if keep {
            *slot = char::from_u32(byte as u32).unwrap_or('\0');
        } else {
            *slot = char::from_u32(256 + extra).unwrap_or('\0');
            extra += 1;
        }
    }
    table
}

/// Hand-written scanner for the GPT-2 pre-tokenizer pattern:
///
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// ```
///
/// The `regex` crate cannot express the trailing negative lookahead, so the
/// alternatives are tried by hand in order, mirroring `eachmatch` semantics.
fn gpt2_scan(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let n = text.len();
    while i < n {
        let rest = &text[i..];
        let matched = contraction_len(rest)
            .or_else(|| optional_class_len(rest, char::is_alphabetic))
            .or_else(|| optional_class_len(rest, char::is_numeric))
            .or_else(|| optional_other_len(rest))
            .or_else(|| whitespace_no_suffix_len(rest))
            .or_else(|| whitespace_len(rest));
        match matched {
            Some(length) if length > 0 => {
                out.push(text[i..i + length].to_string());
                i += length;
            }
            _ => {
                let length = rest.chars().next().map(char::len_utf8).unwrap_or(1);
                i += length;
            }
        }
    }
    out
}

/// `'s|'t|'re|'ve|'m|'ll|'d`.
fn contraction_len(text: &str) -> Option<usize> {
    for pattern in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if text.starts_with(pattern) {
            return Some(pattern.len());
        }
    }
    None
}

fn consume_run(text: &str, predicate: fn(char) -> bool) -> usize {
    let mut length = 0;
    for c in text.chars() {
        if predicate(c) {
            length += c.len_utf8();
        } else {
            break;
        }
    }
    length
}

/// ` ?<class>+` with a single optional leading space.
fn optional_class_len(text: &str, predicate: fn(char) -> bool) -> Option<usize> {
    if let Some(rest) = text.strip_prefix(' ') {
        let run = consume_run(rest, predicate);
        return (run > 0).then_some(1 + run);
    }
    let run = consume_run(text, predicate);
    (run > 0).then_some(run)
}

fn is_other(c: char) -> bool {
    !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric()
}

/// ` ?[^\s\p{L}\p{N}]+` with a single optional leading space.
fn optional_other_len(text: &str) -> Option<usize> {
    if let Some(rest) = text.strip_prefix(' ') {
        let run = consume_run(rest, is_other);
        return (run > 0).then_some(1 + run);
    }
    let run = consume_run(text, is_other);
    (run > 0).then_some(run)
}

/// `\s+(?!\S)`: the longest whitespace run that is not followed by a
/// non-whitespace character.
fn whitespace_no_suffix_len(text: &str) -> Option<usize> {
    let run = consume_run(text, char::is_whitespace);
    if run == 0 {
        return None;
    }
    if run == text.len() {
        return Some(run);
    }
    // The maximal run is followed by a non-whitespace character, so leave the
    // last whitespace character for the lookahead to accept.
    let without_last = text[..run]
        .char_indices()
        .next_back()
        .map(|(index, _)| index)?;
    (without_last > 0).then_some(without_last)
}

/// `\s+`.
fn whitespace_len(text: &str) -> Option<usize> {
    let run = consume_run(text, char::is_whitespace);
    (run > 0).then_some(run)
}

/// `\w+|[^\w\s]+`, used by the `Whitespace` pre-tokenizer.
fn whitespace_pretokenize(text: &str) -> Vec<String> {
    static PATTERN: OnceLock<Option<Regex>> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| Regex::new(r"\w+|[^\w\s]+").ok());
    match pattern {
        Some(regex) => regex
            .find_iter(text)
            .map(|matched| matched.as_str().to_string())
            .collect(),
        None => vec![text.to_string()],
    }
}

// ---------------------------------------------------------------------------
// Token models
// ---------------------------------------------------------------------------

/// `(left, right) -> (merge rank, merged id)`.
type MergeMap = HashMap<(i64, i64), (usize, i64)>;

#[derive(Debug)]
struct Bpe {
    vocab: HashMap<String, i64>,
    merges: MergeMap,
    unk: Option<i64>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    byte_ids: [i64; 256],
}

impl Bpe {
    fn from_spec(spec: &Value) -> Result<Self> {
        let vocab_spec = spec
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| DecisionError::invalid("BPE model is missing a `vocab` object"))?;
        let mut vocab = HashMap::with_capacity(vocab_spec.len());
        for (token, id) in vocab_spec {
            let id = id
                .as_i64()
                .ok_or_else(|| DecisionError::invalid("BPE vocabulary id is not an integer"))?;
            vocab.insert(token.clone(), id);
        }

        for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if spec.get(key).map(|value| !value.is_null()).unwrap_or(false) {
                return Err(DecisionError::unsupported(format!(
                    "BPE `{key}` is not supported"
                )));
            }
        }

        let merges_spec = spec
            .get("merges")
            .and_then(Value::as_array)
            .ok_or_else(|| DecisionError::invalid("BPE model is missing a `merges` array"))?;
        let mut merges: MergeMap = HashMap::with_capacity(merges_spec.len());
        for (index, merge) in merges_spec.iter().enumerate() {
            let (left, right) = merge_pair(merge)?;
            let left_id = *vocab.get(&left).ok_or_else(|| {
                DecisionError::invalid(format!("merge symbol `{left}` is not in the vocabulary"))
            })?;
            let right_id = *vocab.get(&right).ok_or_else(|| {
                DecisionError::invalid(format!("merge symbol `{right}` is not in the vocabulary"))
            })?;
            let mut merged = left;
            merged.push_str(&right);
            let merged_id = *vocab.get(&merged).ok_or_else(|| {
                DecisionError::invalid(format!("merged symbol `{merged}` is not in the vocabulary"))
            })?;
            merges.insert((left_id, right_id), (index + 1, merged_id));
        }

        let unk = match spec.get("unk_token").and_then(Value::as_str) {
            Some(token) => Some(*vocab.get(token).ok_or_else(|| {
                DecisionError::invalid(format!("unk token `{token}` is not in the vocabulary"))
            })?),
            None => None,
        };

        let mut byte_ids = [-1i64; 256];
        for (byte, slot) in byte_ids.iter_mut().enumerate() {
            let key = format!("<0x{byte:02X}>");
            if let Some(id) = vocab.get(&key) {
                *slot = *id;
            }
        }

        Ok(Self {
            vocab,
            merges,
            unk,
            fuse_unk: spec
                .get("fuse_unk")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            byte_fallback: spec
                .get("byte_fallback")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            ignore_merges: spec
                .get("ignore_merges")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            byte_ids,
        })
    }

    fn tokenize(&self, word: &str) -> Vec<i64> {
        if self.ignore_merges {
            if let Some(id) = self.vocab.get(word) {
                return vec![*id];
            }
        }

        let mut symbols: Vec<i64> = Vec::new();
        let mut pending_unk = false;
        for c in word.chars() {
            let key = c.to_string();
            if let Some(id) = self.vocab.get(&key) {
                if pending_unk {
                    if let Some(unk) = self.unk {
                        symbols.push(unk);
                    }
                    pending_unk = false;
                }
                symbols.push(*id);
                continue;
            }

            if self.byte_fallback {
                let mut fallback = Vec::new();
                let mut complete = true;
                for byte in key.bytes() {
                    let id = self.byte_ids[byte as usize];
                    if id < 0 {
                        complete = false;
                        break;
                    }
                    fallback.push(id);
                }
                if complete {
                    symbols.extend(fallback);
                    continue;
                }
            }

            let Some(unk) = self.unk else {
                continue;
            };
            if pending_unk && !self.fuse_unk {
                symbols.push(unk);
            }
            pending_unk = true;
        }
        if pending_unk {
            if let Some(unk) = self.unk {
                symbols.push(unk);
            }
        }

        self.merge(symbols)
    }

    /// Repeatedly apply the lowest-ranked merge, leftmost first on ties.
    fn merge(&self, mut symbols: Vec<i64>) -> Vec<i64> {
        while symbols.len() > 1 {
            let mut best: Option<(usize, usize, i64)> = None;
            for (index, pair) in symbols.windows(2).enumerate() {
                if let Some((rank, merged)) = self.merges.get(&(pair[0], pair[1])) {
                    if best
                        .map(|(best_rank, ..)| *rank < best_rank)
                        .unwrap_or(true)
                    {
                        best = Some((*rank, index, *merged));
                    }
                }
            }
            let Some((_, index, merged)) = best else {
                break;
            };
            symbols[index] = merged;
            symbols.remove(index + 1);
        }
        symbols
    }
}

fn is_merge_pair(value: &Value) -> Option<(String, String)> {
    match value {
        Value::String(text) => {
            let (left, right) = text.split_once(' ')?;
            Some((left.to_string(), right.to_string()))
        }
        Value::Array(items) => {
            let left = items.first()?.as_str()?.to_string();
            let right = items.get(1)?.as_str()?.to_string();
            Some((left, right))
        }
        _ => None,
    }
}

fn merge_pair(value: &Value) -> Result<(String, String)> {
    is_merge_pair(value).ok_or_else(|| {
        DecisionError::invalid("BPE merge must be a string pair or a two-element array")
    })
}

#[derive(Debug)]
struct WordLevel {
    vocab: HashMap<String, i64>,
    unk: Option<i64>,
}

impl WordLevel {
    fn from_spec(spec: &Value) -> Result<Self> {
        let vocab_spec = spec
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| DecisionError::invalid("WordLevel model is missing a `vocab` object"))?;
        let mut vocab = HashMap::with_capacity(vocab_spec.len());
        for (token, id) in vocab_spec {
            let id = id.as_i64().ok_or_else(|| {
                DecisionError::invalid("WordLevel vocabulary id is not an integer")
            })?;
            vocab.insert(token.clone(), id);
        }
        let unk = match spec.get("unk_token").and_then(Value::as_str) {
            Some(token) => Some(*vocab.get(token).ok_or_else(|| {
                DecisionError::invalid(format!("unk token `{token}` is not in the vocabulary"))
            })?),
            None => None,
        };
        Ok(Self { vocab, unk })
    }

    fn tokenize(&self, word: &str) -> Vec<i64> {
        if let Some(id) = self.vocab.get(word) {
            vec![*id]
        } else if let Some(unk) = self.unk {
            vec![unk]
        } else {
            // The reference errors here; encoding is infallible, so skip instead.
            Vec::new()
        }
    }
}

#[derive(Debug)]
enum TokenModel {
    Bpe(Box<Bpe>),
    WordLevel(Box<WordLevel>),
}

impl TokenModel {
    fn from_spec(spec: &Value) -> Result<Self> {
        let kind = spec
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| DecisionError::invalid("token model is missing a string `type`"))?;
        match kind {
            "BPE" => Ok(Self::Bpe(Box::new(Bpe::from_spec(spec)?))),
            "WordLevel" => Ok(Self::WordLevel(Box::new(WordLevel::from_spec(spec)?))),
            other => Err(DecisionError::unsupported(format!(
                "unsupported tokenizer model: {other}"
            ))),
        }
    }

    fn vocab(&self) -> &HashMap<String, i64> {
        match self {
            Self::Bpe(model) => &model.vocab,
            Self::WordLevel(model) => &model.vocab,
        }
    }

    fn tokenize_word(&self, word: &str) -> Vec<i64> {
        match self {
            Self::Bpe(model) => model.tokenize(word),
            Self::WordLevel(model) => model.tokenize(word),
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Reads the special-token surface form from `tokenizer_config.json`.
///
/// Accepts either a bare string or an object with a `content` field, matching
/// the reference.
fn special_surface(config: &Value, name: &str, default: &str) -> String {
    match config.get(name) {
        Some(Value::String(surface)) => surface.clone(),
        Some(Value::Object(object)) => object
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| default.to_string()),
        _ => default.to_string(),
    }
}

/// A byte-level / Metaspace BPE tokenizer loaded from a checkpoint's
/// `tokenizer/` directory.
#[derive(Debug)]
pub struct BpeTokenizer {
    normalizer: Normalizer,
    pre_tokenizer: PreTokenizer,
    model: TokenModel,
    added_raw: AddedMatcher,
    added_normalized: AddedMatcher,
    mask_token: String,
    cls_token_id: i64,
    sep_token_id: i64,
    pad_token_id: i64,
    mask_token_id: i64,
}

impl BpeTokenizer {
    /// Load `dir/tokenizer.json` (and `dir/tokenizer_config.json` if present).
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when a file is missing or malformed, or when it uses a
    /// normalizer / pre-tokenizer / model this port does not support. A missing
    /// `tokenizer_config.json` falls back to the `[CLS]` / `[SEP]` / `[PAD]` /
    /// `[MASK]` defaults looked up from the vocabulary.
    pub fn from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let tokenizer_path = dir.join("tokenizer.json");
        let spec_text = std::fs::read_to_string(&tokenizer_path).map_err(|error| {
            DecisionError::invalid(format!("cannot read tokenizer.json: {error}"))
        })?;
        let spec: Value = serde_json::from_str(&spec_text)
            .map_err(|error| DecisionError::invalid(format!("invalid tokenizer.json: {error}")))?;

        let config_path = dir.join("tokenizer_config.json");
        let config = match std::fs::read_to_string(&config_path) {
            Ok(text) => serde_json::from_str::<Value>(&text).map_err(|error| {
                DecisionError::invalid(format!("invalid tokenizer_config.json: {error}"))
            })?,
            Err(_) => Value::Null,
        };

        Self::from_spec(&spec, &config)
    }

    /// Build a tokenizer from already-parsed `tokenizer.json` and
    /// `tokenizer_config.json` values.
    fn from_spec(spec: &Value, config: &Value) -> Result<Self> {
        let model_spec = spec
            .get("model")
            .ok_or_else(|| DecisionError::invalid("tokenizer.json is missing `model`"))?;
        let model = TokenModel::from_spec(model_spec)?;

        let mut token_to_id = model.vocab().clone();
        let mut added_raw = Vec::new();
        let mut added_normalized = Vec::new();
        if let Some(added) = spec.get("added_tokens").and_then(Value::as_array) {
            for entry in added {
                let content = entry
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        DecisionError::invalid("added token is missing string `content`")
                    })?;
                let id = entry
                    .get("id")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| DecisionError::invalid("added token is missing integer `id`"))?;
                let token = AddedToken {
                    content: content.to_string(),
                    id,
                    lstrip: entry
                        .get("lstrip")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    rstrip: entry
                        .get("rstrip")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    single_word: entry
                        .get("single_word")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                };
                token_to_id.insert(content.to_string(), id);
                if entry
                    .get("normalized")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    added_normalized.push(token);
                } else {
                    added_raw.push(token);
                }
            }
        }

        let normalizer = Normalizer::from_spec(spec.get("normalizer"))?;
        let pre_tokenizer = PreTokenizer::from_spec(spec.get("pre_tokenizer"))?;

        let specs = [
            ("cls_token", "[CLS]"),
            ("sep_token", "[SEP]"),
            ("pad_token", "[PAD]"),
            ("mask_token", "[MASK]"),
        ];
        let mut specials = Vec::with_capacity(specs.len());
        for (name, default) in specs {
            let surface = special_surface(config, name, default);
            let id = *token_to_id.get(&surface).ok_or_else(|| {
                DecisionError::invalid(format!(
                    "tokenizer is missing a valid `{name}` (`{surface}` is not in the vocabulary)"
                ))
            })?;
            specials.push((surface, id));
        }

        Ok(Self {
            normalizer,
            pre_tokenizer,
            model,
            added_raw: AddedMatcher::new(&added_raw),
            added_normalized: AddedMatcher::new(&added_normalized),
            mask_token: specials[3].0.clone(),
            cls_token_id: specials[0].1,
            sep_token_id: specials[1].1,
            pad_token_id: specials[2].1,
            mask_token_id: specials[3].1,
        })
    }

    /// Encode `text` into token ids, without adding special tokens.
    ///
    /// Mirrors `tokenizers.Tokenizer.encode(text, add_special_tokens=False)`
    /// as used by the reference: split on raw added tokens, normalize, split on
    /// normalized added tokens, pre-tokenize and tokenize each word.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let mut ids = Vec::new();
        let mut first = true;
        for piece in split_added(&self.added_raw, text) {
            match piece {
                Piece::Id(id) => ids.push(id),
                Piece::Text(piece) => {
                    let normalized = self.normalizer.apply(&piece);
                    for sub in split_added(&self.added_normalized, &normalized) {
                        match sub {
                            Piece::Id(id) => ids.push(id),
                            Piece::Text(sub) => {
                                if sub.is_empty() {
                                    continue;
                                }
                                for word in pretokenize(&self.pre_tokenizer, &sub, first) {
                                    ids.extend(self.model.tokenize_word(&word));
                                }
                            }
                        }
                    }
                }
            }
            first = false;
        }
        ids
    }
}

impl crate::prompt::Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<i64> {
        BpeTokenizer::encode(self, text)
    }

    fn mask_token(&self) -> &str {
        &self.mask_token
    }

    fn mask_token_id(&self) -> i64 {
        self.mask_token_id
    }

    fn cls_token_id(&self) -> i64 {
        self.cls_token_id
    }

    fn sep_token_id(&self) -> i64 {
        self.sep_token_id
    }

    fn pad_token_id(&self) -> i64 {
        self.pad_token_id
    }
}
