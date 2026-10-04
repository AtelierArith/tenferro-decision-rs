//! Hugging Face Hub checkpoint fetcher.
//!
//! Mirrors the cache layout and environment variables of the Julia runtimes
//! (`extern/Laya.jl/src/agent.jl`, `extern/JeffClient.jl/src/hub.jl`) so a Rust
//! download is interchangeable with a Python/Julia one:
//!
//! - files land in `models--<org>--<name>/snapshots/<commit>/…`
//! - `refs/<revision>` records the resolved commit
//! - `HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN`, `HF_HUB_OFFLINE`
//!   are respected
//! - partial downloads never appear in the snapshot (temp dir + atomic rename)
//!
//! Only the checkpoint files described by a [`CheckpointSpec`] are fetched; a
//! repository that also holds videos or other checkpoints is not mirrored
//! wholesale. The crate is independent of the inference crates: it returns a
//! local snapshot directory suitable for `LayaEngine::load` /
//! `jeff_infer::checkpoint::load_checkpoint`.

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

/// Default Hugging Face endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Errors returned by the fetcher.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    /// The repository id is not `org/name`.
    #[error("invalid Hugging Face repository id: {0:?}")]
    InvalidRepo(String),
    /// The repository/ref is not cached and offline mode is set.
    #[error("{repo}@{revision} is not cached and offline mode is set")]
    Offline {
        /// Repository id.
        repo: String,
        /// Requested revision.
        revision: String,
    },
    /// No file in the repository matched the checkpoint spec.
    #[error("{repo}@{revision} has no matching checkpoint files")]
    NoFiles {
        /// Repository id.
        repo: String,
        /// Requested revision.
        revision: String,
    },
    /// A repository file name could escape the snapshot directory.
    #[error("unsafe file path from the hub: {0:?}")]
    UnsafePath(String),
    /// The downloaded snapshot is missing required files.
    #[error("incomplete checkpoint at {0}")]
    Incomplete(PathBuf),
    /// Network or HTTP failure.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    /// Malformed JSON from the hub API.
    #[error("hub JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Filesystem failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, HubError>;

/// A file-selection rule used by [`CheckpointSpec`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileRule {
    /// Match one exact repository file name.
    Exact(String),
    /// Match every file under a directory prefix (`"tokenizer/"`).
    Prefix(String),
    /// Match a `*`/`?` glob over the whole (possibly nested) file name.
    Glob(String),
}

impl FileRule {
    /// Whether `name` (a repository-relative file name) matches this rule.
    pub fn matches(&self, name: &str) -> bool {
        match self {
            FileRule::Exact(want) => name == want,
            FileRule::Prefix(prefix) => name.starts_with(prefix),
            FileRule::Glob(pattern) => glob_match(pattern, name),
        }
    }
}

/// Checkpoint file selection and the repository/ref to fetch.
#[derive(Clone, Debug)]
pub struct CheckpointSpec {
    /// Hugging Face repository id (`org/name`).
    pub repo: String,
    /// Revision (`main`, a tag, or a commit sha).
    pub revision: String,
    /// Optional subfolder inside the repository.
    pub subfolder: Option<String>,
    /// Files that must exist for the snapshot to be considered complete.
    pub required: Vec<String>,
    /// Include rules; a repository file is fetched iff any rule matches.
    pub rules: Vec<FileRule>,
}

impl CheckpointSpec {
    /// Whether `name` should be fetched.
    pub fn matches(&self, name: &str) -> bool {
        self.rules.iter().any(|rule| rule.matches(name))
    }

    /// The default Laya checkpoint (`Laya.load`'s default).
    ///
    /// Repository `convaiinnovations/laya`, revision `main`. Files:
    /// `model.safetensors`, the agent/encoder configs, `mlx_config.json`, and
    /// `tokenizer/`.
    pub fn laya() -> Self {
        Self {
            repo: "convaiinnovations/laya".to_string(),
            revision: "main".to_string(),
            subfolder: None,
            required: vec![
                "model.safetensors".to_string(),
                "rl_agent_config.json".to_string(),
                "encoder/config.json".to_string(),
            ],
            rules: vec![
                FileRule::Exact("model.safetensors".to_string()),
                FileRule::Exact("rl_agent_config.json".to_string()),
                FileRule::Exact("encoder/config.json".to_string()),
                FileRule::Exact("mlx_config.json".to_string()),
                FileRule::Prefix("tokenizer/".to_string()),
            ],
        }
    }

    /// The pinned Jeff checkpoint from `JeffClient.jl/docs/src/models.md`.
    ///
    /// Repository `mstrasser/Jeff-Qwen3.5-0.8B` at commit
    /// `0f212b3e72acb4dde3f7da61e925d6ab7f819990` (about 1.7 GB).
    pub fn jeff() -> Self {
        Self {
            repo: "mstrasser/Jeff-Qwen3.5-0.8B".to_string(),
            revision: "0f212b3e72acb4dde3f7da61e925d6ab7f819990".to_string(),
            subfolder: None,
            required: vec![
                "config.json".to_string(),
                "decision_config.json".to_string(),
                "readout.safetensors".to_string(),
                "tokenizer.json".to_string(),
                "tokenizer_config.json".to_string(),
            ],
            rules: vec![
                FileRule::Exact("config.json".to_string()),
                FileRule::Exact("decision_config.json".to_string()),
                FileRule::Exact("readout.safetensors".to_string()),
                FileRule::Exact("tokenizer.json".to_string()),
                FileRule::Exact("tokenizer_config.json".to_string()),
                FileRule::Exact("chat_template.jinja".to_string()),
                FileRule::Exact("processor_config.json".to_string()),
                FileRule::Exact("model.safetensors.index.json".to_string()),
                FileRule::Exact("LICENSE".to_string()),
                FileRule::Exact("NOTICE".to_string()),
                FileRule::Glob("model*.safetensors".to_string()),
            ],
        }
    }

    /// Look up a built-in spec by name (`"laya"` or `"jeff"`).
    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "laya" => Some(Self::laya()),
            "jeff" => Some(Self::jeff()),
            _ => None,
        }
    }
}

/// The configured Hub client.
#[derive(Clone, Debug)]
pub struct Hub {
    /// Endpoint base URL, without a trailing slash.
    pub endpoint: String,
    /// Hub cache root (`…/huggingface/hub`).
    pub cache: PathBuf,
    /// Bearer token for private/gated repositories.
    pub token: Option<String>,
    /// Do not download; only use cached snapshots.
    pub offline: bool,
}

impl Hub {
    /// Build a client from the process environment.
    pub fn from_env() -> Self {
        let get = |key: &str| env::var(key).ok();
        Self::from_env_with(&get, home_dir())
    }

    /// Build a client from an environment lookup and a home directory.
    pub fn from_env_with(get: &dyn Fn(&str) -> Option<String>, home: Option<PathBuf>) -> Self {
        let endpoint = get("HF_ENDPOINT")
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string())
            .trim_end_matches('/')
            .to_string();
        let cache =
            resolve_cache_dir(get, home).unwrap_or_else(|| PathBuf::from(".cache/huggingface/hub"));
        let token = get("HF_TOKEN").filter(|value| !value.is_empty());
        let offline = get("HF_HUB_OFFLINE")
            .map(|value| parse_bool(&value))
            .unwrap_or(false);
        Self {
            endpoint,
            cache,
            token,
            offline,
        }
    }

    /// Build a client with explicit settings.
    pub fn new(
        endpoint: impl Into<String>,
        cache: impl Into<PathBuf>,
        token: Option<String>,
        offline: bool,
    ) -> Self {
        Self {
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            cache: cache.into(),
            token,
            offline,
        }
    }

    /// Resolve `spec` to a local directory, downloading if needed.
    ///
    /// Returns the directory that directly contains the checkpoint files
    /// (including `spec.subfolder` when set).
    pub fn resolve(&self, spec: &CheckpointSpec) -> Result<PathBuf> {
        validate_repo(&spec.repo)?;
        let prefix = prefix_for(spec.subfolder.as_deref())?;
        if let Some(dir) = cached_snapshot(&self.cache, &spec.repo, &spec.revision) {
            let target = checkpoint_dir(&dir, &prefix);
            if has_required(&target, spec) {
                return Ok(target);
            }
        }
        if self.offline {
            return Err(HubError::Offline {
                repo: spec.repo.clone(),
                revision: spec.revision.clone(),
            });
        }
        self.download(spec, &prefix)
    }

    fn download(&self, spec: &CheckpointSpec, prefix: &str) -> Result<PathBuf> {
        let client = reqwest::blocking::Client::builder()
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()?;
        let info = self.model_info(&client, &spec.repo, &spec.revision)?;
        let commit = info.sha;
        let files: Vec<String> = info
            .siblings
            .into_iter()
            .map(|sibling| sibling.rfilename)
            .filter(|name| under_prefix(name, prefix) && spec.matches(strip_prefix(name, prefix)))
            .collect();
        if files.is_empty() {
            return Err(HubError::NoFiles {
                repo: spec.repo.clone(),
                revision: spec.revision.clone(),
            });
        }

        let root = repo_root(&self.cache, &spec.repo);
        let snapshot = root.join("snapshots").join(&commit);
        let temp = root.join(format!("download-{}", unique_suffix()));
        fs::create_dir_all(&temp)?;

        let outcome = (|| -> Result<()> {
            for file in &files {
                safe_relative(file)?;
                let destination = snapshot.join(file);
                if destination.is_file() {
                    continue;
                }
                let staged = temp.join(file);
                if let Some(parent) = staged.parent() {
                    fs::create_dir_all(parent)?;
                }
                self.fetch_file(&client, &spec.repo, &commit, file, &staged)?;
                if let Some(parent) = destination.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::rename(&staged, &destination)?;
            }
            Ok(())
        })();
        let _ = fs::remove_dir_all(&temp);
        outcome?;

        let ref_path = root.join("refs").join(&spec.revision);
        if let Some(parent) = ref_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut ref_file = fs::File::create(&ref_path)?;
        ref_file.write_all(commit.as_bytes())?;

        let target = checkpoint_dir(&snapshot, prefix);
        if !has_required(&target, spec) {
            return Err(HubError::Incomplete(target));
        }
        Ok(target)
    }

    fn model_info(
        &self,
        client: &reqwest::blocking::Client,
        repo: &str,
        revision: &str,
    ) -> Result<ModelInfo> {
        let url = format!(
            "{}/api/models/{}/revision/{}",
            self.endpoint,
            repo,
            escape_path(revision)
        );
        let response = self
            .authorize(client.get(&url))
            .send()?
            .error_for_status()?;
        let body = response.bytes()?;
        Ok(serde_json::from_slice::<ModelInfo>(&body)?)
    }

    fn fetch_file(
        &self,
        client: &reqwest::blocking::Client,
        repo: &str,
        commit: &str,
        file: &str,
        destination: &Path,
    ) -> Result<()> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint,
            repo,
            commit,
            escape_path(file)
        );
        let mut response = self
            .authorize(client.get(&url))
            .send()?
            .error_for_status()?;
        let mut output = fs::File::create(destination)?;
        response.copy_to(&mut output)?;
        output.flush()?;
        Ok(())
    }

    fn authorize(
        &self,
        request: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        match &self.token {
            Some(token) => {
                request.header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
            }
            None => request,
        }
    }
}

#[derive(Deserialize)]
struct ModelInfo {
    sha: String,
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

// ------------------------------------------------------------------- helpers

/// Resolve the Hub cache root the way the Julia runtimes do.
pub fn resolve_cache_dir(
    get: &dyn Fn(&str) -> Option<String>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(value) = get("HF_HUB_CACHE") {
        return Some(PathBuf::from(value));
    }
    if let Some(value) = get("HF_HOME") {
        return Some(PathBuf::from(value).join("hub"));
    }
    home.map(|home| home.join(".cache").join("huggingface").join("hub"))
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn validate_repo(repo: &str) -> Result<()> {
    let mut parts = repo.split('/');
    let (Some(namespace), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err(HubError::InvalidRepo(repo.to_string()));
    };
    let valid = |part: &str| {
        !part.is_empty()
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
    };
    if valid(namespace) && valid(name) {
        Ok(())
    } else {
        Err(HubError::InvalidRepo(repo.to_string()))
    }
}

fn prefix_for(subfolder: Option<&str>) -> Result<String> {
    match subfolder {
        None => Ok(String::new()),
        Some(subfolder) => {
            let trimmed = subfolder.trim_matches('/');
            if trimmed.is_empty() {
                return Ok(String::new());
            }
            safe_relative(trimmed)?;
            Ok(format!("{trimmed}/"))
        }
    }
}

fn checkpoint_dir(snapshot: &Path, prefix: &str) -> PathBuf {
    if prefix.is_empty() {
        snapshot.to_path_buf()
    } else {
        snapshot.join(prefix.trim_end_matches('/'))
    }
}

fn under_prefix(name: &str, prefix: &str) -> bool {
    prefix.is_empty() || name.starts_with(prefix)
}

fn strip_prefix<'a>(name: &'a str, prefix: &str) -> &'a str {
    if prefix.is_empty() {
        name
    } else {
        name.strip_prefix(prefix).unwrap_or(name)
    }
}

fn has_required(dir: &Path, spec: &CheckpointSpec) -> bool {
    spec.required.iter().all(|file| dir.join(file).is_file())
}

fn repo_root(cache: &Path, repo: &str) -> PathBuf {
    cache.join(format!("models--{}", repo.replace('/', "--")))
}

fn cached_snapshot(cache: &Path, repo: &str, revision: &str) -> Option<PathBuf> {
    let root = repo_root(cache, repo);
    let commit = fs::read_to_string(root.join("refs").join(revision))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| revision.to_string());
    let dir = root.join("snapshots").join(commit);
    dir.is_dir().then_some(dir)
}

/// Reject file names that could escape the snapshot directory.
pub fn safe_relative(path: &str) -> Result<()> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.contains('\0')
        || path.split('/').any(|part| matches!(part, "" | "." | ".."))
    {
        return Err(HubError::UnsafePath(path.to_string()));
    }
    Ok(())
}

/// Percent-encode each path segment, keeping `[A-Za-z0-9_.~-]`.
pub fn escape_path(path: &str) -> String {
    path.split('/')
        .map(escape_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn escape_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'~' | b'-') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0, 0);
    let mut star: Option<usize> = None;
    let mut mark = 0;
    while ti < text.len() {
        if pi < pattern.len() && (pattern[pi] == '?' || pattern[pi] == text[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pattern.len() && pattern[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(star_index) = star {
            pi = star_index + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < pattern.len() && pattern[pi] == '*' {
        pi += 1;
    }
    pi == pattern.len()
}

fn unique_suffix() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("{}-{count}-{nanos}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_nested_and_flat() {
        assert!(glob_match(
            "model*.safetensors",
            "model-00001-of-00002.safetensors"
        ));
        assert!(glob_match("model*.safetensors", "model.safetensors"));
        assert!(!glob_match("model*.safetensors", "readout.safetensors"));
        assert!(glob_match("*.json", "encoder/config.json"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "abbc"));
    }

    #[test]
    fn presets_filter_the_expected_files() {
        let laya = CheckpointSpec::laya();
        assert!(laya.matches("model.safetensors"));
        assert!(laya.matches("tokenizer/tokenizer.json"));
        assert!(laya.matches("rl_agent_config.json"));
        assert!(!laya.matches("assets/promo.mp4"));

        let jeff = CheckpointSpec::jeff();
        assert!(jeff.matches("model-00001-of-00002.safetensors"));
        assert!(jeff.matches("readout.safetensors"));
        assert!(jeff.matches("tokenizer.json"));
        assert!(!jeff.matches("video.mp4"));
    }

    #[test]
    fn cache_dir_honors_env_precedence() {
        let get = |key: &str| match key {
            "HF_HUB_CACHE" => Some("/custom/hub".to_string()),
            "HF_HOME" => Some("/ignored".to_string()),
            _ => None,
        };
        assert_eq!(
            resolve_cache_dir(&get, Some(PathBuf::from("/home/u"))),
            Some(PathBuf::from("/custom/hub"))
        );

        let get = |key: &str| match key {
            "HF_HOME" => Some("/home/u/.hf".to_string()),
            _ => None,
        };
        assert_eq!(
            resolve_cache_dir(&get, Some(PathBuf::from("/home/u"))),
            Some(PathBuf::from("/home/u/.hf/hub"))
        );

        let get = |_key: &str| None;
        assert_eq!(
            resolve_cache_dir(&get, Some(PathBuf::from("/home/u"))),
            Some(PathBuf::from("/home/u/.cache/huggingface/hub"))
        );
    }

    #[test]
    fn rejects_unsafe_paths_and_repos() {
        assert!(safe_relative("../etc/passwd").is_err());
        assert!(safe_relative("/abs").is_err());
        assert!(safe_relative("a/../../b").is_err());
        assert!(safe_relative("encoder/config.json").is_ok());
        assert!(validate_repo("convaiinnovations/laya").is_ok());
        assert!(validate_repo("mstrasser/Jeff-Qwen3.5-0.8B").is_ok());
        assert!(validate_repo("no-slash").is_err());
        assert!(validate_repo("a/b/c").is_err());
    }

    #[test]
    fn escape_encodes_special_characters() {
        assert_eq!(escape_path("encoder/config.json"), "encoder/config.json");
        assert_eq!(escape_path("a b/c"), "a%20b/c");
    }

    #[test]
    fn cached_snapshot_reads_refs() {
        let dir = std::env::temp_dir().join(format!("hf-fetch-test-{}", unique_suffix()));
        let root = repo_root(&dir, "org/name");
        fs::create_dir_all(root.join("snapshots/deadbeef")).unwrap();
        fs::create_dir_all(root.join("refs")).unwrap();
        fs::write(root.join("refs/main"), "deadbeef").unwrap();
        assert_eq!(
            cached_snapshot(&dir, "org/name", "main"),
            Some(root.join("snapshots/deadbeef"))
        );
        assert_eq!(cached_snapshot(&dir, "org/name", "missing"), None);
        fs::remove_dir_all(&dir).unwrap();
    }
}
