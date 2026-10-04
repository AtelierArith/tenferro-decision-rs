//! Benchmark harness skeleton and reproducibility metadata capture.
//!
//! Every benchmark report the package publishes should record how it was built
//! and on what machine (`docs/agents/specs/docs/05_TESTING_BENCHMARKS.md`).
//! [`BenchMetadata::capture`] collects that once; [`BenchMetadata::to_json`]
//! serializes it next to results.
//!
//! The actual benchmarks live in `benches/` and are added per milestone
//! (Phase 1 primitives, Phase 3 Laya CPU, and so on).

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Machine and build provenance for a benchmark run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchMetadata {
    /// `git rev-parse HEAD`, when available.
    pub git_rev: Option<String>,
    /// `rustc --version`.
    pub rustc: Option<String>,
    /// `cargo --version`.
    pub cargo: Option<String>,
    /// Target triple reported by `rustc -vV`.
    pub target: Option<String>,
    /// Build profile label supplied by the caller (e.g. `release`).
    pub profile: Option<String>,
    /// `std::env::consts::OS`.
    pub os: String,
    /// `std::env::consts::ARCH`.
    pub arch: String,
    /// Logical CPU count.
    pub cpu_count: usize,
    /// Capture time, seconds since the Unix epoch.
    pub timestamp_unix: u64,
}

impl BenchMetadata {
    /// Capture build and machine metadata, shelling out best-effort.
    pub fn capture() -> Self {
        Self {
            git_rev: command_stdout("git", &["rev-parse", "HEAD"]),
            rustc: command_stdout("rustc", &["--version"]),
            cargo: command_stdout("cargo", &["--version"]),
            target: rustc_target(),
            profile: std::env::var("PROFILE").ok(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0),
            timestamp_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Serialize the metadata as pretty JSON.
    pub fn to_json(&self) -> String {
        let value = serde_json::json!({
            "git_rev": self.git_rev,
            "rustc": self.rustc,
            "cargo": self.cargo,
            "target": self.target,
            "profile": self.profile,
            "os": self.os,
            "arch": self.arch,
            "cpu_count": self.cpu_count,
            "timestamp_unix": self.timestamp_unix,
        });
        serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string())
    }
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn rustc_target() -> Option<String> {
    let output = Command::new("rustc").args(["-vV"]).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("host: ").map(str::to_string))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_core_fields() {
        let metadata = BenchMetadata::capture();
        assert!(!metadata.os.is_empty());
        assert!(!metadata.arch.is_empty());
        assert!(metadata.timestamp_unix > 0);
        let json = metadata.to_json();
        assert!(json.contains("\"os\""));
    }
}
