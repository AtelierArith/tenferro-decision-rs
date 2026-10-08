//! `hf-fetch` — download a checkpoint snapshot from the Hugging Face Hub.
//!
//! ```text
//! hf-fetch laya
//! hf-fetch jeff --offline
//! hf-fetch org/name --revision <sha> --include 'model*.safetensors' --include '*.json'
//! ```
//!
//! Prints the local checkpoint directory on success.

use std::path::PathBuf;
use std::process::ExitCode;

use hf_fetch::{CheckpointSpec, FileRule, Hub};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("hf-fetch: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut target: Option<String> = None;
    let mut revision: Option<String> = None;
    let mut subfolder: Option<String> = None;
    let mut cache: Option<PathBuf> = None;
    let mut endpoint: Option<String> = None;
    let mut token: Option<String> = None;
    let mut offline = false;
    let mut progress = false;
    let mut includes: Vec<String> = Vec::new();
    let mut required: Vec<String> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "--offline" => offline = true,
            "--progress" => progress = true,
            "--revision" => revision = Some(next_value(&mut args, "--revision")?),
            "--subfolder" => subfolder = Some(next_value(&mut args, "--subfolder")?),
            "--cache" => cache = Some(PathBuf::from(next_value(&mut args, "--cache")?)),
            "--endpoint" => endpoint = Some(next_value(&mut args, "--endpoint")?),
            "--token" => token = Some(next_value(&mut args, "--token")?),
            "--include" => includes.push(next_value(&mut args, "--include")?),
            "--required" => required.push(next_value(&mut args, "--required")?),
            value if value.starts_with('-') => {
                return Err(format!("unknown option {value:?}; try --help"));
            }
            value => {
                if target.is_some() {
                    return Err("only one repository or preset may be given".to_string());
                }
                target = Some(value.to_string());
            }
        }
    }

    let target = target.ok_or_else(|| "missing repository or preset; try --help".to_string())?;

    let mut spec = match CheckpointSpec::preset(&target) {
        Some(spec) => spec,
        None => generic_spec(&target, includes)?,
    };
    if let Some(revision) = revision {
        spec.revision = revision;
    }
    if subfolder.is_some() {
        spec.subfolder = subfolder;
    }
    if !required.is_empty() {
        spec.required = required;
    }

    let mut hub = Hub::from_env();
    if let Some(cache) = cache {
        hub.cache = cache;
    }
    if let Some(endpoint) = endpoint {
        hub.endpoint = endpoint.trim_end_matches('/').to_string();
    }
    if let Some(token) = token {
        hub.token = Some(token);
    }
    if offline {
        hub.offline = true;
    }

    let mut last_reported = 0u64;
    let mut last_file = String::new();
    let dir = hub
        .resolve_with_progress(&spec, |event| {
            let new_file = last_file != event.file;
            if new_file {
                last_file = event.file.to_string();
                last_reported = 0;
            }
            if progress
                && (new_file
                    || event.downloaded == 0
                    || event.downloaded < last_reported
                    || event.downloaded.saturating_sub(last_reported) >= 8 * 1024 * 1024
                    || event.total == Some(event.downloaded))
            {
                last_reported = event.downloaded;
                match event.total {
                    Some(total) => {
                        eprintln!("{}: {} / {total} bytes", event.file, event.downloaded)
                    }
                    None => eprintln!("{}: {} bytes", event.file, event.downloaded),
                }
            }
        })
        .map_err(|error| error.to_string())?;
    println!("{}", dir.display());
    Ok(())
}

fn generic_spec(repo: &str, includes: Vec<String>) -> Result<CheckpointSpec, String> {
    let rules = if includes.is_empty() {
        vec![
            FileRule::Glob("*.safetensors".to_string()),
            FileRule::Glob("*.json".to_string()),
            FileRule::Prefix("tokenizer/".to_string()),
        ]
    } else {
        includes
            .into_iter()
            .map(|pattern| {
                if let Some(prefix) = pattern.strip_suffix("/**") {
                    FileRule::Prefix(format!("{prefix}/"))
                } else if pattern.contains('*') || pattern.contains('?') {
                    FileRule::Glob(pattern)
                } else {
                    FileRule::Exact(pattern)
                }
            })
            .collect()
    };
    Ok(CheckpointSpec {
        repo: repo.to_string(),
        revision: "main".to_string(),
        subfolder: None,
        required: Vec::new(),
        rules,
    })
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "Usage: hf-fetch <preset|org/name> [options]\n\
         \n\
         Presets: laya, jeff\n\
         \n\
         Options:\n\
         \x20 --revision <rev>      revision/tag/commit (default: main, or the preset's pin)\n\
         \x20 --subfolder <dir>     checkpoint subfolder inside the repository\n\
         \x20 --include <pattern>   file rule for a generic repo (repeatable; 'prefix/**', glob, or exact)\n\
         \x20 --required <file>     file that must exist (repeatable)\n\
         \x20 --offline             only use cached snapshots\n\
         \x20 --progress            report download byte counts on stderr\n\
         \x20 --cache <dir>         override the Hub cache root\n\
         \x20 --endpoint <url>      override HF_ENDPOINT\n\
         \x20 --token <token>       override HF_TOKEN\n\
         \n\
         Prints the resolved checkpoint directory."
    );
}
