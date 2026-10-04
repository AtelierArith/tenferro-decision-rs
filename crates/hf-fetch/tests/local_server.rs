//! End-to-end fetch against a local HTTP server, without touching the network.
//!
//! Exercises the real download path: the model-info API, file resolution,
//! HTTP redirects (as Hugging Face uses for LFS), the `models--org--name`
//! snapshot layout, `refs/<revision>`, and offline reuse from the cache.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::thread;

use hf_fetch::{CheckpointSpec, FileRule, Hub};

enum Reply {
    Ok(&'static str),
    Redirect(&'static str),
}

fn spawn_server(routes: Vec<(&'static str, Reply)>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            if let Err(error) = handle(stream, &routes) {
                eprintln!("test server: {error}");
            }
        }
    });
    format!("http://127.0.0.1:{port}")
}

fn handle(mut stream: TcpStream, routes: &[(&str, Reply)]) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
            break;
        }
    }
    let path = request_line.split_whitespace().nth(1).unwrap_or("");
    let reply = routes
        .iter()
        .find(|(route, _)| *route == path)
        .map(|(_, reply)| reply);
    match reply {
        Some(Reply::Redirect(location)) => {
            write!(
                stream,
                "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )?;
        }
        Some(Reply::Ok(body)) => {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )?;
        }
        None => {
            write!(
                stream,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )?;
        }
    }
    stream.flush()
}

fn temp_cache(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("hf-fetch-it-{name}-{}", std::process::id()))
}

#[test]
fn fetches_snapshot_and_reuses_cache_offline() {
    let base = spawn_server(vec![
        (
            "/api/models/test/repo/revision/main",
            Reply::Ok(
                r#"{"sha":"deadbeef","siblings":[{"rfilename":"config.json"},{"rfilename":"tokenizer/tokenizer.json"},{"rfilename":"video.mp4"},{"rfilename":"big.bin"}]}"#,
            ),
        ),
        (
            "/test/repo/resolve/deadbeef/config.json",
            Reply::Ok(r#"{"ok":true}"#),
        ),
        (
            "/test/repo/resolve/deadbeef/tokenizer/tokenizer.json",
            Reply::Ok(r#"{"tok":1}"#),
        ),
        (
            "/test/repo/resolve/deadbeef/big.bin",
            Reply::Redirect("/cdn/big.bin"),
        ),
        ("/cdn/big.bin", Reply::Ok("BINARY")),
    ]);

    let cache = temp_cache("snapshot");
    let _ = std::fs::remove_dir_all(&cache);

    let spec = CheckpointSpec {
        repo: "test/repo".to_string(),
        revision: "main".to_string(),
        subfolder: None,
        required: vec!["config.json".to_string(), "big.bin".to_string()],
        rules: vec![
            FileRule::Exact("config.json".to_string()),
            FileRule::Prefix("tokenizer/".to_string()),
            FileRule::Exact("big.bin".to_string()),
        ],
    };

    let hub = Hub::new(&base, &cache, None, false);
    let dir = hub.resolve(&spec).unwrap();
    assert_eq!(dir, cache.join("models--test--repo/snapshots/deadbeef"));
    assert_eq!(
        std::fs::read_to_string(dir.join("config.json")).unwrap(),
        r#"{"ok":true}"#
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("tokenizer/tokenizer.json")).unwrap(),
        r#"{"tok":1}"#
    );
    // The redirect target was followed.
    assert_eq!(
        std::fs::read_to_string(dir.join("big.bin")).unwrap(),
        "BINARY"
    );
    // Unmatched files are not fetched.
    assert!(!dir.join("video.mp4").exists());
    assert_eq!(
        std::fs::read_to_string(cache.join("models--test--repo/refs/main")).unwrap(),
        "deadbeef"
    );

    // Offline reuse: no server needed.
    let offline = Hub::new("http://127.0.0.1:1", &cache, None, true);
    assert_eq!(offline.resolve(&spec).unwrap(), dir);

    std::fs::remove_dir_all(&cache).unwrap();
}

#[test]
fn offline_without_cache_errors() {
    let cache = temp_cache("empty");
    let _ = std::fs::remove_dir_all(&cache);
    let hub = Hub::new("http://127.0.0.1:1", &cache, None, true);
    let spec = CheckpointSpec {
        repo: "test/repo".to_string(),
        revision: "main".to_string(),
        subfolder: None,
        required: vec!["config.json".to_string()],
        rules: vec![FileRule::Exact("config.json".to_string())],
    };
    assert!(matches!(
        hub.resolve(&spec),
        Err(hf_fetch::HubError::Offline { .. })
    ));
}

#[test]
fn subfolder_scopes_the_checkpoint() {
    let base = spawn_server(vec![
        (
            "/api/models/test/repo/revision/main",
            Reply::Ok(
                r#"{"sha":"cafe","siblings":[{"rfilename":"a/config.json"},{"rfilename":"b/config.json"},{"rfilename":"a/data.bin"}]}"#,
            ),
        ),
        (
            "/test/repo/resolve/cafe/a/config.json",
            Reply::Ok(r#"{"a":1}"#),
        ),
        ("/test/repo/resolve/cafe/a/data.bin", Reply::Ok("A")),
    ]);

    let cache = temp_cache("subfolder");
    let _ = std::fs::remove_dir_all(&cache);
    let spec = CheckpointSpec {
        repo: "test/repo".to_string(),
        revision: "main".to_string(),
        subfolder: Some("a".to_string()),
        required: vec!["config.json".to_string()],
        rules: vec![
            FileRule::Exact("config.json".to_string()),
            FileRule::Exact("data.bin".to_string()),
        ],
    };
    let hub = Hub::new(&base, &cache, None, false);
    let dir = hub.resolve(&spec).unwrap();
    assert_eq!(dir, cache.join("models--test--repo/snapshots/cafe/a"));
    assert!(dir.join("config.json").is_file());
    assert!(dir.join("data.bin").is_file());
    assert!(
        !cache
            .join("models--test--repo/snapshots/cafe/b/config.json")
            .exists()
    );
    std::fs::remove_dir_all(&cache).unwrap();
}
