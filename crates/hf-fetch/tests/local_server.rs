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
    spawn_authenticated_server(routes, None)
}

fn spawn_authenticated_server(
    routes: Vec<(&'static str, Reply)>,
    expected_token: Option<&'static str>,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            if let Err(error) = handle(stream, &routes, expected_token) {
                eprintln!("test server: {error}");
            }
        }
    });
    format!("http://127.0.0.1:{port}")
}

fn handle(
    mut stream: TcpStream,
    routes: &[(&str, Reply)],
    expected_token: Option<&str>,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut line = String::new();
    let mut authorization = None;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("authorization") {
                authorization = Some(value.trim().to_string());
            }
        }
    }
    if let Some(token) = expected_token {
        if authorization.as_deref() != Some(format!("Bearer {token}").as_str()) {
            write!(
                stream,
                "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )?;
            return stream.flush();
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
    let mut events = Vec::new();
    let dir = hub
        .resolve_with_progress(&spec, |event| {
            events.push((event.file.to_string(), event.downloaded, event.total));
        })
        .unwrap();
    for (file, size) in [
        ("config.json", 11),
        ("tokenizer/tokenizer.json", 9),
        ("big.bin", 6),
    ] {
        let file_events: Vec<_> = events.iter().filter(|event| event.0 == file).collect();
        assert_eq!(file_events.first().unwrap().1, 0);
        assert_eq!(file_events.last().unwrap().1, size);
        assert!(file_events.iter().all(|event| event.2 == Some(size)));
        assert!(file_events.windows(2).all(|pair| pair[0].1 <= pair[1].1));
    }
    let cached = hub
        .resolve_with_progress(&spec, |_| panic!("cached files must not report downloads"))
        .unwrap();
    assert_eq!(cached, dir);
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

#[test]
fn private_repository_uses_hf_token_for_api_files_and_redirects() {
    // Synthetic credential only. Requiring it on every route verifies that the
    // model-info API, resolve URL, and same-origin LFS redirect are authorized.
    let base = spawn_authenticated_server(
        vec![
            (
                "/api/models/test/private/revision/main",
                Reply::Ok(r#"{"sha":"abc123","siblings":[{"rfilename":"weights.bin"}]}"#),
            ),
            (
                "/test/private/resolve/abc123/weights.bin",
                Reply::Redirect("/cdn/private.bin"),
            ),
            ("/cdn/private.bin", Reply::Ok("PRIVATE")),
        ],
        Some("hf_synthetic_test_token"),
    );
    let cache = temp_cache("private");
    let _ = std::fs::remove_dir_all(&cache);
    let get = |key: &str| match key {
        "HF_ENDPOINT" => Some(base.clone()),
        "HF_HUB_CACHE" => Some(cache.to_string_lossy().into_owned()),
        "HF_TOKEN" => Some("hf_synthetic_test_token".to_string()),
        _ => None,
    };
    let hub = Hub::from_env_with(&get, None);
    let spec = CheckpointSpec {
        repo: "test/private".into(),
        revision: "main".into(),
        subfolder: None,
        required: vec!["weights.bin".into()],
        rules: vec![FileRule::Exact("weights.bin".into())],
    };
    for token in [None, Some("wrong-token".into())] {
        let unauthorized = Hub::new(&base, &cache, token, false);
        let error = unauthorized.resolve(&spec).unwrap_err();
        assert!(matches!(error, hf_fetch::HubError::Http(ref error)
            if error.status() == Some(reqwest::StatusCode::UNAUTHORIZED)));
        assert!(!cache.join("models--test--private/refs/main").exists());
    }
    let dir = hub.resolve(&spec).unwrap();
    assert_eq!(std::fs::read(dir.join("weights.bin")).unwrap(), b"PRIVATE");
    std::fs::remove_dir_all(&cache).unwrap();
}

fn scripted_server(responses: Vec<String>) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let mut requests = Vec::new();
        for response in responses {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                request.push_str(&line);
                if line == "\r\n" || line.is_empty() {
                    break;
                }
            }
            requests.push(request);
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        }
        requests
    });
    (base, handle)
}

fn raw_response(status: &str, headers: &str, body: &str, size: usize) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Length: {size}\r\n{headers}Connection: close\r\n\r\n{body}"
    )
}

fn resume_info() -> String {
    let body = r#"{"sha":"resumecommit","siblings":[{"rfilename":"weights.bin"}]}"#;
    raw_response("200 OK", "", body, body.len())
}

fn resume_spec() -> CheckpointSpec {
    CheckpointSpec {
        repo: "test/resume".into(),
        revision: "main".into(),
        subfolder: None,
        required: vec!["weights.bin".into()],
        rules: vec![FileRule::Exact("weights.bin".into())],
    }
}

#[test]
fn interrupted_download_resumes_without_publishing_partial_snapshot() {
    let (base, server) = scripted_server(vec![
        resume_info(),
        raw_response("200 OK", "", "ABC", 10),
        resume_info(),
        raw_response(
            "206 Partial Content",
            "Content-Range: bytes 3-9/10\r\n",
            "DEFGHIJ",
            7,
        ),
    ]);
    let cache = temp_cache("resume");
    let _ = std::fs::remove_dir_all(&cache);
    let hub = Hub::new(base, &cache, None, false);
    let spec = resume_spec();
    assert!(hub.resolve(&spec).is_err());
    let root = cache.join("models--test--resume");
    assert!(!root.join("snapshots/resumecommit/weights.bin").exists());
    assert!(!root.join("refs/main").exists());
    assert_eq!(
        std::fs::read(root.join("downloads/resumecommit/weights.bin")).unwrap(),
        b"ABC"
    );
    let mut events = Vec::new();
    let dir = hub
        .resolve_with_progress(&spec, |event| events.push((event.downloaded, event.total)))
        .unwrap();
    assert_eq!(
        std::fs::read(dir.join("weights.bin")).unwrap(),
        b"ABCDEFGHIJ"
    );
    assert_eq!(events.first(), Some(&(3, Some(10))));
    assert_eq!(events.last(), Some(&(10, Some(10))));
    assert!(!root.join("downloads/resumecommit").exists());
    let requests = server.join().unwrap();
    assert!(!requests[1].to_lowercase().contains("range:"));
    assert!(requests[3].to_lowercase().contains("range: bytes=3-\r\n"));
    std::fs::remove_dir_all(cache).unwrap();
}

#[test]
fn ignored_range_restarts_and_invalid_range_preserves_the_prefix() {
    for (name, status, headers, body, valid) in [
        ("ignored", "200 OK", "", "ABCDEFGHIJ", true),
        (
            "wrong",
            "206 Partial Content",
            "Content-Range: bytes 2-8/9\r\n",
            "DEFGHIJ",
            false,
        ),
        ("missing", "206 Partial Content", "", "DEFGHIJ", false),
    ] {
        let (base, server) = scripted_server(vec![
            resume_info(),
            raw_response(status, headers, body, body.len()),
        ]);
        let cache = temp_cache(name);
        let _ = std::fs::remove_dir_all(&cache);
        let root = cache.join("models--test--resume");
        let partial = root.join("downloads/resumecommit/weights.bin");
        std::fs::create_dir_all(partial.parent().unwrap()).unwrap();
        std::fs::write(&partial, "ABC").unwrap();
        let result = Hub::new(base, &cache, None, false).resolve(&resume_spec());
        if valid {
            assert_eq!(
                std::fs::read(result.unwrap().join("weights.bin")).unwrap(),
                b"ABCDEFGHIJ"
            );
        } else {
            assert!(matches!(result, Err(hf_fetch::HubError::InvalidRange(_))));
            assert_eq!(std::fs::read(partial).unwrap(), b"ABC");
            assert!(!root.join("refs/main").exists());
        }
        server.join().unwrap();
        std::fs::remove_dir_all(cache).unwrap();
    }
}

#[test]
fn unsatisfiable_range_retries_the_complete_file() {
    let (base, server) = scripted_server(vec![
        resume_info(),
        raw_response("416 Range Not Satisfiable", "", "", 0),
        raw_response("200 OK", "", "ABC", 3),
    ]);
    let cache = temp_cache("unsatisfiable");
    let _ = std::fs::remove_dir_all(&cache);
    let partial = cache.join("models--test--resume/downloads/resumecommit/weights.bin");
    std::fs::create_dir_all(partial.parent().unwrap()).unwrap();
    std::fs::write(partial, "TOO LONG").unwrap();
    let dir = Hub::new(base, &cache, None, false)
        .resolve(&resume_spec())
        .unwrap();
    assert_eq!(std::fs::read(dir.join("weights.bin")).unwrap(), b"ABC");
    let requests = server.join().unwrap();
    assert!(requests[1].to_lowercase().contains("range: bytes=8-"));
    assert!(!requests[2].to_lowercase().contains("range:"));
    std::fs::remove_dir_all(cache).unwrap();
}

#[test]
fn concurrent_resolvers_publish_one_complete_file() {
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };
    let base = spawn_server(vec![
        (
            "/api/models/test/resume/revision/main",
            Reply::Ok(r#"{"sha":"resumecommit","siblings":[{"rfilename":"weights.bin"}]}"#),
        ),
        (
            "/test/resume/resolve/resumecommit/weights.bin",
            Reply::Ok("ABCDEFGHIJ"),
        ),
    ]);
    let cache = temp_cache("concurrent");
    let _ = std::fs::remove_dir_all(&cache);
    let barrier = Arc::new(Barrier::new(2));
    let downloads = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let (base, cache, barrier, downloads) = (
                base.clone(),
                cache.clone(),
                barrier.clone(),
                downloads.clone(),
            );
            thread::spawn(move || {
                barrier.wait();
                Hub::new(base, cache, None, false)
                    .resolve_with_progress(&resume_spec(), |event| {
                        if event.downloaded == 0 {
                            downloads.fetch_add(1, Ordering::SeqCst);
                        }
                    })
                    .unwrap()
            })
        })
        .collect();
    let paths: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(paths[0], paths[1]);
    assert_eq!(
        std::fs::read(paths[0].join("weights.bin")).unwrap(),
        b"ABCDEFGHIJ"
    );
    assert_eq!(downloads.load(Ordering::SeqCst), 1);
    std::fs::remove_dir_all(cache).unwrap();
}

#[test]
fn unsafe_revision_and_required_paths_are_rejected_before_cache_access() {
    let hub = Hub::new("http://127.0.0.1:1", temp_cache("unsafe"), None, true);
    let mut spec = resume_spec();
    spec.revision = "../outside".into();
    assert!(matches!(
        hub.resolve(&spec),
        Err(hf_fetch::HubError::UnsafePath(_))
    ));
    spec.revision = "main".into();
    spec.required = vec!["../outside".into()];
    assert!(matches!(
        hub.resolve(&spec),
        Err(hf_fetch::HubError::UnsafePath(_))
    ));
}
