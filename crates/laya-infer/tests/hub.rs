//! Hub integration remains opt-in and honors offline resolution.
#![cfg(feature = "hub")]

use laya_infer::agent::LayaEngine;

#[test]
fn offline_cache_miss_preserves_the_hub_error() {
    let cache = std::env::temp_dir().join(format!(
        "laya-infer-missing-hub-cache-{}",
        std::process::id()
    ));
    assert!(!cache.exists());
    let hub = hf_fetch::Hub::new("http://127.0.0.1:1", cache, None, true);
    let error = LayaEngine::load_from_hub(&hub, &hf_fetch::CheckpointSpec::laya()).unwrap_err();
    match error {
        decision_core::DecisionError::Transport {
            source: Some(source),
            ..
        } => {
            assert!(matches!(
                source.downcast_ref::<hf_fetch::HubError>(),
                Some(hf_fetch::HubError::Offline { .. })
            ));
        }
        other => panic!("expected an offline Hub error, found {other:?}"),
    }
}
