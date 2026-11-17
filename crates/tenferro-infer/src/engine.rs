//! CPU engine construction for the decision engines.
//!
//! tenferro 0.11 never starts threads itself: the caller builds the
//! [`Pool`] and hands it to [`CpuEngine::new`]. Before 0.11 a configured
//! `CpuBackend` followed the process thread budget, so this module keeps that
//! behaviour in one place instead of repeating it at every engine.

use tenferro_cpu::{CpuEngine, Pool};

/// A CPU engine sized from `RAYON_NUM_THREADS`, else the process parallelism.
///
/// The engine owns its pool for its lifetime, so build one per model, not per
/// request. Falls back to a serial engine if the pool cannot be built.
pub fn cpu_engine() -> CpuEngine {
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1);
    match Pool::builder().threads(threads).build() {
        Ok(pool) => CpuEngine::new(pool),
        Err(_) => CpuEngine::serial(),
    }
}
