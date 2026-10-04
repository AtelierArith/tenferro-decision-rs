//! Link the Accelerate framework on macOS for the `cblas_sgemm` fast path.
//!
//! `cblas-sys` only declares the FFI symbols, so the framework is linked here.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
