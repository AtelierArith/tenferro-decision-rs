fn main() {
    #[cfg(feature = "onednn")]
    {
        println!("cargo:rerun-if-env-changed=ONEDNN_INCLUDE_DIR");
        println!("cargo:rerun-if-env-changed=ONEDNN_LIB_DIR");
        println!("cargo:rerun-if-changed=native/prepared_projection.cpp");
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .std("c++17")
            .file("native/prepared_projection.cpp");
        if let Some(include) = std::env::var_os("ONEDNN_INCLUDE_DIR") {
            build.include(include);
        }
        if let Some(library) = std::env::var_os("ONEDNN_LIB_DIR") {
            println!(
                "cargo:rustc-link-search=native={}",
                std::path::Path::new(&library).display()
            );
        }
        build.compile("decision_prepared_projection");
        println!("cargo:rustc-link-lib=dnnl");
    }
}
