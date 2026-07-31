//! Build script for the arithmetic crate.
//!
//! Currently a no-op; retained for future use and to keep Cargo's build-script
//! plumbing warm.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
