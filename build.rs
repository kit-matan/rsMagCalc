//! Bake the git HEAD and build target into the extension, so a run log can name
//! the kernel build it used (`kernels: rust (rsmagcalc 0.1.0, git abc123, release)`)
//! the way pyMagCalc's `magcalc where` names the engine copy.
//!
//! Shelling out to git here is fine: this runs at BUILD time. pyMagCalc's
//! `provenance.py` deliberately avoids a subprocess at RUN time, and so does the
//! Python side of this package -- it only reads what is compiled in.
use std::process::Command;

fn main() {
    let head = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=RSMAGCALC_GIT_HEAD={head}");
    println!(
        "cargo:rustc-env=RSMAGCALC_TARGET={}",
        std::env::var("TARGET").unwrap_or_default()
    );
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");
}
