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
    // Re-run on a new commit, not only on a branch switch: `.git/HEAD` holds
    // `ref: refs/heads/<branch>` and does not change when that branch advances,
    // so watch the ref file it points at as well (and packed-refs, where a gc'd
    // ref lives instead). Without this the baked head goes stale one commit in.
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        if let Some(r) = head.trim().strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed=.git/{r}");
        }
    }
    println!("cargo:rerun-if-changed=.git/packed-refs");
}
