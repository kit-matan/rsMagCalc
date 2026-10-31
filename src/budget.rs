//! The scratch budget: how many bytes of per-item scratch the memory-bound
//! kernels may hold in flight at once. It sizes `capped_pool` (K3-K5, one cap
//! per call) and K7's byte-weighted admission (each cluster admitted by its own
//! scratch).
//!
//! Until 2026-10-31 it was a hard-coded 3 GB, which on a 243 GB, 128-core node
//! held the dimer series at order 8 to 2 threads and at order 9 to 1. It is now,
//! first match wins:
//!   1. the call's own argument (`scratch_budget_gb`);
//!   2. `RSMAGCALC_SCRATCH_GB` in the environment;
//!   3. half the memory this process can still get: the smaller of
//!      /proc/meminfo's MemAvailable and the headroom (limit - usage) of every
//!      memory-limited cgroup above the process (v2 `memory.max`, v1
//!      `memory.limit_in_bytes`) -- a Slurm job's limit lives there;
//!   4. 3 GB, where neither file exists (macOS).
//!
//! The half leaves room for what the estimate does not cover: the caller's own
//! arrays, the results accumulating while the batch runs, and allocator slack.
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) const ENV: &str = "RSMAGCALC_SCRATCH_GB";
const FALLBACK: usize = 3 << 30;
const FRACTION: f64 = 0.5;

/// (budget in bytes, where it came from).
pub(crate) fn scratch_budget(gb: Option<f64>) -> Result<(usize, String), String> {
    budget_from(gb, std::env::var(ENV).ok().as_deref(), Path::new("/"))
}

/// `scratch_budget` with the environment value and the filesystem root given,
/// so the cgroup and meminfo readers can be tested on a fake tree.
pub(crate) fn budget_from(gb: Option<f64>, env: Option<&str>, root: &Path)
                          -> Result<(usize, String), String> {
    if let Some(g) = gb {
        return Ok((gb_bytes(g, "scratch_budget_gb")?, "argument".into()));
    }
    if let Some(s) = env.map(str::trim).filter(|s| !s.is_empty()) {
        let g: f64 = s.parse().map_err(|_| format!("{ENV}={s:?} is not a number of GB"))?;
        return Ok((gb_bytes(g, ENV)?, ENV.into()));
    }
    Ok(match available(root) {
        Some((avail, src)) => (((avail as f64 * FRACTION) as usize).max(1),
                               format!("{FRACTION} x {src}")),
        None => (FALLBACK, "default 3 GB".into()),
    })
}

fn gb_bytes(g: f64, what: &str) -> Result<usize, String> {
    if !(g.is_finite() && g > 0.0) {
        return Err(format!("{what} must be a positive number of GB, got {g}"));
    }
    Ok(((g * (1u64 << 30) as f64) as usize).max(1))
}

/// The memory still available to this process, and which limit set it.
fn available(root: &Path) -> Option<(usize, String)> {
    let mem = meminfo_available(root).map(|b| (b, "MemAvailable".to_string()));
    let cg = cgroup_headroom(root);
    match (mem, cg) {
        (Some(a), Some(b)) => Some(if b.0 < a.0 { b } else { a }),
        (a, b) => a.or(b),
    }
}

fn meminfo_available(root: &Path) -> Option<usize> {
    let s = fs::read_to_string(root.join("proc/meminfo")).ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

fn read_num(p: &Path) -> Option<usize> {
    fs::read_to_string(p).ok()?.trim().parse().ok()
}

/// The smallest (limit - usage) over every memory-limited cgroup from this
/// process's own up to the hierarchy root. "max" (v2) and the v1 near-2^63
/// sentinel mean unlimited.
fn cgroup_headroom(root: &Path) -> Option<(usize, String)> {
    let s = fs::read_to_string(root.join("proc/self/cgroup")).ok()?;
    let mut best: Option<(usize, String)> = None;
    for line in s.lines() {
        let mut it = line.splitn(3, ':');
        let (id, ctl, path) = (it.next()?, it.next()?, it.next()?);
        let (base, lim, use_, name) = if id == "0" && ctl.is_empty() {
            (root.join("sys/fs/cgroup"), "memory.max", "memory.current", "cgroup memory.max")
        } else if ctl.split(',').any(|c| c == "memory") {
            (root.join("sys/fs/cgroup/memory"), "memory.limit_in_bytes",
             "memory.usage_in_bytes", "cgroup memory.limit_in_bytes")
        } else {
            continue;
        };
        let mut dir: PathBuf = base.join(path.trim_start_matches('/'));
        loop {
            if let Some(limit) = read_num(&dir.join(lim)).filter(|&l| l < 1 << 60) {
                let used = read_num(&dir.join(use_)).unwrap_or(0);
                let head = limit.saturating_sub(used);
                if best.as_ref().is_none_or(|b| head < b.0) {
                    best = Some((head, name.to_string()));
                }
            }
            if dir == base || !dir.pop() || !dir.starts_with(&base) {
                break;
            }
        }
    }
    best
}

/// `scratch_budget(gb=None, root=None)` -> (bytes, source): the budget the
/// memory-bound kernels would use now. `root` replaces "/" for the meminfo and
/// cgroup files (tests); the environment variable is read either way.
#[pyo3::pyfunction]
#[pyo3(name = "scratch_budget", signature = (gb=None, root=None))]
fn py_scratch_budget(gb: Option<f64>, root: Option<PathBuf>) -> pyo3::PyResult<(usize, String)> {
    let env = std::env::var(ENV).ok();
    budget_from(gb, env.as_deref(), root.as_deref().unwrap_or(Path::new("/")))
        .map_err(crate::sun_lswt::perr)
}

pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    m.add_function(pyo3::wrap_pyfunction!(py_scratch_budget, m)?)?;
    Ok(())
}
