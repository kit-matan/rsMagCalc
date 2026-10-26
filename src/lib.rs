//! rsmagcalc._core -- Rust kernels for pyMagCalc's beyond-LSWT loops.
//!
//! M0 (scaffold): build provenance and ONE kernel that proves the data contract
//! end to end -- a zero-copy read of NumPy arrays, a result back, and a parity
//! test against pyMagCalc's own implementation. M1: `classical` (K1, the
//! Metropolis sweep, parallel tempering, Wang-Landau, the Landau-Lifshitz
//! integrators and the Langevin thermostat). M2: `cpn` (K2, the CP^(N-1)
//! Metropolis sampler, RK4 integrator and damped quench for SU(N) coherent
//! states). M3: `sun_lswt` (K3, the SU(N) LSWT H(q) assembly, Colpa
//! diagonalization, one-magnon correlations and the shifted-Cholesky guard, on
//! the platform LAPACK through `lapack`). The remaining kernels (KPM, SCGA)
//! follow the order in ../RUST_KERNELS_PLAN.md.
//!
//! Conventions shared with the Python side, and never to be re-derived here:
//!   * the classical energy is E = 1/2 m^T H m + b^T m with H the (3N x 3N)
//!     symmetric Hessian of `thermal_mc.build_supercell` and b the Zeeman term;
//!   * m is (N, 3) row-major, flattened exactly as NumPy's `ravel()` does.
mod classical;
mod cpn;
mod lapack;
mod sun_lswt;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Version, git head and build profile of this extension, for the run log.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("version", env!("CARGO_PKG_VERSION"))?;
    d.set_item("git_head", env!("RSMAGCALC_GIT_HEAD"))?;
    d.set_item("target", env!("RSMAGCALC_TARGET"))?;
    d.set_item(
        "profile",
        if cfg!(debug_assertions) { "debug" } else { "release" },
    )?;
    Ok(d)
}

/// E = 1/2 m^T H m + b^T m  (pyMagCalc `classical_dynamics.energy`).
///
/// Accepts any float64 arrays NumPy hands over, contiguous or not; the shapes are
/// checked, because a silently broadcast mismatch here would be exactly the kind
/// of plausible-but-wrong number the project's history warns about.
#[pyfunction]
fn classical_energy(
    h: PyReadonlyArray2<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    m: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let h = h.as_array();
    let b = b.as_array();
    let m = m.as_array();
    let n3 = b.len();
    if n3 % 3 != 0 {
        return Err(PyValueError::new_err(format!(
            "b must have 3N entries, got {n3}"
        )));
    }
    if h.shape() != [n3, n3] {
        return Err(PyValueError::new_err(format!(
            "H must be ({n3}, {n3}) to match b, got {:?}",
            h.shape()
        )));
    }
    if m.shape() != [n3 / 3, 3] {
        return Err(PyValueError::new_err(format!(
            "m must be ({}, 3) to match b, got {:?}",
            n3 / 3,
            m.shape()
        )));
    }
    // Logical (row-major) order, whatever the memory layout: identical to ravel().
    let mr: Vec<f64> = m.iter().copied().collect();
    let mut e = 0.0;
    for i in 0..n3 {
        let row = h.row(i);
        let hm: f64 = row.iter().zip(&mr).map(|(hij, mj)| hij * mj).sum();
        e += 0.5 * mr[i] * hm + b[i] * mr[i];
    }
    Ok(e)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(classical_energy, m)?)?;
    classical::register(m)?;
    cpn::register(m)?;
    sun_lswt::register(m)?;
    Ok(())
}
