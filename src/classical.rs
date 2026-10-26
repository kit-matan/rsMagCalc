//! K1 -- the classical Metropolis sweep and the Landau-Lifshitz integrator.
//!
//! Replaces, in pyMagCalc: `thermal_mc._sweep` (uniform and flip proposals), the
//! sweep/swap loop of `thermal_mc.parallel_tempering`, the move loop of
//! `thermal_mc.wang_landau`, `classical_dynamics.evolve` (rk4 and implicit
//! midpoint), `classical_dynamics.langevin_step`, and the outer trajectory loop
//! of `classical_dynamics.sampled_correlations`. The FFT and the neutron
//! contraction stay in NumPy.
//!
//! Conventions, shared with the Python side and never re-derived here:
//!   * E = 1/2 m^T H m + b^T m with H the symmetric (3N x 3N) Hessian of
//!     `thermal_mc.build_supercell`; m is (N, 3) row-major, i.e. `ravel()` order;
//!   * `g = H m + b` is the energy GRADIENT (pyMagCalc `local_field` returns it
//!     under that name), so the precession is dS/dt = -S x g and a Metropolis move
//!     at site a costs dE = g_a . d + 1/2 d^T H_aa d with d = v - m_a;
//!   * the "flip" proposal is v = -m_a, Sunny's `propose_flip`: from a polarized
//!     start it never leaves the +/-n axis, which is what makes the sampler Ising;
//!   * RK4 renormalizes |m_i| = S_i after every step; the implicit midpoint rule
//!     does not (it conserves |m_i| by construction, and a rescale would mask a
//!     solver failure), exactly as the Python does.
//!
//! RNG: PCG64 (`rand_pcg`), seeded; deterministic for a given seed and
//! independent of the thread count, because every replica and every trajectory
//! owns its own stream derived from (seed, index). It does NOT reproduce NumPy's
//! PCG64 stream, and the oracle tests do not need it to.
use ndarray::{Array2, Array3, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1,
            PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;
use rayon::prelude::*;

// --------------------------------------------------------------------------
// Dense energy model
// --------------------------------------------------------------------------

/// The classical quadratic form, borrowed zero-copy from NumPy.
#[derive(Clone, Copy)]
struct Dense<'a> {
    n3: usize,
    h: &'a [f64],
    b: &'a [f64],
}

impl<'a> Dense<'a> {
    #[inline]
    fn n(&self) -> usize {
        self.n3 / 3
    }

    /// g = H m + b (the gradient), written into `g`.
    fn gradient(&self, m: &[f64], g: &mut [f64]) {
        let n3 = self.n3;
        for i in 0..n3 {
            let row = &self.h[i * n3..(i + 1) * n3];
            let mut acc = 0.0;
            for (hij, mj) in row.iter().zip(m) {
                acc += hij * mj;
            }
            g[i] = acc + self.b[i];
        }
    }

    /// E = 1/2 m^T H m + b^T m.
    fn energy(&self, m: &[f64]) -> f64 {
        let n3 = self.n3;
        let mut e = 0.0;
        for i in 0..n3 {
            let row = &self.h[i * n3..(i + 1) * n3];
            let mut hm = 0.0;
            for (hij, mj) in row.iter().zip(m) {
                hm += hij * mj;
            }
            e += 0.5 * m[i] * hm + self.b[i] * m[i];
        }
        e
    }

    /// dE for replacing m_a by v, given the current gradient g.
    #[inline]
    fn move_delta(&self, m: &[f64], g: &[f64], a: usize, v: [f64; 3]) -> f64 {
        let n3 = self.n3;
        let s = 3 * a;
        let d = [v[0] - m[s], v[1] - m[s + 1], v[2] - m[s + 2]];
        let mut lin = 0.0;
        let mut quad = 0.0;
        for c in 0..3 {
            lin += g[s + c] * d[c];
            let row = &self.h[(s + c) * n3..(s + c + 1) * n3];
            quad += d[c] * (row[s] * d[0] + row[s + 1] * d[1] + row[s + 2] * d[2]);
        }
        lin + 0.5 * quad
    }

    /// Accept the move: m_a = v and g += H[:, a-block] d, using the symmetry of H
    /// to read rows (contiguous) rather than columns.
    #[inline]
    fn apply_move(&self, m: &mut [f64], g: &mut [f64], a: usize, v: [f64; 3]) {
        let n3 = self.n3;
        let s = 3 * a;
        let d = [v[0] - m[s], v[1] - m[s + 1], v[2] - m[s + 2]];
        for c in 0..3 {
            let dc = d[c];
            if dc == 0.0 {
                continue;
            }
            let row = &self.h[(s + c) * n3..(s + c + 1) * n3];
            for (gk, hk) in g.iter_mut().zip(row) {
                *gk += hk * dc;
            }
        }
        m[s] = v[0];
        m[s + 1] = v[1];
        m[s + 2] = v[2];
    }
}

#[inline]
fn random_unit(rng: &mut Pcg64, s: f64) -> [f64; 3] {
    loop {
        let x: f64 = rng.sample(StandardNormal);
        let y: f64 = rng.sample(StandardNormal);
        let z: f64 = rng.sample(StandardNormal);
        let r = (x * x + y * y + z * z).sqrt();
        if r > 0.0 {
            return [s * x / r, s * y / r, s * z / r];
        }
    }
}

/// A random state of N spins with per-site lengths `s`, into `m` (3N).
fn random_state(rng: &mut Pcg64, s: &[f64], m: &mut [f64]) {
    for (i, si) in s.iter().enumerate() {
        let v = random_unit(rng, *si);
        m[3 * i] = v[0];
        m[3 * i + 1] = v[1];
        m[3 * i + 2] = v[2];
    }
}

/// One Metropolis sweep (N single-spin updates in a random order). Returns the
/// acceptance fraction. `g` is kept consistent with `m` incrementally.
fn sweep_impl(
    d: &Dense<'_>,
    m: &mut [f64],
    g: &mut [f64],
    beta: f64,
    s: &[f64],
    rng: &mut Pcg64,
    flip: bool,
    order: &mut Vec<usize>,
) -> f64 {
    let n = d.n();
    order.clear();
    order.extend(0..n);
    order.shuffle(rng);
    let mut acc = 0usize;
    for &a in order.iter() {
        let v = if flip {
            [-m[3 * a], -m[3 * a + 1], -m[3 * a + 2]]
        } else {
            random_unit(rng, s[a])
        };
        let de = d.move_delta(m, g, a, v);
        if de <= 0.0 || rng.random::<f64>() < (-beta * de).exp() {
            d.apply_move(m, g, a, v);
            acc += 1;
        }
    }
    acc as f64 / n as f64
}

/// Mix (seed, index) into an independent PCG64 stream (splitmix64 finalizer).
pub(crate) fn stream(seed: u64, index: u64) -> Pcg64 {
    let mut z = seed.wrapping_add((index + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    Pcg64::seed_from_u64(z ^ (z >> 31))
}

/// Run `f` inside a rayon pool of `n_workers` threads (None/0 = the global pool).
pub(crate) fn with_pool<T: Send>(n_workers: Option<usize>, f: impl FnOnce() -> T + Send) -> PyResult<T> {
    match n_workers {
        Some(n) if n > 0 => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("rayon pool: {e}")))?;
            Ok(pool.install(f))
        }
        _ => Ok(f()),
    }
}

// --------------------------------------------------------------------------
// Landau-Lifshitz integrators
// --------------------------------------------------------------------------

#[inline]
fn cross(a: &[f64], b: &[f64], out: &mut [f64]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

/// k = -m x (H m + b), using `g` as scratch for the gradient.
fn deriv(d: &Dense<'_>, m: &[f64], g: &mut [f64], k: &mut [f64]) {
    d.gradient(m, g);
    for i in 0..d.n() {
        let s = 3 * i;
        let mut c = [0.0; 3];
        cross(&m[s..s + 3], &g[s..s + 3], &mut c);
        k[s] = -c[0];
        k[s + 1] = -c[1];
        k[s + 2] = -c[2];
    }
}

fn renormalize(m: &mut [f64], s: &[f64]) {
    for (i, si) in s.iter().enumerate() {
        let r = (m[3 * i] * m[3 * i] + m[3 * i + 1] * m[3 * i + 1] + m[3 * i + 2] * m[3 * i + 2])
            .sqrt();
        let f = si / r;
        m[3 * i] *= f;
        m[3 * i + 1] *= f;
        m[3 * i + 2] *= f;
    }
}

struct Scratch {
    g: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    tmp: Vec<f64>,
}

impl Scratch {
    fn new(n3: usize) -> Self {
        Scratch {
            g: vec![0.0; n3],
            k1: vec![0.0; n3],
            k2: vec![0.0; n3],
            k3: vec![0.0; n3],
            k4: vec![0.0; n3],
            tmp: vec![0.0; n3],
        }
    }
}

/// One RK4 step of dm/dt = -m x g, then |m_i| = S_i (pyMagCalc `llg_rk4_step`).
fn rk4_step(d: &Dense<'_>, m: &mut [f64], dt: f64, s: &[f64], w: &mut Scratch) {
    let n3 = d.n3;
    deriv(d, m, &mut w.g, &mut w.k1);
    for i in 0..n3 {
        w.tmp[i] = m[i] + 0.5 * dt * w.k1[i];
    }
    deriv(d, &w.tmp, &mut w.g, &mut w.k2);
    for i in 0..n3 {
        w.tmp[i] = m[i] + 0.5 * dt * w.k2[i];
    }
    deriv(d, &w.tmp, &mut w.g, &mut w.k3);
    for i in 0..n3 {
        w.tmp[i] = m[i] + dt * w.k3[i];
    }
    deriv(d, &w.tmp, &mut w.g, &mut w.k4);
    let f = dt / 6.0;
    for i in 0..n3 {
        m[i] += f * (((w.k1[i] + 2.0 * w.k2[i]) + 2.0 * w.k3[i]) + w.k4[i]);
    }
    renormalize(m, s);
}

/// One implicit-midpoint step by fixed-point iteration (pyMagCalc
/// `implicit_midpoint_step`: same iterate, same stopping rule, no rescale).
/// Returns false if it did not converge in `max_iter`.
fn midpoint_step(
    d: &Dense<'_>,
    m: &mut [f64],
    dt: f64,
    w: &mut Scratch,
    tol: f64,
    max_iter: usize,
) -> bool {
    let n3 = d.n3;
    // w.k1 = m_new, w.tmp = mid, w.k2 = cand
    w.k1.copy_from_slice(m);
    let mut converged = false;
    for _ in 0..max_iter {
        for i in 0..n3 {
            w.tmp[i] = 0.5 * (m[i] + w.k1[i]);
        }
        deriv(d, &w.tmp, &mut w.g, &mut w.k3);
        let mut delta: f64 = 0.0;
        for i in 0..n3 {
            w.k2[i] = m[i] + dt * w.k3[i];
            delta = delta.max((w.k2[i] - w.k1[i]).abs());
        }
        w.k1.copy_from_slice(&w.k2);
        if delta < tol {
            converged = true;
            break;
        }
    }
    m.copy_from_slice(&w.k1);
    converged
}

#[derive(Clone, Copy, PartialEq)]
enum Integrator {
    Rk4,
    Midpoint,
}

fn parse_integrator(name: &str) -> PyResult<Integrator> {
    match name.trim().to_ascii_lowercase().as_str() {
        "rk4" => Ok(Integrator::Rk4),
        "midpoint" => Ok(Integrator::Midpoint),
        other => Err(PyValueError::new_err(format!(
            "integrator must be 'rk4' or 'midpoint', got {other:?}."
        ))),
    }
}

/// Deterministic trajectory into `rec` (n_rec x 3N, row-major). Returns the
/// number of midpoint steps that did not converge.
fn evolve_impl(
    d: &Dense<'_>,
    m: &mut [f64],
    s: &[f64],
    dt: f64,
    n_steps: usize,
    record_every: usize,
    integrator: Integrator,
    rec: &mut [f64],
) -> usize {
    let n3 = d.n3;
    let mut w = Scratch::new(n3);
    let mut irec = 0usize;
    let mut bad = 0usize;
    for step in 0..n_steps {
        if step % record_every == 0 {
            rec[irec * n3..(irec + 1) * n3].copy_from_slice(m);
            irec += 1;
        }
        match integrator {
            Integrator::Rk4 => rk4_step(d, m, dt, s, &mut w),
            Integrator::Midpoint => {
                if !midpoint_step(d, m, dt, &mut w, 1e-12, 50) {
                    bad += 1;
                }
            }
        }
    }
    bad
}

/// Heun step of the stochastic LL equation (pyMagCalc `langevin_step`), with the
/// damping sign that relaxes TOWARD the minimum given that g is the gradient:
///   drift(x) = -x x g + (lam/S) x x (x x g),  noise ~ sqrt(2 lam kT dt / S).
fn langevin_step_impl(
    d: &Dense<'_>,
    m: &mut [f64],
    s: &[f64],
    dt: f64,
    kt: f64,
    lam: f64,
    rng: &mut Pcg64,
    w: &mut Scratch,
) {
    let n3 = d.n3;
    let n = d.n();
    // w.k1 = f0, w.k2 = pred, w.k3 = f1, w.k4 = noise, w.tmp = out
    let drift = |d: &Dense<'_>, x: &[f64], g: &mut [f64], f: &mut [f64]| {
        d.gradient(x, g);
        for i in 0..n {
            let o = 3 * i;
            let mut c1 = [0.0; 3];
            let mut c2 = [0.0; 3];
            cross(&x[o..o + 3], &g[o..o + 3], &mut c1);
            cross(&x[o..o + 3], &c1, &mut c2);
            let a = lam / s[i];
            f[o] = -c1[0] + a * c2[0];
            f[o + 1] = -c1[1] + a * c2[1];
            f[o + 2] = -c1[2] + a * c2[2];
        }
    };
    for i in 0..n {
        let sigma = (2.0 * lam * kt * dt / s[i]).sqrt();
        for c in 0..3 {
            let z: f64 = rng.sample(StandardNormal);
            w.k4[3 * i + c] = sigma * z;
        }
    }
    drift(d, m, &mut w.g, &mut w.k1);
    for i in 0..n {
        let o = 3 * i;
        let mut c = [0.0; 3];
        cross(&m[o..o + 3], &w.k4[o..o + 3], &mut c);
        for k in 0..3 {
            w.k2[o + k] = m[o + k] + dt * w.k1[o + k] - c[k];
        }
    }
    drift(d, &w.k2, &mut w.g, &mut w.k3);
    for i in 0..n {
        let o = 3 * i;
        let mid = [
            0.5 * (m[o] + w.k2[o]),
            0.5 * (m[o + 1] + w.k2[o + 1]),
            0.5 * (m[o + 2] + w.k2[o + 2]),
        ];
        let mut c = [0.0; 3];
        cross(&mid, &w.k4[o..o + 3], &mut c);
        for k in 0..3 {
            w.tmp[o + k] = m[o + k] + 0.5 * dt * (w.k1[o + k] + w.k3[o + k]) - c[k];
        }
    }
    m[..n3].copy_from_slice(&w.tmp);
    renormalize(m, s);
}

// --------------------------------------------------------------------------
// Python-facing helpers
// --------------------------------------------------------------------------

fn dense<'a>(
    h: &'a PyReadonlyArray2<'a, f64>,
    b: &'a PyReadonlyArray1<'a, f64>,
) -> PyResult<Dense<'a>> {
    let n3 = b.len();
    if n3 % 3 != 0 || n3 == 0 {
        return Err(PyValueError::new_err(format!("b must have 3N > 0 entries, got {n3}")));
    }
    if h.shape() != [n3, n3] {
        return Err(PyValueError::new_err(format!(
            "H must be ({n3}, {n3}) to match b, got {:?}",
            h.shape()
        )));
    }
    let hs = h.as_slice().map_err(|_| {
        PyValueError::new_err("H must be a C-contiguous float64 array (np.ascontiguousarray)")
    })?;
    let bs = b
        .as_slice()
        .map_err(|_| PyValueError::new_err("b must be a contiguous float64 array"))?;
    Ok(Dense { n3, h: hs, b: bs })
}

fn check_spins<'a>(s: &'a PyReadonlyArray1<'a, f64>, n: usize) -> PyResult<&'a [f64]> {
    if s.len() != n {
        return Err(PyValueError::new_err(format!(
            "S must have one entry per site ({n}), got {}",
            s.len()
        )));
    }
    s.as_slice()
        .map_err(|_| PyValueError::new_err("S must be a contiguous float64 array"))
}

fn check_m(shape: &[usize], n: usize, name: &str) -> PyResult<()> {
    if shape != [n, 3] {
        return Err(PyValueError::new_err(format!(
            "{name} must be ({n}, 3) to match H, got {shape:?}"
        )));
    }
    Ok(())
}

/// Seeded PCG64 stream, the Rust counterpart of `numpy.random.default_rng(seed)`.
///
/// pyMagCalc hands one of these to the kernels so that successive calls (sweep
/// after sweep) continue one stream, as its NumPy loops do with a Generator.
#[pyclass(name = "Rng", module = "rsmagcalc._core")]
pub struct PyRng {
    pub(crate) inner: Pcg64,
    seed: u64,
}

#[pymethods]
impl PyRng {
    #[new]
    fn new(seed: u64) -> Self {
        PyRng { inner: Pcg64::seed_from_u64(seed), seed }
    }

    /// Uniform in [0, 1).
    fn random(&mut self) -> f64 {
        self.inner.random::<f64>()
    }

    /// `n` standard normals, as a float64 array.
    fn standard_normal<'py>(&mut self, py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f64>> {
        let v: Vec<f64> = (0..n).map(|_| self.inner.sample(StandardNormal)).collect();
        v.into_pyarray(py)
    }

    /// Uniform integer in [0, n).
    fn integers(&mut self, n: u64) -> PyResult<u64> {
        if n == 0 {
            return Err(PyValueError::new_err("integers(n) needs n > 0"));
        }
        Ok(self.inner.random_range(0..n))
    }

    /// The next raw 64-bit output (to seed another generator from this stream).
    fn next_u64(&mut self) -> u64 {
        self.inner.random::<u64>()
    }

    fn __repr__(&self) -> String {
        format!("rsmagcalc.Rng(seed={})", self.seed)
    }
}

/// dE of replacing site `a`'s spin by `v`, from the current gradient `g`:
/// g_a . d + 1/2 d^T H_aa d. Exposed so the identity dE == E(after) - E(before)
/// can be pinned from Python; `sweep` uses the same function.
#[pyfunction]
pub fn move_energy_change(
    m: PyReadonlyArray2<'_, f64>,
    g: PyReadonlyArray1<'_, f64>,
    h: PyReadonlyArray2<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    a: usize,
    v: [f64; 3],
) -> PyResult<f64> {
    let d = dense(&h, &b)?;
    check_m(m.shape(), d.n(), "m")?;
    if a >= d.n() {
        return Err(PyValueError::new_err(format!("site {a} out of range (N = {})", d.n())));
    }
    let ms = m.as_slice().map_err(|_| PyValueError::new_err("m must be C-contiguous"))?;
    let gs = g.as_slice().map_err(|_| PyValueError::new_err("g must be contiguous"))?;
    if gs.len() != d.n3 {
        return Err(PyValueError::new_err("g must have 3N entries"));
    }
    Ok(d.move_delta(ms, gs, a, v))
}

/// One Metropolis sweep in place: `m` (N, 3) and its gradient `g` (3N) are updated,
/// the acceptance fraction is returned. `propose` is "uniform" or "flip".
#[pyfunction]
#[pyo3(signature = (m, g, h, b, beta, s, rng, propose = "uniform"))]
pub fn sweep(
    py: Python<'_>,
    mut m: PyReadwriteArray2<'_, f64>,
    mut g: PyReadwriteArray1<'_, f64>,
    h: PyReadonlyArray2<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    beta: f64,
    s: PyReadonlyArray1<'_, f64>,
    mut rng: PyRefMut<'_, PyRng>,
    propose: &str,
) -> PyResult<f64> {
    let d = dense(&h, &b)?;
    let n = d.n();
    let flip = match propose.trim().to_ascii_lowercase().as_str() {
        "uniform" => false,
        "flip" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "propose must be 'uniform' or 'flip', got {other:?}"
            )))
        }
    };
    check_m(m.shape(), n, "m")?;
    let ss = check_spins(&s, n)?;
    let ms = m
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("m must be a writable C-contiguous array"))?;
    let gs = g
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("g must be a writable contiguous array"))?;
    if gs.len() != d.n3 {
        return Err(PyValueError::new_err("g must have 3N entries"));
    }
    let r = &mut rng.inner;
    let mut order = Vec::with_capacity(n);
    Ok(py.detach(move || sweep_impl(&d, ms, gs, beta, ss, r, flip, &mut order)))
}

/// Replica-exchange Metropolis over a temperature ladder -- the whole
/// sweep/swap/measure loop of pyMagCalc `parallel_tempering`, returning the raw
/// sums it accumulates: (sumE, sumE2, sumM (R, 3), sumM2, n_meas, accept_rate).
/// Replicas are swept in parallel (rayon); each owns a stream derived from
/// (seed, replica), so the result does not depend on the thread count.
#[pyfunction]
#[pyo3(signature = (h, b, s, temperatures, n_sweeps, n_equil, swap_every, measure_every,
                    propose, init, seed, n_workers = None))]
#[allow(clippy::too_many_arguments)]
pub fn parallel_tempering<'py>(
    py: Python<'py>,
    h: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    s: PyReadonlyArray1<'py, f64>,
    temperatures: PyReadonlyArray1<'py, f64>,
    n_sweeps: usize,
    n_equil: usize,
    swap_every: usize,
    measure_every: usize,
    propose: &str,
    init: Option<[f64; 3]>,
    seed: u64,
    n_workers: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    f64,
)> {
    let d = dense(&h, &b)?;
    let n = d.n();
    let n3 = d.n3;
    let ss = check_spins(&s, n)?;
    let flip = match propose.trim().to_ascii_lowercase().as_str() {
        "uniform" => false,
        "flip" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "propose must be 'uniform' or 'flip', got {other:?}"
            )))
        }
    };
    let temps: Vec<f64> = temperatures.as_array().iter().copied().collect();
    let nr = temps.len();
    if nr == 0 {
        return Err(PyValueError::new_err("temperatures must not be empty"));
    }
    if temps.iter().any(|t| !(*t > 0.0)) {
        return Err(PyValueError::new_err("temperatures must be positive"));
    }
    if measure_every == 0 {
        return Err(PyValueError::new_err("measure_every must be >= 1"));
    }
    let betas: Vec<f64> = temps.iter().map(|t| 1.0 / t).collect();

    let out = py.detach(move || -> PyResult<_> {
        with_pool(n_workers, move || {
            let mut master = stream(seed, 0);
            // Replica states: m (nr x 3N), g (nr x 3N), one stream each.
            let mut ms: Vec<Vec<f64>> = Vec::with_capacity(nr);
            let mut gs: Vec<Vec<f64>> = Vec::with_capacity(nr);
            for _ in 0..nr {
                let mut m = vec![0.0; n3];
                match init {
                    Some(dir) => {
                        let r = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                        for i in 0..n {
                            let f = ss[i] / r;
                            m[3 * i] = dir[0] * f;
                            m[3 * i + 1] = dir[1] * f;
                            m[3 * i + 2] = dir[2] * f;
                        }
                    }
                    None => random_state(&mut master, ss, &mut m),
                }
                let mut g = vec![0.0; n3];
                d.gradient(&m, &mut g);
                ms.push(m);
                gs.push(g);
            }
            let mut rngs: Vec<Pcg64> = (0..nr).map(|r| stream(seed, 1 + r as u64)).collect();
            let mut orders: Vec<Vec<usize>> = (0..nr).map(|_| Vec::with_capacity(n)).collect();
            let mut energies: Vec<f64> = vec![0.0; nr];

            let mut sum_e = vec![0.0; nr];
            let mut sum_e2 = vec![0.0; nr];
            let mut sum_m = vec![0.0; 3 * nr];
            let mut sum_m2 = vec![0.0; nr];
            let mut nmeas = 0usize;
            let mut acc_tot = 0.0;
            let mut acc_cnt = 0usize;
            let mut accs = vec![0.0; nr];

            for sweep_i in 0..n_sweeps {
                // Independent replicas: one sweep each, in parallel.
                ms.par_iter_mut()
                    .zip(gs.par_iter_mut())
                    .zip(rngs.par_iter_mut())
                    .zip(orders.par_iter_mut())
                    .zip(accs.par_iter_mut())
                    .enumerate()
                    .for_each(|(r, ((((m, g), rng), order), acc))| {
                        *acc = sweep_impl(&d, m, g, betas[r], ss, rng, flip, order);
                    });
                for a in &accs {
                    acc_tot += a;
                    acc_cnt += 1;
                }
                // Replica swaps between adjacent temperatures.
                if swap_every > 0 && sweep_i % swap_every == 0 {
                    for r in 0..nr {
                        energies[r] = d.energy(&ms[r]);
                    }
                    for r in 0..nr.saturating_sub(1) {
                        let delta = (betas[r] - betas[r + 1]) * (energies[r] - energies[r + 1]);
                        if delta >= 0.0 || master.random::<f64>() < delta.exp() {
                            ms.swap(r, r + 1);
                            gs.swap(r, r + 1);
                            energies.swap(r, r + 1);
                        }
                    }
                }
                if sweep_i >= n_equil && (sweep_i - n_equil) % measure_every == 0 {
                    for r in 0..nr {
                        let e = d.energy(&ms[r]);
                        let mut mv = [0.0; 3];
                        for i in 0..n {
                            mv[0] += ms[r][3 * i];
                            mv[1] += ms[r][3 * i + 1];
                            mv[2] += ms[r][3 * i + 2];
                        }
                        sum_e[r] += e;
                        sum_e2[r] += e * e;
                        sum_m[3 * r] += mv[0];
                        sum_m[3 * r + 1] += mv[1];
                        sum_m[3 * r + 2] += mv[2];
                        sum_m2[r] += mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2];
                    }
                    nmeas += 1;
                }
            }
            let rate = if acc_cnt > 0 { acc_tot / acc_cnt as f64 } else { 0.0 };
            (sum_e, sum_e2, sum_m, sum_m2, nmeas, rate)
        })
    })?;
    let (sum_e, sum_e2, sum_m, sum_m2, nmeas, rate) = out;
    let sum_m = Array2::from_shape_vec((nr, 3), sum_m)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        sum_e.into_pyarray(py),
        sum_e2.into_pyarray(py),
        sum_m.into_pyarray(py),
        sum_m2.into_pyarray(py),
        nmeas,
        rate,
    ))
}

/// Deterministic Landau-Lifshitz trajectory from `m0`: (n_rec, N, 3) recorded every
/// `record_every` steps (the state BEFORE each step, as the Python does), plus the
/// number of implicit-midpoint steps that did not converge (0 for rk4).
#[pyfunction]
#[pyo3(signature = (h, b, s, m0, dt, n_steps, record_every = 1, integrator = "rk4"))]
#[allow(clippy::too_many_arguments)]
pub fn evolve<'py>(
    py: Python<'py>,
    h: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    s: PyReadonlyArray1<'py, f64>,
    m0: PyReadonlyArray2<'py, f64>,
    dt: f64,
    n_steps: usize,
    record_every: usize,
    integrator: &str,
) -> PyResult<(Bound<'py, PyArray3<f64>>, usize)> {
    let d = dense(&h, &b)?;
    let n = d.n();
    let n3 = d.n3;
    let ss = check_spins(&s, n)?;
    check_m(m0.shape(), n, "m0")?;
    let integ = parse_integrator(integrator)?;
    if record_every == 0 {
        return Err(PyValueError::new_err("record_every must be >= 1"));
    }
    let mut m: Vec<f64> = m0.as_array().iter().copied().collect();
    let n_rec = n_steps.div_ceil(record_every);
    let (rec, bad) = py.detach(move || {
        let mut rec = vec![0.0; n_rec * n3];
        let bad = evolve_impl(&d, &mut m, ss, dt, n_steps, record_every, integ, &mut rec);
        (rec, bad)
    });
    let arr = Array3::from_shape_vec((n_rec, n, 3), rec)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((arr.into_pyarray(py), bad))
}

/// `n_steps` Heun steps of the Langevin thermostat from `m0`, continuing the
/// stream `rng`; returns the final (N, 3) configuration.
#[pyfunction]
#[pyo3(signature = (h, b, s, m0, dt, kt, damping, n_steps, rng))]
#[allow(clippy::too_many_arguments)]
pub fn langevin<'py>(
    py: Python<'py>,
    h: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    s: PyReadonlyArray1<'py, f64>,
    m0: PyReadonlyArray2<'py, f64>,
    dt: f64,
    kt: f64,
    damping: f64,
    n_steps: usize,
    mut rng: PyRefMut<'py, PyRng>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d = dense(&h, &b)?;
    let n = d.n();
    let n3 = d.n3;
    let ss = check_spins(&s, n)?;
    check_m(m0.shape(), n, "m0")?;
    if !(kt >= 0.0) {
        return Err(PyValueError::new_err("kT must be >= 0"));
    }
    let mut m: Vec<f64> = m0.as_array().iter().copied().collect();
    let r = &mut rng.inner;
    let m = py.detach(move || {
        let mut w = Scratch::new(n3);
        for _ in 0..n_steps {
            langevin_step_impl(&d, &mut m, ss, dt, kt, damping, r, &mut w);
        }
        m
    });
    let arr = Array2::from_shape_vec((n, 3), m)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// The Wang-Landau move loop (pyMagCalc `wang_landau`, from the point where a
/// configuration inside the window has been found). `m0` must already lie in
/// [e_min, e_max). Returns (log_g, visits, f_final, n_refinements, sweeps_done).
#[pyfunction]
#[pyo3(signature = (h, b, s, m0, e_min, e_max, n_bins, f_init, f_final, flatness,
                    sweeps_per_check, max_sweeps, seed))]
#[allow(clippy::too_many_arguments)]
pub fn wang_landau<'py>(
    py: Python<'py>,
    h: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    s: PyReadonlyArray1<'py, f64>,
    m0: PyReadonlyArray2<'py, f64>,
    e_min: f64,
    e_max: f64,
    n_bins: usize,
    f_init: f64,
    f_final: f64,
    flatness: f64,
    sweeps_per_check: usize,
    max_sweeps: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>, f64, usize, usize)> {
    let d = dense(&h, &b)?;
    let n = d.n();
    let n3 = d.n3;
    let ss = check_spins(&s, n)?;
    check_m(m0.shape(), n, "m0")?;
    if !(e_max > e_min) || n_bins == 0 {
        return Err(PyValueError::new_err(format!(
            "need e_max > e_min and n_bins >= 1, got [{e_min}, {e_max}], {n_bins} bins"
        )));
    }
    let mut m: Vec<f64> = m0.as_array().iter().copied().collect();
    let width = (e_max - e_min) / n_bins as f64;
    let bin_of = move |e: f64| -> Option<usize> {
        if e < e_min || e >= e_max {
            None
        } else {
            Some((((e - e_min) / width) as usize).min(n_bins - 1))
        }
    };
    let e0 = d.energy(&m);
    if bin_of(e0).is_none() {
        return Err(PyValueError::new_err(format!(
            "m0 has energy {e0}, outside the window [{e_min}, {e_max})"
        )));
    }
    let out = py.detach(move || {
        let mut rng = stream(seed, 0);
        let mut g = vec![0.0; n3];
        d.gradient(&m, &mut g);
        let mut log_g: Vec<f64> = vec![0.0; n_bins];
        let mut hist = vec![0i64; n_bins];
        let mut visits = vec![0i64; n_bins];
        let mut e = e0;
        let mut f = f_init;
        let mut n_ref = 0usize;
        let mut swept = 0usize;
        while f > f_final && swept < max_sweeps {
            for _ in 0..sweeps_per_check {
                swept += 1;
                for _ in 0..n {
                    let a = rng.random_range(0..n);
                    let v = random_unit(&mut rng, ss[a]);
                    let de = d.move_delta(&m, &g, a, v);
                    let e_new = e + de;
                    let mut ob = bin_of(e).expect("current energy is inside the window");
                    if let Some(nb) = bin_of(e_new) {
                        let lr = log_g[ob] - log_g[nb];
                        if lr >= 0.0 || rng.random::<f64>() < lr.exp() {
                            d.apply_move(&mut m, &mut g, a, v);
                            e = e_new;
                            ob = nb;
                        }
                    }
                    log_g[ob] += f;
                    hist[ob] += 1;
                    visits[ob] += 1;
                }
            }
            let occ: Vec<i64> = hist.iter().copied().filter(|h| *h > 0).collect();
            if occ.len() > 1 {
                let min = *occ.iter().min().unwrap() as f64;
                let mean = occ.iter().sum::<i64>() as f64 / occ.len() as f64;
                if min >= flatness * mean {
                    f *= 0.5;
                    hist.iter_mut().for_each(|h| *h = 0);
                    n_ref += 1;
                }
            }
        }
        (log_g, visits, f, n_ref, swept)
    });
    let (log_g, visits, f, n_ref, swept) = out;
    Ok((log_g.into_pyarray(py), visits.into_pyarray(py), f, n_ref, swept))
}

/// `n_traj` independent thermal trajectories, in parallel (rayon): each starts
/// from a random state, thermalizes by `therm_sweeps` uniform Metropolis sweeps at
/// `kt`, then evolves `n_steps` of rk4/midpoint recording every `record_every`.
/// Trajectory i uses the stream (seed, traj_start + i), so a run split into chunks
/// gives the same trajectories as one call. Returns (n_traj, n_rec, N, 3) and the
/// count of non-converged midpoint steps.
#[pyfunction]
#[pyo3(signature = (h, b, s, kt, traj_start, n_traj, therm_sweeps, dt, n_steps,
                    record_every = 1, integrator = "rk4", seed = 0, n_workers = None))]
#[allow(clippy::too_many_arguments)]
pub fn sampled_trajectories<'py>(
    py: Python<'py>,
    h: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    s: PyReadonlyArray1<'py, f64>,
    kt: f64,
    traj_start: usize,
    n_traj: usize,
    therm_sweeps: usize,
    dt: f64,
    n_steps: usize,
    record_every: usize,
    integrator: &str,
    seed: u64,
    n_workers: Option<usize>,
) -> PyResult<(Bound<'py, PyArray4<f64>>, usize)> {
    let d = dense(&h, &b)?;
    let n = d.n();
    let n3 = d.n3;
    let ss = check_spins(&s, n)?;
    let integ = parse_integrator(integrator)?;
    if record_every == 0 {
        return Err(PyValueError::new_err("record_every must be >= 1"));
    }
    if !(kt > 0.0) {
        return Err(PyValueError::new_err("kT must be positive"));
    }
    let beta = 1.0 / kt;
    let n_rec = n_steps.div_ceil(record_every);
    let per = n_rec * n3;
    let (rec, bad) = py.detach(move || -> PyResult<_> {
        with_pool(n_workers, move || {
            let mut rec = vec![0.0; n_traj * per];
            let bad: usize = rec
                .par_chunks_mut(per.max(1))
                .enumerate()
                .map(|(i, chunk)| {
                    let mut rng = stream(seed, (traj_start + i) as u64);
                    let mut m = vec![0.0; n3];
                    random_state(&mut rng, ss, &mut m);
                    let mut g = vec![0.0; n3];
                    d.gradient(&m, &mut g);
                    let mut order = Vec::with_capacity(n);
                    for _ in 0..therm_sweeps {
                        sweep_impl(&d, &mut m, &mut g, beta, ss, &mut rng, false, &mut order);
                    }
                    evolve_impl(&d, &mut m, ss, dt, n_steps, record_every, integ, chunk)
                })
                .sum();
            (rec, bad)
        })
    })?;
    let arr = Array4::from_shape_vec((n_traj, n_rec, n, 3), rec)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((arr.into_pyarray(py), bad))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRng>()?;
    m.add_function(wrap_pyfunction!(move_energy_change, m)?)?;
    m.add_function(wrap_pyfunction!(sweep, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_tempering, m)?)?;
    m.add_function(wrap_pyfunction!(evolve, m)?)?;
    m.add_function(wrap_pyfunction!(langevin, m)?)?;
    m.add_function(wrap_pyfunction!(wang_landau, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_trajectories, m)?)?;
    Ok(())
}
