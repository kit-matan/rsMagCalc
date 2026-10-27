//! K4 -- the KPM Chebyshev recursion for the SU(N) LSWT structure factor, and the
//! power-iteration spectral bound that scales it.
//!
//! Replaces, in pyMagCalc `sun/kpm.py`: the moment loop of `kpm_sqw`
//! (beta_{m+1} = 2 A beta_m - beta_{m-1} with A = conj(g H2)/gamma on the three
//! observable vectors, mu_m^{ab} = v_a^H beta_m^b) and `spectral_bound`'s power
//! iteration on D^H D. The Chebyshev coefficients, the +omega cutoff and its
//! smearing, the energy-grid reconstruction and the cross-section contraction stay
//! in NumPy: they are cheap, and the Jackson/cutoff choices are what the tests pin.
//!
//! Conventions, shared with the Python side and never re-derived here:
//!   * the recursion runs on conj(D-hat), NOT D-hat (kpm.py says why: the two agree
//!     for a real D-hat, i.e. every collinear inversion-symmetric model, and differ
//!     on a non-collinear supercell, where D-hat put ~5 % of the weight on bands that
//!     carry none);
//!   * the starting vectors are g v_b with g = diag(I_D, -I_D), and the moments are
//!     v_a^H beta_m^b, assembled from the ANNIHILATION block (Python transposes them
//!     before the contraction; that transpose is the sign of `chiral`);
//!   * gamma <= 0 or non-finite (H(q) identically zero, a Goldstone point) is
//!     replaced by 1, exactly as `kpm_sqw` does; the Python side issues the warning,
//!     so the RAW bound is returned alongside;
//!   * the moment count, when not fixed, is NumPy's
//!     max(round((2 gamma / (0.5 fwhm)) * (-log10 tol)), 8) with round-half-even
//!     (Python's `round`), from the two factors Python passes, in Python's order of
//!     operations, so it is the same integer; the Python side re-derives it and
//!     refuses a mismatch.
//!
//! Where the speed comes from: H(q) of a supercell is SPARSE (S09 at L = 30: 23 400
//! nonzeros of 3.24 M, 0.7 %), and NumPy multiplies it densely. The operator is
//! scanned once per q into CSR when it is at most a quarter full, so the recursion is
//! O(nnz M) instead of O(D^2 M); a denser one (a small entangled cell) is multiplied
//! with the platform zgemm, as NumPy does. The exact zeros skipped are exact zeros of
//! the dense product too, so the two paths differ only in summation order. H(q) itself
//! comes from K3's `SunModel::build_h`, so no 2D x 2D matrix crosses into Python.
use ndarray::{Array1, Array3, Array4};
use numpy::{Complex64, IntoPyArray, PyArray1, PyArray3, PyArray4, PyReadonlyArray1,
            PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::lapack;
use crate::sun_lswt::{capped_pool, perr, qs_vec, SunModel};

type C = Complex64;

const ZERO: C = C { re: 0.0, im: 0.0 };

/// The operator a recursion multiplies by, held densely (row major) or as CSR.
enum Op {
    Dense { a: Vec<C>, n: usize },
    Csr { ptr: Vec<usize>, idx: Vec<u32>, val: Vec<C>, n: usize },
}

impl Op {
    /// `h` (row major n x n) mapped entrywise to conj(h)/scale or h/scale. `sparse`:
    /// Some(true/false) forces the storage, None picks CSR at <= 25 % fill.
    fn new(h: &[C], n: usize, conj: bool, scale: f64, sparse: Option<bool>) -> Op {
        let map = |z: C| if conj { z.conj() / scale } else { z / scale };
        let use_csr = sparse.unwrap_or_else(|| {
            let nnz = h.iter().filter(|z| z.re != 0.0 || z.im != 0.0).count();
            4 * nnz <= n * n
        });
        if !use_csr {
            return Op::Dense { a: h.iter().map(|&z| map(z)).collect(), n };
        }
        let mut ptr = Vec::with_capacity(n + 1);
        let mut idx = Vec::new();
        let mut val = Vec::new();
        ptr.push(0);
        for r in 0..n {
            for (c, &z) in h[r * n..(r + 1) * n].iter().enumerate() {
                if z.re != 0.0 || z.im != 0.0 {
                    idx.push(c as u32);
                    val.push(map(z));
                }
            }
            ptr.push(idx.len());
        }
        Op::Csr { ptr, idx, val, n }
    }

    /// y = A x for three right-hand sides, x and y (n, 3) row major.
    fn mul3(&self, x: &[C], y: &mut [C]) {
        match self {
            // In column-major terms y^T (3 x n) = x^T (3 x n) . A^T, and the row-major
            // A buffer IS A^T column major.
            Op::Dense { a, n } => lapack::zgemm(3, *n, *n, x, 3, a, *n, false, y, 3),
            Op::Csr { ptr, idx, val, n } => {
                for r in 0..*n {
                    let mut s = [ZERO; 3];
                    for p in ptr[r]..ptr[r + 1] {
                        let (a, k) = (val[p], idx[p] as usize * 3);
                        s[0] += a * x[k];
                        s[1] += a * x[k + 1];
                        s[2] += a * x[k + 2];
                    }
                    y[r * 3..r * 3 + 3].copy_from_slice(&s);
                }
            }
        }
    }

    /// y = A x.
    fn mul1(&self, x: &[C], y: &mut [C]) {
        match self {
            Op::Dense { a, n } => lapack::zgemm(1, *n, *n, x, 1, a, *n, false, y, 1),
            Op::Csr { ptr, idx, val, n } => {
                for r in 0..*n {
                    let mut s = ZERO;
                    for p in ptr[r]..ptr[r + 1] {
                        s += val[p] * x[idx[p] as usize];
                    }
                    y[r] = s;
                }
            }
        }
    }

    /// y = A^H x.
    fn mul1_h(&self, x: &[C], y: &mut [C]) {
        match self {
            // op(B) = B^H with B = the buffer read column major = A^T, so conj(A).
            Op::Dense { a, n } => lapack::zgemm(1, *n, *n, x, 1, a, *n, true, y, 1),
            Op::Csr { ptr, idx, val, n } => {
                for z in y.iter_mut() {
                    *z = ZERO;
                }
                for r in 0..*n {
                    let xr = x[r];
                    for p in ptr[r]..ptr[r + 1] {
                        y[idx[p] as usize] += val[p].conj() * xr;
                    }
                }
            }
        }
    }
}

/// `spectral_bound`: pad * sqrt(lambda_max(H^H H)) by `n_iter` power steps from the
/// normalized start vector `x0` (NumPy's, generated on the Python side).
fn power_bound(h: &Op, n: usize, x0: &[C], n_iter: usize, pad: f64) -> f64 {
    let mut x = x0.to_vec();
    let mut u = vec![ZERO; n];
    let mut y = vec![ZERO; n];
    let mut lam = 0.0f64;
    for _ in 0..n_iter {
        h.mul1(&x, &mut u);
        h.mul1_h(&u, &mut y);
        lam = x.iter().zip(&y).map(|(a, b)| (a.conj() * b).re).sum();
        let nrm = y.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if nrm < 1e-300 {
            break;
        }
        for (xi, yi) in x.iter_mut().zip(&y) {
            *xi = yi / nrm;
        }
    }
    pad * lam.max(0.0).sqrt()
}

/// The gamma the recursion uses: the bound, or 1 at a Goldstone point (kpm.py).
#[inline]
fn effective_gamma(g: f64) -> f64 {
    if g.is_finite() && g > 0.0 { g } else { 1.0 }
}

/// NumPy's moment count, bit for bit: max(round((2 g / a) * b), 8), half to even.
#[inline]
fn moment_rule(g: f64, a: f64, b: f64) -> usize {
    let m = (2.0 * g / a * b).round_ties_even();
    if m.is_finite() && m > 8.0 { m as usize } else { 8 }
}

/// mu_m^{ab} = v_a^H T_m(A) g v_b, m < n_mom, into `out` ((n_mom, 3, 3) row major).
/// `v` is (n, 3) row major (v[k * 3 + a]); `a` is conj(g H2)/gamma.
fn moments(a: &Op, n: usize, d: usize, v: &[C], n_mom: usize, out: &mut [C]) {
    let contract = |beta: &[C], o: &mut [C]| {
        for ab in o.iter_mut() {
            *ab = ZERO;
        }
        for k in 0..n {
            let (vk, bk) = (&v[k * 3..k * 3 + 3], &beta[k * 3..k * 3 + 3]);
            for i in 0..3 {
                let ci = vk[i].conj();
                for j in 0..3 {
                    o[i * 3 + j] += ci * bk[j];
                }
            }
        }
    };
    let mut b0: Vec<C> = (0..3 * n).map(|k| v[k] * if k / 3 < d { 1.0 } else { -1.0 }).collect();
    let mut b1 = vec![ZERO; 3 * n];
    let mut y = vec![ZERO; 3 * n];
    a.mul3(&b0, &mut b1);
    contract(&b0, &mut out[0..9]);
    if n_mom > 1 {
        contract(&b1, &mut out[9..18]);
    }
    for m in 2..n_mom {
        a.mul3(&b1, &mut y);
        for (z0, zy) in b0.iter_mut().zip(&y) {
            *z0 = 2.0 * zy - *z0;
        }
        contract(&b0, &mut out[m * 9..m * 9 + 9]);
        std::mem::swap(&mut b0, &mut b1);
    }
}

fn square(h: &PyReadonlyArray2<'_, C>) -> PyResult<(Vec<C>, usize)> {
    let a = h.as_array();
    let n = a.nrows();
    if a.ncols() != n || n == 0 {
        return Err(perr(format!("HMat must be square and non-empty, got {:?}", a.shape())));
    }
    if n % 2 != 0 {
        return Err(perr(format!("HMat must be 2D x 2D (Nambu), got odd size {n}")));
    }
    Ok((a.iter().copied().collect(), n))
}

/// Chebyshev moments of one g H2: (n_moments, 3, 3) complex. `v` is the (3, 2D)
/// observable of `kpm._observable_vectors`; `gamma` is used as given (> 0).
#[pyfunction]
#[pyo3(signature = (hmat, v, n_moments, gamma, sparse=None))]
fn kpm_moments<'py>(py: Python<'py>, hmat: PyReadonlyArray2<'py, C>,
                    v: PyReadonlyArray2<'py, C>, n_moments: usize, gamma: f64,
                    sparse: Option<bool>) -> PyResult<Bound<'py, PyArray3<C>>> {
    let (h, n) = square(&hmat)?;
    let va = v.as_array();
    if va.shape() != [3, n] {
        return Err(perr(format!("v must be (3, {n}), got {:?}", va.shape())));
    }
    if n_moments == 0 {
        return Err(perr("n_moments must be at least 1"));
    }
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(perr(format!("gamma must be finite and positive, got {gamma}")));
    }
    let mut vt = vec![ZERO; 3 * n];
    for k in 0..n {
        for a in 0..3 {
            vt[k * 3 + a] = va[[a, k]];
        }
    }
    let out = py.detach(move || {
        let op = Op::new(&h, n, true, gamma, sparse);
        let mut out = vec![ZERO; n_moments * 9];
        moments(&op, n, n / 2, &vt, n_moments, &mut out);
        out
    });
    Ok(Array3::from_shape_vec((n_moments, 3, 3), out).map_err(|e| perr(e.to_string()))?
        .into_pyarray(py))
}

/// `spectral_bound` of one g H2 from the normalized start vector `x0`.
#[pyfunction]
#[pyo3(signature = (hmat, x0, n_iter=40, pad=1.15, sparse=None))]
fn spectral_bound(py: Python<'_>, hmat: PyReadonlyArray2<'_, C>, x0: PyReadonlyArray1<'_, C>,
                  n_iter: usize, pad: f64, sparse: Option<bool>) -> PyResult<f64> {
    let (h, n) = square(&hmat)?;
    let x0: Vec<C> = x0.as_array().iter().copied().collect();
    if x0.len() != n {
        return Err(perr(format!("x0 must have {n} entries, got {}", x0.len())));
    }
    Ok(py.detach(move || power_bound(&Op::new(&h, n, false, 1.0, sparse), n, &x0, n_iter, pad)))
}

/// The batch over q, H(q) built by K3 inside: (raw gamma (nq,), n_moments (nq,),
/// moments (nq, M_max, 3, 3), zero past each q's own count).
///
/// `gammas` (nq,) fixes the bound per q; otherwise it is the power iteration from
/// `x0`. `n_moments` fixes the count; otherwise `m_rule = (0.5 fwhm, -log10 tol)`.
#[pyfunction]
#[pyo3(signature = (model, qs, x0=None, gammas=None, n_moments=None, m_rule=None,
                    n_iter=40, pad=1.15, sparse=None, n_workers=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn kpm_batch<'py>(py: Python<'py>, model: &Bound<'py, SunModel>,
                  qs: PyReadonlyArray2<'py, f64>, x0: Option<PyReadonlyArray1<'py, C>>,
                  gammas: Option<PyReadonlyArray1<'py, f64>>, n_moments: Option<usize>,
                  m_rule: Option<(f64, f64)>, n_iter: usize, pad: f64,
                  sparse: Option<bool>, n_workers: Option<usize>)
                  -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>,
                               Bound<'py, PyArray4<C>>)> {
    let m: &SunModel = model.get();
    let qs = qs_vec(&qs)?;
    let (n, d, nq) = (m.n, m.d, qs.len());
    let gammas: Option<Vec<f64>> = gammas.map(|g| g.as_array().to_vec());
    if let Some(g) = &gammas {
        if g.len() != nq {
            return Err(perr(format!("gammas must have {nq} entries, got {}", g.len())));
        }
    }
    let x0: Option<Vec<C>> = x0.map(|x| x.as_array().to_vec());
    match (&gammas, &x0) {
        (None, None) => return Err(perr("kpm_batch needs gammas or a start vector x0")),
        (None, Some(x)) if x.len() != n => {
            return Err(perr(format!("x0 must have 2D = {n} entries, got {}", x.len())))
        }
        _ => {}
    }
    match (n_moments, m_rule) {
        (None, None) => return Err(perr("kpm_batch needs n_moments or m_rule")),
        (Some(0), _) => return Err(perr("n_moments must be at least 1")),
        _ => {}
    }
    let per_q = py.detach(move || -> PyResult<Vec<(f64, usize, Vec<C>)>> {
        // H(q) dense plus the scaled operator (dense at worst).
        capped_pool(n_workers, 2 * n * n * 16, || {
            qs.par_iter().enumerate().map(|(iq, &q)| {
                let mut h = vec![ZERO; n * n];
                m.build_h(q, &mut h);
                let g_raw = match &gammas {
                    Some(g) => g[iq],
                    None => power_bound(&Op::new(&h, n, false, 1.0, sparse), n,
                                        x0.as_ref().unwrap(), n_iter, pad),
                };
                let g = effective_gamma(g_raw);
                let nm = match n_moments {
                    Some(k) => k,
                    None => {
                        let (a, b) = m_rule.unwrap();
                        moment_rule(g, a, b)
                    }
                };
                let op = Op::new(&h, n, true, g, sparse);
                drop(h);
                let mut v = vec![ZERO; 3 * n];
                m.observable(q, &mut v);
                let mut out = vec![ZERO; nm * 9];
                moments(&op, n, d, &v, nm, &mut out);
                (g_raw, nm, out)
            }).collect()
        })
    })?;
    let m_max = per_q.iter().map(|r| r.1).max().unwrap_or(n_moments.unwrap_or(0));
    let mut mom = vec![ZERO; nq * m_max * 9];
    for (iq, (_, nm, out)) in per_q.iter().enumerate() {
        mom[iq * m_max * 9..iq * m_max * 9 + nm * 9].copy_from_slice(out);
    }
    let g: Vec<f64> = per_q.iter().map(|r| r.0).collect();
    let nm: Vec<i64> = per_q.iter().map(|r| r.1 as i64).collect();
    Ok((Array1::from(g).into_pyarray(py), Array1::from(nm).into_pyarray(py),
        Array4::from_shape_vec((nq, m_max, 3, 3), mom).map_err(|e| perr(e.to_string()))?
            .into_pyarray(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kpm_moments, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_bound, m)?)?;
    m.add_function(wrap_pyfunction!(kpm_batch, m)?)?;
    Ok(())
}
