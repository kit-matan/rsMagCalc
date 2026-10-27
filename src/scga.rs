//! K5 -- the self-consistent Gaussian approximation (SCGA): the per-q Fourier
//! exchange matrix J(q), its eigenvalues over the lambda mesh, the per-site
//! sum-rule traces of (lambda + J(q))^{-1}, and the 3 x 3 magnetic-moment
//! correlation tensors of the diffuse S(q).
//!
//! Replaces, in pyMagCalc `scga.py`: the per-q body of `fourier_exchange_matrix`
//! (as a batch), the mesh loop of `solve_lambda` that collects the J(q)
//! eigenvalues, the mesh loop of `sublattice_sum_rule_residuals`, and the q loop
//! of `scga_intensities` up to the cross-section contraction (`_contract` stays
//! in NumPy). The 1-D root solve for lambda, its bracketing and the refusal of
//! inequivalent sublattices stay in Python: the brentq callback is already a
//! vectorized sum over the cached eigenvalues, so lambda moves only through the
//! eigenvalues this module returns.
//!
//! Conventions, shared with the Python side and never re-derived here:
//!   * J(q)_{ia,jb} = sum over the directed bonds (i, jc, R, M) of M_ab e^{i q.R},
//!     R the CARTESIAN image of the integer cell offset (NOT the intracell
//!     position -- that phase is the observable's, as in Sunny), summed in bond
//!     order and then made Hermitian as 0.5 (J + J^H);
//!   * the intensity tensor is C = kT W^H (lambda + J)^{-1} W with W the 3N x 3
//!     stack of blocks f_j e^{i q.r_j} g_j^T, i.e. exactly
//!     kT sum_ij f_i f_j conj(e^{i q.r_i}) e^{i q.r_j} g_i blk_ij g_j^T for a REAL
//!     form factor f and a REAL g-tensor (g = I for the bare spin correlations);
//!   * form factors are evaluated in Python (`magcalc.form_factors`) and passed
//!     in per q and site, so this kernel cannot drift from that table.
//!
//! Linear algebra is the platform LAPACK (`lapack.rs`). lambda + J(q) is
//! Hermitian positive definite for lambda above the mesh's lower bound, so the
//! solve is a Cholesky (`zpotrf` + `zpotrs`, and `ztrtri` for the inverse's
//! diagonal) rather than NumPy's explicit inverse; a q off the mesh can still
//! leave it indefinite, and then the solve is NumPy's own LU (`zgesv`). The
//! eigenvalues are `zheevd` with jobz = 'N' on the lower triangle of the
//! column-major J(q), with the queried workspace -- the call `eigvalsh` makes.
//! Threads: rayon over q (`n_workers` sizes the pool).
use ndarray::{Array2, Array3};
use numpy::{Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1,
            PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::lapack;
use crate::sun_lswt::{capped_pool, perr, qs_vec};

type C = Complex64;

const ZERO: C = C { re: 0.0, im: 0.0 };

/// A pyMagCalc `scga._exchange_cache`, packed once: the directed bond list with
/// Cartesian image offsets, the site positions, the g-tensors and spin lengths.
#[pyclass(name = "ScgaModel", module = "rsmagcalc._core", frozen)]
pub struct ScgaModel {
    /// Sites per cell.
    n: usize,
    bi: Vec<usize>,
    bj: Vec<usize>,
    r: Vec<[f64; 3]>,
    /// Bond 3 x 3 exchange matrices, row major.
    m: Vec<[f64; 9]>,
    pos: Vec<[f64; 3]>,
    /// Per-site 3 x 3 g-tensors, row major.
    g: Vec<[f64; 9]>,
    spins: Vec<f64>,
}

#[inline]
fn phase(q: [f64; 3], r: [f64; 3]) -> C {
    let x = q[0] * r[0] + q[1] * r[1] + q[2] * r[2];
    C::new(x.cos(), x.sin())
}

fn lapack_err(what: &str, info: lapack::Int) -> PyErr {
    PyRuntimeError::new_err(format!("{what} failed, info = {info}"))
}

impl ScgaModel {
    #[inline]
    fn n3(&self) -> usize {
        3 * self.n
    }

    /// J(q), row major (3N x 3N), into `a`; `raw` is scratch of the same size.
    fn build(&self, q: [f64; 3], raw: &mut [C], a: &mut [C]) {
        let n3 = self.n3();
        for z in raw.iter_mut() {
            *z = ZERO;
        }
        for k in 0..self.bi.len() {
            let ph = phase(q, self.r[k]);
            let (r0, c0) = (3 * self.bi[k], 3 * self.bj[k]);
            let mk = &self.m[k];
            for x in 0..3 {
                for y in 0..3 {
                    raw[(r0 + x) * n3 + c0 + y] += ph * mk[3 * x + y];
                }
            }
        }
        for r in 0..n3 {
            for c in 0..n3 {
                a[r * n3 + c] = (raw[r * n3 + c] + raw[c * n3 + r].conj()) * 0.5;
            }
        }
    }

    /// lambda I + J(q), COLUMN major, into `mcm` (J(q) row major in `a`).
    fn shifted_colmajor(&self, a: &[C], lam: f64, mcm: &mut [C]) {
        let n3 = self.n3();
        for r in 0..n3 {
            for c in 0..n3 {
                mcm[c * n3 + r] = a[r * n3 + c];
            }
            mcm[r * n3 + r].re += lam;
        }
    }

    /// Per site, Tr_3 of the diagonal 3 x 3 block of (lambda + J(q))^{-1}, into `tr`.
    fn site_traces_at(&self, q: [f64; 3], lam: f64, tr: &mut [f64]) -> PyResult<()> {
        let n3 = self.n3();
        let mut raw = vec![ZERO; n3 * n3];
        let mut a = vec![ZERO; n3 * n3];
        let mut mcm = vec![ZERO; n3 * n3];
        self.build(q, &mut raw, &mut a);
        self.shifted_colmajor(&a, lam, &mut mcm);
        let mut diag = vec![0.0f64; n3];
        if lapack::zpotrf_upper(n3, &mut mcm).is_ok() {
            // M = U^H U, so M^{-1} = U^{-1} U^{-H} and (M^{-1})_kk is the squared
            // norm of row k of U^{-1} (upper triangular: columns k..n3).
            lapack::ztrtri_upper(n3, &mut mcm).map_err(|i| lapack_err("ztrtri", i))?;
            for (k, d) in diag.iter_mut().enumerate() {
                let mut s = 0.0;
                for c in k..n3 {
                    s += mcm[c * n3 + k].norm_sqr();
                }
                *d = s;
            }
        } else {
            // Indefinite: NumPy's `inv` (LU) on the same matrix.
            self.shifted_colmajor(&a, lam, &mut mcm);
            let mut inv = vec![ZERO; n3 * n3];
            for k in 0..n3 {
                inv[k * n3 + k] = C::new(1.0, 0.0);
            }
            lapack::zgesv(n3, n3, &mut mcm, &mut inv).map_err(|i| lapack_err("zgesv", i))?;
            for (k, d) in diag.iter_mut().enumerate() {
                *d = inv[k * n3 + k].re;
            }
        }
        for (i, t) in tr.iter_mut().enumerate() {
            *t = diag[3 * i] + diag[3 * i + 1] + diag[3 * i + 2];
        }
        Ok(())
    }

    /// `site_traces_at` over a q batch, in parallel: (nq * N), q major.
    fn traces(&self, qs: &[[f64; 3]], lam: f64, n_workers: Option<usize>) -> PyResult<Vec<f64>> {
        let (n, n3) = (self.n, self.n3());
        let mut out = vec![0.0f64; qs.len() * n];
        capped_pool(n_workers, 4 * n3 * n3 * 16, || -> PyResult<()> {
            out.par_chunks_mut(n).zip(qs.par_iter())
                .try_for_each(|(t, &q)| self.site_traces_at(q, lam, t))
        })??;
        Ok(out)
    }

    /// C(q) = kT W^H (lambda + J(q))^{-1} W, row major 3 x 3, into `out`.
    fn correlation_at(&self, q: [f64; 3], lam: f64, kt: f64, ff: &[f64], apply_g: bool,
                      out: &mut [C]) -> PyResult<()> {
        let n3 = self.n3();
        let mut raw = vec![ZERO; n3 * n3];
        let mut a = vec![ZERO; n3 * n3];
        let mut mcm = vec![ZERO; n3 * n3];
        self.build(q, &mut raw, &mut a);
        self.shifted_colmajor(&a, lam, &mut mcm);
        // W (3N x 3), column major: block j = f_j e^{i q.r_j} g_j^T.
        let mut w = vec![ZERO; 3 * n3];
        for j in 0..self.n {
            let s = phase(q, self.pos[j]) * ff[j];
            for x in 0..3 {
                for y in 0..3 {
                    // (g_j^T)[x][y] = g_j[y][x]
                    let gt = if apply_g {
                        self.g[j][3 * y + x]
                    } else if x == y {
                        1.0
                    } else {
                        0.0
                    };
                    w[y * n3 + 3 * j + x] = s * gt;
                }
            }
        }
        let mut x = w.clone();
        if lapack::zpotrf_upper(n3, &mut mcm).is_ok() {
            lapack::zpotrs_upper(n3, 3, &mcm, &mut x).map_err(|i| lapack_err("zpotrs", i))?;
        } else {
            self.shifted_colmajor(&a, lam, &mut mcm);
            lapack::zgesv(n3, 3, &mut mcm, &mut x).map_err(|i| lapack_err("zgesv", i))?;
        }
        for p in 0..3 {
            for b in 0..3 {
                let mut s = ZERO;
                for k in 0..n3 {
                    s += w[p * n3 + k].conj() * x[b * n3 + k];
                }
                out[3 * p + b] = s * kt;
            }
        }
        Ok(())
    }
}

#[pymethods]
impl ScgaModel {
    /// `bi`, `bj` (nb,): the directed bonds (i, j mod N) of `_exchange_cache`, in
    /// its order; `r` (nb, 3) their Cartesian image offsets; `m` (nb, 3, 3) the
    /// exchange matrices; `pos` (N, 3) Cartesian site positions; `g` (N, 3, 3);
    /// `spins` (N,).
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(bi: PyReadonlyArray1<'_, i64>, bj: PyReadonlyArray1<'_, i64>,
           r: PyReadonlyArray2<'_, f64>, m: PyReadonlyArray3<'_, f64>,
           pos: PyReadonlyArray2<'_, f64>, g: PyReadonlyArray3<'_, f64>,
           spins: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let (bi, bj, r, m) = (bi.as_array(), bj.as_array(), r.as_array(), m.as_array());
        let (pos, g, spins) = (pos.as_array(), g.as_array(), spins.as_array());
        let n = pos.nrows();
        let nb = bi.len();
        if n == 0 {
            return Err(perr("ScgaModel needs at least one site"));
        }
        if pos.ncols() != 3 || g.shape() != [n, 3, 3] || spins.len() != n {
            return Err(perr(format!(
                "pos (N, 3), g (N, 3, 3) and spins (N,) disagree: {:?}, {:?}, {}",
                pos.shape(), g.shape(), spins.len())));
        }
        if bj.len() != nb || r.shape() != [nb, 3] || m.shape() != [nb, 3, 3] {
            return Err(perr(format!(
                "bond arrays disagree: bi {nb}, bj {}, r {:?}, m {:?}",
                bj.len(), r.shape(), m.shape())));
        }
        let idx = |v: i64| -> PyResult<usize> {
            if v < 0 || v as usize >= n {
                Err(perr(format!("bond site index {v} out of range for N = {n}")))
            } else {
                Ok(v as usize)
            }
        };
        let mat = |a: ndarray::ArrayView2<'_, f64>| -> [f64; 9] {
            let mut o = [0.0; 9];
            for x in 0..3 {
                for y in 0..3 {
                    o[3 * x + y] = a[[x, y]];
                }
            }
            o
        };
        Ok(ScgaModel {
            n,
            bi: bi.iter().map(|&v| idx(v)).collect::<PyResult<_>>()?,
            bj: bj.iter().map(|&v| idx(v)).collect::<PyResult<_>>()?,
            r: r.rows().into_iter().map(|x| [x[0], x[1], x[2]]).collect(),
            m: m.outer_iter().map(mat).collect(),
            pos: pos.rows().into_iter().map(|x| [x[0], x[1], x[2]]).collect(),
            g: g.outer_iter().map(mat).collect(),
            spins: spins.to_vec(),
        })
    }

    #[getter]
    fn n_sites(&self) -> usize {
        self.n
    }

    /// J(q) per q: (nq, 3N, 3N) complex (`fourier_exchange_matrix`).
    #[pyo3(signature = (qs, n_workers=None))]
    fn fourier<'py>(&self, py: Python<'py>, qs: PyReadonlyArray2<'py, f64>,
                    n_workers: Option<usize>) -> PyResult<Bound<'py, PyArray3<C>>> {
        let qs = qs_vec(&qs)?;
        let (n3, nq) = (self.n3(), qs.len());
        let out = py.detach(move || -> PyResult<Vec<C>> {
            let mut out = vec![ZERO; nq * n3 * n3];
            capped_pool(n_workers, 2 * n3 * n3 * 16, || {
                out.par_chunks_mut((n3 * n3).max(1)).zip(qs.par_iter()).for_each(|(a, &q)| {
                    let mut raw = vec![ZERO; n3 * n3];
                    self.build(q, &mut raw, a);
                });
            })?;
            Ok(out)
        })?;
        Ok(Array3::from_shape_vec((nq, n3, n3), out).map_err(|e| perr(e.to_string()))?
            .into_pyarray(py))
    }

    /// The 3N eigenvalues of J(q) per q, ascending: (nq, 3N) (`eigvalsh`).
    #[pyo3(signature = (qs, n_workers=None))]
    fn eigenvalues<'py>(&self, py: Python<'py>, qs: PyReadonlyArray2<'py, f64>,
                        n_workers: Option<usize>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let qs = qs_vec(&qs)?;
        let (n3, nq) = (self.n3(), qs.len());
        let out = py.detach(move || -> PyResult<Vec<f64>> {
            let mut out = vec![0.0f64; nq * n3];
            capped_pool(n_workers, 3 * n3 * n3 * 16, || -> PyResult<()> {
                out.par_chunks_mut(n3).zip(qs.par_iter()).try_for_each(|(w, &q)| {
                    let mut raw = vec![ZERO; n3 * n3];
                    let mut a = vec![ZERO; n3 * n3];
                    let mut acm = vec![ZERO; n3 * n3];
                    self.build(q, &mut raw, &mut a);
                    for r in 0..n3 {
                        for c in 0..n3 {
                            acm[c * n3 + r] = a[r * n3 + c];
                        }
                    }
                    lapack::zheevd_values(n3, &mut acm, w, b'L')
                        .map_err(|i| lapack_err("zheevd", i))
                })
            })??;
            Ok(out)
        })?;
        Ok(Array2::from_shape_vec((nq, n3), out).map_err(|e| perr(e.to_string()))?
            .into_pyarray(py))
    }

    /// Per q and site, Tr_3 [(lambda + J(q))^{-1}]_{ii}: (nq, N).
    #[pyo3(signature = (qs, lam, n_workers=None))]
    fn site_traces<'py>(&self, py: Python<'py>, qs: PyReadonlyArray2<'py, f64>, lam: f64,
                        n_workers: Option<usize>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let qs = qs_vec(&qs)?;
        let nq = qs.len();
        let out = py.detach(move || self.traces(&qs, lam, n_workers))?;
        Ok(Array2::from_shape_vec((nq, self.n), out).map_err(|e| perr(e.to_string()))?
            .into_pyarray(py))
    }

    /// The relative per-site sum-rule residuals over the mesh,
    /// r_i = [(1/N_q) sum_q Tr_3 ((lambda + J(q))^{-1})_ii] / (S_i^2 / kT) - 1,
    /// summed over q in mesh order as `sublattice_sum_rule_residuals` does: (N,).
    #[pyo3(signature = (mesh, lam, kt, n_workers=None))]
    fn sum_rule_residuals<'py>(&self, py: Python<'py>, mesh: PyReadonlyArray2<'py, f64>,
                               lam: f64, kt: f64, n_workers: Option<usize>)
                               -> PyResult<Bound<'py, PyArray1<f64>>> {
        let qs = qs_vec(&mesh)?;
        let (n, nq) = (self.n, qs.len());
        let tr = py.detach(|| self.traces(&qs, lam, n_workers))?;
        let mut acc = vec![0.0f64; n];
        for row in tr.chunks(n) {
            for (a, &t) in acc.iter_mut().zip(row) {
                *a += t;
            }
        }
        let r: Vec<f64> = acc.iter().zip(&self.spins)
            .map(|(&a, &s)| (a / nq as f64) / (s * s / kt) - 1.0)
            .collect();
        Ok(r.into_pyarray(py))
    }

    /// The magnetic-moment (or, without `apply_g`, spin) correlation tensors
    /// C^{ab}(q) of `scga_intensities` before the cross-section contraction:
    /// (nq, 3, 3) complex. `ff` (nq, N): the form factor per q and site (None: 1).
    #[pyo3(signature = (qs, lam, kt, ff=None, apply_g=true, n_workers=None))]
    #[allow(clippy::too_many_arguments)]
    fn correlations<'py>(&self, py: Python<'py>, qs: PyReadonlyArray2<'py, f64>, lam: f64,
                         kt: f64, ff: Option<PyReadonlyArray2<'py, f64>>, apply_g: bool,
                         n_workers: Option<usize>) -> PyResult<Bound<'py, PyArray3<C>>> {
        let qs = qs_vec(&qs)?;
        let (n, n3, nq) = (self.n, self.n3(), qs.len());
        let ff: Vec<f64> = match ff {
            None => vec![1.0; nq * n],
            Some(f) => {
                let f = f.as_array();
                if f.shape() != [nq, n] {
                    return Err(perr(format!("ff must be ({nq}, {n}), got {:?}", f.shape())));
                }
                f.iter().copied().collect()
            }
        };
        let out = py.detach(move || -> PyResult<Vec<C>> {
            let mut out = vec![ZERO; nq * 9];
            capped_pool(n_workers, 4 * n3 * n3 * 16, || -> PyResult<()> {
                out.par_chunks_mut(9).zip(qs.par_iter()).enumerate()
                    .try_for_each(|(iq, (c, &q))| {
                        self.correlation_at(q, lam, kt, &ff[iq * n..(iq + 1) * n], apply_g, c)
                    })
            })??;
            Ok(out)
        })?;
        Ok(Array3::from_shape_vec((nq, 3, 3), out).map_err(|e| perr(e.to_string()))?
            .into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("ScgaModel(N={}, bonds={})", self.n, self.bi.len())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ScgaModel>()?;
    Ok(())
}
