//! K3 -- SU(N) linear spin-wave theory: the per-q H(q) assembly, the Colpa
//! diagonalization, the one-magnon correlation tensors, the general eigenvalues
//! that pyMagCalc's `dispersion` / `max_imaginary` / `stability_report` reduce,
//! and the shifted-Cholesky ground-state guard.
//!
//! Replaces, in pyMagCalc `sun/lswt.py` on `SUNModel`: `hamiltonian`,
//! `_bogoliubov`, the per-q body of `structure_factor` (up to the cross-section
//! contraction, which stays in NumPy so that every engine shares ONE
//! `contract_cross_section`), `dispersion`, `max_imaginary`, and the per-q loops of
//! `is_stable_at` / `assert_stable`. The reference-state tables (`t`, `tb`, `Q`,
//! `QA`, `s0`: the local operators rotated into the |Z> basis) are computed in
//! Python by `SUNModel._prepare` and packed here once per model.
//!
//! Conventions, shared with the Python side and never re-derived here:
//!   * H(q) is g H2 with g = diag(I_D, -I_D): rows D..2D are NEGATED;
//!   * bonds are listed in BOTH directions with NO 1/2 on the hopping;
//!   * the on-site mean-field term c Q_i^a <O_j^b> carries NO phase (it is the
//!     q = 0 sum), and goes into H11 as is and into H22 TRANSPOSED;
//!   * the bond phase is exp(i q . dr) with dr the CARTESIAN displacement (the
//!     full-position gauge), so the one-magnon observable carries no intracell
//!     phase -- only the intra-unit moment offsets d_k of an entangled unit;
//!   * Colpa: H2 = C^H C (C upper), W = C g C^H symmetrised, eigh ascending, then
//!     the order [D positive descending, D negative ascending], Ep = |E|,
//!     T = C^{-1} U sqrt(Ep); the creation amplitudes are columns D.. of T, i.e.
//!     the D most NEGATIVE eigenvectors of W with energies |E_0|..|E_{D-1}|;
//!   * the guard is ONE shifted Cholesky, H2 + shift I > 0, with the shift
//!     supplied by Python (`h2_rel_tolerance` x the q-INDEPENDENT reference scale
//!     of `_reference_h2_scale`; GAP_STATUS Gap 4 #16 says why not per q).
//!
//! Linear algebra is the platform LAPACK (`lapack.rs`: Accelerate's ILP64
//! interface on macOS, the one NumPy itself calls here). Threads: rayon over q,
//! capped so that the per-q scratch (five 2D x 2D complex matrices) stays within
//! a few GB in flight; `n_workers` from Python sizes the pool.
use ndarray::{Array2, Array3, Array4};
use numpy::{Complex64, IntoPyArray, PyArray2, PyArray3, PyArray4, PyReadonlyArray1,
            PyReadonlyArray2, PyReadonlyArray3};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use std::sync::OnceLock;

use crate::classical::with_pool;
use crate::lapack;
use crate::sparse_guard::SparseGuard;

type C = Complex64;

const ZERO: C = C { re: 0.0, im: 0.0 };

create_exception!(rsmagcalc._core, NotPositiveDefinite, PyValueError,
                  "H2(q) is not positive definite: the reference state is not a minimum.");

struct Bond {
    i: usize,
    j: usize,
    dr: [f64; 3],
    /// (a, b, J[a, b]) for the nonzero entries only.
    terms: Vec<(usize, usize, f64)>,
}

struct MomentTerm {
    site: usize,
    d: [f64; 3],
    idx: [usize; 3],
}

/// A pyMagCalc `SUNModel`'s reference-state tables and bond list, packed once.
#[pyclass(name = "SunModel", module = "rsmagcalc._core", frozen)]
pub struct SunModel {
    l: usize,
    nop: usize,
    pub(crate) d: usize,
    pub(crate) n: usize,
    ms: Vec<usize>,
    offs: Vec<usize>,
    s0: Vec<C>,
    t: Vec<C>,
    tb: Vec<C>,
    q: Vec<C>,
    qoff: Vec<usize>,
    qa: Vec<C>,
    qaoff: Vec<usize>,
    qa_nz: Vec<bool>,
    bonds: Vec<Bond>,
    mt: Vec<MomentTerm>,
    n_cells: f64,
    /// The sparse guard's analysis, built on first use (`stable`).
    sparse: OnceLock<Result<SparseGuard, String>>,
}

pub(crate) fn perr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

impl SunModel {
    #[inline]
    fn t_of(&self, i: usize, a: usize) -> &[C] {
        let m = self.ms[i];
        let s = self.nop * self.offs[i] + a * m;
        &self.t[s..s + m]
    }

    #[inline]
    fn tb_of(&self, i: usize, a: usize) -> &[C] {
        let m = self.ms[i];
        let s = self.nop * self.offs[i] + a * m;
        &self.tb[s..s + m]
    }

    #[inline]
    fn q_of(&self, i: usize, a: usize) -> &[C] {
        let m2 = self.ms[i] * self.ms[i];
        let s = self.qoff[i] + a * m2;
        &self.q[s..s + m2]
    }

    #[inline]
    fn qa_of(&self, i: usize) -> &[C] {
        &self.qa[self.qaoff[i]..self.qaoff[i + 1]]
    }

    /// g H2(q), row major (2D x 2D), into `h` (overwritten).
    pub(crate) fn build_h(&self, q: [f64; 3], h: &mut [C]) {
        let n = self.n;
        for z in h.iter_mut() {
            *z = ZERO;
        }
        self.assemble(q, |r, c, v| h[r * n + c] += v);
        for r in self.d..n {
            for z in &mut h[r * n..(r + 1) * n] {
                *z = -*z;
            }
        }
    }

    /// Every contribution to H(q) BEFORE the rows D..2D are negated, i.e. to H2
    /// itself, as `add(row, col, value)`. The sequence of (row, col) -- its length
    /// and its order -- does not depend on q: `build_h` sums it into a dense
    /// matrix, and the sparse guard (`sparse_guard.rs`) records it once and then
    /// sums the same values in the same order into its own slots, so the two
    /// paths hold bit-identical matrix entries.
    #[inline(always)]
    pub(crate) fn assemble(&self, q: [f64; 3], mut add: impl FnMut(usize, usize, C)) {
        let d = self.d;
        for b in &self.bonds {
            let arg = q[0] * b.dr[0] + q[1] * b.dr[1] + q[2] * b.dr[2];
            let ph = C::new(arg.cos(), arg.sin());
            let (bi, bj) = (self.offs[b.i], self.offs[b.j]);
            let (mi, mj) = (self.ms[b.i], self.ms[b.j]);
            for &(a, bb, c) in &b.terms {
                let ti = self.t_of(b.i, a);
                let tbi = self.tb_of(b.i, a);
                let tj = self.t_of(b.j, bb);
                let tbj = self.tb_of(b.j, bb);
                let cph = ph * c;
                for m in 0..mi {
                    let (r1, r2) = (bi + m, d + bi + m);
                    let (x, xb) = (ti[m] * cph, tbi[m] * cph);
                    for nn in 0..mj {
                        add(r1, bj + nn, xb * tj[nn]); // H11: c tb_i t_j ph
                        add(r2, d + bj + nn, x * tbj[nn]); // H22: c t_i tb_j ph
                        add(r1, d + bj + nn, xb * tbj[nn]); // H12: c tb_i tb_j ph
                        add(r2, bj + nn, x * tj[nn]); // H21: c t_i t_j ph
                    }
                }
                // on-site mean field at i, weighted by <O^b>_j -- NO phase
                let qi = self.q_of(b.i, a);
                let s = self.s0[b.j * self.nop + bb] * c;
                for m in 0..mi {
                    for nn in 0..mi {
                        let mf = qi[m * mi + nn] * s;
                        add(bi + m, bi + nn, mf);
                        add(d + bi + nn, d + bi + m, mf); // H22 += mf^T
                    }
                }
            }
        }
        for i in 0..self.l {
            if self.qa_nz[i] {
                let (bi, mi) = (self.offs[i], self.ms[i]);
                let qa = self.qa_of(i);
                for m in 0..mi {
                    for nn in 0..mi {
                        let v = qa[m * mi + nn];
                        add(bi + m, bi + nn, v);
                        add(d + bi + nn, d + bi + m, v);
                    }
                }
            }
        }
    }

    /// The Hermitian H2 = sym(g . gH2), COLUMN major, from the row-major g H2.
    fn h2_colmajor(&self, h: &[C], out: &mut [C]) {
        let (n, d) = (self.n, self.d);
        for c in 0..n {
            let gc = if c < d { 1.0 } else { -1.0 };
            for r in 0..n {
                let gr = if r < d { 1.0 } else { -1.0 };
                out[c * n + r] = 0.5 * (h[r * n + c] * gr + (h[c * n + r] * gc).conj());
            }
        }
    }

    /// The one-magnon observable v (3 x 2D) in the Nambu basis, column major with
    /// leading dimension 3 (so that v[a, k] sits at k * 3 + a).
    pub(crate) fn observable(&self, q: [f64; 3], v: &mut [C]) {
        let d = self.d;
        for z in v.iter_mut() {
            *z = ZERO;
        }
        for mt in &self.mt {
            let arg = q[0] * mt.d[0] + q[1] * mt.d[1] + q[2] * mt.d[2];
            let ph = C::new(arg.cos(), arg.sin());
            let (o, mi) = (self.offs[mt.site], self.ms[mt.site]);
            for a in 0..3 {
                let t = self.t_of(mt.site, mt.idx[a]);
                let tb = self.tb_of(mt.site, mt.idx[a]);
                for m in 0..mi {
                    v[(o + m) * 3 + a] += ph * t[m];
                    v[(d + o + m) * 3 + a] += ph * tb[m];
                }
            }
        }
    }

    /// Cholesky factor C (upper, column major, strict lower zeroed) of H2(q) from
    /// the row-major g H2 in `h`, into `c`. Err(()) when not positive definite.
    fn cholesky_factor(&self, h: &[C], c: &mut [C]) -> Result<(), ()> {
        let n = self.n;
        self.h2_colmajor(h, c);
        lapack::zpotrf_upper(n, c).map_err(|_| ())?;
        for col in 0..n {
            for r in col + 1..n {
                c[col * n + r] = ZERO;
            }
        }
        Ok(())
    }

    /// W = C g C^H symmetrised, then its eigen-decomposition: `w` ascending,
    /// eigenvectors in the columns of `wmat` (column major).
    fn colpa_eig(&self, c: &[C], wmat: &mut [C], w: &mut [f64], scratch: &mut [C]) -> PyResult<()> {
        let (n, d) = (self.n, self.d);
        // scratch = C g
        scratch.copy_from_slice(c);
        for col in d..n {
            for z in &mut scratch[col * n..(col + 1) * n] {
                *z = -*z;
            }
        }
        lapack::zgemm(n, n, n, scratch, n, c, n, true, wmat, n);
        for col in 0..n {
            for r in col..n {
                let a = wmat[col * n + r];
                let b = wmat[r * n + col].conj();
                let s = 0.5 * (a + b);
                wmat[col * n + r] = s;
                wmat[r * n + col] = s.conj();
            }
        }
        lapack::zheevd_vectors(n, wmat, w)
            .map_err(|info| PyRuntimeError::new_err(format!("zheevd failed, info = {info}")))
    }

    /// The sparse guard's analysis, built once per packed model.
    fn sparse_guard(&self) -> Result<&SparseGuard, &str> {
        self.sparse.get_or_init(|| SparseGuard::analyse(self)).as_ref().map_err(|e| e.as_str())
    }

    /// Below this order the dense factorization is a few microseconds and the
    /// analysis is not worth building. Measured on the 120-degree triangular cell,
    /// one thread: 2D = 72 dense 50 / sparse 34 us per q, 2D = 162 257 / 94,
    /// 2D = 288 782 / 163, 2D = 882 10.6 / 0.81 ms.
    const SPARSE_MIN_N: usize = 64;

    /// Sparse when the order is large enough and the sparse factorization needs at
    /// most a quarter of the dense one's flops (it runs nearer BLAS-2 speed on its
    /// small fronts); otherwise -- a dense operator, a small cell, an analysis
    /// error -- the dense `zpotrf`.
    fn use_sparse_guard(&self) -> bool {
        if self.n < Self::SPARSE_MIN_N {
            return false;
        }
        let dense = (self.n as f64).powi(3) / 3.0;
        matches!(self.sparse_guard(), Ok(g) if 4.0 * g.flops <= dense)
    }

    fn not_pd(q: [f64; 3]) -> PyErr {
        NotPositiveDefinite::new_err(format!(
            "SU(N) H(q) is not positive definite at q_cart = {q:?}: the reference state \
             is not a minimum (imaginary magnons). Minimise first."))
    }
}

/// Run `f` over `nq` items with rayon, capping the pool so that at most
/// `budget / per_item` tasks run at once (the per-q matrices are large). The
/// budget is `budget::scratch_budget` (RSMAGCALC_SCRATCH_GB, else half the
/// available memory, else 3 GB).
pub(crate) fn capped_pool<T: Send>(n_workers: Option<usize>, per_item_bytes: usize,
                        f: impl FnOnce() -> T + Send) -> PyResult<T> {
    let (budget, _src) = crate::budget::scratch_budget(None).map_err(perr)?;
    let cap = (budget / per_item_bytes.max(1)).max(1);
    let threads = n_workers.filter(|&k| k > 0).unwrap_or_else(rayon::current_num_threads);
    let k = threads.min(cap);
    if k == threads && n_workers.is_none() {
        with_pool(None, f)
    } else {
        with_pool(Some(k), f)
    }
}

pub(crate) fn qs_vec(qs: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    let a = qs.as_array();
    if a.ncols() != 3 {
        return Err(perr(format!("qs must be (nq, 3), got {:?}", a.shape())));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

#[pymethods]
impl SunModel {
    /// `ms`: bosons per site; `s0` (L, nop); `t`, `tb`: per site (nop, M_i) row
    /// major, concatenated; `q`: per site (nop, M_i, M_i), concatenated; `qa`: per
    /// site (M_i, M_i), concatenated; bonds as `bi`, `bj`, `dr` (nb, 3), `jmat`
    /// (nb, nop, nop) real; moment terms as `mt_site`, `mt_d` (nt, 3), `mt_idx`
    /// (nt, 3); `n_cells` the chemical-cell replica count.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        ms: PyReadonlyArray1<'_, i64>,
        nop: usize,
        s0: PyReadonlyArray1<'_, C>,
        t: PyReadonlyArray1<'_, C>,
        tb: PyReadonlyArray1<'_, C>,
        q: PyReadonlyArray1<'_, C>,
        qa: PyReadonlyArray1<'_, C>,
        bi: PyReadonlyArray1<'_, i64>,
        bj: PyReadonlyArray1<'_, i64>,
        dr: PyReadonlyArray2<'_, f64>,
        jmat: PyReadonlyArray3<'_, f64>,
        mt_site: PyReadonlyArray1<'_, i64>,
        mt_d: PyReadonlyArray2<'_, f64>,
        mt_idx: PyReadonlyArray2<'_, i64>,
        n_cells: f64,
    ) -> PyResult<Self> {
        let ms: Vec<usize> = ms.as_array().iter().map(|&m| m as usize).collect();
        let l = ms.len();
        if l == 0 || nop == 0 {
            return Err(perr("SunModel needs at least one site and one operator"));
        }
        let mut offs = vec![0usize; l + 1];
        let mut qoff = vec![0usize; l + 1];
        let mut qaoff = vec![0usize; l + 1];
        for i in 0..l {
            offs[i + 1] = offs[i] + ms[i];
            qoff[i + 1] = qoff[i] + nop * ms[i] * ms[i];
            qaoff[i + 1] = qaoff[i] + ms[i] * ms[i];
        }
        let d = offs[l];
        let s0: Vec<C> = s0.as_array().iter().copied().collect();
        let t: Vec<C> = t.as_array().iter().copied().collect();
        let tb: Vec<C> = tb.as_array().iter().copied().collect();
        let qv: Vec<C> = q.as_array().iter().copied().collect();
        let qa: Vec<C> = qa.as_array().iter().copied().collect();
        if s0.len() != l * nop || t.len() != nop * d || tb.len() != nop * d
            || qv.len() != qoff[l] || qa.len() != qaoff[l] {
            return Err(perr(format!(
                "table sizes do not match L = {l}, n_ops = {nop}, D = {d}: s0 {}, t {}, \
                 tb {}, Q {}, QA {}", s0.len(), t.len(), tb.len(), qv.len(), qa.len())));
        }
        let qa_nz: Vec<bool> = (0..l)
            .map(|i| qa[qaoff[i]..qaoff[i + 1]].iter().any(|z| z.re != 0.0 || z.im != 0.0))
            .collect();

        let bi = bi.as_array();
        let bj = bj.as_array();
        let dr = dr.as_array();
        let jm = jmat.as_array();
        let nb = bi.len();
        if bj.len() != nb || dr.shape() != [nb, 3] || jm.shape() != [nb, nop, nop] {
            return Err(perr(format!(
                "bond arrays must be bi (nb,), bj (nb,), dr (nb, 3), J (nb, nop, nop) \
                 with nb = {nb}; got bj {}, dr {:?}, J {:?}", bj.len(), dr.shape(), jm.shape())));
        }
        let mut bonds = Vec::with_capacity(nb);
        for k in 0..nb {
            let (i, j) = (bi[k] as usize, bj[k] as usize);
            if i >= l || j >= l {
                return Err(perr(format!("bond {k} refers to site ({i}, {j}) outside 0..{l}")));
            }
            let mut terms = Vec::new();
            for a in 0..nop {
                for b in 0..nop {
                    let c = jm[[k, a, b]];
                    if c != 0.0 {
                        terms.push((a, b, c));
                    }
                }
            }
            bonds.push(Bond { i, j, dr: [dr[[k, 0]], dr[[k, 1]], dr[[k, 2]]], terms });
        }

        let mts = mt_site.as_array();
        let mtd = mt_d.as_array();
        let mti = mt_idx.as_array();
        let nt = mts.len();
        if mtd.shape() != [nt, 3] || mti.shape() != [nt, 3] {
            return Err(perr("moment terms must be mt_site (nt,), mt_d (nt, 3), mt_idx (nt, 3)"));
        }
        let mut mt = Vec::with_capacity(nt);
        for k in 0..nt {
            let site = mts[k] as usize;
            if site >= l {
                return Err(perr(format!("moment term {k} refers to site {site} outside 0..{l}")));
            }
            let idx = [mti[[k, 0]] as usize, mti[[k, 1]] as usize, mti[[k, 2]] as usize];
            if idx.iter().any(|&p| p >= nop) {
                return Err(perr(format!("moment term {k} indexes an operator outside 0..{nop}")));
            }
            mt.push(MomentTerm { site, d: [mtd[[k, 0]], mtd[[k, 1]], mtd[[k, 2]]], idx });
        }
        Ok(SunModel { l, nop, d, n: 2 * d, ms, offs, s0, t, tb, q: qv, qoff, qa, qaoff, qa_nz,
                      bonds, mt, n_cells, sparse: OnceLock::new() })
    }

    #[getter]
    fn d(&self) -> usize {
        self.d
    }

    #[getter]
    fn n_bonds(&self) -> usize {
        self.bonds.len()
    }

    /// g H2(q) for every q: (nq, 2D, 2D) complex, row major.
    #[pyo3(signature = (qs, n_workers=None))]
    fn hamiltonians<'py>(&self, py: Python<'py>, qs: PyReadonlyArray2<'py, f64>,
                         n_workers: Option<usize>) -> PyResult<Bound<'py, PyArray3<C>>> {
        let qs = qs_vec(&qs)?;
        let (n, nq) = (self.n, qs.len());
        let out = py.detach(move || -> PyResult<Vec<C>> {
            let mut out = vec![ZERO; nq * n * n];
            capped_pool(n_workers, n * n * 16, || {
                out.par_chunks_mut(n * n).zip(qs.par_iter()).for_each(|(h, &q)| self.build_h(q, h));
            })?;
            Ok(out)
        })?;
        Ok(Array3::from_shape_vec((nq, n, n), out).map_err(|e| perr(e.to_string()))?
            .into_pyarray(py))
    }

    /// All 2D eigenvalues of g H2(q) per q (unsorted, complex): (nq, 2D). What
    /// pyMagCalc's `dispersion`, `max_imaginary` and `stability_report` reduce.
    #[pyo3(signature = (qs, n_workers=None))]
    fn eigenvalues<'py>(&self, py: Python<'py>, qs: PyReadonlyArray2<'py, f64>,
                        n_workers: Option<usize>) -> PyResult<Bound<'py, PyArray2<C>>> {
        let qs = qs_vec(&qs)?;
        let (n, nq) = (self.n, qs.len());
        let out = py.detach(move || -> PyResult<Vec<C>> {
            let mut out = vec![ZERO; nq * n];
            capped_pool(n_workers, 2 * n * n * 16, || -> PyResult<()> {
                out.par_chunks_mut(n).zip(qs.par_iter()).try_for_each(|(w, &q)| {
                    let mut h = vec![ZERO; n * n];
                    let mut a = vec![ZERO; n * n];
                    self.build_h(q, &mut h);
                    // Hand LAPACK H itself in column-major order -- the layout
                    // NumPy's `eigvals` passes -- not the row-major buffer (H^T):
                    // the eigenvalues agree mathematically but not in rounding,
                    // and on a large cell with degenerate bands of the non-normal
                    // g H2 that rounding difference is ~sqrt(eps) (5e-8 at S09's
                    // 2D = 1800), against ~1e-15 with the same layout.
                    for r in 0..n {
                        for c in 0..n {
                            a[c * n + r] = h[r * n + c];
                        }
                    }
                    lapack::zgeev_values(n, &mut a, w)
                        .map_err(|info| PyRuntimeError::new_err(format!("zgeev failed, info = {info}")))
                })
            })??;
            Ok(out)
        })?;
        Ok(Array2::from_shape_vec((nq, n), out).map_err(|e| perr(e.to_string()))?
            .into_pyarray(py))
    }

    /// Colpa: (omega (nq, D) -- the D positive energies, DESCENDING as
    /// `_bogoliubov` returns them -- and T^T (nq, 2D, 2D): the para-unitary T,
    /// column major, so the Python side transposes the last two axes).
    #[pyo3(signature = (qs, n_workers=None))]
    fn bogoliubov<'py>(&self, py: Python<'py>, qs: PyReadonlyArray2<'py, f64>,
                       n_workers: Option<usize>)
                       -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<C>>)> {
        let qs = qs_vec(&qs)?;
        let (n, d, nq) = (self.n, self.d, qs.len());
        let (om, tt) = py.detach(move || -> PyResult<(Vec<f64>, Vec<C>)> {
            let mut om = vec![0.0f64; nq * d];
            let mut tt = vec![ZERO; nq * n * n];
            capped_pool(n_workers, 5 * n * n * 16, || -> PyResult<()> {
                om.par_chunks_mut(d).zip(tt.par_chunks_mut(n * n)).zip(qs.par_iter())
                    .try_for_each(|((om, t), &q)| {
                        let mut h = vec![ZERO; n * n];
                        let mut c = vec![ZERO; n * n];
                        let mut wm = vec![ZERO; n * n];
                        let mut sc = vec![ZERO; n * n];
                        let mut w = vec![0.0f64; n];
                        self.build_h(q, &mut h);
                        self.cholesky_factor(&h, &mut c).map_err(|_| Self::not_pd(q))?;
                        self.colpa_eig(&c, &mut wm, &mut w, &mut sc)?;
                        // U reordered: [n-1, ..., n-d, 0, ..., d-1], scaled by sqrt|E|
                        for nu in 0..n {
                            let src = if nu < d { n - 1 - nu } else { nu - d };
                            let e = w[src].abs();
                            if nu < d {
                                om[nu] = e;
                            }
                            let s = e.sqrt();
                            for r in 0..n {
                                t[nu * n + r] = wm[src * n + r] * s;
                            }
                        }
                        lapack::ztrsm_left_upper(n, n, &c, t);
                        Ok(())
                    })
            })??;
            Ok((om, tt))
        })?;
        Ok((Array2::from_shape_vec((nq, d), om).map_err(|e| perr(e.to_string()))?.into_pyarray(py),
            Array3::from_shape_vec((nq, n, n), tt).map_err(|e| perr(e.to_string()))?.into_pyarray(py)))
    }

    /// One-magnon correlation tensors: omega (nq, D) and S^{ab}_nu (nq, 3, 3, D),
    /// conj(M^a_nu) M^b_nu / n_cells, exactly the `spin_corr` of
    /// `SUNModel.structure_factor` before its cross-section contraction.
    #[pyo3(signature = (qs, n_workers=None))]
    fn correlations<'py>(&self, py: Python<'py>, qs: PyReadonlyArray2<'py, f64>,
                         n_workers: Option<usize>)
                         -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray4<C>>)> {
        let qs = qs_vec(&qs)?;
        let (n, d, nq) = (self.n, self.d, qs.len());
        let inv_cells = 1.0 / self.n_cells;
        let (om, corr) = py.detach(move || -> PyResult<(Vec<f64>, Vec<C>)> {
            let mut om = vec![0.0f64; nq * d];
            let mut corr = vec![ZERO; nq * 9 * d];
            capped_pool(n_workers, 4 * n * n * 16, || -> PyResult<()> {
                om.par_chunks_mut(d).zip(corr.par_chunks_mut(9 * d)).zip(qs.par_iter())
                    .try_for_each(|((om, corr), &q)| {
                        let mut h = vec![ZERO; n * n];
                        let mut c = vec![ZERO; n * n];
                        let mut wm = vec![ZERO; n * n];
                        let mut sc = vec![ZERO; n * n];
                        let mut w = vec![0.0f64; n];
                        let mut v = vec![ZERO; 3 * n];
                        let mut mm = vec![ZERO; 3 * d];
                        self.build_h(q, &mut h);
                        self.cholesky_factor(&h, &mut c).map_err(|_| Self::not_pd(q))?;
                        self.colpa_eig(&c, &mut wm, &mut w, &mut sc)?;
                        // Y = v C^{-1} (3 x n), then M = Y U[:, :d] sqrt|E|
                        self.observable(q, &mut v);
                        lapack::ztrsm_right_upper(3, n, &c, &mut v, 3);
                        lapack::zgemm(3, d, n, &v, 3, &wm, n, false, &mut mm, 3);
                        for nu in 0..d {
                            let e = w[nu].abs();
                            om[nu] = w[n - 1 - nu].abs();
                            let s = e.sqrt();
                            let m = [mm[nu * 3] * s, mm[nu * 3 + 1] * s, mm[nu * 3 + 2] * s];
                            for a in 0..3 {
                                for b in 0..3 {
                                    corr[(a * 3 + b) * d + nu] = m[a].conj() * m[b] * inv_cells;
                                }
                            }
                        }
                        Ok(())
                    })
            })??;
            Ok((om, corr))
        })?;
        Ok((Array2::from_shape_vec((nq, d), om).map_err(|e| perr(e.to_string()))?.into_pyarray(py),
            Array4::from_shape_vec((nq, 3, 3, d), corr).map_err(|e| perr(e.to_string()))?
                .into_pyarray(py)))
    }

    /// The ground-state guard per q: is H2(q) + shift I positive definite?
    ///
    /// `sparse`: None decides per model (`use_sparse_guard`), True / False force
    /// the sparse multifrontal Cholesky / the dense `zpotrf`. Both decide the same
    /// question on bit-identical matrix entries (`sparse_guard.rs`).
    #[pyo3(signature = (qs, shift, n_workers=None, sparse=None))]
    fn stable(&self, py: Python<'_>, qs: PyReadonlyArray2<'_, f64>, shift: f64,
              n_workers: Option<usize>, sparse: Option<bool>) -> PyResult<Vec<bool>> {
        let qs = qs_vec(&qs)?;
        let (n, nq) = (self.n, qs.len());
        py.detach(move || -> PyResult<Vec<bool>> {
            let use_sparse = match sparse {
                Some(false) => false,
                Some(true) => {
                    self.sparse_guard().map_err(|e| PyRuntimeError::new_err(
                        format!("sparse guard analysis failed: {e}")))?;
                    true
                }
                None => self.use_sparse_guard(),
            };
            let mut out = vec![false; nq];
            if use_sparse {
                let g = self.sparse_guard().unwrap();
                capped_pool(n_workers, g.peak_bytes + g.n_slots_bytes(), || {
                    out.par_iter_mut().zip(qs.par_iter()).for_each(|(ok, &q)| {
                        *ok = g.positive_definite(self, q, shift);
                    });
                })?;
                return Ok(out);
            }
            capped_pool(n_workers, 2 * n * n * 16, || {
                out.par_iter_mut().zip(qs.par_iter()).for_each(|(ok, &q)| {
                    let mut h = vec![ZERO; n * n];
                    let mut a = vec![ZERO; n * n];
                    self.build_h(q, &mut h);
                    self.h2_colmajor(&h, &mut a);
                    for k in 0..n {
                        a[k * n + k].re += shift;
                    }
                    *ok = lapack::zpotrf_upper(n, &mut a).is_ok();
                });
            })?;
            Ok(out)
        })
    }

    /// The sparse guard's analysis and the automatic choice, for tests and timing
    /// notes: n, supervariables, fronts, nnz(L), the complex flop counts of the
    /// sparse and the dense factorization, the largest front, `auto` (what
    /// `stable(sparse=None)` uses) and `error` (an analysis failure, or None).
    fn guard_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        let n = self.n;
        d.set_item("n", n)?;
        d.set_item("flops_dense", (n as f64).powi(3) / 3.0)?;
        let auto = py.detach(|| self.use_sparse_guard());
        match py.detach(|| self.sparse_guard().map(|g| (g.n_super, g.n_fronts(), g.nnz_l,
                                                       g.flops, g.max_front))) {
            Ok((ns, nf, nnz, fl, mf)) => {
                d.set_item("n_super", ns)?;
                d.set_item("n_fronts", nf)?;
                d.set_item("nnz_l", nnz)?;
                d.set_item("flops_sparse", fl)?;
                d.set_item("max_front", mf)?;
                d.set_item("error", py.None())?;
            }
            Err(e) => d.set_item("error", e)?,
        }
        d.set_item("auto", auto)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!("SunModel(L={}, n_ops={}, D={}, bonds={}, n_cells={})",
                self.l, self.nop, self.d, self.bonds.len(), self.n_cells)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SunModel>()?;
    m.add("NotPositiveDefinite", m.py().get_type::<NotPositiveDefinite>())?;
    Ok(())
}
