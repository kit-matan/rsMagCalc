//! K8 -- the dipole engine's LSWT matrix H(q) = 2 g H2(q), built numerically.
//!
//! Replaces, in pyMagCalc, the SymPy Holstein-Primakoff build (`symbolic.gen_HM`)
//! and the `lambdify` of its result. pyMagCalc resolves every term in Python, with
//! the spin model's OWN term builders, into a polynomial in the Cartesian spin
//! components of the over-unit-cell (OUC) sites: one monomial per (coefficient,
//! list of (site, component) factors). That keeps the meaning of each term in one
//! place. This module does only the algebra that `gen_HM` does symbolically:
//!   * the HP expansion of each factor about its site's rotated frame,
//!     S^a = sum_b R[a][b] s_b with s = (sqrt(S/2)(c+c^+), sqrt(S/2)(c-c^+)/i, S - c^+ c),
//!     the sqrt(S/2) of two one-boson factors applied once, as sqrt(S_a S_b)/2, when
//!     they form a pair: sqrt(S/2)^2 is not S/2 in floating point, and gen_HM's
//!     symbolic product is, so a collinear ferromagnet's H(0) would otherwise be
//!     4e-16 instead of exactly 0;
//!   * the product of the factors truncated at boson degree 2 (degree-2 words are
//!     kept as written, with NO reordering: `gen_HM` never normal-orders, and the
//!     placement below puts op1 op2 and op2 op1 in the same element, so the
//!     commutator constant is the only difference and LSWT drops it);
//!   * the Fourier placement of `symbolic._generate_fourier_lookup` and
//!     `_build_TwogH2_matrix`: the 1/2 (k, -k) split, rows (c_k^+, c_-k), columns
//!     (c_k, c_-k^+), the displacement d = r_a - r_b of the word's two sites;
//!   * H = 2 g H2 with g = diag(I, -I), as a sum over displacements
//!     H(q) = sum_d M_d exp(i q.d), q and d Cartesian.
//!
//! The result is a sparse bond model: per displacement d, the nonzero entries of
//! M_d. Its dense form (`bond_model`) is what fMagCalc's `run_sqw_model` takes.
//!
//! Oracle: entry-wise parity with SymPy's `HMat_sym` on every shipped dipole
//! config (pyMagCalc `tests/test_numeric_hq.py`); the building blocks are pinned
//! to closed forms in tests/test_dipole_hq.py here.
use std::collections::HashMap;

use ndarray::{Array2, Array3};
use numpy::{Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1,
            PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::sun_lswt::{capped_pool, perr, qs_vec};

type C = Complex64;

const ZERO: C = C { re: 0.0, im: 0.0 };

/// Displacements equal to 1e-9 A share a bond (the same rounding the SymPy
/// build's float positions give two images of one vector).
const D_KEY_SCALE: f64 = 1e9;

/// One boson operator: annihilation (`dag = false`) or creation, on an OUC site.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Op {
    dag: bool,
    site: u32,
}

/// A word of at most two boson operators, in the order written.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Word {
    len: u8,
    ops: [Op; 2],
}

const EMPTY: Word = Word { len: 0, ops: [Op { dag: false, site: 0 }; 2] };

impl Word {
    fn one(op: Op) -> Word {
        Word { len: 1, ops: [op, op] }
    }

    fn two(a: Op, b: Op) -> Word {
        Word { len: 2, ops: [a, b] }
    }

    /// self * other, or None past degree 2.
    fn times(&self, other: &Word) -> Option<Word> {
        match (self.len, other.len) {
            (0, _) => Some(*other),
            (_, 0) => Some(*self),
            (1, 1) => Some(Word::two(self.ops[0], other.ops[0])),
            _ => None,
        }
    }
}

/// A polynomial in the bosons truncated at degree 2: (word, coefficient).
type Poly = Vec<(Word, C)>;

/// a * b; `pair(x, y)` is the factor sqrt(S_x S_y)/2 that two one-boson words
/// on OUC sites x and y pick up when they form a pair.
fn poly_mul(a: &Poly, b: &Poly, pair: &impl Fn(u32, u32) -> f64) -> Poly {
    let mut out: Poly = Vec::with_capacity(a.len() * b.len());
    for (wa, ca) in a {
        for (wb, cb) in b {
            if let Some(w) = wa.times(wb) {
                let mut v = ca * cb;
                if wa.len == 1 && wb.len == 1 {
                    v *= pair(wa.ops[0].site, wb.ops[0].site);
                }
                match out.iter_mut().find(|(x, _)| *x == w) {
                    Some(slot) => slot.1 += v,
                    None => out.push((w, v)),
                }
            }
        }
    }
    out
}

/// The model's inputs, borrowed for the build.
struct Inputs<'a> {
    n: usize,
    pos: &'a [[f64; 3]],
    spins: &'a [f64],
    rot: &'a [[[f64; 3]; 3]],
}

impl Inputs<'_> {
    /// HP expansion of the global component `a` of OUC site `j`, its one-boson
    /// words WITHOUT their sqrt(S/2) (`pair` applies it).
    fn hp(&self, j: usize, a: usize) -> Poly {
        let jn = j % self.n;
        let r = &self.rot[jn];
        let s_mag = self.spins[jn];
        let site = j as u32;
        let c = Op { dag: false, site };
        let d = Op { dag: true, site };
        vec![
            (EMPTY, C::new(s_mag * r[a][2], 0.0)),
            (Word::one(c), C::new(r[a][0], -r[a][1])),
            (Word::one(d), C::new(r[a][0], r[a][1])),
            (Word::two(d, c), C::new(-r[a][2], 0.0)),
        ]
    }
}

/// Accumulates M_d as a sparse map while words are placed.
struct Builder {
    n: usize,
    keys: HashMap<[i64; 3], usize>,
    dvec: Vec<[f64; 3]>,
    acc: HashMap<(u32, u32), C>,
}

impl Builder {
    fn new(n: usize) -> Self {
        Builder { n, keys: HashMap::new(), dvec: Vec::new(), acc: HashMap::new() }
    }

    fn bond(&mut self, d: [f64; 3]) -> u32 {
        let key = [
            (d[0] * D_KEY_SCALE).round() as i64,
            (d[1] * D_KEY_SCALE).round() as i64,
            (d[2] * D_KEY_SCALE).round() as i64,
        ];
        let next = self.dvec.len();
        let b = *self.keys.entry(key).or_insert(next);
        if b == next {
            self.dvec.push(d);
        }
        b as u32
    }

    fn add(&mut self, bond: u32, r: usize, c: usize, v: C) {
        let dim = 2 * self.n;
        *self.acc.entry((bond, (r * dim + c) as u32)).or_insert(ZERO) += v;
    }

    /// Place coef * op1 op2 where `gen_HM`'s Fourier lookup puts it.
    fn place(&mut self, pos: &[[f64; 3]], op1: Op, op2: Op, coef: C) {
        let n = self.n;
        let (a, b) = (op1.site as usize, op2.site as usize);
        let (u, w) = (a % n, b % n);
        let d = [pos[a][0] - pos[b][0], pos[a][1] - pos[b][1], pos[a][2] - pos[b][2]];
        let mp = self.bond(d); // multiplies exp(+i q.d)
        let mm = self.bond([-d[0], -d[1], -d[2]]); // multiplies exp(-i q.d)
        let h = coef * 0.5;
        match (op1.dag, op2.dag) {
            (true, true) => {
                self.add(mm, u, n + w, h);
                self.add(mp, w, n + u, h);
            }
            (false, false) => {
                self.add(mp, n + w, u, h);
                self.add(mm, n + u, w, h);
            }
            (true, false) => {
                self.add(mm, u, w, h);
                self.add(mp, n + w, n + u, h);
            }
            (false, true) => {
                self.add(mp, w, u, h);
                self.add(mm, n + u, n + w, h);
            }
        }
    }

    /// Sorted entries with 2 g applied (rows n..2n negated).
    fn finish(self) -> DipoleHq {
        let n = self.n;
        let dim = 2 * n;
        let mut ent: Vec<((u32, u32), C)> = self.acc.into_iter().collect();
        ent.sort_unstable_by_key(|e| e.0);
        let mut bond = Vec::with_capacity(ent.len());
        let mut rc = Vec::with_capacity(ent.len());
        let mut val = Vec::with_capacity(ent.len());
        for ((b, k), v) in ent {
            let r = k as usize / dim;
            let g = if r < n { 2.0 } else { -2.0 };
            bond.push(b);
            rc.push(k);
            val.push(v * g);
        }
        DipoleHq { n, dvec: self.dvec, bond, rc, val }
    }
}

fn build(inp: &Inputs, coef: &[f64], ptr: &[i64], site: &[i64], comp: &[i64])
         -> Result<DipoleHq, String> {
    let n_ouc = inp.pos.len();
    let mut cache: HashMap<(usize, usize), Poly> = HashMap::new();
    let mut bld = Builder::new(inp.n);
    let pair = |x: u32, y: u32| {
        (inp.spins[x as usize % inp.n] * inp.spins[y as usize % inp.n]).sqrt() * 0.5
    };
    for m in 0..coef.len() {
        let (lo, hi) = (ptr[m] as usize, ptr[m + 1] as usize);
        let mut p: Poly = vec![(EMPTY, C::new(1.0, 0.0))];
        for f in lo..hi {
            let (j, a) = (site[f], comp[f]);
            if j < 0 || j as usize >= n_ouc || !(0..3).contains(&a) {
                return Err(format!(
                    "monomial {m}: factor (site {j}, component {a}) outside {n_ouc} OUC sites x 3"));
            }
            let key = (j as usize, a as usize);
            let hp = cache.entry(key).or_insert_with(|| inp.hp(key.0, key.1));
            p = poly_mul(&p, hp, &pair);
        }
        let cm = C::new(coef[m], 0.0);
        for (w, v) in p {
            if w.len == 2 && v != ZERO {
                bld.place(inp.pos, w.ops[0], w.ops[1], cm * v);
            }
        }
    }
    Ok(bld.finish())
}

/// The sparse bond model of H(q) = 2 g H2(q): per displacement d, the nonzero
/// entries of M_d.
#[pyclass(name = "DipoleHq", module = "rsmagcalc._core", frozen)]
pub struct DipoleHq {
    n: usize,
    dvec: Vec<[f64; 3]>,
    /// Per entry: its displacement index, row * 2n + col, and value.
    bond: Vec<u32>,
    rc: Vec<u32>,
    val: Vec<C>,
}

impl DipoleHq {
    fn fill(&self, q: [f64; 3], sign: f64, ph: &mut Vec<C>, h: &mut [C]) {
        ph.clear();
        ph.extend(self.dvec.iter().map(|d| {
            let x = sign * (q[0] * d[0] + q[1] * d[1] + q[2] * d[2]);
            C::new(x.cos(), x.sin())
        }));
        for z in h.iter_mut() {
            *z = ZERO;
        }
        for ((&b, &k), v) in self.bond.iter().zip(&self.rc).zip(&self.val) {
            h[k as usize] += v * ph[b as usize];
        }
    }
}

fn rows3(a: &ndarray::ArrayView2<'_, f64>, what: &str) -> PyResult<Vec<[f64; 3]>> {
    if a.ncols() != 3 {
        return Err(perr(format!("{what} must be (n, 3), got {:?}", a.shape())));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

#[pymethods]
impl DipoleHq {
    /// Build the model. `pos_ouc` (n_ouc, 3) Cartesian positions, the n cell sites
    /// first and OUC site j an image of cell site j % n; `spins` (n,) the spin
    /// length of each cell site; `rot` (n, 3, 3) the rotation taking each site's
    /// local frame (moment along z) to the lab; the monomials as CSR: monomial m
    /// is coef[m] * prod_{f in ptr[m]..ptr[m+1]} S^{comp[f]}_{site[f]}.
    #[new]
    fn py_new(py: Python<'_>, n: usize, pos_ouc: PyReadonlyArray2<'_, f64>,
              spins: PyReadonlyArray1<'_, f64>, rot: PyReadonlyArray3<'_, f64>,
              coef: PyReadonlyArray1<'_, f64>, ptr: PyReadonlyArray1<'_, i64>,
              site: PyReadonlyArray1<'_, i64>, comp: PyReadonlyArray1<'_, i64>)
              -> PyResult<Self> {
        let pos = rows3(&pos_ouc.as_array(), "pos_ouc")?;
        let spins: Vec<f64> = spins.as_array().iter().copied().collect();
        let r = rot.as_array();
        if n == 0 || pos.len() < n || pos.len() % n != 0 {
            return Err(perr(format!(
                "pos_ouc must hold a whole number of images of the {n} cell sites, got {}",
                pos.len())));
        }
        if spins.len() != n || r.shape() != [n, 3, 3] {
            return Err(perr(format!(
                "spins must be ({n},) and rot ({n}, 3, 3), got {} and {:?}",
                spins.len(), r.shape())));
        }
        let rot: Vec<[[f64; 3]; 3]> = (0..n)
            .map(|i| {
                let mut m = [[0.0; 3]; 3];
                for a in 0..3 {
                    for b in 0..3 {
                        m[a][b] = r[[i, a, b]];
                    }
                }
                m
            })
            .collect();
        let coef: Vec<f64> = coef.as_array().iter().copied().collect();
        let ptr: Vec<i64> = ptr.as_array().iter().copied().collect();
        let site: Vec<i64> = site.as_array().iter().copied().collect();
        let comp: Vec<i64> = comp.as_array().iter().copied().collect();
        if ptr.len() != coef.len() + 1 || ptr.first() != Some(&0)
            || ptr.windows(2).any(|w| w[1] < w[0])
            || *ptr.last().unwrap() as usize != site.len() || site.len() != comp.len() {
            return Err(perr(format!(
                "monomial CSR is inconsistent: {} coefficients, {} pointers, {} sites, {} \
                 components", coef.len(), ptr.len(), site.len(), comp.len())));
        }
        let inp = Inputs { n, pos: &pos, spins: &spins, rot: &rot };
        py.detach(|| build(&inp, &coef, &ptr, &site, &comp)).map_err(perr)
    }

    /// Rebuild from `entries()` (unpickling, or a model shipped to a worker).
    #[staticmethod]
    fn from_entries(n: usize, dvec: PyReadonlyArray2<'_, f64>, bond: PyReadonlyArray1<'_, i64>,
                    row: PyReadonlyArray1<'_, i64>, col: PyReadonlyArray1<'_, i64>,
                    val: PyReadonlyArray1<'_, C>) -> PyResult<Self> {
        let dvec = rows3(&dvec.as_array(), "dvec")?;
        let dim = 2 * n;
        let (b, r, c, v) = (bond.as_array(), row.as_array(), col.as_array(), val.as_array());
        let ne = b.len();
        if r.len() != ne || c.len() != ne || v.len() != ne {
            return Err(perr("bond, row, col and val must have one length"));
        }
        let mut out = DipoleHq { n, dvec, bond: Vec::with_capacity(ne),
                                 rc: Vec::with_capacity(ne), val: Vec::with_capacity(ne) };
        for e in 0..ne {
            let (bi, ri, ci) = (b[e], r[e], c[e]);
            if bi < 0 || bi as usize >= out.dvec.len() || ri < 0 || ri as usize >= dim
                || ci < 0 || ci as usize >= dim {
                return Err(perr(format!("entry {e} out of range")));
            }
            out.bond.push(bi as u32);
            out.rc.push((ri as usize * dim + ci as usize) as u32);
            out.val.push(v[e]);
        }
        Ok(out)
    }

    /// Number of cell sites; the matrix is 2n x 2n.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    #[getter]
    fn n_bonds(&self) -> usize {
        self.dvec.len()
    }

    #[getter]
    fn n_entries(&self) -> usize {
        self.val.len()
    }

    /// (dvec (nb, 3), bond (ne,), row (ne,), col (ne,), val (ne,)).
    #[allow(clippy::type_complexity)]
    fn entries<'py>(&self, py: Python<'py>)
        -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>,
            Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<C>>) {
        let dim = 2 * self.n;
        let flat: Vec<f64> = self.dvec.iter().flat_map(|d| d.iter().copied()).collect();
        let dv = Array2::from_shape_vec((self.dvec.len(), 3), flat).unwrap();
        let bond: Vec<i64> = self.bond.iter().map(|&b| b as i64).collect();
        let row: Vec<i64> = self.rc.iter().map(|&k| (k as usize / dim) as i64).collect();
        let col: Vec<i64> = self.rc.iter().map(|&k| (k as usize % dim) as i64).collect();
        (dv.into_pyarray(py), bond.into_pyarray(py), row.into_pyarray(py),
         col.into_pyarray(py), self.val.clone().into_pyarray(py))
    }

    /// The dense bond model: (dvec (nb, 3), Mb (nb, 2n, 2n)), H(q) = sum_b Mb[b]
    /// exp(i q.dvec[b]) -- fMagCalc `run_sqw_model`'s input.
    fn bond_model<'py>(&self, py: Python<'py>)
        -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<C>>)> {
        let dim = 2 * self.n;
        let nb = self.dvec.len();
        let mut mb = vec![ZERO; nb * dim * dim];
        for ((&b, &k), v) in self.bond.iter().zip(&self.rc).zip(&self.val) {
            mb[b as usize * dim * dim + k as usize] += v;
        }
        let flat: Vec<f64> = self.dvec.iter().flat_map(|d| d.iter().copied()).collect();
        let dv = Array2::from_shape_vec((nb, 3), flat).map_err(|e| perr(e.to_string()))?;
        let mb = Array3::from_shape_vec((nb, dim, dim), mb).map_err(|e| perr(e.to_string()))?;
        Ok((dv.into_pyarray(py), mb.into_pyarray(py)))
    }

    /// H(q) for every q: (nq, 2n, 2n) complex. `negate` evaluates H(-q).
    #[pyo3(signature = (qs, negate=false, n_workers=None))]
    fn hamiltonians<'py>(&self, py: Python<'py>, qs: PyReadonlyArray2<'py, f64>, negate: bool,
                         n_workers: Option<usize>) -> PyResult<Bound<'py, PyArray3<C>>> {
        let qs = qs_vec(&qs)?;
        let dim = 2 * self.n;
        let nq = qs.len();
        let sign = if negate { -1.0 } else { 1.0 };
        let out = py.detach(move || -> PyResult<Vec<C>> {
            let mut out = vec![ZERO; nq * dim * dim];
            if nq <= 2 {
                // The per-q pools and the fit evaluator ask for one q at a time:
                // no thread pool, which would cost more than the q itself.
                let mut ph = Vec::with_capacity(self.dvec.len());
                for (h, &q) in out.chunks_mut(dim * dim).zip(&qs) {
                    self.fill(q, sign, &mut ph, h);
                }
                return Ok(out);
            }
            capped_pool(n_workers, dim * dim * 16, || {
                out.par_chunks_mut(dim * dim).zip(qs.par_iter()).for_each_init(
                    || Vec::with_capacity(self.dvec.len()),
                    |ph, (h, &q)| self.fill(q, sign, ph, h));
            })?;
            Ok(out)
        })?;
        Ok(Array3::from_shape_vec((nq, dim, dim), out).map_err(|e| perr(e.to_string()))?
            .into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("DipoleHq(n={}, bonds={}, entries={})", self.n, self.dvec.len(), self.val.len())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DipoleHq>()?;
    Ok(())
}
