//! K7 -- the per-cluster perturbation theory of the dimer series expansion.
//!
//! Replaces, in pyMagCalc `sun/dimer_series.py`: `block_effective_series` (the
//! Bloch recursion with column-dependent resolvents and the des Cloizeaux
//! Hermitization) and `_cluster_pt` (embedding the pair operators into the
//! nloc^m product space of one cluster, and running the PT for the ground state
//! and the one-particle states, in total-Sz sectors when the local bases allow
//! it). The cluster enumeration, the isomorphism keys, the linked-cluster
//! subtraction and the resummation stay in Python: they are bookkeeping, and the
//! profile (2026-10-29, Rb2Cu3SnF12 order 5) put 85 % of the time in the PT.
//!
//! The arithmetic is the NumPy code's, term by term, so the two agree to
//! rounding (pinned at 1e-12 in tests/test_dimer_series_parity.py). The only
//! structural differences are savings that change no number:
//!   * V @ Omega_m is computed once per (V_l, m) and reused for the Bloch
//!     recursion and every M_k (NumPy recomputes it inside the M_k double sum);
//!   * Omega_0 is the model-space injection, so Omega_0^+ X is a row gather and
//!     the products of Omega_0 with an Omega_{i>=1} (zero on the model rows by
//!     construction) are skipped as the exact zeros they are;
//!   * the Gram-type terms of S_k and of the diagonal part of M_k are summed in
//!     adjoint pairs, Omega_i^+ W Omega_j + its adjoint (W = 1 or diag E0);
//!   * each total-Sz sector's V is built once per cluster (the ground state and
//!     the Sz = 0 one-particle states share one);
//!   * a sector's sparse V is built directly on the sector's states instead of
//!     on the full product space and then sliced -- the same entries, since the
//!     slice keeps exactly the pairs whose both ends lie in the sector.
//!
//! Conventions shared with the Python side, never re-derived here:
//!   * product-space index = sum_d digit_d nloc^d (little-endian digits, node d);
//!   * a link's pair operator is (nloc^2 x nloc^2) on the index sa*nloc + sb with
//!     sa the digit of the link's FIRST node pa;
//!   * the one-particle model space is [alpha nloc^d for d in nodes for alpha in
//!     1..nloc], d outer -- the row order of the returned delta;
//!   * sectors are keyed by round(2 Sz_total), and the one-particle sectors are
//!     visited in order of first appearance along that list (a dict's order).
//!
//! Threads: `n_workers` scoped threads over the clusters of one batch, each
//! cluster serial. How many run at once is set by memory, cluster by cluster
//! (`Gate`, `budget::scratch_budget`), not by the batch's largest cluster: with
//! the old 3 GB cap on the largest, order 8 ran on 2 threads of 128 and order 9
//! on 1 (2026-10-31).
use ndarray::{Array1, Array3};
use numpy::{Complex64, IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::{Condvar, Mutex};

use crate::sun_lswt::perr;

type C = Complex64;

const ZERO: C = C { re: 0.0, im: 0.0 };
const ONE: C = C { re: 1.0, im: 0.0 };

/// A square sparse matrix in CSR form, each row sorted by column.
pub(crate) struct Csr {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<C>,
}

impl Csr {
    /// out = A x for x, out (n x p) row major. `xnz[c]` false marks an all-zero
    /// row of x (most of Omega_m at low m), whose entries are skipped.
    fn matmul(&self, x: &[C], xnz: &[bool], p: usize, out: &mut [C]) {
        for (r, y) in out.chunks_exact_mut(p).enumerate().take(self.n) {
            y.fill(ZERO);
            let (a, b) = (self.indptr[r], self.indptr[r + 1]);
            for (&c, &v) in self.indices[a..b].iter().zip(&self.data[a..b]) {
                if !xnz[c] {
                    continue;
                }
                for (yb, xb) in y.iter_mut().zip(&x[c * p..c * p + p]) {
                    *yb += v * *xb;
                }
            }
        }
    }
}

/// Which rows of x (n x p, row major) hold a nonzero.
fn row_nonzero(x: &[C], p: usize) -> Vec<bool> {
    x.chunks_exact(p).map(|r| r.iter().any(|&z| z != ZERO)).collect()
}

/// Rows per partial sum in `adj_mul_add`.
const BLOCK: usize = 64;

/// out (p x p) += A^+ diag(w) B for A, B (n x p) row major; w = None is the identity.
///
/// The sum over the n rows is blocked: each run of BLOCK rows is summed on its own
/// and then added in, as NumPy's BLAS does. With a single running sum over all n
/// rows (thousands, at order 6) the order-6 cluster deltas of Rb2Cu3SnF12 sat
/// 1.6e-13 from NumPy, up to five times NumPy's own spread under a relabelling of
/// the cluster's nodes (3e-14); blocked, 1.3 times. Not a general accuracy claim:
/// on a synthetic 20000-state problem both lie inside NumPy's spread.
fn adj_mul_add(a: &[C], b: &[C], w: Option<&[f64]>, n: usize, p: usize, out: &mut [C]) {
    let mut part = vec![ZERO; p * p];
    for (blk, (ab, bb)) in a[..n * p].chunks(BLOCK * p).zip(b[..n * p].chunks(BLOCK * p))
        .enumerate() {
        part.fill(ZERO);
        let rows = ab.chunks_exact(p).zip(bb.chunks_exact(p));
        for (r, (ar, br)) in rows.enumerate() {
            let wr = w.map_or(1.0, |w| w[blk * BLOCK + r]);
            for (&ai, o) in ar.iter().zip(part.chunks_exact_mut(p)) {
                if ai == ZERO {
                    continue;
                }
                let ca = ai.conj() * wr;
                for (oj, bj) in o.iter_mut().zip(br) {
                    *oj += ca * *bj;
                }
            }
        }
        for (o, x) in out.iter_mut().zip(&part) {
            *o += *x;
        }
    }
}

/// out (p x p) += X + X^+ with X = A^+ diag(w) B: the (i, j) and (j, i) terms of a
/// Gram-type sum at once, since (Omega_i^+ W Omega_j)^+ = Omega_j^+ W Omega_i for a
/// real diagonal W.
fn adj_mul_sym_add(a: &[C], b: &[C], w: Option<&[f64]>, n: usize, p: usize, out: &mut [C]) {
    let mut x = vec![ZERO; p * p];
    adj_mul_add(a, b, w, n, p, &mut x);
    for i in 0..p {
        for j in 0..p {
            out[i * p + j] += x[i * p + j] + x[j * p + i].conj();
        }
    }
}

/// out (p x p) += X[rows, :] for X (n x p) row major -- Omega_0^+ X.
fn gather_add(x: &[C], rows: &[usize], p: usize, out: &mut [C]) {
    for (i, &r) in rows.iter().enumerate() {
        for j in 0..p {
            out[i * p + j] += x[r * p + j];
        }
    }
}

/// a b for p x p row-major matrices.
fn mm(a: &[C], b: &[C], p: usize) -> Vec<C> {
    let mut c = vec![ZERO; p * p];
    for i in 0..p {
        for k in 0..p {
            let aik = a[i * p + k];
            if aik == ZERO {
                continue;
            }
            for j in 0..p {
                c[i * p + j] += aik * b[k * p + j];
            }
        }
    }
    c
}

fn eye(p: usize) -> Vec<C> {
    let mut e = vec![ZERO; p * p];
    for i in 0..p {
        e[i * p + i] = ONE;
    }
    e
}

/// pyMagCalc `block_effective_series`: [H_eff_0 .. H_eff_order], each p x p row
/// major, for H(lambda) = diag(e0) + sum_j V_j lambda^j and the model space p_idx.
pub(crate) fn block_effective_series(e0: &[f64], v: &[Csr], p_idx: &[usize], order: usize)
                                     -> Result<Vec<Vec<C>>, String> {
    let n = e0.len();
    let p = p_idx.len();
    let mut model = vec![false; n];
    for &i in p_idx {
        if i >= n {
            return Err(format!("model index {i} out of range for dimension {n}"));
        }
        model[i] = true;
    }
    for vj in v {
        if vj.n != n {
            return Err(format!("V is {} x {}, H0 is {n} x {n}", vj.n, vj.n));
        }
    }
    let eps: Vec<f64> = p_idx.iter().map(|&i| e0[i]).collect();
    if model.iter().any(|&m| !m) {
        let scale = e0.iter().fold(0.0f64, |a, x| a.max(x.abs())).max(1.0);
        let mut dmin = f64::INFINITY;
        for q in 0..n {
            if !model[q] {
                for &e in &eps {
                    dmin = dmin.min((e - e0[q]).abs());
                }
            }
        }
        if dmin < 1e-9 * scale {
            return Err(format!(
                "model space is not separated from the complement (min gap {dmin:.2e} on \
                 scale {scale:.2e}); the perturbative block diagonalization is ill-defined."));
        }
    }

    let jmax = v.len();
    let np_ = n * p;
    let mut om: Vec<Vec<C>> = Vec::with_capacity(order + 1);
    let mut om0 = vec![ZERO; np_];
    for (mu, &i) in p_idx.iter().enumerate() {
        om0[i * p + mu] = ONE;
    }
    let mut om_nz: Vec<Vec<bool>> = vec![row_nonzero(&om0, p)];
    om.push(om0);

    // complement resolvent per model column: 1 / (eps_mu - E0_q), zero on model rows
    let mut inv_den = vec![0.0f64; np_];
    for q in 0..n {
        if !model[q] {
            for mu in 0..p {
                inv_den[q * p + mu] = 1.0 / (eps[mu] - e0[q]);
            }
        }
    }

    // vom[l][m] = V_{l+1} @ Omega_m, for every l + 1 + m <= order
    let mut vom: Vec<Vec<Vec<C>>> = (0..jmax).map(|_| Vec::new()).collect();
    let mut bl: Vec<Vec<C>> = Vec::with_capacity(order + 1);
    let mut b0 = vec![ZERO; p * p];
    for mu in 0..p {
        b0[mu * p + mu] = C::new(eps[mu], 0.0);
    }
    bl.push(b0);
    for k in 1..=order {
        let mut y = vec![ZERO; np_];
        for j in 1..=k.min(jmax) {
            let mut w = vec![ZERO; np_];
            v[j - 1].matmul(&om[k - j], &om_nz[k - j], p, &mut w);
            for (yi, wi) in y.iter_mut().zip(&w) {
                *yi += *wi;
            }
            vom[j - 1].push(w);
        }
        let mut bk = vec![ZERO; p * p];
        for (mu, &i) in p_idx.iter().enumerate() {
            bk[mu * p..(mu + 1) * p].copy_from_slice(&y[i * p..(i + 1) * p]);
        }
        bl.push(bk);
        for j in 1..k {
            // y -= Omega_j @ B_{k-j}; Omega_j (j >= 1) is zero on the model rows,
            // whose y the resolvent zeroes anyway
            let b = &bl[k - j];
            for ((orow, yr), &nz) in om[j].chunks_exact(p).zip(y.chunks_exact_mut(p))
                .zip(&om_nz[j]) {
                if !nz {
                    continue;
                }
                for (&o, brow) in orow.iter().zip(b.chunks_exact(p)) {
                    if o == ZERO {
                        continue;
                    }
                    for (yv, bv) in yr.iter_mut().zip(brow) {
                        *yv -= o * *bv;
                    }
                }
            }
        }
        for (yi, d) in y.iter_mut().zip(&inv_den) {
            *yi *= *d;
        }
        om_nz.push(row_nonzero(&y, p));
        om.push(y);
    }

    // S_k = (Omega^+ Omega)_k ; M_k = (Omega^+ H Omega)_k
    let mut s: Vec<Vec<C>> = Vec::with_capacity(order + 1);
    let mut m: Vec<Vec<C>> = Vec::with_capacity(order + 1);
    for k in 0..=order {
        let mut sk = vec![ZERO; p * p];
        let mut mk = vec![ZERO; p * p];
        if k == 0 {
            for mu in 0..p {
                sk[mu * p + mu] = ONE;
                mk[mu * p + mu] = C::new(eps[mu], 0.0);
            }
        }
        for i in 1..k {
            let j = k - i;
            if i < j {
                adj_mul_sym_add(&om[i], &om[j], None, n, p, &mut sk);
                adj_mul_sym_add(&om[i], &om[j], Some(e0), n, p, &mut mk);
            } else if i == j {
                adj_mul_add(&om[i], &om[j], None, n, p, &mut sk);
                adj_mul_add(&om[i], &om[j], Some(e0), n, p, &mut mk);
            }
        }
        for l in 1..=k.min(jmax) {
            for i in 0..=(k - l) {
                let w = &vom[l - 1][k - l - i];
                if i == 0 {
                    gather_add(w, p_idx, p, &mut mk);
                } else {
                    adj_mul_add(&om[i], w, None, n, p, &mut mk);
                }
            }
        }
        s.push(sk);
        m.push(mk);
    }

    // G = S^{1/2}, T = S^{-1/2} as series (all Hermitian)
    let mut g: Vec<Vec<C>> = vec![eye(p)];
    for k in 1..=order {
        let mut acc = s[k].clone();
        for i in 1..k {
            let gg = mm(&g[i], &g[k - i], p);
            for (a, x) in acc.iter_mut().zip(&gg) {
                *a -= *x;
            }
        }
        for a in acc.iter_mut() {
            *a *= 0.5;
        }
        g.push(acc);
    }
    let mut t: Vec<Vec<C>> = vec![eye(p)];
    for k in 1..=order {
        let mut acc = vec![ZERO; p * p];
        for j in 1..=k {
            let gt = mm(&g[j], &t[k - j], p);
            for (a, x) in acc.iter_mut().zip(&gt) {
                *a -= *x;
            }
        }
        t.push(acc);
    }

    let mut heff = Vec::with_capacity(order + 1);
    for k in 0..=order {
        let mut hk = vec![ZERO; p * p];
        for a in 0..=k {
            for b in 0..=(k - a) {
                let tmt = mm(&mm(&t[a], &m[b], p), &t[k - a - b], p);
                for (h, x) in hk.iter_mut().zip(&tmt) {
                    *h += *x;
                }
            }
        }
        let mut herm = vec![ZERO; p * p];
        for i in 0..p {
            for j in 0..p {
                herm[i * p + j] = 0.5 * (hk[i * p + j] + hk[j * p + i].conj());
            }
        }
        heff.push(herm);
    }
    Ok(heff)
}

/// One link of a cluster: its node positions and pair operator, the operator's
/// nonzero entries grouped by row (sa*nloc + sb).
pub(crate) struct Link {
    pa: usize,
    pb: usize,
    rows: Vec<Vec<(usize, C)>>,
}

impl Link {
    fn new(pa: usize, pb: usize, v: &[C], nloc: usize) -> Self {
        let n2 = nloc * nloc;
        // the threshold `_embed_pair_op` applies
        let rows = (0..n2)
            .map(|r| (0..n2).filter_map(|c| {
                let x = v[r * n2 + c];
                (x.norm() > 1e-14).then_some((c, x))
            }).collect())
            .collect();
        Link { pa, pb, rows }
    }
}

/// One cluster's PT input, owned (built under the GIL, run without it).
pub(crate) struct Cluster {
    /// (m, nloc) local eigen-energies, row major.
    levels: Vec<f64>,
    m: usize,
    nloc: usize,
    links: Vec<Link>,
    /// (m, nloc) total-Sz of each local level, or None for the full space.
    sz: Option<Vec<f64>>,
}

impl Cluster {
    fn dims(&self) -> usize {
        self.nloc.pow(self.m as u32)
    }

    #[inline]
    fn digit(&self, idx: usize, d: usize) -> usize {
        (idx / self.nloc.pow(d as u32)) % self.nloc
    }

    /// Peak scratch of one run, for admitting it into the pool: the Bloch
    /// recursion's Omega_k and V Omega_k, (2 order + 3) complex (rows x p)
    /// blocks on the largest sector, plus 10 % and the per-state index arrays.
    /// Measured against peak RSS (2026-10-31, chains of m = 7-9 dimers at order
    /// m - 1): 0.92-1.07 of the bare blocks without sectors, 0.78 with them.
    /// Until then a sectored cluster was charged dims/2 rows, 2.7x its largest
    /// sector at m = 9.
    fn scratch_bytes(&self, order: usize) -> usize {
        let p = self.m * (self.nloc - 1);
        let rows = self.largest_sector();
        rows * p * 16 * (2 * order + 3) / 10 * 11 + self.dims() * 32
    }

    /// The size of the largest total-Sz sector (the whole space without
    /// sectors), by convolving the nodes' level counts per round(2 Sz).
    fn largest_sector(&self) -> usize {
        let Some(sz) = &self.sz else { return self.dims() };
        let keys: Vec<i64> = sz.iter().map(|x| (x * 2.0).round() as i64).collect();
        let lo: i64 = keys.chunks(self.nloc).map(|c| *c.iter().min().unwrap()).sum();
        let hi: i64 = keys.chunks(self.nloc).map(|c| *c.iter().max().unwrap()).sum();
        let mut hist = vec![0usize; (hi - lo + 1) as usize];
        let mut cur: Vec<(i64, usize)> = vec![(0, 1)];
        for c in keys.chunks(self.nloc) {
            let mut next: std::collections::BTreeMap<i64, usize> = Default::default();
            for &(k, n) in &cur {
                for &d in c {
                    *next.entry(k + d).or_default() += n;
                }
            }
            cur = next.into_iter().collect();
        }
        for (k, n) in cur {
            hist[(k - lo) as usize] = n;
        }
        hist.into_iter().max().unwrap_or(1)
    }

    /// The sparse V restricted to the states `sel` (sorted global indices);
    /// `pos[g]` is g's local index, usize::MAX outside the sector.
    fn sector_v(&self, sel: &[usize], pos: &[usize]) -> Csr {
        let nloc = self.nloc;
        let mut indptr = Vec::with_capacity(sel.len() + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        let mut row: Vec<(usize, C)> = Vec::new();
        for &r in sel {
            row.clear();
            for l in &self.links {
                let (wa, wb) = (nloc.pow(l.pa as u32), nloc.pow(l.pb as u32));
                let (sa, sb) = (self.digit(r, l.pa), self.digit(r, l.pb));
                let base = r - sa * wa - sb * wb;
                for &(cab, x) in &l.rows[sa * nloc + sb] {
                    let lc = pos[base + (cab / nloc) * wa + (cab % nloc) * wb];
                    if lc != usize::MAX {
                        row.push((lc, x));
                    }
                }
            }
            // a stable sort keeps link order within a column, so duplicates sum
            // in the order scipy's V_0 + V_1 + ... does
            row.sort_by_key(|e| e.0);
            let mut k = 0;
            while k < row.len() {
                let (c, mut x) = row[k];
                k += 1;
                while k < row.len() && row[k].0 == c {
                    x += row[k].1;
                    k += 1;
                }
                indices.push(c);
                data.push(x);
            }
            indptr.push(indices.len());
        }
        Csr { n: sel.len(), indptr, indices, data }
    }

    /// pyMagCalc `_cluster_pt`: (gs_int (order+1), delta_int (order+1) x p x p).
    fn pt(&self, order: usize) -> Result<(Vec<f64>, Vec<C>), String> {
        let (m, nloc) = (self.m, self.nloc);
        let dims = self.dims();
        let lv = |d: usize, a: usize| self.levels[d * nloc + a];
        let mut e0 = vec![0.0f64; dims];
        for d in 0..m {
            for (idx, e) in e0.iter_mut().enumerate() {
                *e += lv(d, self.digit(idx, d));
            }
        }
        let e_loc: f64 = (0..m).map(|d| lv(d, 0)).sum();
        let onep: Vec<usize> = (0..m)
            .flat_map(|d| (1..nloc).map(move |a| a * nloc.pow(d as u32)))
            .collect();
        let p = onep.len();
        let mut delta = vec![ZERO; (order + 1) * p * p];
        let mut gs = vec![0.0f64; order + 1];

        let key: Option<Vec<i64>> = self.sz.as_ref().map(|sz| {
            let mut s = vec![0.0f64; dims];
            for d in 0..m {
                for (idx, x) in s.iter_mut().enumerate() {
                    *x += sz[d * nloc + self.digit(idx, d)];
                }
            }
            // half-integer-safe integer key; Rust rounds halves away from zero,
            // NumPy to even, but 2 Sz_total is an integer to rounding here
            s.iter().map(|x| (x * 2.0).round() as i64).collect()
        });

        // the PT for the model states `model` (global indices) within their
        // sector; a sector's (local index map, E0, V) is built once
        let mut sectors: Vec<(Option<i64>, Vec<usize>, Vec<f64>, Csr)> = Vec::new();
        let mut run = |model: &[usize]| -> Result<Vec<Vec<C>>, String> {
            let q = key.as_ref().map(|k| k[model[0]]);
            let si = match sectors.iter().position(|s| s.0 == q) {
                Some(si) => si,
                None => {
                    let sel: Vec<usize> = match &key {
                        None => (0..dims).collect(),
                        Some(k) => (0..dims).filter(|&g| Some(k[g]) == q).collect(),
                    };
                    let mut pos = vec![usize::MAX; dims];
                    for (i, &g) in sel.iter().enumerate() {
                        pos[g] = i;
                    }
                    let v = self.sector_v(&sel, &pos);
                    let e0s: Vec<f64> = sel.iter().map(|&g| e0[g]).collect();
                    sectors.push((q, pos, e0s, v));
                    sectors.len() - 1
                }
            };
            let (_q, pos, e0s, v) = &sectors[si];
            let ploc: Vec<usize> = model.iter().map(|&g| pos[g]).collect();
            if ploc.contains(&usize::MAX) {
                return Err("a model state lies outside its own Sz sector".into());
            }
            block_effective_series(e0s, std::slice::from_ref(v), &ploc, order)
        };

        let hg = run(&[0])?;
        for k in 0..=order {
            gs[k] = hg[k][0].re;
        }
        // one-particle sectors in order of first appearance along onep
        let groups: Vec<Vec<usize>> = match &key {
            None => vec![(0..p).collect()],
            Some(kk) => {
                let mut keys: Vec<i64> = Vec::new();
                let mut members: Vec<Vec<usize>> = Vec::new();
                for (ii, &g) in onep.iter().enumerate() {
                    match keys.iter().position(|&q| q == kk[g]) {
                        Some(j) => members[j].push(ii),
                        None => {
                            keys.push(kk[g]);
                            members.push(vec![ii]);
                        }
                    }
                }
                members
            }
        };
        for cols in &groups {
            let model: Vec<usize> = cols.iter().map(|&ii| onep[ii]).collect();
            let h = run(&model)?;
            let ps = cols.len();
            for k in 0..=order {
                let dk = &mut delta[k * p * p..(k + 1) * p * p];
                for (a, &ca) in cols.iter().enumerate() {
                    for (b, &cb) in cols.iter().enumerate() {
                        let mut x = h[k][a * ps + b];
                        if a == b {
                            x -= gs[k];
                        }
                        dk[ca * p + cb] = x;
                    }
                }
            }
        }
        gs[0] -= e_loc;
        // subtract the purely local order-0 splittings -> delta_int order 0 == 0
        for d in 0..m {
            for a in 1..nloc {
                let i = d * (nloc - 1) + (a - 1);
                delta[i * p + i] -= lv(d, a) - lv(d, 0);
            }
        }
        Ok((gs, delta))
    }
}

/// A Python (indptr, indices, data) CSR triplet -> Csr, checked.
fn csr_from(n: usize, indptr: &[i64], indices: &[i64], data: &[C]) -> PyResult<Csr> {
    let bad = || perr(format!("V is not a valid {n} x {n} CSR matrix"));
    if indptr.len() != n + 1 || indices.len() != data.len()
        || indptr.first() != Some(&0) || indptr.last() != Some(&(data.len() as i64)) {
        return Err(bad());
    }
    if indptr.windows(2).any(|w| w[0] > w[1]) || indices.iter().any(|&c| c < 0 || c as usize >= n) {
        return Err(bad());
    }
    Ok(Csr {
        n,
        indptr: indptr.iter().map(|&x| x as usize).collect(),
        indices: indices.iter().map(|&x| x as usize).collect(),
        data: data.to_vec(),
    })
}

type CsrIn<'py> = (PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, C>);

/// `block_effective_series(e0, v_list, p_idx, order)`, each V a CSR triplet
/// (indptr, indices, data); returns (order+1, p, p).
#[pyfunction]
#[pyo3(name = "block_effective_series")]
fn py_block_effective_series<'py>(py: Python<'py>, e0: PyReadonlyArray1<'py, f64>,
                                  v_list: Vec<CsrIn<'py>>, p_idx: Vec<usize>, order: usize)
                                  -> PyResult<Bound<'py, PyArray3<C>>> {
    let e0: Vec<f64> = e0.as_array().to_vec();
    let n = e0.len();
    let v = v_list.iter()
        .map(|(ip, ix, d)| csr_from(n, ip.as_slice()?, ix.as_slice()?, d.as_slice()?))
        .collect::<PyResult<Vec<_>>>()?;
    let p = p_idx.len();
    let h = py.detach(|| block_effective_series(&e0, &v, &p_idx, order)).map_err(perr)?;
    let flat: Vec<C> = h.into_iter().flatten().collect();
    Ok(Array3::from_shape_vec((order + 1, p, p), flat).map_err(|e| perr(e.to_string()))?
        .into_pyarray(py))
}

type ClusterIn<'py> = (PyReadonlyArray2<'py, f64>,
                       Vec<(usize, usize, PyReadonlyArray2<'py, C>)>,
                       Option<PyReadonlyArray2<'py, f64>>);

fn cluster_from(c: &ClusterIn<'_>) -> PyResult<Cluster> {
    let (levels, links, sz) = c;
    let lv = levels.as_array();
    let (m, nloc) = (lv.nrows(), lv.ncols());
    if m == 0 || nloc < 2 {
        return Err(perr(format!("levels must be (m >= 1, nloc >= 2), got {:?}", lv.shape())));
    }
    if (nloc as f64).powi(m as i32) > 6.7e7 {
        return Err(perr(format!("cluster space nloc^m = {nloc}^{m} is too large")));
    }
    let n2 = nloc * nloc;
    let mut out = Vec::with_capacity(links.len());
    for (pa, pb, v) in links {
        let va = v.as_array();
        if va.shape() != [n2, n2] {
            return Err(perr(format!("pair operator must be ({n2}, {n2}), got {:?}", va.shape())));
        }
        if *pa >= m || *pb >= m || pa == pb {
            return Err(perr(format!("link ({pa}, {pb}) invalid for {m} nodes")));
        }
        let flat: Vec<C> = va.iter().copied().collect();
        out.push(Link::new(*pa, *pb, &flat, nloc));
    }
    let sz = match sz {
        None => None,
        Some(s) => {
            let sa = s.as_array();
            if sa.shape() != [m, nloc] {
                return Err(perr(format!("sz must be ({m}, {nloc}), got {:?}", sa.shape())));
            }
            Some(sa.iter().copied().collect())
        }
    };
    Ok(Cluster { levels: lv.iter().copied().collect(), m, nloc, links: out, sz })
}

/// Admits clusters into the pool by their own scratch, largest first: a worker
/// takes the next cluster only when its bytes fit beside those in flight, and
/// one larger than the whole budget runs alone. Strictly in order, so a large
/// cluster is never overtaken (and starved) by the small ones behind it.
struct Gate {
    budget: usize,
    sizes: Vec<usize>,
    st: Mutex<GateState>,
    cv: Condvar,
}

#[derive(Default)]
struct GateState {
    next: usize,
    bytes: usize,
    running: usize,
    peak_bytes: usize,
    peak_running: usize,
}

impl Gate {
    /// The next position in the admission order and its charge, or None when done.
    fn take(&self) -> Option<(usize, usize)> {
        let mut s = self.st.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if s.next >= self.sizes.len() {
                return None;
            }
            let b = self.sizes[s.next].min(self.budget);
            if s.bytes + b <= self.budget {
                let pos = s.next;
                s.next += 1;
                s.bytes += b;
                s.running += 1;
                s.peak_bytes = s.peak_bytes.max(s.bytes);
                s.peak_running = s.peak_running.max(s.running);
                return Some((pos, b));
            }
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn give_back(&self, b: usize) {
        let mut s = self.st.lock().unwrap_or_else(|e| e.into_inner());
        s.bytes -= b;
        s.running -= 1;
        self.cv.notify_all();
    }
}

/// Returns a cluster's bytes to the gate even if its PT panics, so the other
/// workers cannot wait on them forever.
struct Charge<'a>(&'a Gate, usize);

impl Drop for Charge<'_> {
    fn drop(&mut self) {
        self.0.give_back(self.1);
    }
}

type PtOut = Result<(Vec<f64>, Vec<C>), String>;

struct PoolStats {
    threads: usize,
    budget: usize,
    source: String,
    peak_bytes: usize,
    peak_running: usize,
    max_scratch: usize,
}

fn run_gated(cl: &[Cluster], order: usize, threads: usize, budget: usize, source: String)
             -> (Vec<PtOut>, PoolStats) {
    let sizes: Vec<usize> = cl.iter().map(|c| c.scratch_bytes(order)).collect();
    let mut by_size: Vec<usize> = (0..cl.len()).collect();
    by_size.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
    let gate = Gate {
        budget,
        sizes: by_size.iter().map(|&i| sizes[i]).collect(),
        st: Mutex::new(GateState::default()),
        cv: Condvar::new(),
    };
    let threads = threads.clamp(1, cl.len().max(1));
    let mut out: Vec<Option<PtOut>> = (0..cl.len()).map(|_| None).collect();
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| sc.spawn(|| {
                let mut mine = Vec::new();
                while let Some((pos, b)) = gate.take() {
                    let _charge = Charge(&gate, b);
                    let i = by_size[pos];
                    mine.push((i, cl[i].pt(order)));
                }
                mine
            }))
            .collect();
        for h in hs {
            match h.join() {
                Ok(mine) => mine.into_iter().for_each(|(i, r)| out[i] = Some(r)),
                Err(p) => std::panic::resume_unwind(p),
            }
        }
    });
    let s = gate.st.into_inner().unwrap_or_else(|e| e.into_inner());
    let stats = PoolStats {
        threads,
        budget,
        source,
        peak_bytes: s.peak_bytes,
        peak_running: s.peak_running,
        max_scratch: sizes.iter().copied().max().unwrap_or(0),
    };
    (out.into_iter().map(|r| r.expect("every cluster is taken once")).collect(), stats)
}

/// `cluster_pt_batch(clusters, order, n_workers=None, scratch_budget_gb=None,
/// return_stats=False)`: pyMagCalc `_cluster_pt` for each (levels (m, nloc),
/// [(pa, pb, V)], sz (m, nloc) | None), in parallel over the clusters. Returns
/// [(gs_int (order+1,), delta_int (order+1, p, p))], and with `return_stats`
/// (that list, {threads, peak_concurrent, budget_bytes, budget_source,
/// peak_bytes, max_scratch_bytes, n_clusters}).
///
/// `n_workers` threads (default: rayon's count, i.e. the cores) run the batch;
/// how many run AT ONCE is set by memory: each cluster is charged its own
/// `scratch_bytes`, against the budget of `budget::scratch_budget`.
#[pyfunction]
#[pyo3(signature = (clusters, order, n_workers=None, scratch_budget_gb=None, return_stats=false))]
fn cluster_pt_batch<'py>(py: Python<'py>, clusters: Vec<ClusterIn<'py>>, order: usize,
                         n_workers: Option<usize>, scratch_budget_gb: Option<f64>,
                         return_stats: bool) -> PyResult<Bound<'py, PyAny>> {
    let cl = clusters.iter().map(cluster_from).collect::<PyResult<Vec<_>>>()?;
    let (budget, source) = crate::budget::scratch_budget(scratch_budget_gb).map_err(perr)?;
    let threads = n_workers.filter(|&k| k > 0).unwrap_or_else(rayon::current_num_threads);
    let (res, st) = py.detach(|| run_gated(&cl, order, threads, budget, source));
    let out = PyList::empty(py);
    for (c, r) in cl.iter().zip(res) {
        let (gs, delta) = r.map_err(perr)?;
        let p = c.m * (c.nloc - 1);
        out.append((Array1::from_vec(gs).into_pyarray(py),
                    Array3::from_shape_vec((order + 1, p, p), delta)
                        .map_err(|e| perr(e.to_string()))?.into_pyarray(py)))?;
    }
    if !return_stats {
        return Ok(out.into_any());
    }
    let d = PyDict::new(py);
    d.set_item("threads", st.threads)?;
    d.set_item("peak_concurrent", st.peak_running)?;
    d.set_item("budget_bytes", st.budget)?;
    d.set_item("budget_source", st.source)?;
    d.set_item("peak_bytes", st.peak_bytes)?;
    d.set_item("max_scratch_bytes", st.max_scratch)?;
    d.set_item("n_clusters", cl.len())?;
    Ok((out, d).into_pyobject(py)?.into_any())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_block_effective_series, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_pt_batch, m)?)?;
    Ok(())
}
