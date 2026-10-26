//! K2 -- the CP^(N-1) Metropolis sampler, the RK4 integrator and the damped
//! quench for SU(N) coherent states.
//!
//! Replaces, in pyMagCalc `sun/dynamics.py`: `metropolis_sweep`, the sweep loop
//! of `thermalize` (the adaptive-sigma phase and the recorded fixed-sigma phase;
//! the drift verdict is computed in Python from what this returns), `_h_times_z`
//! + `_rk4` + `_renormalize` (`evolve`), `damped_step` / the `quench` loop, and the
//! trajectory loop of `sampled_correlations`. The time FFT, the neutron
//! contraction, `moment_of` and the topological charges stay in Python.
//!
//! Conventions, shared with the Python side and never re-derived here:
//!   * a site i carries a normalized N_i-vector Z_i and n_ops local operators
//!     O_i^p (N_i x N_i, complex); the sites may differ in N_i (mixed spin), so
//!     every per-site table is RAGGED and addressed through offsets;
//!   * s0_i^p = <Z_i|O^p|Z_i> (complex, not conjugated anywhere: the dipoles are
//!     Hermitian so it is real for them, the biquadratic products are not);
//!   * E = 1/2 sum_bonds Re(s0_i . J . s0_j) + sum_i Re(<Z_i|A_i|Z_i>), the bond
//!     list carrying BOTH directions (pyMagCalc `SUNModel._energy_of`);
//!   * the mean field is h_i = sum_{bonds FROM i} sum_ab J_ab s0_j^b O_i^a + A_i
//!     (`SUNModel.local_field`), and only h_i|Z_i> is ever formed;
//!   * dZ_i/dt = -i h_i Z_i, damped: -i h_i Z_i - lambda (h_i - <h_i>) Z_i; RK4,
//!     then each |Z_i| is renormalized to 1 (`_rk4`);
//!   * the Metropolis proposal is Z_i + sigma (g + i g') renormalized, with g, g'
//!     standard normal N_i-vectors, accepted with min(1, exp(-beta dE)).
//!
//! THE ALGORITHMIC FIX. The Python sampler evaluates the TOTAL energy before and
//! after every proposal, O(L^2) per sweep. Here dE is the change of the bond terms
//! that touch the site plus its on-site term -- exact for any bond list (both
//! directions, a self-bond of a one-cell supercell, an unsymmetrized list), because
//! it is term-by-term the same sum `_energy_of` takes; it is pinned to
//! E(after) - E(before) in tests/test_cpn_parity.py, on a mixed-spin cell too.
//!
//! RNG: PCG64 per stream as in K1 (`classical::stream`); every trajectory of
//! `sampled_trajectories` owns the stream (seed, traj_start + i), so chunked calls
//! and thread counts do not change the result. Not NumPy's stream.
use ndarray::{Array2, Array3};
use numpy::{Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1,
            PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::classical::{stream, with_pool, PyRng};

type C = Complex64;

const ZERO: C = C { re: 0.0, im: 0.0 };

/// The model's static data, packed once from a pyMagCalc `SUNModel`.
#[pyclass(name = "CpnModel", module = "rsmagcalc._core", frozen)]
pub struct CpnModel {
    l: usize,
    n_ops: usize,
    ns: Vec<usize>,
    zoff: Vec<usize>,
    ops: Vec<C>,
    opoff: Vec<usize>,
    a: Vec<C>,
    aoff: Vec<usize>,
    bi: Vec<usize>,
    bj: Vec<usize>,
    j: Vec<C>,
    from: Vec<Vec<usize>>,
    touch: Vec<Vec<usize>>,
}

/// Per-trajectory scratch (never shared between threads).
struct Work {
    s0: Vec<C>,
    snew: Vec<C>,
    c: Vec<C>,
    hz: Vec<C>,
    opz: Vec<C>,
    cand: Vec<C>,
    k1: Vec<C>,
    k2: Vec<C>,
    k3: Vec<C>,
    k4: Vec<C>,
    tmp: Vec<C>,
    order: Vec<usize>,
}

impl Work {
    fn new(m: &CpnModel) -> Self {
        let d = m.d();
        let nmax = m.ns.iter().copied().max().unwrap_or(1);
        Work {
            s0: vec![ZERO; m.l * m.n_ops],
            snew: vec![ZERO; m.n_ops],
            c: vec![ZERO; m.n_ops],
            hz: vec![ZERO; d],
            opz: vec![ZERO; nmax],
            cand: vec![ZERO; nmax],
            k1: vec![ZERO; d],
            k2: vec![ZERO; d],
            k3: vec![ZERO; d],
            k4: vec![ZERO; d],
            tmp: vec![ZERO; d],
            order: Vec::with_capacity(m.l),
        }
    }
}

#[inline]
fn matvec(m: &[C], n: usize, x: &[C], out: &mut [C]) {
    for r in 0..n {
        let row = &m[r * n..(r + 1) * n];
        let mut acc = ZERO;
        for (mrc, xc) in row.iter().zip(x) {
            acc += mrc * xc;
        }
        out[r] = acc;
    }
}

/// x^dagger (M x), with `scratch` holding M x on return.
#[inline]
fn quad(m: &[C], n: usize, x: &[C], scratch: &mut [C]) -> C {
    matvec(m, n, x, &mut scratch[..n]);
    let mut acc = ZERO;
    for r in 0..n {
        acc += x[r].conj() * scratch[r];
    }
    acc
}

#[inline]
fn norm(x: &[C]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

impl CpnModel {
    #[inline]
    fn d(&self) -> usize {
        self.zoff[self.l]
    }

    #[inline]
    fn op(&self, i: usize, p: usize) -> &[C] {
        let n2 = self.ns[i] * self.ns[i];
        let s = self.opoff[i] + p * n2;
        &self.ops[s..s + n2]
    }

    #[inline]
    fn a_of(&self, i: usize) -> &[C] {
        &self.a[self.aoff[i]..self.aoff[i + 1]]
    }

    #[inline]
    fn jmat(&self, k: usize) -> &[C] {
        let n2 = self.n_ops * self.n_ops;
        &self.j[k * n2..(k + 1) * n2]
    }

    /// s0_i^p = <z|O^p|z> for every p, into `out`.
    fn expect_site(&self, i: usize, z: &[C], out: &mut [C], scratch: &mut [C]) {
        let n = self.ns[i];
        for p in 0..self.n_ops {
            out[p] = quad(self.op(i, p), n, z, scratch);
        }
    }

    fn s0_all(&self, z: &[C], s0: &mut [C], scratch: &mut [C]) {
        let nop = self.n_ops;
        for i in 0..self.l {
            let zi = &z[self.zoff[i]..self.zoff[i + 1]];
            let (a, b) = (i * nop, (i + 1) * nop);
            self.expect_site(i, zi, &mut s0[a..b], scratch);
        }
    }

    /// 1/2 Re(s0_i . J_k . s0_j), with site `site`'s expectations replaced by
    /// `snew` when given (the "after" value of a proposal touching it).
    #[inline]
    fn bond_term(&self, k: usize, s0: &[C], site: usize, snew: Option<&[C]>) -> f64 {
        let nop = self.n_ops;
        let (bi, bj) = (self.bi[k], self.bj[k]);
        let si = match snew {
            Some(s) if bi == site => s,
            _ => &s0[bi * nop..(bi + 1) * nop],
        };
        let sj = match snew {
            Some(s) if bj == site => s,
            _ => &s0[bj * nop..(bj + 1) * nop],
        };
        let jm = self.jmat(k);
        let mut acc = ZERO;
        for a in 0..nop {
            let row = &jm[a * nop..(a + 1) * nop];
            let mut jr = ZERO;
            for b in 0..nop {
                jr += row[b] * sj[b];
            }
            acc += si[a] * jr;
        }
        0.5 * acc.re
    }

    /// E = 1/2 sum_bonds Re(s0_i J s0_j) + sum_i Re<Z_i|A_i|Z_i>.
    fn energy_of(&self, z: &[C], s0: &[C], scratch: &mut [C]) -> f64 {
        let mut e = 0.0;
        for k in 0..self.bi.len() {
            e += self.bond_term(k, s0, usize::MAX, None);
        }
        for i in 0..self.l {
            let zi = &z[self.zoff[i]..self.zoff[i + 1]];
            e += quad(self.a_of(i), self.ns[i], zi, scratch).re;
        }
        e
    }

    /// dE of replacing Z_i by `cand` (whose expectations are `snew`).
    fn move_delta(&self, i: usize, z: &[C], s0: &[C], cand: &[C], snew: &[C],
                  scratch: &mut [C]) -> f64 {
        let mut de = 0.0;
        for &k in &self.touch[i] {
            de += self.bond_term(k, s0, i, Some(snew)) - self.bond_term(k, s0, i, None);
        }
        let n = self.ns[i];
        let zi = &z[self.zoff[i]..self.zoff[i + 1]];
        de += quad(self.a_of(i), n, cand, scratch).re - quad(self.a_of(i), n, zi, scratch).re;
        de
    }

    /// h_i |Z_i> for every site, from the current expectations `s0`.
    fn hz_of(&self, z: &[C], s0: &[C], out: &mut [C], c: &mut [C], opz: &mut [C]) {
        let nop = self.n_ops;
        for i in 0..self.l {
            let n = self.ns[i];
            let zi = &z[self.zoff[i]..self.zoff[i + 1]];
            let oi = &mut out[self.zoff[i]..self.zoff[i + 1]];
            for a in 0..nop {
                c[a] = ZERO;
            }
            for &k in &self.from[i] {
                let sj = &s0[self.bj[k] * nop..(self.bj[k] + 1) * nop];
                let jm = self.jmat(k);
                for a in 0..nop {
                    let row = &jm[a * nop..(a + 1) * nop];
                    let mut acc = ZERO;
                    for b in 0..nop {
                        acc += row[b] * sj[b];
                    }
                    c[a] += acc;
                }
            }
            matvec(self.a_of(i), n, zi, oi);
            for a in 0..nop {
                let ca = c[a];
                if ca == ZERO {
                    continue;
                }
                matvec(self.op(i, a), n, zi, &mut opz[..n]);
                for r in 0..n {
                    oi[r] += ca * opz[r];
                }
            }
        }
    }

    /// f(Z) = -i h Z  (damping = 0)  or  -i h Z - lambda (h - <h>) Z.
    fn deriv(&self, z: &[C], damping: f64, w: &mut Work, out: &mut [C]) {
        self.s0_all(z, &mut w.s0, &mut w.opz);
        self.hz_of(z, &w.s0, &mut w.hz, &mut w.c, &mut w.opz);
        let mi = C { re: 0.0, im: -1.0 };
        if damping == 0.0 {
            for k in 0..z.len() {
                out[k] = mi * w.hz[k];
            }
            return;
        }
        for i in 0..self.l {
            let (a, b) = (self.zoff[i], self.zoff[i + 1]);
            let mut eh = 0.0;
            for k in a..b {
                eh += (z[k].conj() * w.hz[k]).re;
            }
            for k in a..b {
                out[k] = mi * w.hz[k] - damping * (w.hz[k] - eh * z[k]);
            }
        }
    }

    fn renormalize(&self, z: &mut [C]) {
        for i in 0..self.l {
            let zi = &mut z[self.zoff[i]..self.zoff[i + 1]];
            let r = norm(zi);
            for v in zi.iter_mut() {
                *v = *v / r;
            }
        }
    }

    /// One RK4 step of dZ/dt = f(Z), then |Z_i| = 1 (pyMagCalc `_rk4`).
    fn rk4(&self, z: &mut [C], dt: f64, damping: f64, w: &mut Work) {
        let d = z.len();
        let mut k1 = std::mem::take(&mut w.k1);
        let mut k2 = std::mem::take(&mut w.k2);
        let mut k3 = std::mem::take(&mut w.k3);
        let mut k4 = std::mem::take(&mut w.k4);
        let mut tmp = std::mem::take(&mut w.tmp);
        self.deriv(z, damping, w, &mut k1);
        let h = 0.5 * dt;
        for k in 0..d {
            tmp[k] = z[k] + h * k1[k];
        }
        self.deriv(&tmp, damping, w, &mut k2);
        for k in 0..d {
            tmp[k] = z[k] + h * k2[k];
        }
        self.deriv(&tmp, damping, w, &mut k3);
        for k in 0..d {
            tmp[k] = z[k] + dt * k3[k];
        }
        self.deriv(&tmp, damping, w, &mut k4);
        let f = dt / 6.0;
        for k in 0..d {
            z[k] = z[k] + f * (((k1[k] + 2.0 * k2[k]) + 2.0 * k3[k]) + k4[k]);
        }
        self.renormalize(z);
        w.k1 = k1;
        w.k2 = k2;
        w.k3 = k3;
        w.k4 = k4;
        w.tmp = tmp;
    }

    /// Trajectory into `rec` (n_rec x D), the state BEFORE each recorded step.
    fn evolve_impl(&self, z: &mut [C], dt: f64, damping: f64, n_steps: usize,
                   record_every: usize, w: &mut Work, rec: &mut [C]) {
        let d = z.len();
        let mut irec = 0usize;
        for step in 0..n_steps {
            if step % record_every == 0 {
                rec[irec * d..(irec + 1) * d].copy_from_slice(z);
                irec += 1;
            }
            self.rk4(z, dt, damping, w);
        }
    }

    /// One Metropolis sweep over the sites in random order; returns the
    /// acceptance fraction. `s0` is kept consistent with `z`.
    fn sweep(&self, z: &mut [C], s0: &mut [C], beta: f64, sigma: f64, rng: &mut Pcg64,
             w: &mut Work) -> f64 {
        let nop = self.n_ops;
        w.order.clear();
        w.order.extend(0..self.l);
        w.order.shuffle(rng);
        let mut acc = 0usize;
        let mut cand = std::mem::take(&mut w.cand);
        let mut snew = std::mem::take(&mut w.snew);
        for idx in 0..self.l {
            let i = w.order[idx];
            let n = self.ns[i];
            let (a, b) = (self.zoff[i], self.zoff[i + 1]);
            for k in 0..n {
                let g: f64 = rng.sample(StandardNormal);
                let gi: f64 = rng.sample(StandardNormal);
                cand[k] = z[a + k] + sigma * C { re: g, im: gi };
            }
            let r = norm(&cand[..n]);
            for k in 0..n {
                cand[k] = cand[k] / r;
            }
            self.expect_site(i, &cand[..n], &mut snew[..nop], &mut w.opz);
            let de = self.move_delta(i, z, s0, &cand[..n], &snew[..nop], &mut w.opz);
            if de <= 0.0 || rng.random::<f64>() < (-beta * de).exp() {
                z[a..b].copy_from_slice(&cand[..n]);
                s0[i * nop..(i + 1) * nop].copy_from_slice(&snew[..nop]);
                acc += 1;
            }
        }
        w.cand = cand;
        w.snew = snew;
        acc as f64 / self.l.max(1) as f64
    }

    /// pyMagCalc `thermalize`'s two phases: adapt sigma over the first half
    /// (multiplicative rule, gain 2/sqrt(k+1), clipped to [1e-4, 10]), then hold
    /// it and record (acceptance, energy) per sweep. Returns (sigma, accs, energies).
    fn thermalize_impl(&self, z: &mut [C], kt: f64, n_sweeps: usize, sigma: f64, adapt: bool,
                       target: f64, rng: &mut Pcg64, w: &mut Work)
                       -> (f64, Vec<f64>, Vec<f64>) {
        let beta = 1.0 / kt;
        let n = n_sweeps.max(1);
        let n_adapt = if adapt { n / 2 } else { 0 };
        let mut sig = sigma;
        let mut s0 = std::mem::take(&mut w.s0);
        self.s0_all(z, &mut s0, &mut w.opz);
        for k in 0..n_adapt {
            let acc = self.sweep(z, &mut s0, beta, sig, rng, w);
            let gain = 2.0 / (k as f64 + 1.0).sqrt();
            sig = (sig * (gain * (acc - target)).exp()).clamp(1e-4, 10.0);
        }
        let mut accs = Vec::with_capacity(n - n_adapt);
        let mut energies = Vec::with_capacity(n - n_adapt);
        for _ in 0..(n - n_adapt) {
            accs.push(self.sweep(z, &mut s0, beta, sig, rng, w));
            energies.push(self.energy_of(z, &s0, &mut w.opz));
        }
        w.s0 = s0;
        (sig, accs, energies)
    }

    fn random_state(&self, rng: &mut Pcg64, z: &mut [C]) {
        for i in 0..self.l {
            let zi = &mut z[self.zoff[i]..self.zoff[i + 1]];
            for v in zi.iter_mut() {
                let g: f64 = rng.sample(StandardNormal);
                let gi: f64 = rng.sample(StandardNormal);
                *v = C { re: g, im: gi };
            }
            let r = norm(zi);
            for v in zi.iter_mut() {
                *v = *v / r;
            }
        }
    }

    fn check_z(&self, z: &[C], name: &str) -> PyResult<()> {
        if z.len() != self.d() {
            return Err(PyValueError::new_err(format!(
                "{name} must be the packed state of {} entries (sum of N_i), got {}",
                self.d(),
                z.len()
            )));
        }
        Ok(())
    }
}

fn to_vec(a: &PyReadonlyArray1<'_, C>) -> Vec<C> {
    a.as_array().iter().copied().collect()
}

fn usize_vec(a: &PyReadonlyArray1<'_, i64>, name: &str, bound: usize) -> PyResult<Vec<usize>> {
    let mut out = Vec::with_capacity(a.len());
    for &v in a.as_array().iter() {
        if v < 0 || (v as usize) >= bound {
            return Err(PyValueError::new_err(format!(
                "{name}: entry {v} out of range [0, {bound})"
            )));
        }
        out.push(v as usize);
    }
    Ok(out)
}

#[pymethods]
impl CpnModel {
    /// `ns`: N_i per site; `ops`: the per-site operator tables concatenated in
    /// site-major, then operator, then row-major order (sum_i n_ops N_i^2 entries);
    /// `a`: the on-site matrices concatenated (sum_i N_i^2, zeros where absent);
    /// `bi`, `bj`, `j`: the bond list, `j` as n_bonds x n_ops x n_ops row-major.
    #[new]
    fn new(
        ns: PyReadonlyArray1<'_, i64>,
        n_ops: usize,
        ops: PyReadonlyArray1<'_, C>,
        a: PyReadonlyArray1<'_, C>,
        bi: PyReadonlyArray1<'_, i64>,
        bj: PyReadonlyArray1<'_, i64>,
        j: PyReadonlyArray1<'_, C>,
    ) -> PyResult<Self> {
        let ns: Vec<usize> = ns
            .as_array()
            .iter()
            .map(|&n| {
                if n < 1 {
                    Err(PyValueError::new_err(format!("N_i must be >= 1, got {n}")))
                } else {
                    Ok(n as usize)
                }
            })
            .collect::<PyResult<_>>()?;
        let l = ns.len();
        if l == 0 {
            return Err(PyValueError::new_err("the model has no sites"));
        }
        if n_ops == 0 {
            return Err(PyValueError::new_err("n_ops must be >= 1"));
        }
        let mut zoff = vec![0usize; l + 1];
        let mut opoff = vec![0usize; l + 1];
        let mut aoff = vec![0usize; l + 1];
        for i in 0..l {
            zoff[i + 1] = zoff[i] + ns[i];
            opoff[i + 1] = opoff[i] + n_ops * ns[i] * ns[i];
            aoff[i + 1] = aoff[i] + ns[i] * ns[i];
        }
        if ops.len() != opoff[l] {
            return Err(PyValueError::new_err(format!(
                "ops must have sum_i n_ops N_i^2 = {} entries, got {}",
                opoff[l],
                ops.len()
            )));
        }
        if a.len() != aoff[l] {
            return Err(PyValueError::new_err(format!(
                "a must have sum_i N_i^2 = {} entries, got {}",
                aoff[l],
                a.len()
            )));
        }
        let bi = usize_vec(&bi, "bi", l)?;
        let bj = usize_vec(&bj, "bj", l)?;
        if bi.len() != bj.len() {
            return Err(PyValueError::new_err("bi and bj must have the same length"));
        }
        if j.len() != bi.len() * n_ops * n_ops {
            return Err(PyValueError::new_err(format!(
                "j must have n_bonds n_ops^2 = {} entries, got {}",
                bi.len() * n_ops * n_ops,
                j.len()
            )));
        }
        let mut from: Vec<Vec<usize>> = vec![Vec::new(); l];
        let mut touch: Vec<Vec<usize>> = vec![Vec::new(); l];
        for k in 0..bi.len() {
            from[bi[k]].push(k);
            touch[bi[k]].push(k);
            if bj[k] != bi[k] {
                touch[bj[k]].push(k);
            }
        }
        Ok(CpnModel {
            l,
            n_ops,
            ns,
            zoff,
            ops: to_vec(&ops),
            opoff,
            a: to_vec(&a),
            aoff,
            bi,
            bj,
            j: to_vec(&j),
            from,
            touch,
        })
    }

    #[getter]
    fn n_sites(&self) -> usize {
        self.l
    }

    #[getter]
    fn n_bonds(&self) -> usize {
        self.bi.len()
    }

    /// Offsets of each site's block in a packed state (length L + 1).
    #[getter]
    fn offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.zoff.iter().map(|&o| o as i64).collect::<Vec<_>>().into_pyarray(py)
    }

    /// E of a packed state (pyMagCalc `sun.dynamics.energy`).
    fn energy(&self, z: PyReadonlyArray1<'_, C>) -> PyResult<f64> {
        let z = to_vec(&z);
        self.check_z(&z, "z")?;
        let mut w = Work::new(self);
        self.s0_all(&z, &mut w.s0, &mut w.opz);
        Ok(self.energy_of(&z, &w.s0, &mut w.opz))
    }

    /// dE of replacing site `i`'s state by `cand` -- the local sum the sweep uses,
    /// exposed so the identity dE == E(after) - E(before) can be pinned.
    fn move_energy_change(&self, z: PyReadonlyArray1<'_, C>, i: usize,
                          cand: PyReadonlyArray1<'_, C>) -> PyResult<f64> {
        let z = to_vec(&z);
        self.check_z(&z, "z")?;
        if i >= self.l {
            return Err(PyValueError::new_err(format!("site {i} out of range (L = {})", self.l)));
        }
        let cand = to_vec(&cand);
        if cand.len() != self.ns[i] {
            return Err(PyValueError::new_err(format!(
                "cand must have N_i = {} entries, got {}",
                self.ns[i],
                cand.len()
            )));
        }
        let mut w = Work::new(self);
        self.s0_all(&z, &mut w.s0, &mut w.opz);
        let mut snew = vec![ZERO; self.n_ops];
        self.expect_site(i, &cand, &mut snew, &mut w.opz);
        Ok(self.move_delta(i, &z, &w.s0, &cand, &snew, &mut w.opz))
    }

    /// h_i |Z_i> for every site, packed (pyMagCalc `_h_times_z`).
    fn h_times_z<'py>(&self, py: Python<'py>, z: PyReadonlyArray1<'py, C>)
                      -> PyResult<Bound<'py, PyArray1<C>>> {
        let z = to_vec(&z);
        self.check_z(&z, "z")?;
        let mut w = Work::new(self);
        self.s0_all(&z, &mut w.s0, &mut w.opz);
        let mut out = vec![ZERO; z.len()];
        self.hz_of(&z, &w.s0, &mut out, &mut w.c, &mut w.opz);
        Ok(out.into_pyarray(py))
    }

    /// The whole `thermalize` sweep loop on the stream `rng`. Returns
    /// (final packed state, sigma, acceptance per recorded sweep, energy per
    /// recorded sweep); the recorded half is the fixed-sigma second phase.
    #[pyo3(signature = (z, kt, n_sweeps, rng, sigma = 0.5, adapt = true, target = 0.5))]
    fn thermalize<'py>(
        &self,
        py: Python<'py>,
        z: PyReadonlyArray1<'py, C>,
        kt: f64,
        n_sweeps: usize,
        mut rng: PyRefMut<'py, PyRng>,
        sigma: f64,
        adapt: bool,
        target: f64,
    ) -> PyResult<(Bound<'py, PyArray1<C>>, f64, Bound<'py, PyArray1<f64>>,
                   Bound<'py, PyArray1<f64>>)> {
        let mut z = to_vec(&z);
        self.check_z(&z, "z")?;
        if !(kt > 0.0) {
            return Err(PyValueError::new_err("kT must be positive"));
        }
        let r = &mut rng.inner;
        let ((sig, accs, energies), z) = py.detach(move || {
            let mut w = Work::new(self);
            let out = self.thermalize_impl(&mut z, kt, n_sweeps, sigma, adapt, target, r, &mut w);
            (out, z)
        });
        Ok((z.into_pyarray(py), sig, accs.into_pyarray(py), energies.into_pyarray(py)))
    }

    /// Deterministic trajectory from a packed `z0`: (n_rec, D), the state BEFORE
    /// each recorded step; `damping` = 0 is the conservative flow.
    #[pyo3(signature = (z0, dt, n_steps, record_every = 1, damping = 0.0))]
    fn evolve<'py>(&self, py: Python<'py>, z0: PyReadonlyArray1<'py, C>, dt: f64,
                   n_steps: usize, record_every: usize, damping: f64)
                   -> PyResult<Bound<'py, PyArray2<C>>> {
        let mut z = to_vec(&z0);
        self.check_z(&z, "z0")?;
        if record_every == 0 {
            return Err(PyValueError::new_err("record_every must be >= 1"));
        }
        let d = z.len();
        let n_rec = n_steps.div_ceil(record_every);
        let rec = py.detach(move || {
            let mut w = Work::new(self);
            let mut rec = vec![ZERO; n_rec * d];
            self.evolve_impl(&mut z, dt, damping, n_steps, record_every, &mut w, &mut rec);
            rec
        });
        let arr = Array2::from_shape_vec((n_rec, d), rec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// `n_steps` damped RK4 steps from `z0` (pyMagCalc `damped_step` / the
    /// `quench` loop between two recorded steps); returns the final packed state.
    #[pyo3(signature = (z0, dt, n_steps, damping))]
    fn quench<'py>(&self, py: Python<'py>, z0: PyReadonlyArray1<'py, C>, dt: f64,
                   n_steps: usize, damping: f64) -> PyResult<Bound<'py, PyArray1<C>>> {
        let mut z = to_vec(&z0);
        self.check_z(&z, "z0")?;
        let z = py.detach(move || {
            let mut w = Work::new(self);
            for _ in 0..n_steps {
                self.rk4(&mut z, dt, damping, &mut w);
            }
            z
        });
        Ok(z.into_pyarray(py))
    }

    /// `n_traj` thermalized CP^(N-1) trajectories in parallel (rayon). Each starts
    /// from `z0` (or a random coherent state when `z0` is None), thermalizes with
    /// the adaptive sampler on the stream (seed, traj_start + i), then evolves.
    /// Returns (traj (n_traj, n_rec, D), sigma (n_traj,), accs (n_traj, n_meas),
    /// energies (n_traj, n_meas)).
    #[pyo3(signature = (z0, kt, traj_start, n_traj, therm_sweeps, sigma, adapt, target,
                        dt, n_steps, record_every = 1, seed = 0, n_workers = None))]
    #[allow(clippy::too_many_arguments)]
    fn sampled_trajectories<'py>(
        &self,
        py: Python<'py>,
        z0: Option<PyReadonlyArray1<'py, C>>,
        kt: f64,
        traj_start: usize,
        n_traj: usize,
        therm_sweeps: usize,
        sigma: f64,
        adapt: bool,
        target: f64,
        dt: f64,
        n_steps: usize,
        record_every: usize,
        seed: u64,
        n_workers: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray3<C>>, Bound<'py, PyArray1<f64>>,
                   Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let z0: Option<Vec<C>> = match z0 {
            Some(a) => {
                let v = to_vec(&a);
                self.check_z(&v, "z0")?;
                Some(v)
            }
            None => None,
        };
        if !(kt > 0.0) {
            return Err(PyValueError::new_err("kT must be positive"));
        }
        if record_every == 0 {
            return Err(PyValueError::new_err("record_every must be >= 1"));
        }
        let d = self.d();
        let n_rec = n_steps.div_ceil(record_every);
        let per = n_rec * d;
        let n = therm_sweeps.max(1);
        let n_meas = n - if adapt { n / 2 } else { 0 };
        let out = py.detach(move || -> PyResult<_> {
            with_pool(n_workers, move || {
                let mut rec = vec![ZERO; n_traj * per];
                let mut sigs = vec![0.0; n_traj];
                let mut accs = vec![0.0; n_traj * n_meas];
                let mut energies = vec![0.0; n_traj * n_meas];
                rec.par_chunks_mut(per.max(1))
                    .zip(sigs.par_iter_mut())
                    .zip(accs.par_chunks_mut(n_meas.max(1)))
                    .zip(energies.par_chunks_mut(n_meas.max(1)))
                    .enumerate()
                    .for_each(|(i, (((chunk, sig), acc), en))| {
                        let mut rng = stream(seed, (traj_start + i) as u64);
                        let mut w = Work::new(self);
                        let mut z = match &z0 {
                            Some(v) => v.clone(),
                            None => {
                                let mut z = vec![ZERO; d];
                                self.random_state(&mut rng, &mut z);
                                z
                            }
                        };
                        let (s, a, e) = self.thermalize_impl(
                            &mut z, kt, therm_sweeps, sigma, adapt, target, &mut rng, &mut w);
                        *sig = s;
                        acc.copy_from_slice(&a);
                        en.copy_from_slice(&e);
                        self.evolve_impl(&mut z, dt, 0.0, n_steps, record_every, &mut w, chunk);
                    });
                (rec, sigs, accs, energies)
            })
        })?;
        let (rec, sigs, accs, energies) = out;
        let rec = Array3::from_shape_vec((n_traj, n_rec, d), rec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let accs = Array2::from_shape_vec((n_traj, n_meas), accs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let energies = Array2::from_shape_vec((n_traj, n_meas), energies)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((rec.into_pyarray(py), sigs.into_pyarray(py), accs.into_pyarray(py),
            energies.into_pyarray(py)))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CpnModel>()?;
    Ok(())
}
