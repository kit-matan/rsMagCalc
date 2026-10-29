"""rsMagCalc -- Rust kernels (PyO3) for pyMagCalc's beyond-LSWT loops.

The contract mirrors fMagCalc's: pyMagCalc never requires this package. It
selects the kernels with `calculation: {kernels: rust}`, checks
``rsmagcalc.backend == "pyo3"``, and otherwise warns and runs its own NumPy
implementation. ``backend`` is ``"pyo3"`` when the compiled core imported and
``None`` when only this pure-Python shell is present (a source checkout that
was never built with maturin).

Which kernels exist, in what order, and what each is pinned to:
``../RUST_KERNELS_PLAN.md`` at the workspace root.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["backend", "build_info", "describe", "classical_energy",
           "Rng", "move_energy_change", "sweep", "parallel_tempering", "anneal",
           "steepest_descent", "evolve",
           "langevin", "wang_landau", "sampled_trajectories", "CpnModel",
           "SunModel", "NotPositiveDefinite", "kpm_start_vector", "kpm_moments",
           "spectral_bound", "ScgaModel", "block_effective_series",
           "cluster_pt_batch"]

try:
    from . import _core
except ImportError:  # pragma: no cover - only without a maturin build
    _core = None

backend = "pyo3" if _core is not None else None


def build_info() -> dict:
    """Version, git head, target and profile of the compiled core.

    Reads only what the build baked in (no subprocess, no filesystem): this is
    called on the startup path of every `magcalc run` that asks for the kernels.
    """
    if _core is None:
        return {"version": __version__, "git_head": "", "target": "",
                "profile": "not built"}
    info = dict(_core.build_info())
    info.setdefault("version", __version__)
    return info


def describe() -> str:
    """One line naming this build, for pyMagCalc's run log and `magcalc where`."""
    info = build_info()
    parts = [f"rsmagcalc {info['version']}"]
    if info.get("git_head"):
        parts.append(f"git {info['git_head']}")
    parts.append(info.get("profile", "?"))
    return "(" + ", ".join(parts) + ")"


def classical_energy(H, b, m) -> float:
    """E = 1/2 m^T H m + b^T m, pyMagCalc `classical_dynamics.energy` in Rust."""
    if _core is None:
        raise RuntimeError(
            "rsmagcalc's compiled core is not built; run `pip install -e . "
            "--no-build-isolation` in rsMagCalc/ or `pip install /path/to/rsMagCalc`."
        )
    import numpy as np

    return _core.classical_energy(
        np.ascontiguousarray(H, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
        np.ascontiguousarray(m, dtype=np.float64),
    )


# --------------------------------------------------------------------------
# K1 -- classical Metropolis sweep and Landau-Lifshitz integrator (M1)
# --------------------------------------------------------------------------
#
# Every function below mirrors one in pyMagCalc's `thermal_mc` /
# `classical_dynamics` and is pinned to it in tests/test_classical_parity.py.
# Inputs: H (3N x 3N float64, symmetric), b (3N), S per site (a scalar is
# broadcast), m (N, 3). The gradient g = H m + b is the quantity pyMagCalc's
# `local_field` returns (the energy GRADIENT, not the field).

def _need_core():
    if _core is None:
        raise RuntimeError(
            "rsmagcalc's compiled core is not built; run `pip install -e . "
            "--no-build-isolation` in rsMagCalc/ or `pip install /path/to/rsMagCalc`."
        )


def _f64(x):
    import numpy as np

    return np.ascontiguousarray(x, dtype=np.float64)


def _spins(S, n):
    """Per-site spin lengths as an (N,) array; a scalar S is broadcast."""
    import numpy as np

    s = np.asarray(S, dtype=np.float64)
    if s.ndim == 0:
        return np.full(n, float(s))
    return _f64(s.reshape(-1))


def _hb(H, b):
    H = _f64(H)
    b = _f64(b).reshape(-1)
    return H, b, b.size // 3


class Rng:
    """Seeded PCG64 stream (`rsmagcalc.Rng(seed)`), the counterpart of
    `numpy.random.default_rng(seed)` for the kernels that continue a stream
    across calls (`sweep`, `langevin`). Deterministic per seed; NOT NumPy's
    stream."""

    __slots__ = ("_r",)

    def __init__(self, seed=0):
        _need_core()
        self._r = _core.Rng(int(seed))

    def random(self):
        return self._r.random()

    def standard_normal(self, size=None):
        import numpy as np

        if size is None:
            return float(self._r.standard_normal(1)[0])
        shape = (int(size),) if np.isscalar(size) else tuple(int(k) for k in size)
        n = int(np.prod(shape)) if shape else 1
        return self._r.standard_normal(n).reshape(shape)

    def integers(self, n):
        return int(self._r.integers(int(n)))

    def next_u64(self):
        return int(self._r.next_u64())

    def permutation(self, n):
        return self._r.permutation(int(n))

    def __repr__(self):
        return repr(self._r)


def move_energy_change(m, g, H, b, a, v):
    """dE of replacing spin `a` by `v`, from the gradient g: g_a.d + 1/2 d^T H_aa d."""
    _need_core()
    H, b, n = _hb(H, b)
    return _core.move_energy_change(_f64(m), _f64(g).reshape(-1), H, b, int(a),
                                    [float(x) for x in v])


def sweep(m, g, H, b, beta, S, rng, propose="uniform"):
    """One Metropolis sweep IN PLACE on `m` (N, 3) and its gradient `g` (3N);
    returns the acceptance fraction. `m` and `g` must be writable C-contiguous
    float64 arrays (the in-place contract cannot survive a copy)."""
    import numpy as np

    _need_core()
    if not isinstance(rng, Rng):
        raise TypeError("sweep needs an rsmagcalc.Rng, got %r" % type(rng).__name__)
    for name, arr in (("m", m), ("g", g)):
        if not (isinstance(arr, np.ndarray) and arr.dtype == np.float64
                and arr.flags.c_contiguous and arr.flags.writeable):
            raise TypeError(f"{name} must be a writable C-contiguous float64 array")
    H, b, n = _hb(H, b)
    return _core.sweep(m, g.reshape(-1), H, b, float(beta), _spins(S, n), rng._r,
                       str(propose))


def parallel_tempering(H, b, S, temperatures, n_sweeps, n_equil, swap_every=1,
                       measure_every=1, propose="uniform", init=None, seed=0,
                       n_workers=None):
    """The whole replica-exchange loop of pyMagCalc `parallel_tempering`.

    Returns a dict of the raw accumulators pyMagCalc turns into a ThermalResult:
    sumE, sumE2 (R,), sumM (R, 3), sumM2 (R,), n_meas, accept_rate. Replicas are
    swept in parallel; `n_workers` sizes the thread pool (None = all cores)."""
    _need_core()
    H, b, n = _hb(H, b)
    init = None if init is None else [float(x) for x in init]
    sE, sE2, sM, sM2, nmeas, rate = _core.parallel_tempering(
        H, b, _spins(S, n), _f64(temperatures).reshape(-1), int(n_sweeps),
        int(n_equil), int(swap_every), int(measure_every), str(propose), init,
        int(seed), None if n_workers is None else int(n_workers))
    return {"sumE": sE, "sumE2": sE2, "sumM": sM, "sumM2": sM2,
            "n_meas": int(nmeas), "accept_rate": float(rate)}


def anneal(H, b, c, S, n_sweeps, T_start, T_end, seeds, m0=None, polish=True,
           max_iter=500, tol=1e-10, n_workers=None):
    """pyMagCalc `annealing.anneal` for one or several runs at once.

    Run r draws from the stream of `seeds[r]` and starts from `m0` (N, 3) when
    r == 0 and one is given, else from a random state. The schedule, proposal mix,
    best-state tracking and keep-if-better steepest-descent polish are
    `annealing.anneal`'s. Runs execute in parallel (`n_workers`, None = all cores);
    a run's result does not depend on the thread count or on the other seeds.
    Returns (m_best (R, N, 3), e_best (R,)), e_best including `c`."""
    _need_core()
    import numpy as np

    H, b, n = _hb(H, b)
    seeds = [int(x) for x in np.atleast_1d(seeds)]
    m0 = None if m0 is None else _f64(np.asarray(m0, dtype=np.float64).reshape(-1, 3))
    return _core.anneal(H, b, float(c), _spins(S, n), int(n_sweeps), float(T_start),
                        float(T_end), seeds, m0, bool(polish), int(max_iter),
                        float(tol), None if n_workers is None else int(n_workers))


def steepest_descent(H, b, c, S, m0, max_iter=500, tol=1e-10):
    """pyMagCalc `annealing.steepest_descent` (deterministic): (m_best (N, 3), e_best)."""
    _need_core()
    import numpy as np

    H, b, n = _hb(H, b)
    m0 = _f64(np.asarray(m0, dtype=np.float64).reshape(-1, 3))
    return _core.steepest_descent(H, b, float(c), _spins(S, n), m0, int(max_iter),
                                  float(tol))


def evolve(H, b, S, m0, dt, n_steps, record_every=1, integrator="rk4"):
    """Deterministic LL trajectory: (n_rec, N, 3), plus the number of implicit
    midpoint steps that did not converge (always 0 for rk4)."""
    _need_core()
    H, b, n = _hb(H, b)
    return _core.evolve(H, b, _spins(S, n), _f64(m0), float(dt), int(n_steps),
                        int(record_every), str(integrator))


def langevin(H, b, S, m0, dt, kT, damping, n_steps, rng):
    """`n_steps` Heun steps of the Langevin thermostat (pyMagCalc `langevin_step`),
    continuing the stream `rng`; returns the final (N, 3) configuration."""
    _need_core()
    if not isinstance(rng, Rng):
        raise TypeError("langevin needs an rsmagcalc.Rng, got %r" % type(rng).__name__)
    H, b, n = _hb(H, b)
    return _core.langevin(H, b, _spins(S, n), _f64(m0), float(dt), float(kT),
                          float(damping), int(n_steps), rng._r)


def wang_landau(H, b, S, m0, e_min, e_max, n_bins, f_init, f_final, flatness,
                sweeps_per_check, max_sweeps, seed=0):
    """The Wang-Landau move loop from a configuration `m0` already inside the
    window. Returns (log_g, visits, f_final, n_refinements, sweeps_done)."""
    _need_core()
    H, b, n = _hb(H, b)
    return _core.wang_landau(H, b, _spins(S, n), _f64(m0), float(e_min), float(e_max),
                             int(n_bins), float(f_init), float(f_final),
                             float(flatness), int(sweeps_per_check), int(max_sweeps),
                             int(seed))


def sampled_trajectories(H, b, S, kT, traj_start, n_traj, therm_sweeps, dt, n_steps,
                         record_every=1, integrator="rk4", seed=0, n_workers=None):
    """`n_traj` thermalized LL trajectories in parallel: (n_traj, n_rec, N, 3) and
    the non-converged midpoint count. Trajectory i uses the stream
    (seed, traj_start + i), so chunked calls reproduce one big call."""
    _need_core()
    H, b, n = _hb(H, b)
    return _core.sampled_trajectories(
        H, b, _spins(S, n), float(kT), int(traj_start), int(n_traj), int(therm_sweeps),
        float(dt), int(n_steps), int(record_every), str(integrator), int(seed),
        None if n_workers is None else int(n_workers))


# --------------------------------------------------------------------------
# K2 -- CP^(N-1) Metropolis sampler, RK4 integrator and damped quench (M2)
# --------------------------------------------------------------------------
#
# Mirrors pyMagCalc `sun/dynamics.py` and is pinned to it in
# tests/test_cpn_parity.py. A `CpnModel` packs a `SUNModel`'s static data once
# (per-site N_i, operator tables, on-site matrices, bond list -- ragged, so a
# mixed-spin cell is fine); states travel as ONE packed complex vector of
# sum_i N_i entries (`pack` / `unpack`), which is what keeps a 1600-site quench
# from paying a Python object per site per step.

class CpnModel:
    """A pyMagCalc `SUNModel`'s static data, packed for the CP^(N-1) kernels.

    `Ns`: N_i per site; `ops[i][p]`: the N_i x N_i local operators (the same
    count n_ops on every site); `onsite`: `(i, A_i)` pairs (summed per site);
    `bonds`: `(i, j, J)` with J n_ops x n_ops, BOTH directions listed, as in
    `SUNModel.bonds` (the `dr` entry is not needed and not taken).
    """

    __slots__ = ("_m", "Ns", "L", "n_ops", "offsets", "uniform")

    def __init__(self, Ns, ops, onsite, bonds):
        import numpy as np

        _need_core()
        Ns = [int(n) for n in Ns]
        L = len(Ns)
        if L == 0:
            raise ValueError("CpnModel needs at least one site")
        n_ops = len(ops[0])
        if any(len(ol) != n_ops for ol in ops):
            raise ValueError("every site must expose the same number of operators")
        ops_flat = []
        for i in range(L):
            for p in range(n_ops):
                o = np.asarray(ops[i][p], dtype=np.complex128)
                if o.shape != (Ns[i], Ns[i]):
                    raise ValueError(f"ops[{i}][{p}] must be ({Ns[i]}, {Ns[i]}), "
                                     f"got {o.shape}")
                ops_flat.append(o.ravel())
        A = [np.zeros((n, n), dtype=np.complex128) for n in Ns]
        for (k, Amat) in onsite:
            Amat = np.asarray(Amat, dtype=np.complex128)
            if Amat.shape != (Ns[k], Ns[k]):
                raise ValueError(f"onsite[{k}] must be ({Ns[k]}, {Ns[k]}), "
                                 f"got {Amat.shape}")
            A[int(k)] += Amat
        bonds = list(bonds)
        bi = np.array([int(b[0]) for b in bonds], dtype=np.int64)
        bj = np.array([int(b[1]) for b in bonds], dtype=np.int64)
        if bonds:
            J = np.array([np.asarray(b[-1], dtype=np.complex128) for b in bonds])
            if J.shape != (len(bonds), n_ops, n_ops):
                raise ValueError(f"every bond J must be ({n_ops}, {n_ops}); got "
                                 f"stacked shape {J.shape}")
            J = J.ravel()
        else:
            J = np.zeros(0, dtype=np.complex128)
        self._m = _core.CpnModel(
            np.asarray(Ns, dtype=np.int64), int(n_ops),
            np.ascontiguousarray(np.concatenate(ops_flat)),
            np.ascontiguousarray(np.concatenate([a.ravel() for a in A])),
            bi, bj, np.ascontiguousarray(J))
        self.Ns, self.L, self.n_ops = Ns, L, n_ops
        self.offsets = np.asarray(self._m.offsets)
        self.uniform = len(set(Ns)) == 1

    # -- state packing ------------------------------------------------------
    def pack(self, Z):
        """A list of per-site states (or an (L, N) array) -> one complex vector."""
        import numpy as np

        if self.uniform and isinstance(Z, np.ndarray) and Z.shape == (self.L, self.Ns[0]):
            return np.ascontiguousarray(Z, dtype=np.complex128).ravel()
        out = np.empty(int(self.offsets[-1]), dtype=np.complex128)
        for i in range(self.L):
            z = np.asarray(Z[i], dtype=np.complex128).reshape(-1)
            if z.size != self.Ns[i]:
                raise ValueError(f"Z[{i}] must have {self.Ns[i]} entries, got {z.size}")
            out[self.offsets[i]:self.offsets[i + 1]] = z
        return out

    def unpack(self, flat):
        """One packed vector -> a list of per-site arrays (views, not copies)."""
        import numpy as np

        flat = np.asarray(flat)
        if self.uniform:
            return list(flat.reshape(self.L, self.Ns[0]))
        return [flat[self.offsets[i]:self.offsets[i + 1]] for i in range(self.L)]

    # -- kernels --------------------------------------------------------------
    def energy(self, Z):
        return self._m.energy(self.pack(Z))

    def move_energy_change(self, Z, i, cand):
        """dE of replacing Z[i] by `cand`, from the bonds touching i only."""
        import numpy as np

        return self._m.move_energy_change(self.pack(Z), int(i),
                                          np.ascontiguousarray(cand, dtype=np.complex128))

    def h_times_z(self, Z):
        """h_i |Z_i> for every site (pyMagCalc `_h_times_z`), as a list."""
        return self.unpack(self._m.h_times_z(self.pack(Z)))

    def thermalize(self, Z, kT, n_sweeps, rng, sigma=0.5, adapt=True, target=0.5):
        """pyMagCalc `thermalize`'s sweep loop on the stream `rng` (an `Rng`).
        Returns (Z as a list, sigma, acceptance per recorded sweep, energy per
        recorded sweep) -- the recorded sweeps are the fixed-sigma second half."""
        if not isinstance(rng, Rng):
            raise TypeError("thermalize needs an rsmagcalc.Rng, got %r" % type(rng).__name__)
        flat, sig, accs, energies = self._m.thermalize(
            self.pack(Z), float(kT), int(n_sweeps), rng._r, float(sigma), bool(adapt),
            float(target))
        return self.unpack(flat), float(sig), accs, energies

    def evolve(self, Z0, dt, n_steps, record_every=1, damping=0.0):
        """Deterministic trajectory: a list over recorded steps of per-site lists."""
        rec = self._m.evolve(self.pack(Z0), float(dt), int(n_steps), int(record_every),
                             float(damping))
        return [self.unpack(row) for row in rec]

    def quench(self, Z0, dt, n_steps, damping):
        """`n_steps` damped RK4 steps; the final state as a list."""
        return self.unpack(self._m.quench(self.pack(Z0), float(dt), int(n_steps),
                                          float(damping)))

    def sampled_trajectories(self, Z0, kT, traj_start, n_traj, therm_sweeps, sigma,
                             adapt, target, dt, n_steps, record_every=1, seed=0,
                             n_workers=None):
        """`n_traj` thermalized trajectories in parallel, starting from `Z0` (or
        from random coherent states when `Z0` is None). Returns (traj as an
        (n_traj, n_rec, D) packed array, sigma per trajectory, acceptance
        (n_traj, n_meas), energies (n_traj, n_meas)); trajectory i owns the stream
        (seed, traj_start + i), so chunked calls reproduce one call."""
        z0 = None if Z0 is None else self.pack(Z0)
        return self._m.sampled_trajectories(
            z0, float(kT), int(traj_start), int(n_traj), int(therm_sweeps), float(sigma),
            bool(adapt), float(target), float(dt), int(n_steps), int(record_every),
            int(seed), None if n_workers is None else int(n_workers))


# --------------------------------------------------------------------------
# K3 -- SU(N) LSWT: H(q) assembly, Colpa diagonalization, one-magnon
# correlations, general eigenvalues, the shifted-Cholesky guard (M3)
# --------------------------------------------------------------------------
#
# Mirrors pyMagCalc `sun/lswt.py` (`SUNModel.hamiltonian`, `_bogoliubov`,
# `structure_factor` up to the cross-section contraction, `dispersion` /
# `max_imaginary` through `eigenvalues`, and `is_stable_at`) and is pinned to it in
# tests/test_sun_lswt_parity.py. A `SunModel` packs the reference-state tables
# `SUNModel._prepare` built (t, tb, Q, QA, s0 -- ragged per site, so mixed spin
# is fine) and the bond list once; every method then runs a q BATCH in parallel.

class NotPositiveDefinite(ValueError):
    """Raised by `SunModel.bogoliubov` / `correlations` when H2(q) is not positive
    definite at some q (the reference state is not a minimum). pyMagCalc turns it
    into the same `numpy.linalg.LinAlgError` its NumPy path raises."""


if _core is not None:  # the Rust exception IS this one, so `except` works either way
    NotPositiveDefinite = _core.NotPositiveDefinite  # noqa: F811


class SunModel:
    """A pyMagCalc `SUNModel`'s reference-state tables, packed for the LSWT kernels.

    Arguments are the `SUNModel` attributes of the same names: `Ms` (bosons per
    site), `s0` (L, n_ops), `t` / `tb` (per site (n_ops, M_i)), `Q` (per site
    (n_ops, M_i, M_i)), `QA` (per site (M_i, M_i)), `bonds` as `(i, j, dr_cart, J)`
    with J (n_ops, n_ops) REAL and both directions listed, `moment_terms` (per site a
    list of `(d_k, (ix, iy, iz))`), `n_cells`.
    """

    __slots__ = ("_m", "L", "D", "n_ops", "Ms")

    def __init__(self, Ms, s0, t, tb, Q, QA, bonds, moment_terms, n_cells=1, n_ops=None):
        import numpy as np

        _need_core()
        Ms = [int(m) for m in Ms]
        L = len(Ms)
        if L == 0:
            raise ValueError("SunModel needs at least one site")
        s0 = np.ascontiguousarray(s0, dtype=np.complex128)
        if n_ops is None:
            n_ops = int(s0.shape[1])
        if s0.shape != (L, n_ops):
            raise ValueError(f"s0 must be ({L}, {n_ops}), got {s0.shape}")

        def _cat(parts, shapes, name):
            flat = []
            for i, (p, shp) in enumerate(zip(parts, shapes)):
                a = np.asarray(p, dtype=np.complex128)
                if a.shape != shp:
                    raise ValueError(f"{name}[{i}] must be {shp}, got {a.shape}")
                flat.append(a.ravel())
            return np.ascontiguousarray(np.concatenate(flat)) if flat \
                else np.zeros(0, dtype=np.complex128)

        t_flat = _cat(t, [(n_ops, m) for m in Ms], "t")
        tb_flat = _cat(tb, [(n_ops, m) for m in Ms], "tb")
        q_flat = _cat(Q, [(n_ops, m, m) for m in Ms], "Q")
        qa_flat = _cat(QA, [(m, m) for m in Ms], "QA")

        bonds = list(bonds)
        nb = len(bonds)
        bi = np.array([int(b[0]) for b in bonds], dtype=np.int64)
        bj = np.array([int(b[1]) for b in bonds], dtype=np.int64)
        dr = np.zeros((nb, 3), dtype=np.float64)
        J = np.zeros((nb, n_ops, n_ops), dtype=np.float64)
        for k, (i, j, d, jm) in enumerate(bonds):
            dr[k] = np.asarray(d, dtype=np.float64).reshape(3)
            jm = np.asarray(jm)
            if jm.shape != (n_ops, n_ops):
                raise ValueError(f"bond {k}: J must be ({n_ops}, {n_ops}), got {jm.shape}")
            if np.iscomplexobj(jm) and np.any(jm.imag):
                raise ValueError(f"bond {k}: J must be real (SUNModel stores real J)")
            J[k] = jm.real
        mt_site, mt_d, mt_idx = [], [], []
        for i, terms in enumerate(moment_terms):
            for (d, idx) in terms:
                mt_site.append(i)
                mt_d.append(np.asarray(d, dtype=np.float64).reshape(3))
                mt_idx.append([int(x) for x in idx])
        mt_site = np.asarray(mt_site, dtype=np.int64)
        mt_d = (np.asarray(mt_d, dtype=np.float64).reshape(-1, 3) if mt_site.size
                else np.zeros((0, 3)))
        mt_idx = (np.asarray(mt_idx, dtype=np.int64).reshape(-1, 3) if mt_site.size
                  else np.zeros((0, 3), dtype=np.int64))
        self._m = _core.SunModel(
            np.asarray(Ms, dtype=np.int64), int(n_ops), s0.ravel(), t_flat, tb_flat,
            q_flat, qa_flat, bi, bj, np.ascontiguousarray(dr), np.ascontiguousarray(J),
            mt_site, np.ascontiguousarray(mt_d), np.ascontiguousarray(mt_idx),
            float(n_cells))
        self.L, self.D, self.n_ops, self.Ms = L, int(self._m.d), int(n_ops), Ms

    @staticmethod
    def _qs(qs):
        import numpy as np

        return np.ascontiguousarray(np.asarray(qs, dtype=np.float64).reshape(-1, 3))

    def hamiltonians(self, qs, n_workers=None):
        """g H2(q) per q: (nq, 2D, 2D) complex (`SUNModel.hamiltonian`)."""
        return self._m.hamiltonians(self._qs(qs), n_workers)

    def eigenvalues(self, qs, n_workers=None):
        """All 2D eigenvalues of g H2(q) per q, unsorted: (nq, 2D) complex."""
        return self._m.eigenvalues(self._qs(qs), n_workers)

    def bogoliubov(self, qs, n_workers=None):
        """Colpa per q: (omega (nq, D) descending, T (nq, 2D, 2D)) as
        `SUNModel._bogoliubov` returns them. Raises NotPositiveDefinite."""
        om, tt = self._m.bogoliubov(self._qs(qs), n_workers)
        return om, tt.transpose(0, 2, 1)

    def correlations(self, qs, n_workers=None):
        """(omega (nq, D), S^{ab}_nu (nq, 3, 3, D)): the per-band one-magnon
        correlation tensors of `SUNModel.structure_factor`, before the
        cross-section contraction and the form factor."""
        return self._m.correlations(self._qs(qs), n_workers)

    def stable(self, qs, shift, n_workers=None, sparse=None):
        """Per q: is H2(q) + shift I positive definite (`SUNModel.is_stable_at`)?

        `sparse`: None lets the model choose (a multifrontal sparse Cholesky on a
        minimum-degree ordering when it needs at most a quarter of the dense flops
        and 2D >= 64, else one dense `zpotrf`); True / False force either. Both
        test the same bit-identical shifted matrix; see `guard_info`."""
        import numpy as np

        return np.asarray(self._m.stable(self._qs(qs), float(shift), n_workers, sparse),
                          dtype=bool)

    def guard_info(self):
        """The sparse guard's analysis: n, n_super, n_fronts, nnz_l, flops_sparse,
        flops_dense, max_front, `auto` (the choice `stable(sparse=None)` makes) and
        `error` (None unless the analysis failed, in which case auto is dense)."""
        return dict(self._m.guard_info())

    def kpm_moments(self, qs, n_moments=None, gammas=None, fwhm=None, tol=None,
                    n_iter=40, pad=1.15, seed=0, sparse=None, n_workers=None):
        """KPM over a q batch, H(q) built inside (K4): `(gamma_raw (nq,),
        n_moments (nq,), moments (nq, M_max, 3, 3))`, each q's moments zero past
        its own count.

        `gammas` fixes the bound per q, else it is `spectral_bound` from NumPy's
        start vector for `seed`. `n_moments` fixes the count, else it is
        `sun.kpm`'s rule from `fwhm` and `tol`. `gamma_raw` is the bound BEFORE
        the Goldstone substitution (<= 0 or non-finite -> 1), so the caller can
        say so. `sparse`: None picks CSR when H(q) is at most a quarter full.
        """
        import numpy as np

        qs = self._qs(qs)
        x0 = None
        if gammas is not None:
            gammas = np.ascontiguousarray(gammas, dtype=np.float64).reshape(-1)
        else:
            x0 = kpm_start_vector(2 * self.D, seed)
        rule = None
        if n_moments is None:
            if fwhm is None or tol is None:
                raise ValueError("kpm_moments needs n_moments, or fwhm and tol")
            # The two factors of sun.kpm's rule, computed as it computes them.
            rule = (0.5 * fwhm, float(-np.log10(tol)))
        else:
            n_moments = int(n_moments)
        return _core.kpm_batch(self._m, qs, x0, gammas, n_moments, rule, int(n_iter),
                               float(pad), sparse, n_workers)

    def __repr__(self):
        return repr(self._m)


# --------------------------------------------------------------------------
# K4: the KPM Chebyshev recursion and its spectral bound
# --------------------------------------------------------------------------
# Mirrors pyMagCalc `sun/kpm.py` (the moment loop of `kpm_sqw` and
# `spectral_bound`) and is pinned to it in tests/test_kpm_parity.py. The batch
# entry point is `SunModel.kpm_moments`; these two take an already-built g H2.

def kpm_start_vector(n, seed=0):
    """`sun.kpm.spectral_bound`'s start vector, drawn exactly as it draws it
    (NumPy's generator), so the Rust bound iterates from the same point."""
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x /= np.linalg.norm(x)
    return np.ascontiguousarray(x, dtype=np.complex128)


def kpm_moments(HMat, v, n_moments, gamma, sparse=None):
    """mu_m^{ab} = v_a^H T_m(conj(HMat)/gamma) g v_b: (n_moments, 3, 3) complex.
    `v` is the (3, 2D) observable of `sun.kpm._observable_vectors`."""
    import numpy as np

    _need_core()
    return _core.kpm_moments(np.ascontiguousarray(HMat, dtype=np.complex128),
                             np.ascontiguousarray(v, dtype=np.complex128),
                             int(n_moments), float(gamma), sparse)


def spectral_bound(HMat, n_iter=40, pad=1.15, seed=0, sparse=None):
    """pad * sqrt(lambda_max(HMat^H HMat)) by power iteration (`sun.kpm.spectral_bound`)."""
    import numpy as np

    _need_core()
    H = np.ascontiguousarray(HMat, dtype=np.complex128)
    return _core.spectral_bound(H, kpm_start_vector(H.shape[0], seed), int(n_iter),
                                float(pad), sparse)


# --------------------------------------------------------------------------
# K5: the self-consistent Gaussian approximation (SCGA)
# --------------------------------------------------------------------------
# Mirrors pyMagCalc `scga.py` (`fourier_exchange_matrix`, the eigenvalue mesh of
# `solve_lambda`, `sublattice_sum_rule_residuals`, and `scga_intensities` up to
# `_contract`) and is pinned to it in tests/test_scga_parity.py. The lambda root
# solve, its bracketing, the sublattice refusal and the cross-section contraction
# stay in pyMagCalc.

class ScgaModel:
    """A pyMagCalc `scga._exchange_cache`, packed for the SCGA kernels.

    `bonds` is the cache's directed bond list `(i, jc, R_cart, M)` in its order
    (M real 3 x 3), `pos` (N, 3) Cartesian site positions, `g` the per-site 3 x 3
    g-tensors, `spins` (N,). `from_cache` takes the dict itself.
    """

    __slots__ = ("_m", "N")

    def __init__(self, bonds, pos, g, spins):
        import numpy as np

        _need_core()
        pos = _f64(np.asarray(pos, dtype=np.float64).reshape(-1, 3))
        N = pos.shape[0]
        bonds = list(bonds)
        nb = len(bonds)
        bi = np.array([int(b[0]) for b in bonds], dtype=np.int64)
        bj = np.array([int(b[1]) for b in bonds], dtype=np.int64)
        r = np.zeros((nb, 3))
        m = np.zeros((nb, 3, 3))
        for k, (_i, _j, R, M) in enumerate(bonds):
            r[k] = np.asarray(R, dtype=np.float64).reshape(3)
            M = np.asarray(M)
            if np.iscomplexobj(M) and np.any(M.imag):
                raise ValueError(f"bond {k}: M must be real")
            m[k] = np.real(M).reshape(3, 3)
        g = _f64(np.asarray(g, dtype=np.float64).reshape(N, 3, 3))
        spins = _f64(np.asarray(spins, dtype=np.float64).reshape(N))
        self._m = _core.ScgaModel(bi, bj, _f64(r), _f64(m), pos, g, spins)
        self.N = N

    @classmethod
    def from_cache(cls, cache):
        return cls(cache["bonds"], cache["pos"], cache["g"], cache["spins"])

    @staticmethod
    def _qs(qs):
        import numpy as np

        return np.ascontiguousarray(np.asarray(qs, dtype=np.float64).reshape(-1, 3))

    def fourier(self, qs, n_workers=None):
        """J(q) per q: (nq, 3N, 3N) complex (`fourier_exchange_matrix`)."""
        return self._m.fourier(self._qs(qs), n_workers)

    def eigenvalues(self, qs, n_workers=None):
        """The eigenvalues of J(q) per q, ascending: (nq, 3N) (`eigvalsh`)."""
        return self._m.eigenvalues(self._qs(qs), n_workers)

    def site_traces(self, qs, lam, n_workers=None):
        """Tr_3 of each site's diagonal block of (lam + J(q))^{-1}: (nq, N)."""
        return self._m.site_traces(self._qs(qs), float(lam), n_workers)

    def sum_rule_residuals(self, mesh, lam, kT, n_workers=None):
        """`sublattice_sum_rule_residuals` over `mesh`: (N,)."""
        return self._m.sum_rule_residuals(self._qs(mesh), float(lam), float(kT), n_workers)

    def correlations(self, qs, lam, kT, ff=None, apply_g=True, n_workers=None):
        """C^{ab}(q) = kT sum_ij f_i f_j conj(e^{iq.r_i}) e^{iq.r_j} g_i
        [(lam + J(q))^{-1}]_ij g_j^T: (nq, 3, 3) complex, before the cross-section
        contraction. `ff` (nq, N) real form factors (None: 1); `apply_g` False uses
        g = I (the bare spin correlations)."""
        import numpy as np

        qs = self._qs(qs)
        if ff is not None:
            ff = _f64(np.asarray(ff, dtype=np.float64).reshape(len(qs), self.N))
        return self._m.correlations(qs, float(lam), float(kT), ff, bool(apply_g),
                                    n_workers)

    def __repr__(self):
        return repr(self._m)


# --------------------------------------------------------------------------
# K7: the per-cluster perturbation theory of the dimer series expansion
# --------------------------------------------------------------------------
# Mirrors pyMagCalc `sun/dimer_series.py` (`block_effective_series` and
# `_cluster_pt`) and is pinned to it in tests/test_dimer_series_parity.py. The
# cluster enumeration, the isomorphism keys, the linked-cluster subtraction and
# the resummation stay in pyMagCalc.

def _csr_triplet(V, n):
    """(indptr, indices, data) of an n x n matrix: dense (exact zeros dropped) or
    anything with `.tocsr()` (scipy.sparse; duplicates summed)."""
    import numpy as np

    if hasattr(V, "tocsr"):
        c = V.tocsr(copy=True)
        c.sum_duplicates()
        if c.shape != (n, n):
            raise ValueError(f"V is {c.shape}, H0 is ({n}, {n})")
        return (np.ascontiguousarray(c.indptr, dtype=np.int64),
                np.ascontiguousarray(c.indices, dtype=np.int64),
                np.ascontiguousarray(c.data, dtype=np.complex128))
    A = np.asarray(V, dtype=np.complex128)
    if A.shape != (n, n):
        raise ValueError(f"V is {A.shape}, H0 is ({n}, {n})")
    rows, cols = np.nonzero(A)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return (indptr, np.ascontiguousarray(cols, dtype=np.int64),
            np.ascontiguousarray(A[rows, cols]))


def block_effective_series(E0, V_list, P_idx, order):
    """[H_eff_0 .. H_eff_order] (each p x p) for H(lambda) = diag(E0) + sum_j
    V_list[j-1] lambda^j and the model space P_idx (pyMagCalc
    `sun.dimer_series.block_effective_series`). Each V is dense or scipy sparse."""
    import numpy as np

    _need_core()
    E0 = _f64(np.asarray(E0, dtype=np.float64).ravel())
    vs = [_csr_triplet(V, len(E0)) for V in V_list]
    H = _core.block_effective_series(E0, vs, [int(i) for i in P_idx], int(order))
    return list(H)


def cluster_pt_batch(clusters, order, n_workers=None):
    """pyMagCalc `sun.dimer_series._cluster_pt` for many clusters, in parallel.

    `clusters`: iterable of (levels_list, links_local, sz_list) exactly as
    `_cluster_pt` takes them -- per-node local energies (nloc,), links
    [(pa, pb, V (nloc^2, nloc^2))] with V on the index sa*nloc + sb of node pa
    first, and per-node level Sz (nloc,) or None. Returns [(gs_int (order+1,),
    delta_int (order+1, p, p))] in the same order.
    """
    import numpy as np

    _need_core()
    packed = []
    for levels_list, links_local, sz_list in clusters:
        lv = _f64(np.array([np.asarray(l, dtype=np.float64) for l in levels_list]))
        links = [(int(pa), int(pb), np.ascontiguousarray(V, dtype=np.complex128))
                 for (pa, pb, V) in links_local]
        sz = (None if sz_list is None else
              _f64(np.array([np.asarray(s, dtype=np.float64) for s in sz_list])))
        packed.append((lv, links, sz))
    return _core.cluster_pt_batch(packed, int(order), n_workers)
