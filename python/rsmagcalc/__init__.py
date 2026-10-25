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
           "Rng", "move_energy_change", "sweep", "parallel_tempering", "evolve",
           "langevin", "wang_landau", "sampled_trajectories"]

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
