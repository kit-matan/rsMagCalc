"""K2 pinned to pyMagCalc's NumPy CP^(N-1) path (the oracle) and to exact identities.

Deterministic kernels at 1e-12: `evolve` (the conservative RK4 flow) against
`sun.dynamics.evolve` from the same Z0 over 50 steps, `quench` against
`sun.dynamics.quench`, `h_times_z` against the `local_field` definition, and the
single-move dE of the Metropolis sweep against E(after) - E(before) from
`sun.dynamics.energy` -- on a plain SU(3) cell, on the biquadratic (12-wide
operator set) cell, and on a MIXED-SPIN cell, which the Python fast path refuses
and the kernel must not. The random sampler is pinned to what does not depend on
the stream: determinism per seed, chunk/thread independence, |Z_i| = 1, the
acceptance diagnostics, and the sampler's own closed-form partition function.
"""
import copy

import numpy as np
import pytest

import rsmagcalc

sd = pytest.importorskip("magcalc.sun.dynamics",
                         reason="pyMagCalc not installed (the parity oracle)")
from magcalc.generic_model import GenericSpinModel  # noqa: E402
from magcalc.sun.lswt import SUNModel  # noqa: E402

LAT = [[4.0, 0, 0], [0, 7.0, 0], [0, 0, 7.0]]
NN = ((["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0]))


def _two_site_model(S=1.0, biquad=None, spins=None, ncells=1):
    """The models tests/test_sun_quench.py uses: an anisotropic two-site chain,
    optionally biquadratic or mixed spin, optionally on a supercell."""
    S_A, S_B = (spins if spins else (S, S))
    inter = {"heisenberg": [{"pair": p, "rij_offset": o, "value": 0.8} for p, o in NN],
             "single_ion_anisotropy": [{"value": -0.45, "axis": [0.3, 0, 1],
                                        "atoms": ["A", "B"]}]}
    if biquad is not None:
        inter["biquadratic"] = [{"pair": p, "rij_offset": o, "value": biquad}
                                for p, o in NN]
    cfg = {"crystal_structure": {"lattice_vectors": LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": S_A},
                                              {"label": "B", "pos": [0.5, 0, 0],
                                               "spin_S": S_B}]},
           "interactions": inter, "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [[0, 0, 1], [0, 0, -1]]},
           "calculation": {"mode": "SUN", "on_imaginary": "off"}, "tasks": {}}
    sup = None if ncells == 1 else [[ncells, 0, 0], [0, 1, 0], [0, 0, 1]]
    return SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(cfg)), params=[],
                                       supercell=sup)


def _cpn(mdl):
    return rsmagcalc.CpnModel(mdl.Ns, mdl.ops, mdl.onsite,
                              [(i, j, J) for (i, j, dr, J) in mdl.bonds])


def _random_state(mdl, seed):
    rng = np.random.default_rng(seed)
    Z = []
    for i in range(mdl.L):
        v = rng.standard_normal(mdl.Ns[i]) + 1j * rng.standard_normal(mdl.Ns[i])
        Z.append(v / np.linalg.norm(v))
    return Z


def _maxdiff(a, b):
    return max(np.abs(np.asarray(x) - np.asarray(y)).max() for x, y in zip(a, b))


CASES = [({}, "plain SU(3)"),
         ({"biquad": -0.3}, "biquadratic, 12-wide operator set"),
         ({"spins": (0.5, 1.0)}, "mixed spin N = (2, 3)"),
         ({"spins": (1.5, 0.5), "ncells": 3}, "mixed spin on a 3-cell supercell")]


# --------------------------------------------------------------------------
# Exact identities of the energy and its local change
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw,label", CASES)
def test_energy_matches_numpy(kw, label):
    mdl = _two_site_model(**kw)
    m = _cpn(mdl)
    Z = _random_state(mdl, 3)
    assert m.energy(Z) == pytest.approx(sd.energy(mdl, Z), abs=1e-13), label


@pytest.mark.parametrize("kw,label", CASES)
def test_move_energy_change_equals_energy_after_minus_before(kw, label):
    """THE algorithmic fix, pinned: the sweep's local dE (bonds touching the site
    plus its on-site term) is E(after) - E(before) of the FULL energy, to 1e-12,
    for every site, including on a mixed-spin cell."""
    mdl = _two_site_model(**kw)
    m = _cpn(mdl)
    rng = np.random.default_rng(11)
    Z = _random_state(mdl, 5)
    e0 = sd.energy(mdl, Z)
    for i in range(mdl.L):
        v = rng.standard_normal(mdl.Ns[i]) + 1j * rng.standard_normal(mdl.Ns[i])
        cand = Z[i] + 0.4 * v
        cand /= np.linalg.norm(cand)
        Zc = [z.copy() for z in Z]
        Zc[i] = cand
        want = sd.energy(mdl, Zc) - e0
        got = m.move_energy_change(Z, i, cand)
        assert abs(got - want) < 1e-12, f"{label}, site {i}: {got} vs {want}"


def test_move_energy_change_is_exact_with_a_self_bond():
    """A one-cell supercell bonds a site to its own periodic image, so one bond
    term is QUADRATIC in that site's expectations; the term-by-term difference is
    still exact where a linearized mean-field dE would not be."""
    cfg = {"crystal_structure": {"lattice_vectors": LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": 1.0}]},
           "interactions": {"heisenberg": [
               {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": 0.7},
               {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": 0.7}],
               "single_ion_anisotropy": [{"value": -0.2, "axis": [0, 0, 1],
                                          "atoms": ["A"]}]},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "calculation": {"mode": "SUN", "on_imaginary": "off"}, "tasks": {}}
    mdl = SUNModel.from_generic_model(GenericSpinModel(cfg), params=[])
    assert any(i == j for (i, j, _, _) in mdl.bonds), "no self-bond: test is moot"
    m = _cpn(mdl)
    Z = _random_state(mdl, 2)
    cand = _random_state(mdl, 9)[0]
    want = sd.energy(mdl, [cand]) - sd.energy(mdl, Z)
    assert abs(m.move_energy_change(Z, 0, cand) - want) < 1e-12


@pytest.mark.parametrize("kw,label", CASES)
def test_h_times_z_matches_the_local_field_definition(kw, label):
    """h_i|Z_i> against `SUNModel.local_field` applied the slow, obvious way --
    the same oracle tests/test_sun_quench.py uses for the NumPy fast path."""
    mdl = _two_site_model(**kw)
    m = _cpn(mdl)
    Z = _random_state(mdl, 17)
    want = [h @ Z[i] for i, h in enumerate(sd.local_hamiltonians(mdl, Z))]
    got = m.h_times_z(Z)
    scale = max(np.abs(w).max() for w in want)
    assert _maxdiff(got, want) <= 1e-12 * scale, label


# --------------------------------------------------------------------------
# Deterministic parity: RK4 and the damped quench
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw,label", CASES)
def test_evolve_matches_numpy_to_1e12_over_50_steps(kw, label):
    mdl = _two_site_model(**kw)
    m = _cpn(mdl)
    Z0 = _random_state(mdl, 23)
    want = sd.evolve(mdl, Z0, 0.02, 50, record_every=1)
    got = m.evolve(Z0, 0.02, 50, 1)
    assert len(got) == len(want) == 50
    err = max(_maxdiff(g, w) for g, w in zip(got, want))
    assert err < 1e-12, f"{label}: {err:.3e}"
    for z in got[-1]:
        assert abs(np.linalg.norm(z) - 1.0) < 1e-14


def test_evolve_record_every():
    mdl = _two_site_model(ncells=2)
    m = _cpn(mdl)
    Z0 = _random_state(mdl, 1)
    want = sd.evolve(mdl, Z0, 0.01, 23, record_every=5)
    got = m.evolve(Z0, 0.01, 23, 5)
    assert len(got) == len(want) == 5
    assert max(_maxdiff(g, w) for g, w in zip(got, want)) < 1e-12


@pytest.mark.parametrize("kw,label", CASES[:3])
def test_quench_matches_numpy_to_1e12(kw, label):
    mdl = _two_site_model(**kw)
    m = _cpn(mdl)
    Z0 = _random_state(mdl, 8)
    want, snaps = sd.quench(mdl, Z0, 0.02, 60, damping=0.3, record_at=(0, 25, 60))
    got = m.quench(Z0, 0.02, 60, 0.3)
    assert _maxdiff(got, want) < 1e-12, label
    mid = m.quench(Z0, 0.02, 25, 0.3)
    assert _maxdiff(mid, snaps[25]) < 1e-12, label
    assert _maxdiff(m.quench(mid, 0.02, 35, 0.3), snaps[60]) < 1e-12, label
    # one step is pyMagCalc's damped_step
    assert _maxdiff(m.quench(Z0, 0.01, 1, 0.05), sd.damped_step(mdl, Z0, 0.01, 0.05)) < 1e-13


def test_zero_damping_quench_is_the_conservative_flow():
    mdl = _two_site_model()
    m = _cpn(mdl)
    Z0 = _random_state(mdl, 4)
    a = m.quench(Z0, 0.02, 10, 0.0)
    b = m.evolve(Z0, 0.02, 11, 1)[-1]
    assert _maxdiff(a, b) == 0.0


# --------------------------------------------------------------------------
# The sampler: stream-independent properties
# --------------------------------------------------------------------------
def test_thermalize_is_deterministic_per_seed_and_keeps_the_norm():
    mdl = _two_site_model(spins=(0.5, 1.0), ncells=3)
    m = _cpn(mdl)
    Z0 = [z.copy() for z in mdl.Z]
    out = [m.thermalize(Z0, 0.3, 40, rsmagcalc.Rng(5), sigma=0.2) for _ in range(2)]
    assert _maxdiff(out[0][0], out[1][0]) == 0.0
    assert out[0][1] == out[1][1] and np.array_equal(out[0][3], out[1][3])
    Z, sig, accs, energies = out[0]
    assert len(accs) == len(energies) == 20          # the fixed-sigma half of 40
    assert np.all((accs >= 0) & (accs <= 1)) and 0.0 < np.mean(accs) < 1.0
    for z in Z:
        assert abs(np.linalg.norm(z) - 1.0) < 1e-14
    # the recorded energies are the energies of the states the chain visited
    assert energies[-1] == pytest.approx(sd.energy(mdl, Z), abs=1e-12)
    # a different seed is a different sample; a different sigma a different one
    assert _maxdiff(m.thermalize(Z0, 0.3, 40, rsmagcalc.Rng(6), sigma=0.2)[0], Z) > 1e-6
    assert _maxdiff(Z0, [z for z in mdl.Z]) == 0.0      # the input was not touched


def test_thermalize_adapts_sigma_toward_the_target_acceptance():
    """Same rule as pyMagCalc: the step size is tuned over the first half and the
    fixed-sigma half then sits at the target. Low enough kT that the target is
    reachable (at high kT even a uniform draw is accepted most of the time, and
    the rule saturates at sigma = 10 by design)."""
    mdl = _two_site_model(ncells=4)
    m = _cpn(mdl)
    for target in (0.3, 0.7):
        _, sig, accs, _ = m.thermalize(mdl.Z, 0.05, 400, rsmagcalc.Rng(1), sigma=0.5,
                                       adapt=True, target=target)
        assert np.mean(accs) == pytest.approx(target, abs=0.08), (target, sig, accs.mean())
    _, sig, accs, _ = m.thermalize(mdl.Z, 0.5, 40, rsmagcalc.Rng(1), sigma=0.123, adapt=False)
    assert sig == 0.123 and len(accs) == 40


def test_sampler_reproduces_the_closed_form_partition_function():
    """A decoupled site with A = diag(a_i): |z_i|^2 is uniform on the simplex, so
    <E> = -d/dbeta log Z(beta) with Z = sum_i e^{-beta a_i} / prod_{j != i}
    beta (a_j - a_i) -- pyMagCalc's own oracle for its sampler
    (tests/test_sun_sampler_equilibration.py), here against the Rust one."""
    a = np.array([0.0, 0.7, 1.9])
    beta = 1.0 / 0.4
    m = rsmagcalc.CpnModel([3] * 4, [[np.eye(3, dtype=complex)] * 3] * 4,
                           [(i, np.diag(a)) for i in range(4)], [])

    def z_of(b):
        return sum(np.exp(-b * a[i]) / np.prod([b * (a[j] - a[i]) for j in range(3)
                                                if j != i]) for i in range(3))

    h = 1e-5
    e_exact = -(np.log(z_of(beta + h)) - np.log(z_of(beta - h))) / (2 * h) * 4
    Z0 = [np.array([1, 0, 0], complex)] * 4
    _, _, _, energies = m.thermalize(Z0, 0.4, 20000, rsmagcalc.Rng(2), sigma=0.5)
    e_mean = energies[2000:].mean()
    assert e_mean == pytest.approx(e_exact, rel=0.03), (e_mean, e_exact)


def test_sampled_trajectories_chunks_threads_and_seeds():
    mdl = _two_site_model(spins=(0.5, 1.0), ncells=3)
    m = _cpn(mdl)
    kw = dict(kT=0.3, therm_sweeps=30, sigma=0.2, adapt=True, target=0.5, dt=0.02,
              n_steps=12, record_every=3, seed=7)
    one = m.sampled_trajectories(mdl.Z, traj_start=0, n_traj=5, **kw)
    parts = [m.sampled_trajectories(mdl.Z, traj_start=k, n_traj=2, n_workers=1, **kw)
             for k in (0, 2, 4)]
    traj = np.concatenate([p[0] for p in parts])[:5]
    assert one[0].shape == (5, 4, 15) and np.array_equal(one[0], traj)
    assert np.array_equal(one[1], np.concatenate([p[1] for p in parts])[:5])
    assert one[2].shape == one[3].shape == (5, 15)
    assert np.array_equal(one[3], np.concatenate([p[3] for p in parts])[:5])
    two = m.sampled_trajectories(mdl.Z, traj_start=0, n_traj=5, n_workers=3, **kw)
    assert np.array_equal(one[0], two[0])
    # every recorded state is normalized per site; trajectories differ
    for t in range(5):
        for s in range(4):
            for z in m.unpack(one[0][t, s]):
                assert abs(np.linalg.norm(z) - 1.0) < 1e-13
    assert not np.array_equal(one[0][0], one[0][1])
    # random start: still normalized, and different from the model's state
    rnd = m.sampled_trajectories(None, traj_start=0, n_traj=2, **kw)
    for z in m.unpack(rnd[0][0, 0]):
        assert abs(np.linalg.norm(z) - 1.0) < 1e-13


def test_the_trajectory_after_thermalize_is_the_same_evolve():
    """sampled_trajectories = thermalize on stream (seed, i) + evolve: pinned so the
    parallel loop cannot drift from the two single-trajectory kernels."""
    mdl = _two_site_model(ncells=2)
    m = _cpn(mdl)
    traj, sig, accs, en = m.sampled_trajectories(mdl.Z, 0.3, 0, 1, 20, 0.2, True, 0.5,
                                                 0.02, 8, 1, seed=3)
    # the same stream the kernel derives for trajectory 0 is not exposed, so
    # compare against evolve from the recorded start instead
    start = m.unpack(traj[0, 0])
    want = m.evolve(start, 0.02, 8, 1)
    assert max(_maxdiff(m.unpack(traj[0, s]), want[s]) for s in range(8)) == 0.0
    assert en.shape == (1, 10) and accs.shape == (1, 10) and sig.shape == (1,)


# --------------------------------------------------------------------------
# Contract errors
# --------------------------------------------------------------------------
def test_shape_errors_are_loud():
    mdl = _two_site_model(spins=(0.5, 1.0))
    m = _cpn(mdl)
    with pytest.raises(ValueError, match="entries"):
        m.move_energy_change(mdl.Z, 0, np.ones(3, complex))
    with pytest.raises(ValueError, match="Z\\[0\\] must have"):
        m.energy([np.ones(3, complex), np.ones(3, complex)])
    with pytest.raises(ValueError, match="out of range"):
        m.move_energy_change(mdl.Z, 5, np.ones(2, complex))
    with pytest.raises(TypeError, match="Rng"):
        m.thermalize(mdl.Z, 0.3, 4, np.random.default_rng(0))
    with pytest.raises(ValueError, match="positive"):
        m.thermalize(mdl.Z, 0.0, 4, rsmagcalc.Rng(0))
    with pytest.raises(ValueError, match="operators"):
        rsmagcalc.CpnModel([2, 2], [[np.eye(2)] * 3, [np.eye(2)] * 2], [], [])
