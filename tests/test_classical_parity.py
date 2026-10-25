"""K1 pinned to pyMagCalc's NumPy path (the oracle) and to exact identities.

Deterministic kernels at 1e-12: `evolve` (rk4 and implicit midpoint) against
`classical_dynamics.evolve` from the same m0, and the single-move dE of the
Metropolis sweep against E(after) - E(before). Random kernels cannot be matched
stream for stream (RUST_KERNELS_PLAN.md §2, RNG), so they are pinned to the
properties that do not depend on the stream: determinism per seed, independence
of the thread count and of chunking, conservation of |S_i|, the Ising invariant
of the flip proposal, and the exact classical-dimer thermodynamics that
pyMagCalc's own sampler tests use.
"""
import numpy as np
import pytest

import rsmagcalc

cd = pytest.importorskip("magcalc.classical_dynamics",
                         reason="pyMagCalc not installed (the parity oracle)")


def _random_problem(n, seed, scale=0.3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3 * n, 3 * n))
    H = scale * 0.5 * (A + A.T)
    b = 0.1 * rng.standard_normal(3 * n)
    return H, b


def _random_state(n, S, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, 3))
    m *= np.asarray(S, float).reshape(-1, 1) / np.linalg.norm(m, axis=1, keepdims=True)
    return m


def _L(x):
    return 1.0 / np.tanh(x) - 1.0 / x


# --------------------------------------------------------------------------
# Deterministic parity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("integrator", ["rk4", "midpoint"])
@pytest.mark.parametrize("n,seed", [(1, 0), (5, 1), (12, 2)])
def test_evolve_matches_numpy_to_1e12(integrator, n, seed):
    H, b = _random_problem(n, seed)
    S = 1.0
    m0 = _random_state(n, S, seed + 100)
    dt = 0.3 * cd.suggest_timestep(H, b, m0, S)
    want = cd.evolve(H, b, S, m0, dt, 50, record_every=1, integrator=integrator)
    got, bad = rsmagcalc.evolve(H, b, S, m0, dt, 50, 1, integrator)
    assert bad == 0
    assert got.shape == want.shape == (50, n, 3)
    assert np.abs(got - want).max() < 1e-12


def test_evolve_record_every_and_shapes():
    H, b = _random_problem(4, 7)
    m0 = _random_state(4, 1.0, 3)
    want = cd.evolve(H, b, 1.0, m0, 0.01, 23, record_every=5)
    got, _ = rsmagcalc.evolve(H, b, 1.0, m0, 0.01, 23, 5)
    assert got.shape == want.shape == (5, 4, 3)
    assert np.abs(got - want).max() < 1e-12


def test_evolve_renormalizes_to_per_site_spin_lengths():
    """Mixed spin: rk4 rescales each site to ITS length. The NumPy `evolve` takes a
    scalar S, so the per-site contract is checked against the identity itself."""
    n = 6
    H, b = _random_problem(n, 11)
    S = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 1.0])
    m0 = _random_state(n, S, 4)
    got, _ = rsmagcalc.evolve(H, b, S, m0, 0.005, 200, 1, "rk4")
    assert np.linalg.norm(got[-1], axis=1) == pytest.approx(S, abs=1e-12)
    # and with a uniform S it is the NumPy answer
    m1 = _random_state(n, 1.5, 4)
    want = cd.evolve(H, b, 1.5, m1, 0.005, 50)
    got, _ = rsmagcalc.evolve(H, b, np.full(n, 1.5), m1, 0.005, 50)
    assert np.abs(got - want).max() < 1e-12


def test_unknown_integrator_raises():
    H, b = _random_problem(2, 1)
    with pytest.raises(ValueError, match="rk4.*midpoint"):
        rsmagcalc.evolve(H, b, 1.0, _random_state(2, 1.0, 1), 0.01, 2, 1, "verlet")


@pytest.mark.parametrize("seed", range(4))
def test_single_move_energy_change_is_exact(seed):
    """dE from the gradient == E(after) - E(before), the identity `sweep` relies on
    (and the one every later sampler kernel will be pinned to, K2 included)."""
    n = 9
    H, b = _random_problem(n, seed, scale=1.0)
    S = np.array([1.0, 2.0, 0.5] * 3)
    m = _random_state(n, S, seed + 50)
    g = H @ m.ravel() + b
    rng = np.random.default_rng(seed + 200)
    for _ in range(20):
        a = int(rng.integers(n))
        v = rng.standard_normal(3)
        v *= S[a] / np.linalg.norm(v)
        after = m.copy()
        after[a] = v
        want = cd.energy(H, b, after) - cd.energy(H, b, m)
        got = rsmagcalc.move_energy_change(m, g, H, b, a, v)
        assert got == pytest.approx(want, abs=1e-12, rel=1e-12)


# --------------------------------------------------------------------------
# The sweep: invariants of the stochastic kernel
# --------------------------------------------------------------------------
def test_sweep_keeps_the_gradient_consistent_and_the_lengths_fixed():
    n = 10
    H, b = _random_problem(n, 5, scale=1.0)
    S = np.array([1.0, 1.5] * 5)
    m = _random_state(n, S, 8)
    g = H @ m.ravel() + b
    rng = rsmagcalc.Rng(1)
    accs = [rsmagcalc.sweep(m, g, H, b, 1.0, S, rng) for _ in range(30)]
    assert all(0.0 <= a <= 1.0 for a in accs) and np.mean(accs) > 0.05
    assert np.abs(g - (H @ m.ravel() + b)).max() < 1e-10
    assert np.linalg.norm(m, axis=1) == pytest.approx(S, abs=1e-12)


def test_flip_proposal_from_a_polarized_start_stays_ising():
    """Sunny's `propose_flip`: v = -m_a never leaves the +/-n axis, which is what
    makes a continuous-spin engine reproduce the 2-D Ising tutorial."""
    n = 16
    H, b = _random_problem(n, 9, scale=1.0)
    m = np.tile([0.0, 0.0, 1.0], (n, 1))
    g = H @ m.ravel() + b
    rng = rsmagcalc.Rng(2)
    for _ in range(50):
        rsmagcalc.sweep(m, g, H, b, 0.5, 1.0, rng, propose="flip")
    assert np.abs(m[:, :2]).max() == 0.0
    assert np.allclose(np.abs(m[:, 2]), 1.0)


def test_sweep_is_deterministic_per_seed_and_in_place():
    n = 6
    H, b = _random_problem(n, 3, scale=1.0)
    outs = []
    for _ in range(2):
        m = _random_state(n, 1.0, 1)
        g = H @ m.ravel() + b
        rng = rsmagcalc.Rng(42)
        for _ in range(10):
            rsmagcalc.sweep(m, g, H, b, 1.0, 1.0, rng)
        outs.append(m.copy())
    assert np.array_equal(outs[0], outs[1])
    m2 = _random_state(n, 1.0, 1)
    g2 = H @ m2.ravel() + b
    rng = rsmagcalc.Rng(43)
    for _ in range(10):
        rsmagcalc.sweep(m2, g2, H, b, 1.0, 1.0, rng)
    assert not np.array_equal(outs[0], m2)


def test_sweep_refuses_a_copy_it_could_not_write_back_to():
    n = 3
    H, b = _random_problem(n, 1)
    m = _random_state(n, 1.0, 1)
    g = H @ m.ravel() + b
    with pytest.raises(TypeError, match="C-contiguous"):
        rsmagcalc.sweep(np.asfortranarray(m), g, H, b, 1.0, 1.0, rsmagcalc.Rng(0))
    with pytest.raises(TypeError, match="Rng"):
        rsmagcalc.sweep(m, g, H, b, 1.0, 1.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        rsmagcalc.sweep(m, g, H, b[:-3], 1.0, 1.0, rsmagcalc.Rng(0))


# --------------------------------------------------------------------------
# Parallel tempering and Wang-Landau: the exact classical dimer
# --------------------------------------------------------------------------
def _dimers(n_dimers, J=1.0):
    """`n_dimers` isolated classical Heisenberg dimers: E = J sum S_1.S_2."""
    N = 2 * n_dimers
    H = np.zeros((3 * N, 3 * N))
    for k in range(n_dimers):
        a, c = 2 * k, 2 * k + 1
        for x in range(3):
            H[3 * a + x, 3 * c + x] = J
            H[3 * c + x, 3 * a + x] = J
    return H, np.zeros(3 * N), N


def test_parallel_tempering_reproduces_the_exact_dimer_energy():
    """<E>/spin = -J S^2 L(J S^2 / kT) / 2 -- the closed form pyMagCalc's
    test_thermal_mc.py pins its NumPy sampler to."""
    H, b, N = _dimers(18)
    temps = np.array([0.5, 0.9, 1.5, 3.0])
    out = rsmagcalc.parallel_tempering(H, b, 1.0, temps, 6000, 2000, swap_every=1,
                                       seed=3, n_workers=2)
    E = out["sumE"] / out["n_meas"] / N
    exact = -_L(1.0 / temps) / 2.0
    assert E == pytest.approx(exact, abs=0.01)
    assert 0.0 < out["accept_rate"] < 1.0


def test_parallel_tempering_is_deterministic_and_thread_count_independent():
    H, b, N = _dimers(6)
    temps = np.array([0.4, 0.8, 1.6])
    a = rsmagcalc.parallel_tempering(H, b, 1.0, temps, 400, 100, seed=7, n_workers=1)
    c = rsmagcalc.parallel_tempering(H, b, 1.0, temps, 400, 100, seed=7, n_workers=3)
    d = rsmagcalc.parallel_tempering(H, b, 1.0, temps, 400, 100, seed=8, n_workers=1)
    for k in ("sumE", "sumE2", "sumM", "sumM2"):
        assert np.array_equal(a[k], c[k])
    assert not np.array_equal(a["sumE"], d["sumE"])


def test_polarized_init_and_flip_give_the_ising_magnetization_sign():
    """A polarized start is kept by the flip proposal: on a ferromagnetic Ising
    chain well below the domain-wall energy 2J the replica's <M_z> stays near +N S
    (a 1-D chain has no true order, so kT must be small against 2J, not Tc)."""
    N = 24
    H = np.zeros((3 * N, 3 * N))
    for i in range(N):
        j = (i + 1) % N
        for x in range(3):
            H[3 * i + x, 3 * j + x] -= 1.0
            H[3 * j + x, 3 * i + x] -= 1.0
    out = rsmagcalc.parallel_tempering(H, np.zeros(3 * N), 1.0, np.array([0.25]),
                                       500, 100, swap_every=0, propose="flip",
                                       init=[0, 0, 1], seed=1)
    mz = out["sumM"][0, 2] / out["n_meas"] / N
    assert mz > 0.9


def test_wang_landau_density_of_states_of_a_dimer_is_flat():
    """cos(theta) of two random unit vectors is uniform, so g(E) is CONSTANT on
    [-J S^2, J S^2] -- a closed form for the whole density of states."""
    H, b, N = _dimers(1)
    m0 = _random_state(N, 1.0, 2)
    log_g, visits, f, n_ref, swept = rsmagcalc.wang_landau(
        H, b, 1.0, m0, -1.001, 1.001, 40, 1.0, 1e-5, 0.8, 400, 4_000_000, seed=1)
    occ = visits > 0
    lg = log_g[occ] - log_g[occ].min()
    assert occ.sum() > 30 and n_ref >= 8 and f <= 1e-5
    assert lg.max() - lg.min() < 0.35
    centres = np.linspace(-1.001, 1.001, 41)
    centres = 0.5 * (centres[1:] + centres[:-1])
    assert abs(np.polyfit(centres[occ], lg, 1)[0]) < 0.25


def test_wang_landau_refuses_a_start_outside_the_window():
    H, b, N = _dimers(1)
    with pytest.raises(ValueError, match="outside the window"):
        rsmagcalc.wang_landau(H, b, 1.0, _random_state(N, 1.0, 2), 50.0, 60.0,
                              10, 1.0, 0.5, 0.8, 10, 100)


# --------------------------------------------------------------------------
# Langevin
# --------------------------------------------------------------------------
def _free_spins(N, h):
    b = np.zeros(3 * N)
    b[2::3] = -h
    return np.zeros((3 * N, 3 * N)), b


def test_langevin_relaxes_toward_the_minimum_and_keeps_the_lengths():
    """The sign trap pyMagCalc's test_integrators.py records: g is the GRADIENT,
    so the damping must drive <S^z> UP for E = -h S^z."""
    N, h = 50, 0.8
    H, b = _free_spins(N, h)
    m = _random_state(N, 1.0, 5)
    start = m[:, 2].mean()
    out = rsmagcalc.langevin(H, b, 1.0, m, 0.02, 1e-12, 0.5, 3000, rsmagcalc.Rng(0))
    assert out[:, 2].mean() > start
    assert out[:, 2].mean() == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(out, axis=1) == pytest.approx(1.0, abs=1e-12)


def test_langevin_samples_the_langevin_function():
    N, h, kT = 200, 0.8, 0.7     # dense H: 200 free spins keep this under 3 s
    H, b = _free_spins(N, h)
    m = _random_state(N, 1.0, 1)
    rng = rsmagcalc.Rng(2)
    m = rsmagcalc.langevin(H, b, 1.0, m, 0.02, kT, 0.5, 3000, rng)
    acc = []
    for _ in range(300):
        m = rsmagcalc.langevin(H, b, 1.0, m, 0.02, kT, 0.5, 10, rng)
        acc.append(m[:, 2].mean())
    assert np.mean(acc) == pytest.approx(_L(h / kT), abs=0.015)


def test_langevin_is_deterministic_per_seed():
    H, b = _free_spins(10, 0.5)
    m = _random_state(10, 1.0, 1)
    a = rsmagcalc.langevin(H, b, 1.0, m, 0.02, 0.5, 0.5, 50, rsmagcalc.Rng(9))
    c = rsmagcalc.langevin(H, b, 1.0, m, 0.02, 0.5, 0.5, 50, rsmagcalc.Rng(9))
    assert np.array_equal(a, c)


# --------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------
def test_sampled_trajectories_are_chunk_and_thread_independent():
    n = 8
    H, b = _random_problem(n, 21, scale=1.0)
    full, bad = rsmagcalc.sampled_trajectories(H, b, 1.0, 0.5, 0, 4, 30, 0.01, 16,
                                               seed=5, n_workers=4)
    assert full.shape == (4, 16, n, 3) and bad == 0
    first, _ = rsmagcalc.sampled_trajectories(H, b, 1.0, 0.5, 0, 2, 30, 0.01, 16,
                                              seed=5, n_workers=1)
    second, _ = rsmagcalc.sampled_trajectories(H, b, 1.0, 0.5, 2, 2, 30, 0.01, 16,
                                               seed=5, n_workers=1)
    assert np.array_equal(full, np.concatenate([first, second]))
    assert np.linalg.norm(full, axis=-1) == pytest.approx(1.0, abs=1e-12)
    # the trajectories are genuinely different thermal samples
    assert not np.array_equal(full[0], full[1])


def test_each_trajectory_is_an_ll_evolution_of_its_own_start():
    """The recorded start of each trajectory, evolved by the parity-pinned `evolve`,
    reproduces the rest of it: thermalization and evolution are composed as the
    NumPy `sampled_correlations` composes them."""
    n = 6
    H, b = _random_problem(n, 22, scale=1.0)
    traj, _ = rsmagcalc.sampled_trajectories(H, b, 1.0, 0.8, 0, 2, 20, 0.01, 40,
                                             record_every=2, seed=3)
    for t in traj:
        want = cd.evolve(H, b, 1.0, t[0], 0.01, 40, record_every=2)
        assert np.abs(t - want).max() < 1e-12
