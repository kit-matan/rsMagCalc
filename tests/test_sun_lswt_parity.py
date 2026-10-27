"""K3 pinned to pyMagCalc's NumPy SU(N) LSWT path (the oracle) and to exact identities.

`SunModel.hamiltonians` against `SUNModel._hamiltonian_numpy` elementwise at 1e-13;
`eigenvalues` against `numpy.linalg.eigvals` of the same matrix (the sorted real
parts, i.e. `dispersion`, and the imaginary parts `max_imaginary` reads);
`correlations` against `_one_magnon_correlations_numpy` -- energies and the 3x3
per-band tensors at 1e-10, summed over degenerate groups where a band is
degenerate (the per-band split inside a degenerate eigenspace is a basis choice,
not physics); `bogoliubov`'s T against the para-unitary identities and, column by
column up to a phase, against `_bogoliubov`; and the guard's verdicts against
`is_stable_at` on the fixtures of pyMagCalc's tests/test_kpm_stability.py (a
stationary maximum, the frustrated chain on both sides of its transition, the
non-collinear 120-degree supercell with its Goldstone zeros at Gamma and K), on
a mixed-spin cell, on the biquadratic 12-wide operator set and on the entangled
engine's dimer chain (constituent offsets in the moment operator).

Every model is built by pyMagCalc itself; nothing here is a typed-in number.
"""
import copy
import os

import numpy as np
import pytest

import rsmagcalc

lswt = pytest.importorskip("magcalc.sun.lswt",
                           reason="pyMagCalc not installed (the parity oracle)")
from magcalc.generic_model import GenericSpinModel  # noqa: E402
from magcalc.sun.lswt import SUNModel, apply_bond_disorder  # noqa: E402
from magcalc.sun.operators import coherent_from_direction, spin_matrices  # noqa: E402

pytestmark = pytest.mark.skipif(rsmagcalc.backend is None,
                                reason="rsmagcalc core not built")


# --------------------------------------------------------------------------
# Models (pyMagCalc's own test fixtures, rebuilt here)
# --------------------------------------------------------------------------
LAT = [[4.0, 0, 0], [0, 7.0, 0], [0, 0, 7.0]]
NN = ((["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0]))
B_MAT = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T


def _two_site(S=1.0, biquad=None, spins=None, ncells=1, sia=-0.45, J=0.8, kernels="numpy"):
    S_A, S_B = (spins if spins else (S, S))
    inter = {"heisenberg": [{"pair": p, "rij_offset": o, "value": J} for p, o in NN]}
    if sia:
        inter["single_ion_anisotropy"] = [{"value": sia, "axis": [0.3, 0, 1],
                                           "atoms": ["A", "B"]}]
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
    mdl = SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(cfg)), params=[],
                                      supercell=sup, kernels=kernels)
    mdl.minimize_energy(n_restarts=4, seed=0)
    return mdl


SQ_LAT = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 10.0]]
SQ_B = 2 * np.pi * np.linalg.inv(np.array(SQ_LAT, float)).T
SQ_OFFS = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])


def _square(J, S=0.5):
    """One site, z = 4, FERROMAGNETIC state: with J > 0 a stationary MAXIMUM."""
    cfg = {"crystal_structure": {"lattice_vectors": SQ_LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                               "spin_S": S}]},
           "interactions": {"heisenberg": [{"pair": ["A", "A"], "rij_offset": list(o),
                                            "value": J} for o in SQ_OFFS]},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": []}
    return SUNModel.from_generic_model(GenericSpinModel(cfg), [])


CHAIN_LAT = [[1.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]
CHAIN_B = 2 * np.pi * np.linalg.inv(np.array(CHAIN_LAT, float)).T


def _chain(J1=-1.0, J2=0.5, S=0.5):
    """A ferromagnetic chain frustrated by J2: unstable to a spiral for J2 > |J1|/4."""
    cfg = {"crystal_structure": {"lattice_vectors": CHAIN_LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                               "spin_S": S}]},
           "interactions": {"heisenberg":
                            [{"pair": ["A", "A"], "rij_offset": list(o), "value": J1}
                             for o in ([1, 0, 0], [-1, 0, 0])] +
                            [{"pair": ["A", "A"], "rij_offset": list(o), "value": J2}
                             for o in ([2, 0, 0], [-2, 0, 0])]},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": []}
    return SUNModel.from_generic_model(GenericSpinModel(cfg), [])


TRI = np.array([[3.0, 0, 0], [-1.5, 3 * np.sqrt(3) / 2, 0], [0, 0, 10.0]])
TRI_B = 2 * np.pi * np.linalg.inv(TRI).T
TRI_OFFS = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0], [-1, -1, 0])
K120 = np.array([1 / 3, 1 / 3, 0.0])


def _triangular_120(n):
    """The exact 120-degree state on an n x n cell: NON-COLLINEAR, Goldstone zeros
    at Gamma and at K."""
    cfg = {"crystal_structure": {"lattice_vectors": TRI.tolist(),
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": 0.5}]},
           "interactions": {"heisenberg": [{"pair": ["A", "A"], "rij_offset": list(o),
                                            "value": 1.0} for o in TRI_OFFS]},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": []}
    m = SUNModel.from_generic_model(GenericSpinModel(cfg), [],
                                    supercell=[[n, 0, 0], [0, n, 0], [0, 0, 1]])
    ang = 2 * np.pi * ((np.asarray(m.pos, float) @ np.linalg.inv(TRI)) @ K120)
    m.Z = [coherent_from_direction(s, d) for s, d in
           zip(m.S, np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1))]
    m._prepare()
    assert abs(m.energy_per_site() + 0.375) < 1e-10
    return m


def _dimer_chain(J=1.0, Jp=0.3, n_units=1):
    """The entangled engine's coupled-dimer chain: one effective SU(4) site per
    dimer, the singlet as reference, the constituent offsets in the moment
    operator (tests/test_entangled_units.py)."""
    S = spin_matrices(0.5)
    I2 = np.eye(2)
    S1 = [np.kron(S[a], I2) for a in range(3)]
    S2 = [np.kron(I2, S[a]) for a in range(3)]
    A = J * sum(S1[a] @ S2[a] for a in range(3))
    Z = np.linalg.eigh(A)[1][:, 0]
    C = np.zeros((6, 6))
    C[3:6, 0:3] = Jp * np.eye(3)      # S2 of unit i couples to S1 of unit i+1
    bonds, onsite, ops, mts = [], [], [], []
    for u in range(n_units):
        nxt = (u + 1) % n_units
        bonds.append((u, nxt, np.array([1.0, 0, 0]), C))
        bonds.append((nxt, u, np.array([-1.0, 0, 0]), C.T))
        onsite.append((u, A))
        ops.append(S1 + S2)
        mts.append([(np.array([-0.2, 0.1, 0.0]), (0, 1, 2)),
                    (np.array([0.2, -0.1, 0.0]), (3, 4, 5))])
    m = SUNModel(spins=[1.5] * n_units, coherent_states=[Z] * n_units, bonds=bonds,
                 onsite=onsite, operators=ops, moment_terms=mts)
    m.pos = np.array([[float(u), 0, 0] for u in range(n_units)])
    return m


def _pack(mdl):
    return rsmagcalc.SunModel(mdl.Ms, mdl.s0, mdl.t, mdl.tb, mdl.Q, mdl.QA, mdl.bonds,
                              mdl.moment_terms, n_cells=getattr(mdl, "n_cells", 1))


def _qs(B, n=7, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=(n, 3)) @ B


def _grouped(w, corr, tol=1e-8):
    """Sum the per-band tensors over groups of degenerate energies (descending w)."""
    out, cur, acc = [], None, None
    for k in range(len(w)):
        if cur is None or abs(w[k] - cur) > tol:
            if cur is not None:
                out.append((cur, acc))
            cur, acc = w[k], corr[:, :, k].copy()
        else:
            acc = acc + corr[:, :, k]
    out.append((cur, acc))
    return out


CASES = [
    (lambda: _two_site(), B_MAT, "plain SU(3)"),
    (lambda: _two_site(biquad=-0.3), B_MAT, "biquadratic, 12-wide operator set"),
    (lambda: _two_site(spins=(0.5, 1.0)), B_MAT, "mixed spin N = (2, 3)"),
    (lambda: _two_site(spins=(1.5, 0.5), ncells=3), B_MAT, "mixed spin, 3-cell supercell"),
    (lambda: _triangular_120(3), TRI_B, "non-collinear 120-degree supercell"),
    (lambda: _dimer_chain(n_units=2), 2 * np.pi * np.eye(3), "entangled dimer chain"),
]


# --------------------------------------------------------------------------
# H(q), eigenvalues, Colpa, correlations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("make,B,label", CASES, ids=[c[2] for c in CASES])
def test_hamiltonian_matches_numpy_elementwise(make, B, label):
    mdl = make()
    rs = _pack(mdl)
    qs = _qs(B)
    H = rs.hamiltonians(qs)
    assert H.shape == (len(qs), 2 * mdl.D, 2 * mdl.D)
    for k, q in enumerate(qs):
        ref = mdl._hamiltonian_numpy(q)
        assert np.abs(H[k] - ref).max() < 1e-13, f"{label}: q {k}"
    # a single q equals its place in the batch (the batch is the only path)
    assert np.array_equal(rs.hamiltonians(qs[2:3])[0], H[2])


@pytest.mark.parametrize("make,B,label", CASES, ids=[c[2] for c in CASES])
def test_eigenvalues_match_numpy_eigvals(make, B, label):
    mdl = make()
    rs = _pack(mdl)
    qs = _qs(B)
    ev = rs.eigenvalues(qs)
    for k, q in enumerate(qs):
        ref = np.linalg.eigvals(mdl._hamiltonian_numpy(q))
        got = np.sort(ev[k].real)
        assert np.abs(got - np.sort(ref.real)).max() < 1e-10, f"{label}: q {k}"
        assert abs(np.abs(ev[k].imag).max() - np.abs(ref.imag).max()) < 1e-10


@pytest.mark.parametrize("make,B,label", CASES, ids=[c[2] for c in CASES])
def test_correlations_match_numpy_structure_factor(make, B, label):
    mdl = make()
    rs = _pack(mdl)
    qs = _qs(B)
    om, corr = rs.correlations(qs)
    assert om.shape == (len(qs), mdl.D) and corr.shape == (len(qs), 3, 3, mdl.D)
    for k, q in enumerate(qs):
        w_ref, c_ref = mdl._one_magnon_correlations_numpy(q)
        assert np.abs(om[k] - w_ref).max() < 1e-10, f"{label}: energies at q {k}"
        got, ref = _grouped(om[k], corr[k]), _grouped(w_ref, c_ref)
        assert len(got) == len(ref)
        for (wg, cg), (wr, cr) in zip(got, ref):
            assert abs(wg - wr) < 1e-10
            assert np.abs(cg - cr).max() < 1e-10, f"{label}: tensors at q {k}, w = {wg}"
        # the unpolarized cross-section, per band, where the bands are distinct
        if len(got) == mdl.D:
            qh = q / np.linalg.norm(q)
            proj = np.eye(3) - np.outer(qh, qh)
            ig = np.einsum("ab,abm->m", proj, corr[k]).real
            ir = np.einsum("ab,abm->m", proj, c_ref).real
            assert np.abs(ig - ir).max() < 1e-10


@pytest.mark.parametrize("make,B,label", CASES[:4], ids=[c[2] for c in CASES[:4]])
def test_bogoliubov_T_is_para_unitary_and_matches_up_to_column_phases(make, B, label):
    mdl = make()
    rs = _pack(mdl)
    qs = _qs(B, n=3)
    om, T = rs.bogoliubov(qs)
    D = mdl.D
    g = np.diag([1.0] * D + [-1.0] * D)
    for k, q in enumerate(qs):
        w_ref, T_ref = mdl._bogoliubov(q)
        assert np.abs(om[k] - w_ref).max() < 1e-10
        Tk = T[k]
        # para-unitarity and the diagonalization T^H H2 T = diag(Ep)
        H2 = g @ mdl._hamiltonian_numpy(q)
        H2 = 0.5 * (H2 + H2.conj().T)
        assert np.abs(Tk.conj().T @ g @ Tk - g).max() < 1e-10
        Ep = np.concatenate([om[k], om[k]])
        assert np.abs(Tk.conj().T @ H2 @ Tk - np.diag(Ep)).max() < 1e-9
        # each column equals the NumPy column up to a phase, where non-degenerate
        for c in range(2 * D):
            e = Ep[c]
            if np.sum(np.abs(Ep - e) < 1e-8) > 1:
                continue
            a, b = Tk[:, c], T_ref[:, c]
            ph = np.vdot(b, a) / max(np.vdot(b, b).real, 1e-300)
            assert abs(abs(ph) - 1) < 1e-8
            assert np.abs(a - ph * b).max() < 1e-9


def test_batch_is_independent_of_the_thread_count():
    mdl = _two_site(spins=(0.5, 1.0), ncells=3)
    rs = _pack(mdl)
    qs = _qs(B_MAT, n=11)
    a = rs.correlations(qs, n_workers=1)
    b = rs.correlations(qs, n_workers=3)
    c = rs.correlations(qs)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[0], c[0])
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[1], c[1])
    assert np.array_equal(rs.hamiltonians(qs, n_workers=1), rs.hamiltonians(qs, n_workers=4))
    assert np.array_equal(rs.eigenvalues(qs, n_workers=1), rs.eigenvalues(qs, n_workers=4))


def test_a_non_minimum_raises_NotPositiveDefinite_where_numpy_raises_LinAlgError():
    mdl = _square(1.0)           # the ferromagnet on an antiferromagnet: a maximum
    rs = _pack(mdl)
    q = np.array([[0.3, 0.2, 0.0]]) @ SQ_B
    with pytest.raises(np.linalg.LinAlgError):
        mdl._bogoliubov(q[0])
    with pytest.raises(rsmagcalc.NotPositiveDefinite):
        rs.bogoliubov(q)
    with pytest.raises(rsmagcalc.NotPositiveDefinite):
        rs.correlations(q)
    assert issubclass(rsmagcalc.NotPositiveDefinite, ValueError)


# --------------------------------------------------------------------------
# The ground-state guard: verdict by verdict against is_stable_at
# --------------------------------------------------------------------------
def _shift(mdl, rel_tol=lswt.H2_REL_TOL):
    return float(rel_tol) * mdl._reference_h2_scale()


def _verdicts(mdl, qs, rel_tol=lswt.H2_REL_TOL):
    """The kernel's verdicts (automatic path) and `is_stable_at`'s. Both kernel
    paths -- the sparse multifrontal Cholesky and the dense zpotrf -- are forced
    as well and must agree with the automatic one on every q, so every guard case
    below pins both (tests/test_sparse_guard.py has the rest)."""
    rs = _pack(mdl)
    shift = _shift(mdl, rel_tol)
    got = rs.stable(qs, shift)
    for sparse in (True, False):
        assert np.array_equal(rs.stable(qs, shift, sparse=sparse), got), f"sparse={sparse}"
    ref = np.array([mdl.is_stable_at(q, rel_tol=rel_tol) for q in qs])
    return got, ref


@pytest.mark.parametrize("J", [1.0, 2.5])
def test_guard_refuses_a_stationary_maximum_everywhere_except_gamma(J):
    mdl = _square(J)
    qs = np.array([[0.3, 0.2, 0], [0.5, 0.5, 0], [0.1, 0, 0], [0.25, 0.1, 0],
                   [0, 0, 0]]) @ SQ_B
    got, ref = _verdicts(mdl, qs)
    assert np.array_equal(got, ref)
    assert list(got[:-1]) == [False] * 4 and got[-1]   # H2 = 0 exactly at Gamma


def test_guard_on_the_frustrated_chain_on_both_sides_of_its_transition():
    qs = np.array([[h, 0, 0] for h in np.linspace(0, 0.5, 11)]) @ CHAIN_B
    for J2, expect in ((0.1, True), (0.24, True), (0.26, False), (0.5, False)):
        mdl = _chain(J2=J2)
        got, ref = _verdicts(mdl, qs)
        assert np.array_equal(got, ref), f"J2 = {J2}"
        assert bool(got.all()) == expect, f"J2 = {J2}: {got}"


def test_guard_passes_the_non_collinear_state_at_gamma_and_K_and_the_tolerance_matters():
    mdl = _triangular_120(3)
    qs = np.vstack([np.array([[0, 0, 0], K120, [0.5, 0, 0], [0.2, 0.1, 0]]) @ TRI_B,
                    _qs(TRI_B, n=6)])
    got, ref = _verdicts(mdl, qs)
    assert np.array_equal(got, ref) and got.all()
    # the same q-INDEPENDENT shift as the Python guard: with no shift the Goldstone
    # zeros at Gamma and K decide on round-off, exactly as the docstring warns
    # (the DENSE path: a round-off verdict depends on the elimination order, so
    # the sparse one need not repeat it -- which is the shift's whole point)
    rs = _pack(mdl)
    unshifted = rs.stable(qs[:2], 0.0, sparse=False)
    assert list(unshifted) == [bool(lswt._is_positive_definite(mdl.h2_matrix(q)))
                               for q in qs[:2]]


def test_guard_verdicts_on_a_disordered_supercell_agree_with_the_exact_criterion():
    mdl = _triangular_120(3)
    apply_bond_disorder(mdl, 0.6, seed=4)
    qs = _qs(TRI_B, n=12, seed=9)
    got, ref = _verdicts(mdl, qs)
    assert np.array_equal(got, ref)
    exact = np.array([float(np.linalg.eigvalsh(mdl.h2_matrix(q)).min()) > -_shift(mdl)
                      for q in qs])
    assert np.array_equal(got, exact)
    assert not got.all(), "sigma = 0.6 should destabilise something"


@pytest.mark.parametrize("make,B,label", CASES, ids=[c[2] for c in CASES])
def test_guard_agrees_on_every_model(make, B, label):
    mdl = make()
    qs = _qs(B, n=9, seed=5)
    for rel_tol in (lswt.H2_REL_TOL, 1e-3):
        got, ref = _verdicts(mdl, qs, rel_tol)
        assert np.array_equal(got, ref), f"{label}, rel_tol {rel_tol}"


# --------------------------------------------------------------------------
# Packing: shape checks, and the entangled offsets really enter
# --------------------------------------------------------------------------
def test_packing_rejects_mismatched_tables():
    mdl = _two_site()
    with pytest.raises(ValueError):
        rsmagcalc.SunModel(mdl.Ms, mdl.s0, mdl.t[:1], mdl.tb, mdl.Q, mdl.QA, mdl.bonds,
                           mdl.moment_terms)
    with pytest.raises(ValueError):
        rsmagcalc.SunModel(mdl.Ms, mdl.s0, mdl.t, mdl.tb, mdl.Q, mdl.QA,
                           [(0, 1, np.zeros(3), np.zeros((2, 2)))], mdl.moment_terms)
    assert "SunModel(" in repr(_pack(mdl))


def test_the_moment_offsets_of_an_entangled_unit_enter_the_kernel():
    """The dimer selection rule: the staggered combination carries the triplon, so
    the intensity varies as (1 - cos q.d) with the CONSTITUENT offset d -- the one
    piece of the structure factor that is not in the full-position gauge."""
    m = _dimer_chain(n_units=1)
    d = m.moment_terms[0][1][0] - m.moment_terms[0][0][0]
    rs = _pack(m)
    qs = np.array([[0.0, 0, 0], np.pi * d / np.dot(d, d)])      # q.d = 0 and q.d = pi
    om, corr = rs.correlations(qs)
    tr = np.einsum("qaam->qm", corr).real                         # (nq, D) trace
    assert np.abs(tr[0]).max() < 1e-12                 # q = 0: total spin, no weight
    assert tr[1].sum() > 1e-3                          # q.d = pi: the staggered combination
    w_ref, c_ref = m._one_magnon_correlations_numpy(qs[1])
    assert np.abs(np.einsum("aam->m", c_ref).real - tr[1]).max() < 1e-12
