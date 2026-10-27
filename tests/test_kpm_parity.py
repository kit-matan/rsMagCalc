"""K4 pinned to pyMagCalc's NumPy KPM (the oracle): moments, bound, end to end.

The Chebyshev moments of `SunModel.kpm_moments` (the batch, H(q) built inside by
K3) and of the standalone `kpm_moments(HMat, v, ...)` against
`magcalc.sun.kpm._moments_numpy` at the SAME gamma, to 1e-12 relative to the
largest moment, on every K3 parity fixture -- the plain and biquadratic SU(3)
cells, mixed spin, a mixed-spin supercell, the NON-COLLINEAR 120-degree supercell
(where the recursion must run on conj(D-hat): the first past KPM bug) and the
entangled dimer chain -- with the operator forced dense, forced sparse, and left to
choose. The spectral bound against `spectral_bound` (same start vector, NumPy's).
End to end, `kpm_sqw` under "rust" against "numpy" in the antisymmetric `chiral`
channel on the ferromagnet that caught the second past bug (the sign of chiral is
the transpose of the moments), and at the Goldstone point where H(q) is
identically zero. The moment count from `tol` equals NumPy's integer.

Every model is built by pyMagCalc itself; nothing here is a typed-in number.
"""
import logging
import os
import sys

import numpy as np
import pytest

import rsmagcalc

kpm = pytest.importorskip("magcalc.sun.kpm",
                          reason="pyMagCalc not installed (the parity oracle)")

sys.path.insert(0, os.path.dirname(__file__))
from test_sun_lswt_parity import (CASES, _pack, _qs, _triangular_120, TRI_B,  # noqa: E402
                                  SUNModel, GenericSpinModel)

pytestmark = pytest.mark.skipif(rsmagcalc.backend is None,
                                reason="rsmagcalc core not built")

RTOL = 1e-12


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def _ferromagnet(S=1.0):
    """Square-lattice FM (tests/test_kpm.py at s = 1: a non-zero chiral signal for q
    out of the plane). At s = 1/2 (one boson) H(Gamma) is IDENTICALLY zero; at s = 1
    the quadrupolar band keeps it gapped."""
    lat = [[1., 0, 0], [0, 1, 0], [0, 0, 10]]
    cfg = {"crystal_structure": {"lattice_vectors": lat,
                                 "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                               "spin_S": S}]},
           "interactions": {"heisenberg": [{"pair": ["A", "A"], "rij_offset": o,
                                            "value": -1.0}
                                           for o in ([1, 0, 0], [-1, 0, 0],
                                                     [0, 1, 0], [0, -1, 0])]},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": []}
    return (SUNModel.from_generic_model(GenericSpinModel(cfg), []),
            2 * np.pi * np.linalg.inv(np.array(lat, float)).T)


ALL = CASES + [(lambda: _triangular_120(6), TRI_B, "non-collinear 120-degree, 36 sites"),
               (lambda: _ferromagnet()[0], _ferromagnet()[1], "ferromagnet")]
IDS = [c[2] for c in ALL]


def _numpy_moments(mdl, q, n_moments, gamma):
    H = mdl._hamiltonian_numpy(q)
    return kpm._moments_numpy(H, kpm._observable_vectors(mdl, q), n_moments, gamma)


# --------------------------------------------------------------------------
# Moments
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sparse", [None, True, False], ids=["auto", "csr", "dense"])
@pytest.mark.parametrize("make,B,label", ALL, ids=IDS)
def test_batch_moments_match_numpy_at_the_same_gamma(make, B, label, sparse):
    mdl = make()
    rs = _pack(mdl)
    qs = _qs(B, n=4)
    gam = np.array([kpm.spectral_bound(mdl._hamiltonian_numpy(q)) for q in qs])
    M = 150
    g_raw, nm, mom = rs.kpm_moments(qs, n_moments=M, gammas=gam, sparse=sparse)
    assert np.array_equal(g_raw, gam) and list(nm) == [M] * len(qs)
    assert mom.shape == (len(qs), M, 3, 3)
    for k, q in enumerate(qs):
        ref = _numpy_moments(mdl, q, M, gam[k])
        assert _rel(mom[k], ref) < RTOL, f"{label}: q {k}, {_rel(mom[k], ref):.1e}"


@pytest.mark.parametrize("sparse", [None, True, False], ids=["auto", "csr", "dense"])
@pytest.mark.parametrize("make,B,label", ALL, ids=IDS)
def test_standalone_moments_and_bound_match_numpy(make, B, label, sparse):
    mdl = make()
    for q in _qs(B, n=2, seed=11):
        H = mdl._hamiltonian_numpy(q)
        v = kpm._observable_vectors(mdl, q)
        g_np = kpm.spectral_bound(H)
        g_rs = rsmagcalc.spectral_bound(H, sparse=sparse)
        assert abs(g_rs - g_np) < 1e-12 * g_np, f"{label}: bound {g_rs} vs {g_np}"
        got = rsmagcalc.kpm_moments(H, v, 97, g_np, sparse=sparse)
        assert _rel(got, kpm._moments_numpy(H, v, 97, g_np)) < RTOL, label


@pytest.mark.parametrize("make,B,label", ALL, ids=IDS)
def test_batch_bound_and_moment_count_are_numpys(make, B, label):
    """With neither fixed, the kernel draws the bound from NumPy's start vector and
    applies `sun.kpm`'s moment rule: the same gamma, the same INTEGER count."""
    mdl = make()
    rs = _pack(mdl)
    qs = _qs(B, n=5, seed=7)
    for fwhm, tol in ((0.2, 0.02), (0.05, 1e-3), (1.3, 0.3)):
        g_raw, nm, mom = rs.kpm_moments(qs, fwhm=fwhm, tol=tol)
        assert mom.shape == (len(qs), int(nm.max()), 3, 3)
        for k, q in enumerate(qs):
            g_np = kpm.spectral_bound(mdl._hamiltonian_numpy(q))
            assert abs(g_raw[k] - g_np) < 1e-12 * g_np, label
            assert nm[k] == kpm._n_moments_for(g_raw[k], fwhm, tol), label
            ref = _numpy_moments(mdl, q, int(nm[k]), g_raw[k])
            assert _rel(mom[k, :nm[k]], ref) < RTOL, label
            assert not np.any(mom[k, nm[k]:]), "padding past a q's own count"


def test_a_long_recursion_on_the_81_site_supercell():
    """S09's own geometry at 9 x 9 (2D = 162, CSR on its own), 600 moments: the
    accumulated round-off of the two summation orders stays at 1e-12."""
    mdl = _triangular_120(9)
    rs = _pack(mdl)
    qs = np.array([[0.2, 0.1, 0], [0.37, -0.11, 0]]) @ TRI_B
    gam = np.array([kpm.spectral_bound(mdl._hamiltonian_numpy(q)) for q in qs])
    _, _, mom = rs.kpm_moments(qs, n_moments=600, gammas=gam)
    for k, q in enumerate(qs):
        assert _rel(mom[k], _numpy_moments(mdl, q, 600, gam[k])) < RTOL


def test_a_single_q_equals_its_place_in_the_batch():
    mdl = _triangular_120(3)
    rs = _pack(mdl)
    qs = _qs(TRI_B, n=5)
    g, nm, mom = rs.kpm_moments(qs, fwhm=0.2, tol=0.02, n_workers=3)
    g1, nm1, mom1 = rs.kpm_moments(qs[3:4], fwhm=0.2, tol=0.02)
    assert g1[0] == g[3] and nm1[0] == nm[3]
    assert np.array_equal(mom1[0], mom[3, :nm[3]])


# --------------------------------------------------------------------------
# End to end through pyMagCalc's kpm_sqw
# --------------------------------------------------------------------------
def test_chiral_channel_rust_equals_numpy():
    """The antisymmetric channel pins the (a, b) order of the moments; a transposed
    tensor flips its sign (relative error exactly 2)."""
    sm, B = _ferromagnet()
    egrid = np.linspace(0, 10, 300)
    for q_rlu in ([0.2, 0.1, 0.4], [0.3, 0.0, 0.6], [0.25, 0.35, 0.5]):
        q = np.array(q_rlu, float) @ B
        ref = kpm.kpm_sqw(sm, q, egrid, 0.3, n_moments=400, cross_section="chiral",
                          kernels="numpy").intensities
        assert np.abs(ref).max() > 0.1
        for hm in (None, sm.hamiltonian(q)):
            got = kpm.kpm_sqw(sm, q, egrid, 0.3, n_moments=400, cross_section="chiral",
                              kernels="rust", hmat=hm).intensities
            assert _rel(got, ref) < 1e-11


def test_kpm_sqw_batch_rust_equals_numpy_on_the_non_collinear_supercell():
    sm = _triangular_120(6)
    qs = np.array([[0.2, 0.1, 0], [0.37, -0.11, 0], [0.13, 0.29, 0]]) @ TRI_B
    egrid = np.arange(0.0, 3.0001, 0.02)
    ref = kpm.kpm_sqw_batch(sm, qs, egrid, 0.15, kernels="numpy")
    got = kpm.kpm_sqw_batch(sm, qs, egrid, 0.15, kernels="rust")
    for r, g in zip(ref, got):
        assert g.n_moments == r.n_moments and abs(g.gamma - r.gamma) < 1e-12 * r.gamma
        assert _rel(g.intensities, r.intensities) < 1e-11


def test_goldstone_point_behaves_as_numpy_warning_included(caplog):
    """H(q) identically zero (the ferromagnet at Gamma): bound 0, gamma -> 1, the
    warning, and the same (zero) column under both kernels."""
    sm, _ = _ferromagnet(S=0.5)
    rs = _pack(sm)
    assert not np.any(sm._hamiltonian_numpy(np.zeros(3)))
    g_raw, nm, mom = rs.kpm_moments(np.zeros((1, 3)), fwhm=0.3, tol=0.02)
    assert g_raw[0] == 0.0 and nm[0] == kpm._n_moments_for(1.0, 0.3, 0.02)
    assert _rel(mom[0], _numpy_moments(sm, np.zeros(3), int(nm[0]), 1.0)) < RTOL
    egrid = np.linspace(0, 10, 50)
    out = {}
    for kern in ("numpy", "rust"):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="magcalc.sun.kpm"):
            r = kpm.kpm_sqw(sm, np.zeros(3), egrid, 0.3, kernels=kern)
        assert "identically zero" in caplog.text, kern
        assert r.gamma == 1.0 and np.all(np.isfinite(r.intensities))
        out[kern] = r.intensities
    assert np.array_equal(out["numpy"], out["rust"])


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------
def test_empty_batch():
    rs = _pack(_triangular_120(3))
    g, nm, mom = rs.kpm_moments(np.zeros((0, 3)), n_moments=20, gammas=np.zeros(0))
    assert g.shape == (0,) and nm.shape == (0,) and mom.shape == (0, 20, 3, 3)
    g, nm, mom = rs.kpm_moments(np.zeros((0, 3)), fwhm=0.2, tol=0.02)
    assert mom.shape[0] == 0


def test_bad_arguments_are_refused():
    mdl = _triangular_120(3)
    rs = _pack(mdl)
    q = np.zeros((1, 3))
    H = mdl._hamiltonian_numpy(np.array([0.3, 0.1, 0.0]))
    v = kpm._observable_vectors(mdl, np.array([0.3, 0.1, 0.0]))
    with pytest.raises(ValueError):
        rs.kpm_moments(q)                                     # no count, no rule
    with pytest.raises(ValueError):
        rs.kpm_moments(q, n_moments=0, gammas=[1.0])
    with pytest.raises(ValueError):
        rs.kpm_moments(np.zeros((2, 3)), n_moments=5, gammas=[1.0])
    with pytest.raises(ValueError):
        rsmagcalc.kpm_moments(H, v, 10, 0.0)                  # gamma must be > 0
    with pytest.raises(ValueError):
        rsmagcalc.kpm_moments(H, v[:, :-1], 10, 1.0)          # v does not match
    with pytest.raises(ValueError):
        rsmagcalc.kpm_moments(H[:, :-1], v, 10, 1.0)          # not square
