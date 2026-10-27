"""The guard's sparse path (`src/sparse_guard.rs`) decides exactly what the dense
one decides.

`SunModel.stable` answers "is H2(q) + shift I positive definite?" either by one
dense `zpotrf` of the 2D x 2D matrix or by a multifrontal sparse Cholesky on a
minimum-degree ordering; `stable(sparse=None)` picks per model. The matrix entries
are bit-identical between the two (the same contributions summed in the same
order), and a symmetric permutation does not change positive definiteness, so
only the rounding of the factorization differs. The oracles here are exact, not
the dense path alone:

* the margin: with lambda = min eig H2(q) from `eigvalsh`, the verdict at
  shift = -lambda +- 1e-9 x scale must be (lambda + shift > 0) on both paths --
  a sparse factorization that dropped or misplaced ANY entry or update would
  move lambda by far more than that;
* the Goldstone zeros of the clean 120-degree state at Gamma and K, where the
  sparse path is the automatic one: passed with the guard's shift, refused with
  a slightly negative one (so the zero is really there and really seen);
* Sunny's sigma = 1/3 disorder on the 120-degree state, which is NOT a minimum
  (pyMagCalc tests/test_s09_disorder_kpm.py): still refused, q by q identical;
* a dense operator (every site coupled to every other) keeps the dense path.

The guard cases of test_sun_lswt_parity.py force both paths as well.
"""
import os
import sys

import numpy as np
import pytest

import rsmagcalc

lswt = pytest.importorskip("magcalc.sun.lswt",
                           reason="pyMagCalc not installed (the parity oracle)")

sys.path.insert(0, os.path.dirname(__file__))
from test_sun_lswt_parity import (CASES, CHAIN_LAT, K120, TRI_B, _pack, _qs,  # noqa: E402
                                  _triangular_120, apply_bond_disorder, GenericSpinModel,
                                  SUNModel)

pytestmark = pytest.mark.skipif(rsmagcalc.backend is None,
                                reason="rsmagcalc core not built")


def _scale(mdl):
    return mdl._reference_h2_scale()


def _disordered(n, sigma, seed):
    m = _triangular_120(n)
    apply_bond_disorder(m, sigma, seed=seed)
    m.minimize_energy(n_restarts=1, max_iter=2000, tol=1e-15)
    return m


def _both(rs, qs, shift):
    sp = rs.stable(qs, shift, sparse=True)
    de = rs.stable(qs, shift, sparse=False)
    return sp, de


@pytest.mark.parametrize("n,sigma,seed", [(6, 0.6, 4), (12, 0.6, 1), (12, 0.0, 0)])
def test_the_verdict_is_exact_at_the_margin(n, sigma, seed):
    mdl = _disordered(n, sigma, seed) if sigma else _triangular_120(n)
    rs = _pack(mdl)
    delta = 1e-9 * _scale(mdl)
    qs = np.vstack([np.array([[0, 0, 0], K120]) @ TRI_B, _qs(TRI_B, n=6, seed=seed)])
    for q in qs:
        lam = float(np.linalg.eigvalsh(mdl.h2_matrix(q)).min())
        for shift, expect in ((-lam + delta, True), (-lam - delta, False)):
            sp, de = _both(rs, q[None], shift)
            assert sp[0] == de[0] == expect, (q, lam, shift, sp, de)


def test_goldstone_zeros_on_a_cell_where_sparse_is_automatic():
    mdl = _triangular_120(12)
    rs = _pack(mdl)
    info = rs.guard_info()
    assert info["auto"] and info["error"] is None
    qs = np.vstack([np.array([[0, 0, 0], K120, [0.5, 0, 0]]) @ TRI_B, _qs(TRI_B, n=6)])
    shift = lswt.H2_REL_TOL * _scale(mdl)
    got = rs.stable(qs, shift)
    ref = np.array([mdl.is_stable_at(q) for q in qs])
    assert got.all() and np.array_equal(got, ref)
    assert np.array_equal(rs.stable(qs, shift, sparse=False), got)
    # an EXACT zero at Gamma and K: any negative shift refuses them, on both paths
    neg = -1e-9 * _scale(mdl)
    sp, de = _both(rs, qs[:2], neg)
    assert not sp.any() and not de.any()


def test_sunny_s_sigma_one_third_is_still_refused():
    """The relaxed 120-degree state at sigma = 1/3 is not a minimum on most
    realizations; the path is the one pyMagCalc's slow S09 test scans."""
    path = np.concatenate([
        np.linspace(np.zeros(3), K120, 14, endpoint=False),
        np.linspace(K120, np.array([0.5, 0.0, 0.0]), 14, endpoint=False),
        np.linspace(np.array([0.5, 0.0, 0.0]), np.zeros(3), 14)]) @ TRI_B
    refused = 0
    for seed in range(3):
        mdl = _disordered(9, 1 / 3, seed)
        rs = _pack(mdl)
        assert rs.guard_info()["auto"]           # 2D = 162: the sparse path
        shift = lswt.H2_REL_TOL * _scale(mdl)
        sp, de = _both(rs, path, shift)
        ref = np.array([mdl.is_stable_at(q) for q in path])
        assert np.array_equal(sp, de) and np.array_equal(sp, ref), seed
        refused += not sp.all()
    assert refused >= 2, f"only {refused} of 3 realizations refused"


def _all_to_all_chain(n_cells=40):
    """A ferromagnetic chain with a coupling at EVERY distance up to half the
    cell: every site touches every other, so H2 has no sparsity to exploit."""
    offs = [[k, 0, 0] for k in range(1, n_cells // 2 + 1)]
    heis = [{"pair": ["A", "A"], "rij_offset": o, "value": -1.0 / o[0]} for o in offs]
    heis += [{"pair": ["A", "A"], "rij_offset": [-o[0], 0, 0], "value": -1.0 / o[0]}
             for o in offs]
    cfg = {"crystal_structure": {"lattice_vectors": CHAIN_LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                               "spin_S": 0.5}]},
           "interactions": {"heisenberg": heis},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": []}
    return SUNModel.from_generic_model(GenericSpinModel(cfg), [],
                                       supercell=[[n_cells, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_a_dense_operator_keeps_the_dense_path():
    mdl = _all_to_all_chain()
    rs = _pack(mdl)
    info = rs.guard_info()
    assert info["n"] >= 64 and info["error"] is None
    assert not info["auto"], info                # fill: no flop saving
    qs = _qs(2 * np.pi * np.linalg.inv(np.array(CHAIN_LAT, float)).T, n=5)
    shift = lswt.H2_REL_TOL * _scale(mdl)
    sp, de = _both(rs, qs, shift)
    assert np.array_equal(sp, de) and de.all()   # a ferromagnet: stable everywhere


def test_small_cells_stay_dense_and_every_case_analyses():
    for make, B, label in CASES:
        mdl = make()
        rs = _pack(mdl)
        info = rs.guard_info()
        n = info["n"]
        assert info["error"] is None, label
        assert n <= info["nnz_l"] <= n * (n + 1) // 2, (label, info)
        assert info["auto"] == (n >= 64 and 4 * info["flops_sparse"] <= info["flops_dense"])


def test_threads_do_not_change_a_verdict():
    mdl = _disordered(12, 0.6, 1)
    rs = _pack(mdl)
    qs = _qs(TRI_B, n=24, seed=2)
    shift = lswt.H2_REL_TOL * _scale(mdl)
    one = rs.stable(qs, shift, n_workers=1, sparse=True)
    assert np.array_equal(one, rs.stable(qs, shift, n_workers=4, sparse=True))
    assert not one.all(), "sigma = 0.6 should destabilise something"
