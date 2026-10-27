"""K5 pinned to pyMagCalc's NumPy SCGA (the oracle): J(q), eigenvalues, residuals, lambda, C(q).

`ScgaModel` packed from `scga._exchange_cache` against the NumPy code it replaces:

  * J(q) (`fourier_exchange_matrix`) and its mesh eigenvalues (`eigvalsh`) -- these
    are BIT-identical (same bond order, same symmetrization, same `zheevd` call with
    the queried workspace), which is why lambda is too; asserted at 1e-12 relative
    so that a harmless reordering would not fail, and exact equality reported;
  * the per-site sum-rule residuals of `sublattice_sum_rule_residuals` at 1e-12;
  * lambda itself from `solve_lambda(kernels="rust")` against "numpy" at 1e-12
    relative, and the residual it reports;
  * the correlation tensors C(q) against `scga.correlation_tensor` (the NumPy
    loop) at 1e-12 relative to the largest entry, with and without `apply_g`;
  * S(q) end to end through `scga_intensities` under both kernels.

Fixtures: the square and kagome antiferromagnets `tests/test_scga.py` compares to
Sunny, the Heisenberg chain of the closed form, a kagome with an ANISOTROPIC
off-diagonal g-tensor and Cu2+ form factors (so the f_i, the phases and g_i^T all
enter), a square lattice with DM (complex, non-real-symmetric J(q)), and a 16-site
pyrochlore cell. Off the mesh, below lambda_min, lambda + J(q) is indefinite and
the kernel switches from Cholesky to NumPy's LU; that branch is pinned too.

Every model is built by pyMagCalc itself; nothing here is a typed-in number.
"""
import logging

import numpy as np
import pytest

import rsmagcalc

scga = pytest.importorskip("magcalc.scga",
                           reason="pyMagCalc not installed (the parity oracle)")
from magcalc.generic_model import GenericSpinModel  # noqa: E402

pytestmark = pytest.mark.skipif(rsmagcalc.backend is None,
                                reason="rsmagcalc core not built")

RTOL = 1e-12
G_ANISO = [[1.8, 0, 0.25], [0, 4.32, 0], [0.25, 0, 1.8]]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _fm():
    return {"type": "pattern", "pattern_type": "ferromagnetic", "direction": [0, 0, 1]}


def _chain():
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 10, 0], [0, 0, 10]],
                                 "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                               "spin_S": 0.5}]},
           "interactions": {"heisenberg": [
               {"pair": ["A", "A"], "rij_offset": o, "value": 1.0}
               for o in ([1, 0, 0], [-1, 0, 0])]},
           "parameters": {}, "parameter_order": [], "magnetic_structure": _fm()}
    return GenericSpinModel(cfg), [], 0.8, 200


def _square(dm=0.0):
    inter = [{"type": "heisenberg", "pair": ["A", "A"], "rij_offset": o, "value": 1.0}
             for o in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])]
    if dm:
        inter += [{"type": "dm", "pair": ["A", "A"], "rij_offset": [s, 0, 0],
                   "value": [0.0, s * dm, 0.0]} for s in (1, -1)]
        inter += [{"type": "dm", "pair": ["A", "A"], "rij_offset": [0, s, 0],
                   "value": [-s * dm, 0.0, 0.0]} for s in (1, -1)]
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 1, 0], [0, 0, 10]],
                                 "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                               "spin_S": 1.0}]},
           "interactions": inter if dm else {"heisenberg": inter},
           "parameters": {}, "parameter_order": [], "magnetic_structure": _fm()}
    return GenericSpinModel(cfg), [], 2.0, 20


def _kagome(ion=None, g=None):
    t = np.radians(120.0)
    lv = [[1., 0, 0], [np.cos(t), np.sin(t), 0], [0, 0, 10]]
    atoms = [{"label": lab, "pos": p, "spin_S": 1.0}
             for lab, p in (("A", [0.5, 0, 0]), ("B", [0, 0.5, 0]), ("C", [0.5, 0.5, 0]))]
    for a in atoms:
        if ion:
            a["ion"] = ion
        if g is not None:
            a["g"] = g
    cfg = {"crystal_structure": {"lattice_vectors": lv, "atoms_uc": atoms},
           "interactions": {"symmetry_rules": [{"type": "heisenberg", "distance": 0.5,
                                                "value": "J1"}]},
           "parameters": {"J1": 1.0}, "parameter_order": ["J1"],
           "magnetic_structure": _fm()}
    return GenericSpinModel(cfg), [1.0], 1.5, 12


def _pyrochlore():
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    tet = np.array([[0, 0, 0], [0, .25, .25], [.25, 0, .25], [.25, .25, 0]])
    pos = [(f + t) % 1.0 for f in fcc for t in tet]
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 1., 0], [0, 0, 1.]],
                                 "atoms_uc": [{"label": f"A{i}", "pos": [float(x) for x in p],
                                               "spin_S": 1.0} for i, p in enumerate(pos)]},
           "interactions": {"symmetry_rules": [{"type": "heisenberg",
                                                "distance": float(np.sqrt(2) / 4),
                                                "value": "J1"}]},
           "parameters": {"J1": 1.0}, "parameter_order": ["J1"],
           "magnetic_structure": _fm()}
    return GenericSpinModel(cfg), [1.0], 2.0, 5


CASES = {
    "chain": _chain,
    "square": _square,
    "square_dm": lambda: _square(dm=0.3),
    "kagome": _kagome,
    "kagome_g_ff": lambda: _kagome(ion="Cu2+", g=G_ANISO),
    "pyrochlore16": _pyrochlore,
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    m, p, kT, nq = CASES[request.param]()
    lam, lam_min, res, cache, mesh = scga.solve_lambda(m, p, kT, nq, kernels="numpy")
    rm = rsmagcalc.ScgaModel.from_cache(cache)
    rng = np.random.default_rng(7)
    lat = np.asarray(m.unit_cell(), float)
    B = 2 * np.pi * np.linalg.inv(lat).T
    qs = np.vstack([np.zeros(3), rng.uniform(-1.5, 1.5, (24, 3)) @ B])
    return dict(name=request.param, m=m, p=p, kT=kT, nq=nq, lam=lam, lam_min=lam_min,
                res=res, cache=cache, mesh=mesh, rm=rm, qs=qs)


def test_fourier_matrix(case):
    A_np = np.array([scga.fourier_exchange_matrix(case["m"], case["p"], q, case["cache"])
                     for q in case["qs"]])
    A_rs = case["rm"].fourier(case["qs"])
    assert A_rs.shape == A_np.shape
    assert _rel(A_rs, A_np) < RTOL
    assert np.array_equal(A_rs, A_np), "J(q) is no longer bit-identical to NumPy's"


def test_mesh_eigenvalues(case):
    ev_np = np.array([np.linalg.eigvalsh(scga.fourier_exchange_matrix(
        case["m"], case["p"], q, case["cache"])) for q in case["mesh"]])
    ev_rs = case["rm"].eigenvalues(case["mesh"])
    assert _rel(ev_rs, ev_np) < RTOL
    assert np.array_equal(ev_rs, ev_np), "mesh eigenvalues no longer bit-identical"


def test_lambda_and_residual(case):
    lam, lam_min, res, _c, _mesh = scga.solve_lambda(case["m"], case["p"], case["kT"],
                                                     case["nq"], kernels="rust")
    assert abs(lam - case["lam"]) <= RTOL * abs(case["lam"])
    assert abs(lam_min - case["lam_min"]) <= RTOL * max(abs(case["lam_min"]), 1.0)
    assert abs(res - case["res"]) < 1e-12


@pytest.mark.parametrize("dlam", [0.0, 0.37])
def test_sum_rule_residuals(case, dlam):
    lam = case["lam"] + dlam
    r_np = scga.sublattice_sum_rule_residuals(case["m"], case["p"], case["kT"], lam,
                                              case["cache"], case["mesh"], kernels="numpy")
    r_rs = case["rm"].sum_rule_residuals(case["mesh"], lam, case["kT"])
    assert r_rs.shape == (case["cache"]["N"],)
    assert np.max(np.abs(r_rs - r_np)) < RTOL
    # and through pyMagCalc's own entry point
    r_py = scga.sublattice_sum_rule_residuals(case["m"], case["p"], case["kT"], lam,
                                              case["cache"], case["mesh"], kernels="rust")
    assert np.max(np.abs(r_py - r_np)) < RTOL


def _ff(case):
    ions = case["cache"]["ions"]
    qmag = np.array([float(np.linalg.norm(q)) for q in case["qs"]])
    ff = np.ones((len(qmag), case["cache"]["N"]))
    for i, ion in enumerate(ions):
        if ion:
            ff[:, i] = scga.get_form_factor(ion, qmag)
    return ff


@pytest.mark.parametrize("apply_g", [True, False])
def test_correlation_tensors(case, apply_g):
    C_np = np.array([scga.correlation_tensor(case["m"], case["p"], q, case["kT"],
                                             case["lam"], case["cache"], apply_g)
                     for q in case["qs"]])
    C_rs = case["rm"].correlations(case["qs"], case["lam"], case["kT"], _ff(case), apply_g)
    assert C_rs.shape == (len(case["qs"]), 3, 3)
    assert _rel(C_rs, C_np) < RTOL


def test_indefinite_branch_matches_numpy_lu(case):
    """lambda below lambda_min: lambda + J(q) indefinite at some q, the Cholesky fails
    and the kernel solves by LU as NumPy's `inv` does."""
    lam = case["lam_min"] - 0.3 * max(1.0, abs(case["lam_min"]))
    A = [scga.fourier_exchange_matrix(case["m"], case["p"], q, case["cache"])
         for q in case["qs"]]
    n3 = 3 * case["cache"]["N"]
    indef = [np.linalg.eigvalsh(lam * np.eye(n3) + a).min() < 0 for a in A]
    assert any(indef)
    C_np = np.array([scga.correlation_tensor(case["m"], case["p"], q, case["kT"], lam,
                                             case["cache"], True) for q in case["qs"]])
    C_rs = case["rm"].correlations(case["qs"], lam, case["kT"], _ff(case), True)
    for k in range(len(C_np)):
        assert _rel(C_rs[k], C_np[k]) < 1e-10, (k, indef[k])
    tr_np = np.array([[np.real(np.trace(np.linalg.inv(lam * np.eye(n3) + a)
                                        [3 * i:3 * i + 3, 3 * i:3 * i + 3]))
                       for i in range(case["cache"]["N"])] for a in A])
    tr_rs = case["rm"].site_traces(case["qs"], lam)
    assert _rel(tr_rs, tr_np) < 1e-10


@pytest.mark.parametrize("cs", ["perp", "trace", "zz"])
def test_intensities_end_to_end(case, cs):
    kw = dict(cross_section=cs, lam=case["lam"], mesh=case["mesh"], cache=case["cache"])
    I_np = scga.scga_intensities(case["m"], case["p"], case["qs"], case["kT"],
                                 kernels="numpy", **kw)
    I_rs = scga.scga_intensities(case["m"], case["p"], case["qs"], case["kT"],
                                 kernels="rust", **kw)
    assert _rel(I_rs, I_np) < RTOL


def test_threads_do_not_change_the_numbers(case):
    rm, qs, lam, kT = case["rm"], case["qs"], case["lam"], case["kT"]
    a = rm.correlations(qs, lam, kT, _ff(case), True, n_workers=1)
    b = rm.correlations(qs, lam, kT, _ff(case), True, n_workers=4)
    assert np.array_equal(a, b)
    assert np.array_equal(rm.sum_rule_residuals(case["mesh"], lam, kT, n_workers=1),
                          rm.sum_rule_residuals(case["mesh"], lam, kT, n_workers=3))


def test_empty_batches():
    m, p, kT, nq = _kagome()
    cache = scga._exchange_cache(m, p)
    rm = rsmagcalc.ScgaModel.from_cache(cache)
    e = np.zeros((0, 3))
    assert rm.correlations(e, 3.0, kT).shape == (0, 3, 3)
    assert rm.eigenvalues(e).shape == (0, 9)
    assert rm.fourier(e).shape == (0, 9, 9)
    assert rm.site_traces(e, 3.0).shape == (0, 3)


def test_shape_errors_are_loud():
    m, p, kT, nq = _kagome()
    cache = scga._exchange_cache(m, p)
    rm = rsmagcalc.ScgaModel.from_cache(cache)
    with pytest.raises(ValueError, match="ff must be"):
        rm._m.correlations(np.zeros((2, 3)), 3.0, kT, np.ones((2, 2)), True, None)
    with pytest.raises(ValueError, match="qs must be"):
        rm._m.eigenvalues(np.zeros((2, 2)), None)
    bad = list(cache["bonds"])
    bad[0] = (7, bad[0][1], bad[0][2], bad[0][3])
    with pytest.raises(ValueError, match="out of range"):
        rsmagcalc.ScgaModel(bad, cache["pos"], cache["g"], cache["spins"])


def test_the_g_and_form_factor_fixture_exercises_both(caplog):
    """Guards the guard: the anisotropic-g / Cu2+ case must actually differ from g = 2,
    f = 1, or its parity would prove nothing about those two inputs."""
    m, p, kT, nq = _kagome(ion="Cu2+", g=G_ANISO)
    cache = scga._exchange_cache(m, p)
    assert all(ion for ion in cache["ions"])
    assert not np.allclose(cache["g"][0], 2 * np.eye(3))
    q = np.array([[2.0, 1.0, 0.3]])
    rm = rsmagcalc.ScgaModel.from_cache(cache)
    with caplog.at_level(logging.WARNING):
        f = scga.get_form_factor("Cu2+", float(np.linalg.norm(q)))
    assert 0.0 < f < 0.99
    a = rm.correlations(q, 4.0, kT, np.full((1, 3), f), True)
    b = rm.correlations(q, 4.0, kT, None, False)
    assert _rel(a, b) > 1e-2
