"""K7 pinned to pyMagCalc's NumPy dimer series (the oracle): the PT engine and the cluster PT.

  * `block_effective_series` against `magcalc.sun.dimer_series.block_effective_series`
    at 1e-12 relative: a quasi-degenerate model space, a multi-order V series
    (eigenvalue_series's use), p = 1, scipy-sparse input, and a model space that
    is not the lowest states;
  * `cluster_pt_batch` against `_cluster_pt` at 1e-12 relative, with and without
    the total-Sz sectors: the alternating chain's clusters, Rb2Cu3SnF12's (6
    dimers, out-of-plane DM, when pyMagCalc's examples are present) and random
    nloc = 3 clusters with complex Hermitian pair operators on arbitrary node
    pairs (pa > pb included), so no symmetry of the physical models hides a bug;
  * at order 6 the kernel's rounding is NumPy's: on Rb2Cu3SnF12's six-link
    clusters it differs from NumPy by at most 3x what NumPy differs from itself
    under a relabelling of the cluster's nodes (the same physics, summed in another
    order) -- measured 1.3x. A regression bound, not a discriminator: the earlier
    unsummed-in-blocks Gram loop reached 2-3x on these clusters (5x on others);
  * the thread count changes nothing: n_workers = 1 and the full pool agree bit
    for bit (each cluster runs serially, in the same order);
  * the refusals: a model space degenerate with its complement raises what the
    NumPy engine raises, and malformed shapes or links raise ValueError instead of
    reading out of bounds.

Nothing here is a typed-in number: every reference is pyMagCalc's own code.
"""
import importlib.util
import os

import numpy as np
import pytest

import rsmagcalc

ds = pytest.importorskip("magcalc.sun.dimer_series")

TOL = 1e-12


def _close(a, b):
    scale = max(float(np.abs(a).max()), 1.0)
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()) <= TOL * scale


def _herm(rng, n):
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return 0.5 * (A + A.conj().T)


# --------------------------------------------------------------------------
# block_effective_series
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["quasi_degenerate", "series_of_V", "single",
                                  "sparse", "excited_model"])
def test_block_effective_series_matches_numpy(case):
    rng = np.random.default_rng(11)
    N, order = 16, 6
    E0 = np.sort(np.concatenate([[1.0, 1.02, 1.05], 3 + 2 * rng.random(N - 3)]))
    P = [0, 1, 2]
    V = [_herm(rng, N)]
    if case == "series_of_V":
        V = [_herm(rng, N) for _ in range(4)]
    elif case == "single":
        P = [0]
    elif case == "sparse":
        sp = pytest.importorskip("scipy.sparse")
        A = V[0] * (rng.random((N, N)) < 0.3)
        V = [sp.csr_matrix(0.5 * (A + A.conj().T))]
    elif case == "excited_model":
        E0 = np.concatenate([[0.0, 0.4], [2.0, 2.0, 2.1], 4 + rng.random(N - 5)])
        P = [2, 3, 4]
    ref = ds.block_effective_series(E0, V, P, order)
    got = rsmagcalc.block_effective_series(E0, V, P, order)
    assert len(got) == order + 1
    for k in range(order + 1):
        assert got[k].shape == (len(P), len(P))
        assert _close(got[k], ref[k]), k


def test_degenerate_model_space_raises_like_numpy():
    E0 = np.array([0.0, 1.0, 1.0, 2.0])
    V = [np.ones((4, 4))]
    with pytest.raises(ValueError, match="not separated"):
        ds.block_effective_series(E0, V, [1], 3)
    with pytest.raises(ValueError, match="not separated"):
        rsmagcalc.block_effective_series(E0, V, [1], 3)


# --------------------------------------------------------------------------
# cluster_pt_batch
# --------------------------------------------------------------------------
def _chain_tasks(max_size=4):
    I3 = np.eye(3)
    m = ds.DimerSeriesModel.from_spin_arrays(
        np.diag([1.0, 10.0, 10.0]), [0.5, 0.5], [[0, 0, 0], [0.35, 0, 0]],
        [(0, 1, (0, 0, 0), I3), (1, 0, (1, 0, 0), 0.6 * I3)], units=[[0, 1]])
    return _model_tasks(m, max_size, 10)


def _model_tasks(m, max_size, per_size):
    levels = m._enumerate(max_size)
    out = []
    for size in range(1, max_size + 1):
        for form in sorted(levels[size])[:per_size]:
            _k, node_order = m._struct(form)
            lv = [m.levels[d] for (d, _c) in node_order]
            sz = ([m.sz_levels[d] for (d, _c) in node_order]
                  if m.sz_levels is not None else None)
            out.append((lv, m._local_links(form, node_order), sz))
    return out


def _rb_model():
    import magcalc

    path = os.path.join(os.path.dirname(magcalc.__file__), "..", "examples",
                        "entangled", "Rb2Cu3SnF12", "series_dispersion.py")
    if not os.path.exists(path):
        pytest.skip("pyMagCalc's examples are not present (a wheel install)")
    spec = importlib.util.spec_from_file_location("rb_series", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model()[0]


def _rb_tasks():
    return _model_tasks(_rb_model(), 3, 8)


def _random_tasks(n=6):
    """nloc = 3 clusters of 2-4 nodes, random levels and complex Hermitian pair
    operators on arbitrary node pairs; no sectors (the operators conserve nothing)."""
    rng = np.random.default_rng(5)
    out = []
    for t in range(n):
        m = 2 + t % 3
        lv = [np.sort(np.concatenate([[0.0], 1.0 + rng.random(2)])) for _ in range(m)]
        pairs = [(i, i + 1) for i in range(m - 1)] + [(m - 1, 0)] * (m > 2)
        links = [(pa, pb, 0.3 * _herm(rng, 9)) for (pa, pb) in pairs]
        out.append((lv, links, None))
    return out


# random pair operators conserve no Sz, so they run without sectors only
@pytest.mark.parametrize("model, sectors", [
    ("chain", True), ("chain", False), ("Rb2Cu3SnF12", True), ("Rb2Cu3SnF12", False),
    ("random", False)])
def test_cluster_pt_matches_numpy(model, sectors):
    tasks = {"chain": _chain_tasks, "Rb2Cu3SnF12": _rb_tasks,
             "random": _random_tasks}[model]()
    if not sectors:
        tasks = [(lv, links, None) for (lv, links, _sz) in tasks]
    order = 4
    got = rsmagcalc.cluster_pt_batch(tasks, order)
    assert len(got) == len(tasks)
    for (lv, links, sz), (g, d) in zip(tasks, got):
        g0, d0 = ds._cluster_pt(lv, links, order, sz)
        assert g.shape == g0.shape and d.shape == d0.shape
        assert _close(g, g0) and _close(d, d0)


def test_order6_rounding_within_numpys_own_spread():
    m = _rb_model()
    order, nexc = 6, m.nexc
    levels = m._enumerate(order)[order]
    rng = np.random.default_rng(3)
    spread = np.zeros(order + 1)
    tasks, refs = [], []
    for form in sorted(levels)[:12]:
        _k, node_order = m._struct(form)
        lv = [m.levels[d] for (d, _c) in node_order]
        sz = [m.sz_levels[d] for (d, _c) in node_order]
        links = m._local_links(form, node_order)
        _g, d0 = ds._cluster_pt(lv, links, order, sz)
        # the same cluster with its nodes relabelled, mapped back
        k = len(lv)
        perm = rng.permutation(k)
        inv = np.argsort(perm)
        _g1, d1 = ds._cluster_pt([lv[inv[j]] for j in range(k)],
                                 [(int(perm[a]), int(perm[b]), V) for (a, b, V) in links],
                                 order, [sz[inv[j]] for j in range(k)])
        idx = np.concatenate([[perm[i] * nexc + a for a in range(nexc)] for i in range(k)])
        spread = np.maximum(spread, np.abs(d1[:, idx][:, :, idx] - d0).max(axis=(1, 2)))
        tasks.append((lv, links, sz))
        refs.append(d0)
    got = rsmagcalc.cluster_pt_batch(tasks, order)
    dev = np.zeros(order + 1)
    for (_g, d), d0 in zip(got, refs):
        dev = np.maximum(dev, np.abs(d - d0).max(axis=(1, 2)))
    assert spread[order] > 0                      # the relabelling really reorders sums
    assert np.all(dev <= 3 * np.maximum(spread, 1e-15)), (dev, spread)


def test_thread_count_changes_nothing():
    tasks = _chain_tasks(5)
    a = rsmagcalc.cluster_pt_batch(tasks, 5, n_workers=1)
    b = rsmagcalc.cluster_pt_batch(tasks, 5)
    for (ga, da), (gb, db) in zip(a, b):
        assert np.array_equal(ga, gb) and np.array_equal(da, db)


def test_empty_batch():
    assert rsmagcalc.cluster_pt_batch([], 3) == []


@pytest.mark.parametrize("bad", ["V_shape", "link_self", "link_range", "sz_shape",
                                 "one_level"])
def test_malformed_clusters_raise(bad):
    lv = [np.array([0.0, 1.0, 1.0, 1.0])] * 2
    V = np.zeros((16, 16), complex)
    links, sz = [(0, 1, V)], [np.array([0.0, 1.0, 0.0, -1.0])] * 2
    if bad == "V_shape":
        links = [(0, 1, np.zeros((9, 9), complex))]
    elif bad == "link_self":
        links = [(1, 1, V)]
    elif bad == "link_range":
        links = [(0, 2, V)]
    elif bad == "sz_shape":
        sz = [np.zeros(3)] * 2
    elif bad == "one_level":
        lv, links, sz = [np.array([0.0])] * 2, [], None
    with pytest.raises(ValueError):
        rsmagcalc.cluster_pt_batch([(lv, links, sz)], 2)
