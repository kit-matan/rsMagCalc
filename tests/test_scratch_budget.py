"""The scratch budget and K7's byte-weighted admission (2026-10-31).

Until then every memory-bound kernel capped its pool at a hard-coded 3 GB divided
by the LARGEST item's scratch. On a 243 GB, 128-core node that held the dimer
series to 2 threads at order 8 and 1 at order 9, whatever `n_workers` said. Pinned
here:

  * the budget's precedence -- argument, $RSMAGCALC_SCRATCH_GB, half of the
    smaller of MemAvailable and every cgroup's headroom, 3 GB -- on fake
    /proc and /sys trees (cgroup v2 and v1, nested limits, the "unlimited"
    sentinels);
  * the admission: every cluster's charge fits the budget beside those in
    flight, one larger than the budget runs alone, and the results do not
    depend on the budget or the thread count (bit for bit);
  * the per-cluster estimate's sector count, against the exact identity that m
    dimers with level Sz (0, -1, 0, +1) have C(2m, m) states at total Sz = 0
    (the coefficient of x^0 in (x^-1/2 + x^1/2)^(2m)).
"""
import math

import numpy as np
import pytest

import rsmagcalc

pytestmark = pytest.mark.skipif(rsmagcalc.backend != "pyo3",
                                reason="rsmagcalc core not built")

GiB = 1 << 30


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("RSMAGCALC_SCRATCH_GB", raising=False)


def _budget(root=None, gb=None):
    return tuple(rsmagcalc._core.scratch_budget(gb, None if root is None else str(root)))


def _tree(tmp_path, meminfo_gb=None, cgroup=None, files=()):
    """A fake root: /proc/meminfo, /proc/self/cgroup and cgroup files."""
    (tmp_path / "proc/self").mkdir(parents=True)
    if meminfo_gb is not None:
        (tmp_path / "proc/meminfo").write_text(
            f"MemTotal:  999999999 kB\nMemFree:  1 kB\n"
            f"MemAvailable:  {int(meminfo_gb * GiB) // 1024} kB\n")
    if cgroup is not None:
        (tmp_path / "proc/self/cgroup").write_text(cgroup)
    for rel, text in files:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return tmp_path


# --------------------------------------------------------------------------
# The budget
# --------------------------------------------------------------------------
def test_no_files_falls_back_to_3_gb(tmp_path):
    assert _budget(_tree(tmp_path)) == (3 * GiB, "default 3 GB")


def test_half_of_mem_available(tmp_path):
    assert _budget(_tree(tmp_path, meminfo_gb=100)) == (50 * GiB, "0.5 x MemAvailable")


def test_cgroup_v2_nested_limit_wins_when_tighter(tmp_path):
    # Slurm's shape: the job cgroup carries the limit, the task cgroup below it
    # says "max"; the headroom is limit - current of the LIMITED level
    root = _tree(tmp_path, meminfo_gb=100, cgroup="0::/slurm/job_7/task_0\n", files=[
        ("sys/fs/cgroup/slurm/job_7/task_0/memory.max", "max\n"),
        ("sys/fs/cgroup/slurm/job_7/task_0/memory.current", f"{GiB}\n"),
        ("sys/fs/cgroup/slurm/job_7/memory.max", f"{10 * GiB}\n"),
        ("sys/fs/cgroup/slurm/job_7/memory.current", f"{2 * GiB}\n"),
        ("sys/fs/cgroup/memory.max", "max\n")])
    assert _budget(root) == (4 * GiB, "0.5 x cgroup memory.max")


def test_cgroup_v2_looser_than_meminfo_is_ignored(tmp_path):
    root = _tree(tmp_path, meminfo_gb=20, cgroup="0::/job\n", files=[
        ("sys/fs/cgroup/job/memory.max", f"{200 * GiB}\n"),
        ("sys/fs/cgroup/job/memory.current", "0\n")])
    assert _budget(root) == (10 * GiB, "0.5 x MemAvailable")


def test_cgroup_v1_memory_controller(tmp_path):
    root = _tree(tmp_path, meminfo_gb=100,
                 cgroup="12:cpu,cpuacct:/slurm/uid_1/job_9\n"
                        "4:memory:/slurm/uid_1/job_9\n", files=[
        ("sys/fs/cgroup/memory/slurm/uid_1/job_9/memory.limit_in_bytes", f"{8 * GiB}\n"),
        ("sys/fs/cgroup/memory/slurm/uid_1/job_9/memory.usage_in_bytes", f"{2 * GiB}\n"),
        # the v1 root reports "unlimited" as a near-2^63 number
        ("sys/fs/cgroup/memory/memory.limit_in_bytes", "9223372036854771712\n")])
    assert _budget(root) == (3 * GiB, "0.5 x cgroup memory.limit_in_bytes")


def test_usage_above_limit_gives_the_minimum_not_a_wrap(tmp_path):
    root = _tree(tmp_path, cgroup="0::/j\n", files=[
        ("sys/fs/cgroup/j/memory.max", f"{GiB}\n"),
        ("sys/fs/cgroup/j/memory.current", f"{2 * GiB}\n")])
    assert _budget(root) == (1, "0.5 x cgroup memory.max")


def test_environment_beats_the_files_and_the_argument_beats_both(tmp_path, monkeypatch):
    root = _tree(tmp_path, meminfo_gb=100)
    monkeypatch.setenv("RSMAGCALC_SCRATCH_GB", "7.5")
    assert _budget(root) == (int(7.5 * GiB), "RSMAGCALC_SCRATCH_GB")
    assert _budget(root, gb=2) == (2 * GiB, "argument")
    assert rsmagcalc.scratch_budget() == (int(7.5 * GiB), "RSMAGCALC_SCRATCH_GB")


@pytest.mark.parametrize("bad", ["lots", "0", "-1", "nan"])
def test_a_bad_environment_value_raises(monkeypatch, bad):
    monkeypatch.setenv("RSMAGCALC_SCRATCH_GB", bad)
    with pytest.raises(ValueError, match="RSMAGCALC_SCRATCH_GB"):
        rsmagcalc.scratch_budget()


@pytest.mark.parametrize("bad", [0.0, -2.0, float("inf")])
def test_a_bad_argument_raises(bad):
    with pytest.raises(ValueError, match="scratch_budget_gb"):
        rsmagcalc.scratch_budget(bad)


# --------------------------------------------------------------------------
# K7's admission
# --------------------------------------------------------------------------
def _dimer_chain(m, V_scale=0.3):
    """m S = 1/2 dimers in a chain: singlet 0, triplet 1, an Sz-conserving
    Heisenberg coupling between neighbouring dimers' facing spins."""
    s = 0.5 * np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    r = 1 / np.sqrt(2)
    # singlet, T-1, T0, T+1 in the (up, down) product basis idx = 2 s1 + s2
    B = np.array([[0, r, -r, 0], [0, 0, 0, 1], [0, r, r, 0], [1, 0, 0, 0]],
                 dtype=complex).T
    I2 = np.eye(2)
    S1 = [B.conj().T @ np.kron(x, I2) @ B for x in s]
    S2 = [B.conj().T @ np.kron(I2, x) @ B for x in s]
    V = V_scale * sum(np.kron(S2[k], S1[k]) for k in range(3))
    lv = [np.array([0.0, 1.0, 1.0, 1.0])] * m
    sz = [np.array([0.0, -1.0, 0.0, 1.0])] * m
    return lv, [(d, d + 1, V) for d in range(m - 1)], sz


def _expected_scratch(m, order, rows):
    p = 3 * m
    return rows * p * 16 * (2 * order + 3) // 10 * 11 + 4 ** m * 32


@pytest.mark.parametrize("m", [2, 4, 6])
def test_sector_count_is_the_central_binomial(m):
    order = 3
    lv, links, sz = _dimer_chain(m)
    _out, st = rsmagcalc.cluster_pt_batch([(lv, links, sz)], order, return_stats=True)
    assert st["max_scratch_bytes"] == _expected_scratch(m, order, math.comb(2 * m, m))
    _out, st = rsmagcalc.cluster_pt_batch([(lv, links, None)], order, return_stats=True)
    assert st["max_scratch_bytes"] == _expected_scratch(m, order, 4 ** m)


def _batch():
    """Clusters of three sizes, each several times."""
    return [_dimer_chain(m, 0.2 + 0.05 * k) for k in range(4) for m in (3, 4, 5)]


def test_results_do_not_depend_on_budget_or_threads():
    tasks, order = _batch(), 4
    ref = rsmagcalc.cluster_pt_batch(tasks, order, n_workers=1)
    for kw in ({"n_workers": 4, "scratch_budget_gb": 1e-9},
               {"n_workers": 4, "scratch_budget_gb": 100.0},
               {"n_workers": 32}):
        got = rsmagcalc.cluster_pt_batch(tasks, order, **kw)
        assert len(got) == len(ref)
        for (g0, d0), (g1, d1) in zip(ref, got):
            assert np.array_equal(g0, g1) and np.array_equal(d0, d1)


def test_a_budget_below_one_cluster_runs_them_one_at_a_time():
    _out, st = rsmagcalc.cluster_pt_batch(_batch(), 4, n_workers=4,
                                          scratch_budget_gb=1e-9, return_stats=True)
    assert st["peak_concurrent"] == 1
    assert st["budget_source"] == "argument"
    assert st["threads"] == 4 and st["n_clusters"] == 12


def test_admitted_bytes_never_exceed_the_budget():
    tasks, order = _batch(), 4
    _o, st = rsmagcalc.cluster_pt_batch(tasks, order, return_stats=True)
    big = st["max_scratch_bytes"]
    # room for the largest cluster and a bit: never two of the largest at once
    gb = 1.5 * big / (1 << 30)
    _o, st = rsmagcalc.cluster_pt_batch(tasks, order, n_workers=8,
                                        scratch_budget_gb=gb, return_stats=True)
    assert st["peak_bytes"] <= st["budget_bytes"]
    assert st["budget_bytes"] == int(gb * (1 << 30))


def test_threads_are_capped_by_the_batch_not_by_the_largest_cluster():
    # the old rule capped the POOL at budget / largest; now the pool is
    # n_workers (or the batch size) and memory admits cluster by cluster
    tasks = [_dimer_chain(3)] * 3
    _o, st = rsmagcalc.cluster_pt_batch(tasks, 3, n_workers=16,
                                        scratch_budget_gb=1e-9, return_stats=True)
    assert st["threads"] == 3 and st["peak_concurrent"] == 1


def test_default_call_still_returns_a_plain_list():
    out = rsmagcalc.cluster_pt_batch([_dimer_chain(2)], 2)
    assert isinstance(out, list) and len(out) == 1 and len(out[0]) == 2
