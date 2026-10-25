"""The data contract, pinned against pyMagCalc (the oracle, as for fMagCalc).

`classical_energy` is the M0 kernel: it exists to prove that a NumPy array
reaches Rust unchanged (layout, flattening order, dtype) and that the number
coming back is pyMagCalc's. Every later kernel reads the same (H, b, m) and will
be pinned the same way, plus the physics oracles in RUST_KERNELS_PLAN.md.
"""
import numpy as np
import pytest

import rsmagcalc

magcalc_cd = pytest.importorskip(
    "magcalc.classical_dynamics",
    reason="pyMagCalc not installed (the parity oracle)",
)


def _random_problem(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3 * n, 3 * n))
    H = 0.5 * (A + A.T)
    b = rng.standard_normal(3 * n)
    m = rng.standard_normal((n, 3))
    m *= 1.5 / np.linalg.norm(m, axis=1, keepdims=True)
    return H, b, m


@pytest.mark.parametrize("n,seed", [(1, 0), (2, 1), (7, 2), (40, 3)])
def test_classical_energy_matches_pymagcalc(n, seed):
    H, b, m = _random_problem(n, seed)
    expected = magcalc_cd.energy(H, b, m)
    got = rsmagcalc.classical_energy(H, b, m)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_non_contiguous_views_are_read_in_logical_order():
    """A Fortran-ordered H or a strided m must give the same answer: the kernel
    reads logical (row-major) order, like `ravel()`, never raw memory."""
    H, b, m = _random_problem(5, 4)
    expected = magcalc_cd.energy(H, b, m)
    Hf = np.asfortranarray(H)
    m_strided = np.ascontiguousarray(np.repeat(m, 2, axis=0))[::2]
    assert not m_strided.flags.c_contiguous
    got = rsmagcalc.classical_energy(Hf, b, m_strided)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_shape_mismatch_is_an_error_not_a_broadcast():
    H, b, m = _random_problem(3, 5)
    with pytest.raises(ValueError):
        rsmagcalc.classical_energy(H, b[:-3], m)
    with pytest.raises(ValueError):
        rsmagcalc.classical_energy(H, b, m[:-1])
