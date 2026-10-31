"""K8: the dipole H(q) = 2 g H2(q) builder against closed forms (2026-10-31).

`DipoleHq` takes monomials in the spin components and does the Holstein-Primakoff
expansion, the Fourier placement and 2g. pyMagCalc pins it entry by entry to its
SymPy build (tests/test_numeric_hq.py there); these pin it to linear spin-wave
results the code never sees:

  * a ferromagnetic chain, omega = 2 |J| S (1 - cos qa), H(0) EXACTLY zero;
  * the same with an easy-plane anisotropy and the moment in the plane:
    omega^2 = A (A + 2 D S), A = 2|J|S(1 - cos qa), gapless at q = 0 -- the
    anomalous (c c, c+ c+) placement;
  * a Neel chain, omega = 2 J S |sin(q d)| on two sublattices, the second rotated
    by pi (the rotation handling);
  * a single Zeeman monomial, h S^z, which carries -h c+ c;
  * a collinear ferromagnet's biquadratic B (S_i.S_j)^2 = a bilinear J = 2 B S^2
    (four-factor products);
  * moment along x with the anisotropy along x: the same gap as along z;
then the bond-model form, H(-q), pickling, and the input checks.
"""
import itertools
import pickle

import numpy as np
import pytest

import rsmagcalc

pytestmark = pytest.mark.skipif(rsmagcalc.backend != "pyo3",
                                reason="rsmagcalc core not built")

A = 3.0


def _chain(n_images=1):
    """One site at 0 and its images at +-a, +-2a ..."""
    xs = [0.0] + [s * k * A for k in range(1, n_images + 1) for s in (1, -1)]
    return np.array([[x, 0, 0] for x in xs])


def _bond(J, i, j):
    return [(0.5 * J, ((i, c), (j, c))) for c in range(3)]


def _qs(n=9):
    return np.array([[q, 0, 0] for q in np.linspace(-np.pi / A, np.pi / A, n)])


def _omega(H):
    """The positive eigenvalues of g H2, sorted."""
    ev = np.sort(np.linalg.eigvals(H).real)
    return ev[len(ev) // 2:]


def test_ferromagnetic_chain():
    J, S = -1.3, 1.5
    mono = _bond(J, 0, 1) + _bond(J, 0, 2)
    m = rsmagcalc.DipoleHq(1, _chain(), [S], np.eye(3)[None], mono)
    q = _qs()
    H = m.hamiltonians(q)
    w = 2 * abs(J) * S * (1 - np.cos(q[:, 0] * A))
    assert np.abs(H[:, 0, 0] - w).max() < 1e-14
    assert np.abs(H[:, 1, 1] + w).max() < 1e-14
    assert np.abs(H[:, 0, 1]).max() == 0 and np.abs(H[:, 1, 0]).max() == 0
    # exactly zero at q = 0: the sqrt(S/2)^2 is applied as S/2
    assert np.all(m.hamiltonians(np.zeros((1, 3)))[0] == 0)


def test_easy_plane_ferromagnet_with_the_moment_in_the_plane():
    J, S, D = -1.0, 1.0, 0.4
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)    # local z -> lab x
    mono = _bond(J, 0, 1) + _bond(J, 0, 2) + [(D, ((0, 2), (0, 2)))]
    m = rsmagcalc.DipoleHq(1, _chain(), [S], R[None], mono)
    q = _qs()
    Aq = 2 * abs(J) * S * (1 - np.cos(q[:, 0] * A))
    # g H2 = [[a, b], [-b*, -a]] has omega^2 = a^2 - |b|^2. Read it off the entries:
    # at q = 0 the mode is a Jordan block, where an eigensolver returns ~sqrt(eps).
    H = m.hamiltonians(q)
    w2 = H[:, 0, 0].real ** 2 - np.abs(H[:, 0, 1]) ** 2
    assert np.abs(w2 - Aq * (Aq + 2 * D * S)).max() < 1e-13
    assert np.abs(H[:, 0, 1] + H[:, 1, 0].conj()).max() < 1e-15


def test_neel_chain():
    J, S, d = 1.0, 2.0, A / 2
    pos = np.array([[0, 0, 0], [d, 0, 0], [-A, 0, 0], [-A + d, 0, 0], [A, 0, 0],
                    [A + d, 0, 0]], float)   # cell (A, B), then the images at -a, +a
    Rpi = np.diag([1.0, -1.0, -1.0])         # B points along -z
    mono = (_bond(J, 0, 1) + _bond(J, 0, 3)      # A - B (cell), A - B (left image)
            + _bond(J, 1, 0) + _bond(J, 1, 4))   # B - A (cell), B - A (right image)
    m = rsmagcalc.DipoleHq(2, pos, [S, S], np.stack([np.eye(3), Rpi]), mono)
    q = _qs()
    w = 2 * J * S * np.abs(np.sin(q[:, 0] * d))
    for h, wk in zip(m.hamiltonians(q), w):
        assert np.allclose(_omega(h), [wk, wk], atol=1e-7 if wk < 1e-6 else 1e-12)


def test_a_zeeman_monomial_carries_a_number_operator():
    h, S = -0.37, 2.5
    m = rsmagcalc.DipoleHq(1, np.zeros((1, 3)), [S], np.eye(3)[None], [(h, ((0, 2),))])
    H = m.hamiltonians(np.array([[0.2, 0, 0]]))[0]
    assert np.allclose(H, np.diag([-h, h]), atol=1e-15)


def test_collinear_biquadratic_is_a_bilinear_2BS2():
    B, S = 0.05, 1.5

    def biq(i, j):
        out = []
        for a, b in itertools.product(range(3), repeat=2):
            out.append((0.5 * B, ((i, a), (j, a), (i, b), (j, b))))
        return out
    mb = rsmagcalc.DipoleHq(1, _chain(), [S], np.eye(3)[None], biq(0, 1) + biq(0, 2))
    mj = rsmagcalc.DipoleHq(1, _chain(), [S], np.eye(3)[None],
                            _bond(2 * B * S * S, 0, 1) + _bond(2 * B * S * S, 0, 2))
    q = _qs()
    assert np.abs(mb.hamiltonians(q) - mj.hamiltonians(q)).max() < 1e-13


def test_the_rotation_carries_the_anisotropy_axis():
    D, S = -0.6, 1.0
    gap = 2 * abs(D) * S
    mz = rsmagcalc.DipoleHq(1, np.zeros((1, 3)), [S], np.eye(3)[None], [(D, ((0, 2), (0, 2)))])
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
    mx = rsmagcalc.DipoleHq(1, np.zeros((1, 3)), [S], R[None], [(D, ((0, 0), (0, 0)))])
    q = np.zeros((1, 3))
    assert _omega(mz.hamiltonians(q)[0])[0] == pytest.approx(gap, abs=1e-14)
    assert _omega(mx.hamiltonians(q)[0])[0] == pytest.approx(gap, abs=1e-14)


def _some_model():
    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.normal(size=(2, 3, 3)))[0]
    pos = np.array([[0, 0, 0], [1.1, 0.3, 0], [3, 0, 0], [4.1, 0.3, 0], [-3, 0, 0],
                    [-1.9, 0.3, 0]], float)
    mono = []
    for (i, j) in [(0, 1), (1, 0), (0, 3), (1, 4), (0, 2), (1, 5)]:
        for a, b in itertools.product(range(3), repeat=2):
            mono.append((rng.normal(), ((i, a), (j, b))))
    mono.append((0.3, ((0, 0), (0, 1), (0, 2), (0, 2))))
    mono.append((-0.2, ((1, 1),)))
    return rsmagcalc.DipoleHq(2, pos, [1.0, 2.5], R, mono)


def test_bond_model_negation_and_pickling():
    m = _some_model()
    q = np.random.default_rng(1).normal(size=(7, 3))
    H = m.hamiltonians(q)
    dvec, Mb = m.bond_model()
    dense = np.einsum("qb,bij->qij", np.exp(1j * q @ dvec.T), Mb)
    assert np.abs(dense - H).max() < 1e-13
    assert np.array_equal(m.hamiltonians(q, negate=True), m.hamiltonians(-q))
    m2 = pickle.loads(pickle.dumps(m))
    assert np.array_equal(m2.hamiltonians(q), H)
    # one q (the serial path) and many (the thread pool) give the same matrix
    assert np.array_equal(m.hamiltonians(q[:1])[0], H[0])


def test_rows_of_the_hole_block_are_negated():
    """H = 2 g H2 with H2 Hermitian: g H is Hermitian only up to the sign of
    its lower rows, so g H must be Hermitian."""
    m = _some_model()
    for h in m.hamiltonians(np.random.default_rng(2).normal(size=(3, 3))):
        n = h.shape[0] // 2
        gh = h.copy()
        gh[n:] *= -1
        assert np.abs(gh - gh.conj().T).max() < 1e-12


@pytest.mark.parametrize("kw,msg", [
    ({"spins": [1.0, 2.0]}, "spins must be"),
    ({"monomials": [(1.0, ((5, 0), (0, 0)))]}, "outside"),
    ({"monomials": [(1.0, ((0, 3), (0, 0)))]}, "outside"),
])
def test_bad_inputs_raise(kw, msg):
    args = dict(n=1, pos_ouc=_chain(), spins=[1.0], rot=np.eye(3)[None],
                monomials=[(1.0, ((0, 0), (1, 0)))])
    args.update(kw)
    with pytest.raises(ValueError, match=msg):
        rsmagcalc.DipoleHq(args["n"], args["pos_ouc"], args["spins"], args["rot"],
                           args["monomials"])
