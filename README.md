# rsMagCalc

Rust kernels, behind PyO3, for the loops in pyMagCalc that are Python-bound per
site or per q: the classical Metropolis sweep and Landau–Lifshitz integrator, the
annealing ground-state search, the CP^(N−1) sampler, SU(N) LSWT assembly and Bogoliubov diagonalization, the KPM
Chebyshev recursion, SCGA, the per-cluster perturbation theory of the dimer series
expansion, and the dipole engine's LSWT matrix H(q) built numerically in place of its
SymPy expansion. What each kernel covers, the oracle it is pinned to and
its measured speed-up are recorded in pyMagCalc's
[`GAP_STATUS.md`](https://github.com/kit-matan/pyMagCalc/blob/master/GAP_STATUS.md)
under "Rust kernels K1" to "K5", "anneal on K1's dense model", "Rust kernels K7" and "Rust kernels K8". This is **not** a port of pyMagCalc: the config,
symmetry, term resolution, runner, fitting and plotting stay in Python, and
the dipole LSWT hot path stays with [fMagCalc](https://github.com/kit-matan/fMagCalc).
New to MagCalc? Start with pyMagCalc's
[`MAGCALC_GUIDE.md`](https://github.com/kit-matan/pyMagCalc/blob/master/MAGCALC_GUIDE.md).

The contract is fMagCalc's:

- pyMagCalc never requires this package. `calculation: {kernels: rust}` selects it;
  if it is missing or a kernel fails, pyMagCalc logs a WARNING and runs its own
  NumPy implementation.
- Every kernel is pinned to pyMagCalc's NumPy path (parity, tight tolerance) **and**
  to the independent oracle its pyMagCalc tests already use (Sunny, closed forms,
  sum rules), by running those tests under both kernels.
- The run log names the build it used (`kernels: rust (rsmagcalc 0.1.0, git …,
  release)`), as it names the engine copy.

## Build

You need a Rust toolchain (`rustup`) and maturin. To use the kernels:

```bash
pip install maturin
pip install git+https://github.com/kit-matan/rsMagCalc
python -c "import rsmagcalc; print(rsmagcalc.backend, rsmagcalc.describe())"   # pyo3 (...)
```

To work on them, in a clone:

```bash
cd rsMagCalc
pip install -e . --no-build-isolation   # compiles rsmagcalc._core; editable, like pyMagCalc
pytest                                  # parity against pyMagCalc, which must be installed
```

`maturin develop --release` does the same but only inside a virtualenv or conda
environment (it refuses a bare pyenv interpreter); the pip form works anywhere.
Rebuild after editing `src/` with the same `pip install -e .` command -- an editable
install makes the Python sources live, not the compiled core.

**Do not strip the extension, and do not pass `--strip` to maturin.** On macOS 27
(Xcode 27.1, ld-27037.1) every strip pass -- cargo's default `strip = "debuginfo"`,
maturin's `--strip`, Apple's `strip -x` -- leaves the LINKEDIT string pool 4 bytes off
an 8-byte boundary whenever the indirect symbol table has an odd length, and the new
dyld refuses the file with `mis-aligned LINKEDIT string pool`. The symptom is
`rsmagcalc.backend == None` after a build that reported success. `Cargo.toml` sets
`strip = "none"` and `pyproject.toml` sets `strip = false` for this reason (measured
2026-10-26); `rust-lld` is not an alternative here, it cannot read the SDK's `.tbd`
stubs. If a wheel built elsewhere fails to import on macOS 27, check
`otool -l _core.abi3.so | grep stroff` against a multiple of 8 before anything else.

`maturin build --release` produces a wheel. abi3 (`pyo3/abi3-py39`) means one wheel
per platform, importable from every CPython ≥ 3.9. Wheels are built without
`-C target-cpu=native`; set `RUSTFLAGS` for a local build that may use it.

## Status

M0: scaffold. `build_info()` / `describe()` for provenance, and one kernel,
`classical_energy(H, b, m)`, that proves the zero-copy NumPy contract and is pinned
to `magcalc.classical_dynamics.energy`.

M1 (K1, `src/classical.rs`): the classical Metropolis sweep (`sweep`, uniform and
flip proposals, in place on a `Rng` stream), the whole replica-exchange loop
(`parallel_tempering`, rayon over replicas), the Wang–Landau move loop
(`wang_landau`), the Landau–Lifshitz integrators (`evolve`: rk4 with per-step
renormalization, implicit midpoint without), the Heun Langevin thermostat
(`langevin`) and `sampled_trajectories` (rayon over trajectories; trajectory i owns
the PCG64 stream (seed, i), so chunked calls reproduce one call and no result
depends on the thread count). Dense H (3N×3N) for now; `g = H m + b` is the energy
GRADIENT, as pyMagCalc's `local_field` returns it. Pinned in
`tests/test_classical_parity.py`: `evolve` matches `magcalc.classical_dynamics.evolve`
to 1e-12 (measured 2e-16) over 50 rk4 and 50 midpoint steps, the single-move dE
equals E(after) − E(before) to 1e-12, plus the exact classical-dimer thermodynamics
and the Langevin function. pyMagCalc selects it with `calculation: {kernels: rust}`
and runs its own oracle tests under both kernel sets.

The ground-state search (2026-10-28) runs on the same dense model and move
primitives: `anneal` is pyMagCalc's `annealing.anneal` (a site drawn with
replacement, Sunny's `LocalSampler` uniform/flip/delta proposal mix with the adaptive
delta width, a geometric schedule, the best state kept, then the keep-if-better
steepest-descent polish), for several independent runs at once in parallel (run r
owns the stream of `seeds[r]`, so a batch reproduces the per-seed calls whatever the
thread count). `steepest_descent` is `annealing.steepest_descent`, deterministic and
matched to it to 1e-12 with mixed spins and a large on-site block. Pinned in
`tests/test_classical_parity.py`, including the mixed-spin trimer's closed-triangle
energy reached by the Metropolis loop alone, with the polish off.

M2 (K2, `src/cpn.rs`, `rsmagcalc.CpnModel`): the CP^(N−1) Metropolis sampler for
SU(N) coherent states with pyMagCalc's adaptive step size (`thermalize`, returning
the per-sweep acceptance and energy record the equilibration verdict is made
from), the RK4 integrator (`evolve`), the damped quench (`quench`) and
`sampled_trajectories` (rayon over trajectories, stream (seed, i) each). The
model's per-site operator tables, on-site matrices and bond list are packed once
as RAGGED arrays, so a mixed-spin cell runs through the kernel where pyMagCalc's
vectorized path refuses it. The energy change of a proposal is the term-by-term
change of the bond terms touching the site plus its on-site term — exact for any
bond list — where the NumPy sampler re-evaluates the total energy twice per site.
Pinned in `tests/test_cpn_parity.py`: that dE equals E(after) − E(before) to 1e-12
on plain SU(3), biquadratic (12-wide operator set), mixed-spin and self-bonded
cells; `h_i|Z_i>` against the `local_field` definition; `evolve` and `quench`
against `magcalc.sun.dynamics` to 1e-12 over 50 steps; the sampler against the
closed-form partition function of a decoupled site.

M3 (K3, `src/sun_lswt.rs`, `rsmagcalc.SunModel`): SU(N) linear spin-wave theory.
A `SUNModel`'s reference-state tables (`t`, `tb`, `Q`, `QA`, `s0`, ragged per site
so mixed spin works) and its bond list are packed once; every method then runs a
q BATCH in parallel (rayon): `hamiltonians` (g H2, the per-q assembly that is a
Python loop over bonds × n_ops² in pyMagCalc), `eigenvalues` (the general problem
`dispersion` / `max_imaginary` reduce), `bogoliubov` (Colpa with the explicit
[D positive descending, D negative] order), `correlations` (the per-band one-magnon
tensors conj(M^a) M^b / n_cells, before the cross-section contraction, which stays
in NumPy) and `stable` (the ground-state guard: one shifted Cholesky, the shift
supplied by pyMagCalc from its q-INDEPENDENT reference scale). The linear algebra
is the platform LAPACK (`src/lapack.rs`: Accelerate's ILP64 interface on macOS,
linked with `-framework Accelerate`), not faer: inside a degenerate eigenspace only
the same eigensolver reproduces NumPy's per-band split, which keeps band-by-band
oracle tests meaningful under both kernels. Linux links the system `lapack`/`blas`
(LP64) and is untested. Pinned in `tests/test_sun_lswt_parity.py` (37 tests): H(q)
elementwise to 1e-13, energies and per-band tensors to 1e-10, T para-unitary and
equal to NumPy's columns up to phase, the guard verdict by verdict on pyMagCalc's
`test_kpm_stability` fixtures, mixed spin, the biquadratic operator set, a
non-collinear and a disordered supercell, and the entangled dimer chain.

M4 (K4, `src/kpm.rs`): the KPM Chebyshev recursion of pyMagCalc's `sun/kpm.py`
(moments mu_m^{ab} = v_a^H T_m(conj(g H2)/gamma) g v_b) and its power-iteration
spectral bound. `kpm_moments(HMat, v, M, gamma)` and `spectral_bound(HMat, ...)`
take a matrix already built; `SunModel.kpm_moments(qs, ...)` runs a q batch in
parallel with H(q) built inside by K3, so no 2D x 2D matrix crosses into Python.
H(q) is scanned once per q into CSR when it is at most a quarter full (a supercell's
is: 0.7 % at S09's L = 30), else multiplied with zgemm. The bound starts from
NumPy's own start vector (drawn in the Python wrapper), and the moment count from
`fwhm`/`tol` is pyMagCalc's integer. The Chebyshev coefficients, the +omega cutoff
and the reconstruction stay in pyMagCalc. Pinned in `tests/test_kpm_parity.py`
(63 tests): moments at 1e-12 at the same gamma, dense, CSR and auto, on every K3
fixture plus 36- and 81-site non-collinear supercells; the bound; the chiral
channel end to end; the Goldstone point. S09 L = 30 KPM: 1.4 s/q -> 43 ms/q with
pyMagCalc's stability guard included (`pyMagCalc/GAP_STATUS.md`, "Rust kernels K4").

The guard on a sparse Cholesky (`src/sparse_guard.rs`): `SunModel.stable` factors
H2(q) + shift I with a multifrontal Cholesky on a minimum-degree ordering
(analysed once per model; Accelerate `zpotrf`/`ztrsm`/`zherk` per front) when
2D >= 64 and that needs at most a quarter of the dense flops, else one dense
`zpotrf` as before; `sparse=` forces either and `guard_info()` reports the choice.
The matrix entries are bit-identical to the dense path's (`SunModel::assemble`
emits its contributions in a q-independent order), so only the factorization's
rounding differs. Pinned in `tests/test_sparse_guard.py` (exact verdicts at
shift = -min eig +- 1e-9 scale, the Goldstone zeros, Sunny's sigma = 1/3 refused,
a dense operator kept dense) and by forcing both paths in every guard case of
`tests/test_sun_lswt_parity.py`. S09 L = 30: guard 38 -> 1.05 ms/q, a KPM q
43 -> 7.6 ms (`pyMagCalc/GAP_STATUS.md`, "guard 3 on a SPARSE Cholesky").

M5 (K5, `src/scga.rs`): the SCGA loops of pyMagCalc's `scga.py`. `ScgaModel` is
packed once from `scga._exchange_cache` (directed bonds with Cartesian image
offsets, positions, g-tensors, spins); `fourier(qs)` is `fourier_exchange_matrix`,
`eigenvalues(mesh)` the J(q) spectrum `solve_lambda`'s root solve sums over
(zheevd jobz N with the queried workspace, the call `eigvalsh` makes -- bit-identical,
so lambda is too), `sum_rule_residuals(mesh, lam, kT)` the per-site check behind the
inequivalent-sublattice refusal (Cholesky + ztrtri for the inverse's diagonal), and
`correlations(qs, lam, kT, ff, apply_g)` the 3 x 3 tensors kT W^H (lam + J)^-1 W
(Cholesky + zpotrs on 3 right-hand sides; NumPy's LU when lam + J(q) is
indefinite). The root solve, the form factors (passed in per q and site) and the
cross-section contraction stay in pyMagCalc. Pinned in `tests/test_scga_parity.py`
(75 tests) at 1e-12 on the chain, square, square + DM, kagome, kagome with an
anisotropic g and Cu2+ form factors, and a 16-site pyrochlore. That cell's
`solve_lambda` 3.9 s -> 0.17 s, a 1681-q map 2.5 s -> 27 ms
(`pyMagCalc/GAP_STATUS.md`, "Rust kernels K5").
K6, the mixed-spin dipole S(Q,ω), did not come here. The dipole path is fMagCalc's,
and fMagCalc lacked only the per-site √(S_i/2) prefactor, so it went there instead
(fMagCalc `eae20e3`; pyMagCalc `GAP_STATUS.md`, "K6"). That completed the plan:
the work here was K1–K5 plus the sparse guard.

K7 (`src/dimer_series.rs`, 2026-10-29) came after it, from a profile of every
shipped example under the compiled kernels: the dimer series expansion
(`sun/dimer_series.py`) was the one loop left that was Python-bound. Two functions:
`block_effective_series(E0, V_list, P_idx, order)` is pyMagCalc's Bloch recursion
with column-dependent resolvents plus the des Cloizeaux Hermitization (each V dense
or scipy-sparse, passed as CSR), and `cluster_pt_batch(clusters, order, n_workers)`
is `_cluster_pt` for a list of clusters, threaded over the clusters: the pair operators
embedded into the nloc^m product space, the ground state and each one-particle
total-Sz sector run separately when the local bases carry Sz, the interaction parts
returned. The enumeration, isomorphism keys, linked-cluster subtraction and
Dlog-Padé stay in pyMagCalc. The arithmetic is the NumPy code's term by term, with
three savings that change no number (V·Ω_m computed once, Ω_0 as a row gather, the
Gram terms of S_k and M_k summed in adjoint pairs). Pinned in
`tests/test_dimer_series_parity.py` at 1e-12 on random quasi-degenerate,
multi-order and sparse PT problems, and on the alternating chain's, Rb2Cu3SnF12's
and random nloc = 3 clusters, with and without sectors (`pyMagCalc/GAP_STATUS.md`,
"Rust kernels K7").

Since 2026-10-31 the one-particle block that shares its sector with the vacuum (the
T0 states under out-of-plane DM, or every one-particle state without sectors) is no
longer the des Cloizeaux H_eff. That H_eff is not cluster additive once the dressed
vacuum has one-particle components. The block now goes through the projective
cluster-additive transformation of Hörmann and Schmidt, SciPost Phys. 15, 097 (2023),
which removes each one-particle state's vacuum component before orthonormalizing.
The same change was made in pyMagCalc's `_cluster_pt`. The other sectors are
unchanged bit for bit (`pyMagCalc/GAP_STATUS.md`, "The dimer series' T0 sector made
cluster additive").

K8 (`src/dipole_hq.rs`, 2026-10-31) builds the dipole engine's LSWT matrix
H(q) = 2 g H2(q) numerically, replacing pyMagCalc's SymPy Holstein–Primakoff build
(`symbolic.gen_HM`; FeI₂ 35 s → 0.66 s, SW28 26 s → 0.6 s) under `calculation:
{hamiltonian_builder: numeric}` or `auto`, its default. pyMagCalc
(`numeric_hq.HamiltonianTerms`) resolves every term with the spin model's own
builders into monomials in the spin components, coefficients evaluated for the
current parameters. `DipoleHq(n, pos_ouc, spins, rot, monomials)` then does what
`gen_HM` does symbolically: the HP expansion of each factor about its site's
frame, truncated at boson degree 2 with no reordering, the Fourier placement of
`gen_HM`'s lookup, and 2g. It returns a sparse bond model H(q) = Σ_d M_d e^{iq·d}
with `hamiltonians(qs, negate)` (threaded over q), `bond_model()` (the dense form
fMagCalc's `run_sqw_model` takes) and `entries()`. It pickles as its entries, so
pyMagCalc's per-q process pools receive it directly. Two numerical details:
- The √(S/2) of two one-boson factors is applied once, as √(S_a S_b)/2, when they
  pair. √(S/2)² is not S/2 in floating point, and a collinear ferromagnet's H(0)
  would otherwise come out 4e-16 instead of exactly 0.
- Rows of the hole block are negated after assembly.

Pinned in pyMagCalc `tests/test_numeric_hq.py` entry by entry to `HMat_sym` (≤ 5e-15
relative on every buildable shipped dipole config, at random rotations, and on a
triclinic mixed-spin model carrying every term), and here in `tests/test_dipole_hq.py`
to closed forms: the ferromagnetic and Néel chains, an easy-plane ferromagnet's
gapless mode, a Zeeman monomial, the collinear biquadratic's bilinear equivalent
(pyMagCalc `GAP_STATUS.md`, "Rust kernels K8").

### Memory decides how many items run at once

The kernels whose per-item scratch is large (SU(N) LSWT, KPM and SCGA per q; the
dimer series per cluster) run as many items at once as fit a **scratch budget**.
`rsmagcalc.scratch_budget()` reports it as (bytes, source), and the first match wins:

1. the call's own `scratch_budget_gb` (the dimer series takes one);
2. `RSMAGCALC_SCRATCH_GB` in the environment, in GB;
3. half the memory this process can still get: the smaller of `/proc/meminfo`'s
   MemAvailable and the headroom (limit − usage) of every memory-limited cgroup above
   it (v2 `memory.max`, v1 `memory.limit_in_bytes`). A Slurm job's limit is there;
4. 3 GB, where neither file exists (macOS).

Until 2026-10-31 it was a fixed 3 GB, divided by the largest item's scratch. On a
243 GB, 128-core node that left the dimer series on 2 threads at order 8 and 1 at
order 9, whatever `n_workers` said. SU(N) LSWT, KPM and SCGA still cap their pool at
budget / per-q scratch, which is the same for every q of a call. The dimer series
instead charges each cluster its own scratch, admits the clusters largest first,
and starts the next one only when it fits beside those already running; a cluster
larger than the whole budget runs alone. That estimate, (2·order + 3) complex
rows × p blocks on the largest total-Sz sector, plus 10 % and the index arrays, was
measured against peak RSS on chains of 7–9 dimers: the measured peak is 0.55–0.97 of
it. `cluster_pt_batch(..., return_stats=True)` also returns what the pool did:
`threads`, `peak_concurrent`, `budget_bytes`, `budget_source`, `peak_bytes`,
`max_scratch_bytes` and `n_clusters`. The results do not depend on the budget or on
the thread count, bit for bit (`tests/test_scratch_budget.py`).
