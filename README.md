# rsMagCalc

Rust kernels, behind PyO3, for the loops in pyMagCalc that are Python-bound per
site or per q: the classical Metropolis sweep and Landau–Lifshitz integrator, the
CP^(N−1) sampler, SU(N) LSWT assembly and Bogoliubov diagonalization, the KPM
Chebyshev recursion, and SCGA. The scope, the order, and the oracle each kernel is
pinned to are in `../RUST_KERNELS_PLAN.md` at the workspace root. This is **not** a
port of pyMagCalc: the config, symmetry, symbolic model builder, runner, fitting and
plotting stay in Python, and the dipole LSWT hot path stays with fMagCalc.

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

```bash
cd rsMagCalc
pip install maturin
pip install -e . --no-build-isolation   # compiles rsmagcalc._core; editable, like pyMagCalc
python -c "import rsmagcalc; print(rsmagcalc.backend, rsmagcalc.describe())"
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
K5–K6 follow the plan.
