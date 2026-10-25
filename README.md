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
maturin develop --release        # builds rsmagcalc._core into the active interpreter
python -c "import rsmagcalc; print(rsmagcalc.backend, rsmagcalc.describe())"
pytest                           # parity against pyMagCalc, which must be installed
```

`maturin build --release` produces a wheel. abi3 (`pyo3/abi3-py39`) means one wheel
per platform, importable from every CPython ≥ 3.9. Wheels are built without
`-C target-cpu=native`; set `RUSTFLAGS` for a local build that may use it.

## Status

M0: scaffold. `build_info()` / `describe()` for provenance, and one kernel,
`classical_energy(H, b, m)`, that proves the zero-copy NumPy contract and is pinned
to `magcalc.classical_dynamics.energy`. Kernels K1–K6 follow the plan.
