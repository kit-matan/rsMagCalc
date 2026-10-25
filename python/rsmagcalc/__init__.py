"""rsMagCalc -- Rust kernels (PyO3) for pyMagCalc's beyond-LSWT loops.

The contract mirrors fMagCalc's: pyMagCalc never requires this package. It
selects the kernels with `calculation: {kernels: rust}`, checks
``rsmagcalc.backend == "pyo3"``, and otherwise warns and runs its own NumPy
implementation. ``backend`` is ``"pyo3"`` when the compiled core imported and
``None`` when only this pure-Python shell is present (a source checkout that
was never built with maturin).

Which kernels exist, in what order, and what each is pinned to:
``../RUST_KERNELS_PLAN.md`` at the workspace root.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["backend", "build_info", "describe", "classical_energy"]

try:
    from . import _core
except ImportError:  # pragma: no cover - only without a maturin build
    _core = None

backend = "pyo3" if _core is not None else None


def build_info() -> dict:
    """Version, git head, target and profile of the compiled core.

    Reads only what the build baked in (no subprocess, no filesystem): this is
    called on the startup path of every `magcalc run` that asks for the kernels.
    """
    if _core is None:
        return {"version": __version__, "git_head": "", "target": "",
                "profile": "not built"}
    info = dict(_core.build_info())
    info.setdefault("version", __version__)
    return info


def describe() -> str:
    """One line naming this build, for pyMagCalc's run log and `magcalc where`."""
    info = build_info()
    parts = [f"rsmagcalc {info['version']}"]
    if info.get("git_head"):
        parts.append(f"git {info['git_head']}")
    parts.append(info.get("profile", "?"))
    return "(" + ", ".join(parts) + ")"


def classical_energy(H, b, m) -> float:
    """E = 1/2 m^T H m + b^T m, pyMagCalc `classical_dynamics.energy` in Rust."""
    if _core is None:
        raise RuntimeError(
            "rsmagcalc's compiled core is not built; run `pip install -e . "
            "--no-build-isolation` in rsMagCalc/ or `pip install /path/to/rsMagCalc`."
        )
    import numpy as np

    return _core.classical_energy(
        np.ascontiguousarray(H, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
        np.ascontiguousarray(m, dtype=np.float64),
    )
