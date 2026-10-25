"""Test-session guard: never open a GUI window.

The same guard as `pyMagCalc/tests/conftest.py` (separate repo, so a copy): the
parity tests import pyMagCalc, whose plotting layer calls `plt.show()` on any
non-Agg backend, and a native window blocks an unattended run forever.
"""
import os

if not os.environ.get("MAGCALC_TEST_GUI"):
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except ImportError:
        pass
    os.environ["MAGCALC_NO_GUI"] = "1"


def pytest_runtest_setup(item):
    if os.environ.get("MAGCALC_TEST_GUI"):
        return
    try:
        import matplotlib
    except ImportError:
        return
    if not matplotlib.get_backend().lower().startswith("agg"):
        matplotlib.use("Agg", force=True)
