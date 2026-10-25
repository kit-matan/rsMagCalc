"""The compiled core is present and names itself.

These do not skip: a checkout of this repo that is not built is a broken
checkout, and a skip here would be the silent-hole shape pyMagCalc's notes warn
about (`pyMagCalc/CLAUDE.md`, "A skipped module is a silent hole").
"""
import rsmagcalc


def test_compiled_core_is_present():
    assert rsmagcalc.backend == "pyo3", (
        "rsmagcalc imported without its compiled core; build it with "
        "`maturin develop --release` in rsMagCalc/."
    )


def test_build_info_names_the_build():
    info = rsmagcalc.build_info()
    assert info["version"] == rsmagcalc.__version__
    assert info["profile"] in ("release", "debug")
    assert info["target"]  # the Rust target triple, set by build.rs


def test_describe_is_one_line_for_the_run_log():
    line = rsmagcalc.describe()
    assert "\n" not in line
    assert line.startswith("(rsmagcalc ")
