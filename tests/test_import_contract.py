"""Import-contract regression tests (Wave 8).

Contract under test:
  * Importing `utils` or any utils submodule performs NO console output at
    import time and succeeds regardless of optional visualization
    dependencies or the stdio encoding.
  * Unavailability of an optional capability is only reported when the
    relevant function is called (ResultAnalyzer.plot_tree).
  * Public re-exports of utils/__init__.py stay intact.

Tests spawn isolated subprocesses with a forced PYTHONIOENCODING so the
cp1252 failure mode (UnicodeEncodeError on redirected stdio) is simulated
portably; optional modules are blocked via sys.modules[...] = None instead
of mutating the environment.
"""

import os
import subprocess
import sys

import importlib.util

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BLOCK_NX = "import sys; sys.modules['networkx']=None; "


def _run(code, encoding="cp1252", blocked=""):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONIOENCODING", "PYTHONUTF8")
    }
    env["PYTHONIOENCODING"] = encoding
    proc = subprocess.run(
        [sys.executable, "-c", blocked + code],
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    return (
        proc.returncode,
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
    )


def test_import_utils_silent_without_networkx_cp1252():
    rc, out, err = _run("import utils", blocked=BLOCK_NX)
    assert rc == 0, f"stderr:\n{err}"
    assert out == "", f"unexpected import-time stdout: {out!r}"


def test_import_utils_fine_tune_independent_of_analysis_optional_deps():
    rc, out, err = _run("import utils.fine_tune; print('FT_OK')", blocked=BLOCK_NX)
    assert rc == 0, f"stderr:\n{err}"
    assert out.strip() == "FT_OK"


def test_import_emits_no_output_normal_env_utf8():
    rc, out, err = _run("import utils", encoding="utf-8")
    assert rc == 0, f"stderr:\n{err}"
    assert out == "", f"import must be silent, got: {out!r}"


def test_missing_required_dep_degrades_gracefully_for_unrelated_submodule():
    # matplotlib missing: analysis cannot load, but importing fine_tune via
    # the package must still succeed (documented broad-except degradation).
    rc, out, err = _run(
        "import utils.fine_tune; print('FT_OK')",
        blocked="import sys; sys.modules['matplotlib']=None; ",
    )
    assert rc == 0, f"stderr:\n{err}"
    assert out.strip() == "FT_OK"


def test_plot_tree_reports_unavailability_only_at_call_time():
    spec = importlib.util.spec_from_file_location(
        "analysis_under_test", os.path.join(_REPO_ROOT, "utils", "analysis.py")
    )
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)

    analysis.HAS_NETWORKX = False
    result = analysis.ResultAnalyzer.plot_tree(object())
    assert result is None


def test_public_reexports_intact():
    code = (
        "from utils import ResultAnalyzer, plot_fitness, "
        "plot_tree_visualization, create_analysis_report; "
        "from utils.analysis import ResultAnalyzer as RA2; "
        "from utils.fine_tune import refine_with_scipy; print('API_OK')"
    )
    rc, out, err = _run(code, encoding="utf-8")
    assert rc == 0, f"stderr:\n{err}"
    assert out.strip() == "API_OK"
