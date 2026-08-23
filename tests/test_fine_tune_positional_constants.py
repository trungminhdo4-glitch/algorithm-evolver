"""Regression tests: refine_with_scipy must parameterize constants per occurrence.

utils.fine_tune.refine_with_scipy used expr.atoms(sp.Number) plus a
value-keyed subs() map. atoms() returns a SET of values and sympy carries
signs on Mul, so equal-valued constants at different structural positions
(e.g. amplitude drawn equal to frequency) collapsed into ONE shared fit
parameter and curve_fit could never refine them independently - violating
the documented 'replace all constants with parameters' contract.

The module is loaded directly from its file here because utils/__init__.py
eagerly imports analysis.py, whose NetworkX fallback crashes on cp1252
consoles when networkx is absent (documented finding, separate wave).
"""

import importlib.util
import math
import os

import numpy as np
import sympy as sp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "fine_tune_under_test", os.path.join(_REPO_ROOT, "utils", "fine_tune.py")
)
fine_tune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fine_tune)

refine_with_scipy = fine_tune.refine_with_scipy


def _mse(refined_str, t, truth):
    func = sp.lambdify(sp.Symbol("t"), sp.sympify(refined_str), modules=["numpy"])
    return float(np.mean((func(t) - truth(t)) ** 2))


def test_colliding_floats_get_independent_parameters():
    # Same value at a multiplicative and an additive position.
    t = np.linspace(0.0, 4 * np.pi, 400)
    truth = lambda x: 2.5 * x + 3.0
    refined, consts = refine_with_scipy(
        "2.6*t + 2.6", [[x] for x in t], truth(t), input_names=["t"]
    )
    assert len(consts) == 2, f"expected 2 independent parameters, got {consts}"
    fitted = [float(v) for v in consts.values()]
    assert any(abs(v - 2.5) < 1e-6 for v in fitted), consts
    assert any(abs(v - 3.0) < 1e-6 for v in fitted), consts
    assert _mse(refined, t, truth) < 1e-10


def test_sign_carry_twins_stay_independent():
    # sympy normalizes exp(-0.9*t) so its constant shares a magnitude with the
    # cos frequency; value-keyed substitution degenerated this fit (pinned
    # parameter, covariance warning) instead of refining all three positions.
    t = np.linspace(0.0, 4 * np.pi, 400)
    truth = lambda x: 2.0 * np.exp(-0.6 * x) * np.cos(5.0 * x)
    refined, consts = refine_with_scipy(
        "1.0*exp(-0.9*t)*cos(0.9*t)", [[x] for x in t], truth(t), input_names=["t"]
    )
    assert len(consts) == 3, f"expected 3 independent parameters, got {consts}"
    assert _mse(refined, t, truth) < 1e-8


def test_distinct_constants_refinement_parity_preserved():
    # Pre-existing showcase behavior must survive the fix.
    v = np.linspace(10, 100, 20)
    angle = np.linspace(0.1, 1.4, 20)
    y = (v**2 * np.sin(2 * angle)) / 9.81
    refined, consts = refine_with_scipy(
        "v**1.5*sin(2.0*angle)*0.5",
        list(zip(v, angle)),
        y,
        input_names=["v", "angle"],
    )
    assert len(consts) == 3
    fitted = sorted(float(val) for val in consts.values())
    assert abs(fitted[0] - 1 / 9.81) < 1e-6  # multiplicative constant
    assert all(abs(fitted[i] - 2.0) < 1e-3 for i in (1, 2))  # exponent, freq
    func = sp.lambdify(
        (sp.Symbol("v"), sp.Symbol("angle")), sp.sympify(refined), modules=["numpy"]
    )
    assert float(np.mean((func(v, angle) - y) ** 2)) < 1e-12


def test_integer_structure_untouched():
    refined, consts = refine_with_scipy(
        "2*t + 1", [[x] for x in np.linspace(0, 1, 10)], None, input_names=["t"]
    )
    assert consts == {}
    assert refined == "2*t + 1"


def test_no_float_occurrences_means_no_parameters():
    refined, consts = refine_with_scipy(
        "sin(t)", [[x] for x in np.linspace(0, 1, 10)], None, input_names=["t"]
    )
    assert consts == {}
    assert refined == "sin(t)"
