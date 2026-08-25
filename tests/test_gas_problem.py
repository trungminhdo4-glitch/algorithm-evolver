from deap import gp

from core.physics import DimensionalChecker
from problems.ideal_gas import IdealGasProblem


def test_documented_gas_law_has_temperature_dimension():
    problem = IdealGasProblem()
    pset = problem.create_primitive_set()
    individual = gp.PrimitiveTree.from_string(
        "protected_div(mul(P, V), mul(n, R))", pset
    )

    unit, consistent = DimensionalChecker(problem.pset_units).check_tree(individual)

    assert consistent
    assert unit == problem.target_unit
