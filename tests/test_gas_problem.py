import random

from deap import gp

from core.evolution import EvolutionaryEngine
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


def test_cached_fitness_tracks_dimensional_penalty_factor():
    problem = IdealGasProblem()
    generation = {"value": 80}
    evaluated_ids = []

    def evaluate(individual, toolbox):
        evaluated_ids.append(id(individual))
        return problem.evaluate(individual, toolbox, generation=generation["value"])

    engine = EvolutionaryEngine(
        pset=problem.create_primitive_set(),
        evaluate_func=evaluate,
        population_size=6,
    )
    random.seed(42)
    population = engine.toolbox.population(n=engine.population_size)

    problem.invalidate_stale_fitness(population, generation["value"])
    population, _ = engine.run_nsga2(
        generations=0, seed=42, verbose=False, population=population
    )
    assert all(individual.fitness.valid for individual in population)

    generation["value"] = 120
    assert problem.penalty_factor(80) == 0.8
    assert problem.penalty_factor(120) == 1.0
    problem.invalidate_stale_fitness(population, generation["value"])
    assert all(not individual.fitness.valid for individual in population)

    evaluated_ids.clear()
    reused_ids = [id(individual) for individual in population]
    population, _ = engine.run_nsga2(
        generations=0, seed=42, verbose=False, population=population
    )
    assert evaluated_ids == reused_ids
    assert all(individual.fitness.valid for individual in population)

    evaluated_ids.clear()
    cached_fitness = [individual.fitness.values for individual in population]
    generation["value"] = 160
    assert problem.penalty_factor(160) == 1.0
    problem.invalidate_stale_fitness(population, generation["value"])
    assert all(individual.fitness.valid for individual in population)

    population, _ = engine.run_nsga2(
        generations=0, seed=42, verbose=False, population=population
    )
    assert evaluated_ids == []
    assert [individual.fitness.values for individual in population] == cached_fitness
