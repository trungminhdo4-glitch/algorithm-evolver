"""Regression tests: cached fitness must not survive a curriculum window change.

The damped-oscillation curriculum evaluates generations
<= CURRICULUM_SPLIT_GENERATION on the t <= 1.0 subset and later
generations on the full domain. DEAP caches fitness values and
eaMuPlusLambda re-evaluates only individuals with invalid fitness, so
survivors carried across the boundary used to keep subset-window MSEs
that are incomparable with full-domain scores (selection corruption and,
for reported candidates, wrong discovery metrics).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deap import tools

from problems.damped_oscillation import DampedOscillationProblem
from core.evolution import EvolutionaryEngine

POP = 24
GENS = 23  # crosses the curriculum boundary at generation 21


def _build_engine(problem):
    pset = problem.create_primitive_set()
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,
        population_size=POP,
        cxpb=0.8,
        mutpb=0.2,
        tournsize=5,
        max_height=7,
    )
    engine.toolbox.register("select", tools.selNSGA2)
    return engine


def _run_curriculum_loop(problem, engine):
    """Same shape as experiments/run_oscillation.py, at test scale."""
    population = engine.toolbox.population(n=POP)
    for gen in range(1, GENS + 1):
        problem.current_generation = gen
        problem.invalidate_stale_fitness(population, gen)
        population, _ = engine.run_nsga2(
            generations=1, seed=42 + gen, verbose=False, population=population
        )
    return population


def test_evaluation_window_flips_at_documented_boundary():
    problem = DampedOscillationProblem()
    assert problem.evaluation_window(0) == "initial_rise"
    assert (
        problem.evaluation_window(problem.CURRICULUM_SPLIT_GENERATION) == "initial_rise"
    )
    assert (
        problem.evaluation_window(problem.CURRICULUM_SPLIT_GENERATION + 1)
        == "full_domain"
    )
    assert problem.evaluation_window(10_000) == "full_domain"


def test_evaluate_uses_window_authority_for_data_selection():
    problem = DampedOscillationProblem()
    subset_points = sum(1 for inp, _ in problem.train_data if inp[0] <= 1.0)

    problem.current_generation = problem.CURRICULUM_SPLIT_GENERATION
    early = problem.evaluation_window()
    problem.current_generation = problem.CURRICULUM_SPLIT_GENERATION + 1
    late = problem.evaluation_window()

    assert early == "initial_rise"
    assert late == "full_domain"
    # The subset window is a strict prefix of the training data.
    assert 0 < subset_points < len(problem.train_data)


def test_no_stale_fitness_survives_curriculum_boundary():
    problem = DampedOscillationProblem()
    engine = _build_engine(problem)
    population = _run_curriculum_loop(problem, engine)

    assert problem.evaluation_window() == "full_domain"
    stale = []
    for idx, ind in enumerate(population):
        assert ind.fitness.valid, f"individual {idx} ended with invalid fitness"
        fresh_mse = problem.evaluate(ind, engine.toolbox)[0]
        if not math.isclose(ind.fitness.values[0], fresh_mse, rel_tol=1e-9):
            stale.append((idx, ind.fitness.values[0], fresh_mse))
    assert not stale, f"stale fitness survived the curriculum switch: {stale}"


def test_reported_front_matches_full_domain_truth_after_boundary():
    problem = DampedOscillationProblem()
    engine = _build_engine(problem)
    population = _run_curriculum_loop(problem, engine)

    front = tools.sortNondominated(population, len(population), first_front_only=True)[
        0
    ]
    for rank, ind in enumerate(front):
        fresh_mse = problem.evaluate(ind, engine.toolbox)[0]
        assert math.isclose(ind.fitness.values[0], fresh_mse, rel_tol=1e-9), (
            f"front member {rank} reports {ind.fitness.values[0]} "
            f"but true full-domain MSE is {fresh_mse}"
        )
