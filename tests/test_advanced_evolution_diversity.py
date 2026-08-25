import operator
import random

from deap import gp

from core.advanced_evolution import AdvancedEvolutionaryEngine


def test_diversity_injection_is_evaluated_before_population_consumers():
    population_size = 10
    evaluations = 0

    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)

    def evaluate(individual, toolbox):
        nonlocal evaluations
        evaluations += 1
        return (float(len(individual)),)

    engine = AdvancedEvolutionaryEngine(
        pset=pset,
        evaluate_func=evaluate,
        population_size=population_size,
        cxpb=0.0,
        mutpb=0.0,
        bloat_control=False,
        maintain_diversity=True,
        multi_objective=True,
        seed=42,
    )
    engine._calculate_diversity = lambda population: 0.0

    population, logbook, hall_of_fame = engine.run(
        generations=10, verbose=False, progress_bar=False
    )

    assert evaluations == population_size + 1
    assert logbook[-1]["nevals"] == 1
    assert all(ind.fitness.valid for ind in population)
    assert all(len(ind.fitness.values) == 2 for ind in population)
    assert all(ind.fitness.valid for ind in hall_of_fame)


def test_hall_of_fame_observes_best_offspring_before_injection(monkeypatch):
    population_size = 10
    evaluations = 0
    mutations = 0

    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    unique_tree = gp.PrimitiveTree.from_string(
        "add(ARG0, add(ARG0, add(ARG0, add(ARG0, ARG0))))", pset
    )

    def evaluate(individual, toolbox):
        nonlocal evaluations
        evaluations += 1
        error = 0.0 if evaluations == population_size * 10 + 1 else 100.0
        return (error,)

    def mutate(individual):
        nonlocal mutations
        mutations += 1
        if mutations == population_size * 9 + 1:
            individual[0:len(individual)] = unique_tree
        return (individual,)

    engine = AdvancedEvolutionaryEngine(
        pset=pset,
        evaluate_func=evaluate,
        population_size=population_size,
        cxpb=0.0,
        mutpb=1.0,
        bloat_control=False,
        maintain_diversity=True,
        multi_objective=True,
        seed=42,
    )
    engine._calculate_diversity = lambda population: 0.0
    engine.toolbox.register("mutate", mutate)

    original_randint = random.randint
    monkeypatch.setattr(
        random,
        "randint",
        lambda low, high: 0
        if (low, high) == (0, population_size - 1)
        else original_randint(low, high),
    )

    population, _, hall_of_fame = engine.run(
        generations=10, verbose=False, progress_bar=False
    )

    assert evaluations == population_size * 11 + 1
    assert all(ind.fitness.values[0] == 100.0 for ind in population)
    assert hall_of_fame[0].fitness.values[0] == 0.0
