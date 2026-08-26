import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deap import gp

from core import advanced_evolution
from core.advanced_evolution import AdvancedEvolutionaryEngine


def _make_engine(multi_objective=False, fail=False, fail_later=False,
                 label="primary", n_jobs=2):
    captured = {}
    original_fitness = (123.0, 7.0) if multi_objective else (123.0,)
    fitness_weights = (-1.0, -0.1) if multi_objective else (-1.0,)
    original_weighted_fitness = tuple(
        value * weight
        for value, weight in zip(original_fitness, fitness_weights)
    )
    worker_marker = {"source": label, "values": (1, 2, 3)}

    def local_add(left, right):
        return left + right

    pset = gp.PrimitiveSet(f"MAIN_{label}", 1)
    pset.addPrimitive(local_add, 2, name="local_add")

    def local_evaluate(individual, toolbox):
        assert type(individual) is captured["individual_type"]
        assert type(individual.fitness) is captured["fitness_type"]
        assert toolbox is captured["toolbox"]
        assert individual.pset is type(individual).pset
        assert individual.pset is captured["pset"]
        assert individual.pset is toolbox.compile.keywords["pset"]
        assert individual.pset.context["local_add"](2, 3) == 5
        assert individual.worker_marker is worker_marker
        for registration in (
            "compile",
            "evaluate",
            "select",
            "mate",
            "mutate",
            "sequence_score",
        ):
            assert hasattr(toolbox, registration)
        if fail or (fail_later and getattr(individual, "fail_later", False)):
            raise RuntimeError("intentional worker failure")
        assert individual.fitness.worker_marker is worker_marker
        assert individual.fitness.wvalues == original_weighted_fitness
        function = toolbox.compile(expr=individual)
        function(1.0)
        return toolbox.sequence_score(individual.sequence),

    engine = AdvancedEvolutionaryEngine(
        pset=pset,
        evaluate_func=local_evaluate,
        population_size=8,
        cxpb=0.0,
        mutpb=0.0,
        n_jobs=n_jobs,
        multi_objective=multi_objective,
        bloat_control=False,
        maintain_diversity=False,
        seed=17,
    )
    captured.update(
        individual_type=engine.individual_type,
        fitness_type=engine.fitness_type,
        pset=pset,
    )
    engine.toolbox.register("sequence_score", float)

    def add_evaluation_offset(evaluate):
        def evaluate_with_offset(*args, **kwargs):
            fitness = evaluate(*args, **kwargs)
            return (fitness[0] + 0.25,) + fitness[1:]

        return evaluate_with_offset

    engine.toolbox.decorate("evaluate", add_evaluation_offset)
    if fail_later:
        def mark_for_failure(individual):
            individual.worker_marker = worker_marker
            individual.fail_later = True
            return individual,

        engine.toolbox.register("mutate", mark_for_failure)
        engine.mutpb = 1.0

    captured["toolbox"] = engine.toolbox
    original_population = engine.toolbox.population

    def population_with_custom_attributes(n):
        population = original_population(n=n)
        for sequence, individual in enumerate(population):
            individual.sequence = sequence
            individual.worker_marker = worker_marker
            individual.fitness.worker_marker = worker_marker
            individual.fitness.values = original_fitness
        return population

    engine.toolbox.register("population", population_with_custom_attributes)
    return engine


def _run(engine, generations=2, progress_bar=False):
    return engine.run(
        generations=generations, verbose=False, progress_bar=progress_bar
    )


def _snapshot(result):
    population, logbook, hall_of_fame = result
    return (
        [(str(ind), ind.fitness.values) for ind in population],
        list(logbook),
        [(str(ind), ind.fitness.values) for ind in hall_of_fame],
    )


def _assert_result(result, objective_count):
    population, logbook, hall_of_fame = result
    assert len(population) == 8
    assert len(logbook) == 3
    assert hall_of_fame
    assert all(len(ind.fitness.values) == objective_count for ind in population)
    if objective_count == 2:
        assert all(
            abs(ind.fitness.values[1] - len(ind)) < 1e-9
            for ind in population
        )


def _run_scenario(name):
    if name == "single":
        _assert_result(_run(_make_engine()), 1)
    elif name == "multi":
        _assert_result(_run(_make_engine(multi_objective=True)), 2)
    elif name == "repeated":
        engine = _make_engine()
        first = _run(engine)
        second = _run(engine)
        _assert_result(first, 1)
        _assert_result(second, 1)
        assert _snapshot(first) == _snapshot(second)
    elif name == "batch_order":
        population, _, _ = _run(_make_engine(), generations=0)
        assert [ind.fitness.values[0] for ind in population] == [
            value + 0.25 for value in range(8)
        ]
    elif name == "toolbox_parity":
        serial = _run(_make_engine(label="serial", n_jobs=1), generations=0)
        parallel = _run(
            _make_engine(label="parallel", n_jobs=2), generations=0
        )
        assert [ind.fitness.values for ind in serial[0]] == [
            ind.fitness.values for ind in parallel[0]
        ]
    elif name == "same_mode_concurrent":
        engines = [
            _make_engine(label="first"),
            _make_engine(label="second"),
        ]
        assert engines[0].individual_type is not engines[1].individual_type
        assert engines[0].pset is not engines[1].pset
        with ThreadPoolExecutor(max_workers=2) as thread_pool:
            results = list(thread_pool.map(_run, engines))
        for result in results:
            _assert_result(result, 1)
    elif name == "failure_cleanup":
        engine = _make_engine(fail=True)
        try:
            _run(engine)
        except RuntimeError as error:
            assert "intentional worker failure" in str(error)
        else:
            raise AssertionError("worker evaluation did not fail")

        deadline = time.monotonic() + 5.0
        while mp.active_children() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not mp.active_children(), "worker processes survived failed run"
    elif name == "later_failure_cleanup":
        class TrackingProgress:
            def __init__(self):
                self.closed = False

            def update(self, amount):
                pass

            def close(self):
                self.closed = True
                raise RuntimeError("cosmetic progress close failure")

        progress = TrackingProgress()
        original_tqdm = advanced_evolution.tqdm
        advanced_evolution.tqdm = lambda **kwargs: progress
        try:
            try:
                _run(
                    _make_engine(fail_later=True),
                    generations=2,
                    progress_bar=True,
                )
            except RuntimeError as error:
                assert "intentional worker failure" in str(error)
            else:
                raise AssertionError("later worker evaluation did not fail")
        finally:
            advanced_evolution.tqdm = original_tqdm

        assert progress.closed, "progress bar survived failed run"
        deadline = time.monotonic() + 5.0
        while mp.active_children() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not mp.active_children(), "worker processes survived failed run"
    else:
        raise AssertionError(f"unknown scenario: {name}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    _run_scenario(sys.argv[1])
    print(f"scenario {sys.argv[1]} passed")
