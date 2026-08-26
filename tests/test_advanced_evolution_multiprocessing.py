import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from deap import creator, gp

from core import advanced_evolution
from core.advanced_evolution import (
    AdvancedEvolutionaryEngine,
    _batch_count,
    _serialize_batch,
    _split_batches,
)


SCENARIO_SCRIPT = Path(__file__).with_name(
    "advanced_evolution_spawn_scenario.py"
)
SCENARIO_TIMEOUT = 45


def _terminate_process_tree(process):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.communicate()


def _make_engine(multi_objective=False, n_jobs=1):
    pset = gp.PrimitiveSet(
        "MULTI" if multi_objective else "SINGLE",
        1,
    )
    pset.addPrimitive(lambda left, right: left + right, 2, name="local_add")

    def evaluate(individual, toolbox):
        return float(len(individual)),

    return AdvancedEvolutionaryEngine(
        pset,
        evaluate,
        population_size=4,
        n_jobs=n_jobs,
        multi_objective=multi_objective,
        maintain_diversity=False,
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "single",
        "multi",
        "repeated",
        "batch_order",
        "toolbox_parity",
        "same_mode_concurrent",
        "failure_cleanup",
        "later_failure_cleanup",
    ],
)
def test_spawn_multiprocessing_scenarios(scenario):
    popen_options = {}
    if os.name == "nt":
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_options["start_new_session"] = True

    process = subprocess.Popen(
        [sys.executable, str(SCENARIO_SCRIPT), scenario],
        cwd=SCENARIO_SCRIPT.parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen_options,
    )
    try:
        stdout, stderr = process.communicate(timeout=SCENARIO_TIMEOUT)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        pytest.fail(
            f"spawn scenario {scenario!r} exceeded {SCENARIO_TIMEOUT}s"
        )

    assert process.returncode == 0, (
        f"spawn scenario {scenario!r} failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert f"scenario {scenario} passed" in stdout


@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_objective_modes_use_distinct_creator_classes_in_either_order(order):
    engines = [_make_engine(multi_objective=mode) for mode in order]
    by_mode = dict(zip(order, engines))
    single = by_mode[False].toolbox.individual()
    multi = by_mode[True].toolbox.individual()

    assert type(single) is by_mode[False].individual_type
    assert type(single.fitness) is by_mode[False].fitness_type
    assert type(multi) is by_mode[True].individual_type
    assert type(multi.fitness) is by_mode[True].fitness_type
    assert single.pset is by_mode[False].pset
    assert multi.pset is by_mode[True].pset
    assert type(single) is not type(multi)
    assert len(single.fitness.weights) == 1
    assert len(multi.fitness.weights) == 2


@pytest.mark.parametrize("multi_objective", [False, True])
def test_same_mode_engines_have_unique_types_and_psets(multi_objective):
    first = _make_engine(multi_objective=multi_objective)
    second = _make_engine(multi_objective=multi_objective)
    first_individual = first.toolbox.individual()
    second_individual = second.toolbox.individual()

    assert first.individual_type is not second.individual_type
    assert first.fitness_type is not second.fitness_type
    assert type(first_individual) is first.individual_type
    assert type(second_individual) is second.individual_type
    assert first_individual.pset is first.pset
    assert second_individual.pset is second.pset

    for engine in (first, second):
        population, _, _ = engine.run(
            generations=0, verbose=False, progress_bar=False
        )
        assert all(ind.fitness.valid for ind in population)


def test_creator_namespace_does_not_grow_and_old_engines_remain_usable():
    creator_names = set(vars(creator))
    engines = [_make_engine(multi_objective=index % 2 == 0) for index in range(50)]

    assert set(vars(creator)) == creator_names
    for engine in (engines[0], engines[-1]):
        population, _, _ = engine.run(
            generations=0, verbose=False, progress_bar=False
        )
        assert len(population) == 4
        assert all(ind.fitness.valid for ind in population)


def test_batch_serialization_restores_creator_name_collisions(monkeypatch):
    engine = _make_engine()
    population = engine.toolbox.population(n=2)
    fitness_name = engine.fitness_type.__name__
    individual_name = engine.individual_type.__name__
    previous_fitness = object()
    previous_individual = object()
    setattr(creator, fitness_name, previous_fitness)
    setattr(creator, individual_name, previous_individual)

    try:
        payload = _serialize_batch(
            engine.toolbox,
            population,
            engine.fitness_type,
            engine.individual_type,
        )
        assert payload
        assert getattr(creator, fitness_name) is previous_fitness
        assert getattr(creator, individual_name) is previous_individual

        def fail_serialization(value):
            raise RuntimeError("serialization failed")

        monkeypatch.setattr(
            advanced_evolution.cloudpickle,
            "dumps",
            fail_serialization,
        )
        with pytest.raises(RuntimeError, match="serialization failed"):
            _serialize_batch(
                engine.toolbox,
                population,
                engine.fitness_type,
                engine.individual_type,
            )
        assert getattr(creator, fitness_name) is previous_fitness
        assert getattr(creator, individual_name) is previous_individual
    finally:
        delattr(creator, individual_name)
        delattr(creator, fitness_name)


def test_concurrent_batch_serialization_is_locked_and_restores_names(
    monkeypatch,
):
    engines = [_make_engine(), _make_engine()]
    populations = [engine.toolbox.population(n=2) for engine in engines]
    original_dumps = advanced_evolution.cloudpickle.dumps
    state_lock = threading.Lock()
    active_serializers = 0
    max_active_serializers = 0

    def observed_dumps(value):
        nonlocal active_serializers, max_active_serializers
        with state_lock:
            active_serializers += 1
            max_active_serializers = max(
                max_active_serializers, active_serializers
            )
        try:
            time.sleep(0.05)
            return original_dumps(value)
        finally:
            with state_lock:
                active_serializers -= 1

    monkeypatch.setattr(advanced_evolution.cloudpickle, "dumps", observed_dumps)
    with ThreadPoolExecutor(max_workers=2) as thread_pool:
        payloads = list(thread_pool.map(
            lambda pair: _serialize_batch(
                pair[0].toolbox,
                pair[1],
                pair[0].fitness_type,
                pair[0].individual_type,
            ),
            zip(engines, populations),
        ))

    assert all(payloads)
    assert max_active_serializers == 1
    for engine in engines:
        assert not hasattr(creator, engine.fitness_type.__name__)
        assert not hasattr(creator, engine.individual_type.__name__)


def test_deterministic_balanced_batches():
    assert _split_batches(range(10), 3) == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert _split_batches([], 3) == []
    assert _batch_count(8, 2) == 4
    assert _batch_count(100, 61) == 50
    assert _batch_count(1000, 61) == 244
    assert _batch_count(100, 61) < 100


@pytest.mark.parametrize("n_jobs", [0, 1])
def test_zero_and_one_job_use_serial_map(monkeypatch, n_jobs):
    def unexpected_executor(*args, **kwargs):
        raise AssertionError("serial run created a process executor")

    monkeypatch.setattr(advanced_evolution, "ProcessPoolExecutor",
                        unexpected_executor)
    engine = _make_engine(n_jobs=n_jobs)
    population, _, _ = engine.run(
        generations=0, verbose=False, progress_bar=False
    )

    assert len(population) == 4
    assert all(ind.fitness.valid for ind in population)


def test_minus_one_uses_default_spawn_executor_for_large_cpu_count(monkeypatch):
    captured = {}

    class FakeExecutor:
        _max_workers = 61

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def shutdown(self):
            captured["shutdown"] = True

    cpu_count_calls = 0

    def large_cpu_count():
        nonlocal cpu_count_calls
        cpu_count_calls += 1
        return 128

    monkeypatch.setattr(advanced_evolution, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(advanced_evolution.mp, "cpu_count", large_cpu_count)
    engine = _make_engine(n_jobs=-1)
    monkeypatch.setattr(engine, "_run", lambda *args: "completed")

    assert engine.run(verbose=False, progress_bar=False) == "completed"
    assert engine.n_jobs == -1
    assert cpu_count_calls == 0
    assert captured["max_workers"] is None
    assert captured["mp_context"].get_start_method() == "spawn"
    assert captured["shutdown"] is True


def test_progress_close_failure_does_not_mask_run_error_or_skip_shutdown(
    monkeypatch,
):
    state = {"closed": False, "shutdown": False}

    class PrimaryRunError(RuntimeError):
        pass

    class FakeExecutor:
        _max_workers = 2

        def __init__(self, **kwargs):
            pass

        def shutdown(self):
            state["shutdown"] = True

    class FailingProgress:
        def close(self):
            state["closed"] = True
            raise RuntimeError("cosmetic close failure")

    primary_error = PrimaryRunError("primary run failure")
    engine = _make_engine(n_jobs=2)

    def fail_run(*args):
        args[-1]["bar"] = FailingProgress()
        raise primary_error

    monkeypatch.setattr(advanced_evolution, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(engine, "_run", fail_run)

    with pytest.raises(PrimaryRunError) as raised:
        engine.run(verbose=False, progress_bar=True)

    assert raised.value is primary_error
    assert state == {"closed": True, "shutdown": True}
