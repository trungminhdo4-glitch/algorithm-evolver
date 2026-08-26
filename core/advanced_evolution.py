"""
Erweiterte Evolution Engine mit allen Verbesserungen
KEINE Import-Fehler mehr - Vollständig getestet!
"""
import operator
import random
import itertools
import sys
import threading
import numpy as np
from deap import base, creator, tools, gp
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import cloudpickle
from tqdm import tqdm
import sympy as sp


_CREATOR_COUNTER = itertools.count()
_CREATOR_LOCK = threading.Lock()
_BATCHES_PER_WORKER = 4


def _evaluate_registered(individual, evaluate_func, toolbox,
                         multi_objective, bloat_control):
    """Evaluate through a worker-safe toolbox registration."""
    error = evaluate_func(individual, toolbox)[0]
    size = len(individual)

    if bloat_control and size > 100:
        error += (size - 100) * 0.01
    if multi_objective:
        return error, size
    return error,


def _evaluate_batch_in_worker(serialized_batch):
    """Evaluate a batch whose shared identities were restored in one load."""
    toolbox, individuals = cloudpickle.loads(serialized_batch)
    return [toolbox.evaluate(individual) for individual in individuals]


def _serialize_batch(toolbox, individuals, fitness_type, individual_type):
    """Expose DEAP creator types only while its reducer builds the payload."""
    fitness_name = fitness_type.__name__
    individual_name = individual_type.__name__
    with _CREATOR_LOCK:
        had_fitness = hasattr(creator, fitness_name)
        had_individual = hasattr(creator, individual_name)
        previous_fitness = getattr(creator, fitness_name, None)
        previous_individual = getattr(creator, individual_name, None)
        setattr(creator, fitness_name, fitness_type)
        setattr(creator, individual_name, individual_type)
        try:
            return cloudpickle.dumps((toolbox, individuals))
        finally:
            if had_individual:
                setattr(creator, individual_name, previous_individual)
            elif hasattr(creator, individual_name):
                delattr(creator, individual_name)
            if had_fitness:
                setattr(creator, fitness_name, previous_fitness)
            elif hasattr(creator, fitness_name):
                delattr(creator, fitness_name)


def _split_batches(individuals, batch_count):
    """Split individuals into stable, balanced, contiguous batches."""
    individuals = list(individuals)
    if not individuals:
        return []

    batch_count = min(batch_count, len(individuals))
    batch_size, extra = divmod(len(individuals), batch_count)
    batches = []
    start = 0
    for index in range(batch_count):
        end = start + batch_size + (index < extra)
        batches.append(individuals[start:end])
        start = end
    return batches


def _batch_count(item_count, worker_count):
    """Bound work-stealing batches while avoiding one payload per item."""
    if item_count == 0:
        return 0
    return min(
        worker_count * _BATCHES_PER_WORKER,
        max(1, item_count // 2),
    )


class AdvancedEvolutionaryEngine:
    def __init__(self, pset, evaluate_func, 
                 population_size=100, cxpb=0.6, mutpb=0.3,
                 tournsize=5, max_height=10, n_jobs=1,
                 multi_objective=False, bloat_control=True,
                 maintain_diversity=True, seed=42):
        
        self.pset = pset
        self.evaluate_func = evaluate_func
        self.population_size = population_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.tournsize = tournsize
        self.max_height = max_height
        self.n_jobs = n_jobs
        self.multi_objective = multi_objective
        self.bloat_control = bloat_control
        self.maintain_diversity = maintain_diversity
        self.seed = seed
        
        self._setup_creator()
        self._setup_toolbox()
        self._setup_statistics()
    
    def _setup_creator(self):
        """Setup DEAP Creator Klassen"""
        mode = "Multi" if self.multi_objective else "Single"
        weights = (-1.0, -0.1) if self.multi_objective else (-1.0,)

        with _CREATOR_LOCK:
            while True:
                suffix = next(_CREATOR_COUNTER)
                fitness_name = f"AdvancedFitness{mode}_{suffix}"
                individual_name = f"AdvancedIndividual{mode}_{suffix}"
                if not hasattr(creator, fitness_name) and not hasattr(
                    creator, individual_name
                ):
                    break

            creator.create(fitness_name, base.Fitness, weights=weights)
            self.fitness_type = getattr(creator, fitness_name)
            creator.create(individual_name, gp.PrimitiveTree,
                           fitness=self.fitness_type, pset=self.pset)
            self.individual_type = getattr(creator, individual_name)
            delattr(creator, individual_name)
            delattr(creator, fitness_name)
    
    def _setup_toolbox(self):
        """Setup DEAP Toolbox"""
        self.toolbox = base.Toolbox()
        
        # Expressions-Generierung
        self.toolbox.register("expr", gp.genHalfAndHalf,
                             pset=self.pset, min_=1, max_=3)
        
        # Individuen-Generierung
        self.toolbox.register("individual", tools.initIterate,
                             self.individual_type, self.toolbox.expr)
        
        # Population-Generierung
        self.toolbox.register("population", tools.initRepeat,
                             list, self.toolbox.individual)
        
        # Evaluierungsfunktion. The toolbox cycle is intentional and does not
        # include the engine, allowing the complete toolbox to reach workers.
        self.toolbox.register(
            "evaluate",
            _evaluate_registered,
            evaluate_func=self.evaluate_func,
            toolbox=self.toolbox,
            multi_objective=self.multi_objective,
            bloat_control=self.bloat_control,
        )
        
        # Compiler
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        # Selektions- und Variationsoperatoren
        self.toolbox.register("select", tools.selTournament,
                             tournsize=self.tournsize)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform,
                             expr=self.toolbox.expr_mut, pset=self.pset)
        
        # Größen-Limits
        self.toolbox.decorate("mate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.max_height))
        self.toolbox.decorate("mutate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.max_height))
    
    def _setup_statistics(self):
        """Setup für Statistiken"""
        self.stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0] if not self.multi_objective else ind.fitness.values[0])
        self.stats_size = tools.Statistics(len)
        self.stats_height = tools.Statistics(operator.attrgetter("height"))
        
        self.stats = tools.MultiStatistics(
            fitness=self.stats_fit,
            size=self.stats_size,
            height=self.stats_height
        )
        
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.stats.register("std", np.std)
    
    def _calculate_diversity(self, population):
        """Berechnet die Diversität der Population"""
        if len(population) < 2:
            return 0
        
        try:
            from scipy.spatial.distance import pdist
            features = []
            for ind in population:
                features.append([
                    len(ind),
                    ind.height,
                    len(set([node.name for node in ind]))
                ])
            
            features = np.array(features)
            if len(features) > 1:
                distances = pdist(features)
                return float(np.mean(distances))
        except:
            pass
        return 0
    
    def run(self, generations=40, verbose=True, progress_bar=True):
        """Haupt-Evolutionsschleife"""
        executor = None
        progress_state = {"bar": None}
        try:
            if self.n_jobs == -1 or self.n_jobs > 1:
                executor = ProcessPoolExecutor(
                    max_workers=None if self.n_jobs == -1 else self.n_jobs,
                    mp_context=mp.get_context("spawn"),
                )
                worker_count = executor._max_workers

                def evaluate_all(individuals):
                    individuals = list(individuals)
                    batches = _split_batches(
                        individuals,
                        _batch_count(len(individuals), worker_count),
                    )
                    payloads = (
                        _serialize_batch(
                            self.toolbox,
                            batch,
                            self.fitness_type,
                            self.individual_type,
                        )
                        for batch in batches
                    )
                    batch_fitnesses = executor.map(
                        _evaluate_batch_in_worker, payloads
                    )
                    return list(itertools.chain.from_iterable(batch_fitnesses))
            else:
                def evaluate_all(individuals):
                    return self.toolbox.map(self.toolbox.evaluate, individuals)

            return self._run(
                generations,
                verbose,
                progress_bar,
                evaluate_all,
                progress_state,
            )
        finally:
            primary_error = sys.exc_info()[1]
            close_error = None
            shutdown_error = None
            try:
                if progress_state["bar"] is not None:
                    progress_state["bar"].close()
            except BaseException as error:
                close_error = error
            try:
                if executor is not None:
                    executor.shutdown()
            except BaseException as error:
                shutdown_error = error

            if primary_error is None:
                if shutdown_error is not None:
                    raise shutdown_error
                if close_error is not None:
                    raise close_error

    def _run(self, generations, verbose, progress_bar, evaluate_all,
             progress_state):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialisiere Population
        population = self.toolbox.population(n=self.population_size)
        hall_of_fame = tools.HallOfFame(1)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (self.stats.fields if hasattr(self.stats, 'fields') else [])
        
        # Initiale Evaluierung
        fitnesses = evaluate_all(population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        record = self.stats.compile(population)
        logbook.record(gen=0, nevals=len(population), **record)
        
        if verbose:
            print(f"Gen 0: Fitness={record['fitness']['min']:.4f}, Size={record['size']['avg']:.1f}")
        
        # Progress Bar
        if progress_bar:
            pbar = tqdm(total=generations, desc="Generationen")
            progress_state["bar"] = pbar
        
        # Evolution
        for gen in range(1, generations + 1):
            # Selektion
            if self.multi_objective:
                # Für Multi-Objective: NSGA-II Selektion
                offspring = tools.selNSGA2(population, len(population))
            else:
                offspring = self.toolbox.select(population, len(population))
            
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Bloat Control - Größenbegrenzung
            if self.bloat_control:
                for ind in offspring:
                    if len(ind) > 150:
                        # Kürze zu große Individuen
                        new_ind = gp.PrimitiveTree(ind[:100])
                        ind[:] = new_ind[:]
                        if hasattr(ind.fitness, 'values'):
                            del ind.fitness.values

            # Evaluiere neue Individuen
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = evaluate_all(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Ersetze Population
            population[:] = offspring

            # Hall of Fame aktualisieren
            hall_of_fame.update(population)

            # Diversitätserhaltung
            if self.maintain_diversity and gen % 10 == 0:
                diversity = self._calculate_diversity(population)
                if diversity < 5.0:
                    # Füge neue zufällige Individuen hinzu
                    n_new = max(1, int(self.population_size * 0.1))
                    for _ in range(n_new):
                        idx = random.randint(0, len(population)-1)
                        population[idx] = self.toolbox.individual()

            # Evaluiere injizierte Individuen
            injected_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = evaluate_all(injected_ind)
            for ind, fit in zip(injected_ind, fitnesses):
                ind.fitness.values = fit

            if injected_ind:
                hall_of_fame.update(population)
            
            # Logging
            record = self.stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind) + len(injected_ind), **record)
            
            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: Fitness={record['fitness']['min']:.4f}, "
                      f"Size={record['size']['avg']:.1f}")
            
            if progress_bar:
                pbar.update(1)
        
        # Speichere Logbook für Visualisierung
        self.logbook = logbook
        
        return population, logbook, hall_of_fame
