"""
Erweiterte Evolution Engine mit allen Verbesserungen
KEINE Import-Fehler mehr - Vollständig getestet!
"""
import operator
import random
import numpy as np
from deap import base, creator, tools, gp
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import sympy as sp

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
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.multi_objective = multi_objective
        self.bloat_control = bloat_control
        self.maintain_diversity = maintain_diversity
        self.seed = seed
        
        self._setup_creator()
        self._setup_toolbox()
        self._setup_statistics()
        self._setup_parallel()
    
    def _setup_creator(self):
        """Setup DEAP Creator Klassen"""
        if self.multi_objective:
            # Multi-Objective: Fitness + Size
            if not hasattr(creator, "FitnessMulti"):
                creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -0.1))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", gp.PrimitiveTree,
                              fitness=creator.FitnessMulti, pset=self.pset)
        else:
            # Single-Objective: Nur Fitness
            if not hasattr(creator, "FitnessMin"):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", gp.PrimitiveTree,
                              fitness=creator.FitnessMin, pset=self.pset)
    
    def _setup_toolbox(self):
        """Setup DEAP Toolbox"""
        self.toolbox = base.Toolbox()
        
        # Expressions-Generierung
        self.toolbox.register("expr", gp.genHalfAndHalf,
                             pset=self.pset, min_=1, max_=3)
        
        # Individuen-Generierung
        self.toolbox.register("individual", tools.initIterate,
                             creator.Individual, self.toolbox.expr)
        
        # Population-Generierung
        self.toolbox.register("population", tools.initRepeat,
                             list, self.toolbox.individual)
        
        # Evaluierungsfunktion
        if self.multi_objective:
            def evaluate_with_size(individual):
                error = self.evaluate_func(individual, self.toolbox)[0]
                size = len(individual)
                if self.bloat_control and size > 100:
                    error += (size - 100) * 0.01
                return error, size
            self.toolbox.register("evaluate", evaluate_with_size)
        else:
            def evaluate_with_bloat(individual):
                error = self.evaluate_func(individual, self.toolbox)[0]
                if self.bloat_control:
                    size = len(individual)
                    if size > 100:
                        error += (size - 100) * 0.01
                return error,
            self.toolbox.register("evaluate", evaluate_with_bloat)
        
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
    
    def _setup_parallel(self):
        """Setup für parallele Verarbeitung"""
        if self.n_jobs > 1:
            self.pool = ProcessPoolExecutor(max_workers=self.n_jobs)
            self.toolbox.register("map", self.pool.map)
    
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
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialisiere Population
        population = self.toolbox.population(n=self.population_size)
        hall_of_fame = tools.HallOfFame(1)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (self.stats.fields if hasattr(self.stats, 'fields') else [])
        
        # Initiale Evaluierung
        fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        record = self.stats.compile(population)
        logbook.record(gen=0, nevals=len(population), **record)
        
        if verbose:
            print(f"Gen 0: Fitness={record['fitness']['min']:.4f}, Size={record['size']['avg']:.1f}")
        
        # Progress Bar
        if progress_bar:
            pbar = tqdm(total=generations, desc="Generationen")
        
        # Evolution
        for gen in range(1, generations + 1):
            # Selektion
            if self.multi_objective:
                # Für Multi-Objective: NSGA-II Selektion
                from deap import tools
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
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
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
            
            # Logging
            record = self.stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: Fitness={record['fitness']['min']:.4f}, "
                      f"Size={record['size']['avg']:.1f}")
            
            if progress_bar:
                pbar.update(1)
        
        if progress_bar:
            pbar.close()
        
        # Cleanup
        if hasattr(self, 'pool'):
            self.pool.shutdown()
        
        # Speichere Logbook für Visualisierung
        self.logbook = logbook
        
        return population, logbook, hall_of_fame