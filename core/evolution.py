"""
Core evolutionary engine - reusable across all problems
"""
import operator
import random
import numpy as np
from deap import base, creator, tools, algorithms, gp

def _get_fitness(ind):
    return ind.fitness.values

def _evaluate_wrapper(evaluate_func, pset, individual):
    # Compile locally for multiprocessing safety
    func = gp.compile(individual, pset)
    return evaluate_func(individual, func)

class EvolutionaryEngine:
    def __init__(self, pset, evaluate_func, population_size=100, 
                 cxpb=0.6, mutpb=0.3, tournsize=5, max_height=10):
        self.pset = pset
        self.evaluate_func = evaluate_func
        self.population_size = population_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.tournsize = tournsize
        self.max_height = max_height
        
        self._setup_creator()
        self._setup_toolbox()
        self._setup_statistics()
    
    def _setup_creator(self):
        try:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        except RuntimeError:
            pass
            
        try:
            creator.create("Individual", gp.PrimitiveTree, 
                          fitness=creator.FitnessMin, pset=self.pset)
        except RuntimeError:
            pass
    
    def _setup_toolbox(self):
        self.toolbox = base.Toolbox()
        
        self.toolbox.register("expr", gp.genHalfAndHalf, 
                             pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, 
                             creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, 
                             list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", _evaluate_wrapper, self.evaluate_func, self.pset)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        
        self.toolbox.register("select", tools.selTournament, 
                             tournsize=self.tournsize)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, 
                             expr=self.toolbox.expr_mut, pset=self.pset)
        
        self.toolbox.decorate("mate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.max_height))
        self.toolbox.decorate("mutate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.max_height))
    
    def _setup_statistics(self):
        self.stats_fit = tools.Statistics(_get_fitness)
        self.stats_size = tools.Statistics(len)
        self.stats = tools.MultiStatistics(fitness=self.stats_fit, 
                                          size=self.stats_size)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
    
    def run(self, generations=40, seed=42, verbose=True):
        random.seed(seed)
        population = self.toolbox.population(n=self.population_size)
        hall_of_fame = tools.HallOfFame(1)
        
        population, log = algorithms.eaSimple(
            population, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
            ngen=generations, stats=self.stats, halloffame=hall_of_fame,
            verbose=verbose
        )
        
        return population, log, hall_of_fame

    def run_nsga2(self, generations=40, seed=42, verbose=True, population=None):
        """
        Runs the NSGA-II multi-objective evolutionary algorithm.
        """
        random.seed(seed)
        if population is None:
            population = self.toolbox.population(n=self.population_size)
        
        # NSGA-II uses a different selection mechanism
        # We ensure it's registered in the runner, but this method uses eaMuPlusLambda
        population, log = algorithms.eaMuPlusLambda(
            population, self.toolbox, 
            mu=self.population_size, 
            lambda_=self.population_size,
            cxpb=self.cxpb, mutpb=self.mutpb,
            ngen=generations, stats=self.stats, 
            halloffame=None,
            verbose=verbose
        )
        
        return population, log