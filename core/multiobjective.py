# Datei: core/multiobjective.py
from deap import algorithms, tools

class MultiObjectiveEngine(EvolutionaryEngine):
    def __init__(self, **kwargs):
        # Zwei Ziele: 1. Minimieren Fehler, 2. Minimieren Programmgröße
        try:
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -0.1))
        except RuntimeError:
            pass
            
        try:
            creator.create("IndividualMO", gp.PrimitiveTree,
                          fitness=creator.FitnessMulti, pset=self.pset)
        except RuntimeError:
            pass
        
        super().__init__(**kwargs)
    
    def evaluate_with_size(self, individual, toolbox):
        """Berechnet Fehler UND Programmgröße"""
        error = self.evaluate_func(individual, toolbox)[0]
        size = len(individual)  # Anzahl der Knoten im Baum
        return error, size
    
    def run_multiobjective(self, generations=50):
        # NSGA-II Algorithmus für Multi-Objective
        population = self.toolbox.population(n=self.population_size)
        
        # Fitness zuweisen
        for ind in population:
            ind.fitness.values = self.evaluate_with_size(ind, self.toolbox)
        
        # NSGA-II
        for gen in range(generations):
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            
            # Crossover & Mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluierung der Nachkommen
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Kombiniere Population und wähle nächste Generation
            population = tools.selNSGA2(population + offspring, self.population_size)
        
        return population