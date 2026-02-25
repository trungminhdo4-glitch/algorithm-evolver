# Datei: core/transfer_learning.py
import pickle

class TransferLearningEngine(EvolutionaryEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = []
    
    def save_best_individuals(self, population, n_best=10):
        """Speichert beste Individuen für zukünftige Probleme"""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
        self.knowledge_base.extend(sorted_pop[:n_best])
    
    def warm_start(self, new_problem):
        """Verwendet vorheriges Wissen für neues Problem"""
        if self.knowledge_base:
            # Initialisiere Population mit vorherigem Wissen
            population = []
            
            # 50% aus Wissen, 50% zufällig
            n_from_knowledge = self.population_size // 2
            
            for i in range(n_from_knowledge):
                # Passe bestehende Programme an neues Problem an
                base_ind = random.choice(self.knowledge_base)
                adapted = self._adapt_individual(base_ind, new_problem)
                population.append(adapted)
            
            # Rest zufällig
            for _ in range(self.population_size - n_from_knowledge):
                population.append(self.toolbox.individual())
            
            return population
        
        return self.toolbox.population(n=self.population_size)
    
    def _adapt_individual(self, individual, new_problem):
        """Passt Individuum an neues Problem an"""
        # Einfache Mutation zur Anpassung
        mutant = self.toolbox.clone(individual)
        self.toolbox.mutate(mutant)
        return mutant