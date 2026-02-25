# Datei: core/diversity.py
import numpy as np
from scipy.spatial.distance import pdist, squareform

class DiversityPreserver:
    def __init__(self, population):
        self.population = population
    
    def calculate_diversity(self):
        """Berechnet genetische Diversität der Population"""
        # Konvertiere Individuen zu Feature-Vektoren
        features = []
        for ind in self.population:
            # Merkmale: Größe, Tiefe, Operationen-Verteilung
            size = len(ind)
            depth = ind.height
            op_counts = self._count_operations(ind)
            features.append([size, depth] + list(op_counts.values()))
        
        # Berechne paarweise Distanzen
        features_array = np.array(features)
        distances = pdist(features_array, metric='euclidean')
        avg_distance = np.mean(distances)
        
        return avg_distance
    
    def maintain_diversity(self, population, min_diversity=10.0):
        """Fügt Diversität hinzu wenn nötig"""
        current_diversity = self.calculate_diversity()
        
        if current_diversity < min_diversity:
            # Ersetze ähnliche Individuen
            self._replace_similar_individuals(population)
            
            # Füge zufällige Neue hinzu
            n_new = int(len(population) * 0.1)  # 10% neue
            for _ in range(n_new):
                population[random.randint(0, len(population)-1)] = \
                    self.toolbox.individual()
    
    def _count_operations(self, individual):
        """Zählt Häufigkeit verschiedener Operationen"""
        ops = {}
        for node in individual:
            op_name = node.name
            ops[op_name] = ops.get(op_name, 0) + 1
        return ops