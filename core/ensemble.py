# Datei: core/ensemble.py
import numpy as np
class EnsembleEvolver:
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
    
    def evolve_ensemble(self, problem, engine_params):
        """Evolviert mehrere Modelle und kombiniert sie"""
        for i in range(self.n_models):
            # Unterschiedliche Start-Seeds
            engine = EvolutionaryEngine(
                seed=42 + i * 100,
                **engine_params
            )
            
            population, _, hall_of_fame = engine.run(verbose=False)
            best_model = hall_of_fame[0]
            self.models.append(best_model)
        
        return self.models
    
    def ensemble_predict(self, inputs, toolbox):
        """Vorhersage durch Voting oder Averaging"""
        predictions = []
        for model in self.models:
            func = toolbox.compile(expr=model)
            try:
                pred = func(*inputs)
                predictions.append(pred)
            except:
                predictions.append(0)
        
        # Durchschnitt oder Median
        return np.median(predictions)  # Robust gegenüber Ausreißern