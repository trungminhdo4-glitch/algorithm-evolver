# Datei: core/hyperparameter_optimization.py
from sklearn.model_selection import ParameterGrid
import optuna

class HyperparameterOptimizer:
    def __init__(self, problem_class):
        self.problem = problem_class()
    
    def optimize_with_optuna(self, n_trials=50):
        def objective(trial):
            # Hyperparameter-Vorschläge
            pop_size = trial.suggest_int('population_size', 50, 500)
            cxpb = trial.suggest_float('cxpb', 0.4, 0.9)
            mutpb = trial.suggest_float('mutpb', 0.1, 0.5)
            max_height = trial.suggest_int('max_height', 5, 20)
            generations = trial.suggest_int('generations', 30, 200)
            
            # Engine mit diesen Parametern
            pset = self.problem.create_primitive_set()
            engine = EvolutionaryEngine(
                pset=pset,
                evaluate_func=self.problem.evaluate,
                population_size=pop_size,
                cxpb=cxpb,
                mutpb=mutpb,
                max_height=max_height
            )
            
            # Lauf
            _, _, hall_of_fame = engine.run(
                generations=generations,
                verbose=False
            )
            
            # Fitness als Objective
            return hall_of_fame[0].fitness.values[0]
        
        # Optuna Studie
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params