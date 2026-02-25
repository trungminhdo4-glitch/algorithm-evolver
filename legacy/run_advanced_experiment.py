# Datei: run_advanced_experiment.py
"""
Neue modulare Architektur mit allen Verbesserungen
"""
import yaml
from core.parallel_evolution import ParallelEvolutionaryEngine
from core.multiobjective import MultiObjectiveEngine
from core.ensemble import EnsembleEvolver
from core.transfer_learning import TransferLearningEngine
from utils.advanced_visualization import AdvancedVisualizer

class AdvancedExperimentRunner:
    def __init__(self, config_path="config/experiment_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Modulare Komponenten
        self.engines = {
            'parallel': ParallelEvolutionaryEngine,
            'multiobjective': MultiObjectiveEngine,
            'ensemble': EnsembleEvolver,
            'transfer': TransferLearningEngine
        }
    
    def run_experiment(self, experiment_name):
        config = self.config['experiments'][experiment_name]
        
        # Dynamisch Engine erstellen
        engine_class = self.engines[config['engine']]
        engine = engine_class(**config['params'])
        
        # Features aktivieren
        for feature in config['features']:
            self._activate_feature(engine, feature)
        
        # Experiment durchführen
        results = engine.run()
        
        # Visualisierung
        if config.get('visualize', True):
            visualizer = AdvancedVisualizer()
            visualizer.plot_convergence_dashboard(engine.logbook)
            visualizer.plot_3d_function(results['best_individual'], engine.toolbox)
        
        return results
    
    def _activate_feature(self, engine, feature_name):
        """Aktiviert Features dynamisch"""
        if feature_name == 'bloat_control':
            engine.enable_bloat_control(max_nodes=150)
        elif feature_name == 'diversity_preservation':
            engine.maintain_diversity = True
        # ... weitere Features