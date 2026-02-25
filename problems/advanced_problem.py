# Datei: problems/advanced_problem.py
import pandas as pd
from sklearn.model_selection import train_test_split

class DataDrivenProblem:
    def __init__(self, data_path=None):
        self.data = self.load_data(data_path)
        self.train_data, self.test_data = self.split_data()
    
    def load_data(self, path):
        """Lädt Daten für datengetriebene Probleme"""
        if path and path.endswith('.csv'):
            df = pd.read_csv(path)
            return df
        else:
            # Fallback: generiere Testdaten
            return self.generate_test_data()
    
    def create_primitive_set_from_data(self):
        """Erstellt Primitive Set basierend auf Datentypen"""
        pset = gp.PrimitiveSetTyped("MAIN", self.input_types, self.output_type)
        
        # Füge Operationen basierend auf Datentypen hinzu
        if self.output_type == float:
            self._add_numeric_primitives(pset)
        elif self.output_type == bool:
            self._add_boolean_primitives(pset)
        
        return pset
    
    def evaluate_with_cross_validation(self, individual, toolbox, k_folds=5):
        """Cross-Validation statt einfacher Evaluation"""
        scores = []
        
        for fold in range(k_folds):
            train_fold, val_fold = self.get_fold(fold, k_folds)
            
            # Trainiere auf Fold
            func = toolbox.compile(expr=individual)
            train_score = self._calculate_score(func, train_fold)
            
            # Validierung
            val_score = self._calculate_score(func, val_fold)
            
            scores.append(val_score)
        
        return np.mean(scores),  # Durchschnittliche Validierungs-Fehler