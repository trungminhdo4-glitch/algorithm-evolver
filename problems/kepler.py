"""
Kepler's Law problem: Discover T = a^1.5 from NASA data.
"""
import math
from deap import gp

class KeplerProblem:
    """
    Defines Kepler's Third Law problem: T = a^1.5
    a: Semi-major axis (AU)
    T: Orbital period (years)
    """
    
    def __init__(self):
        self.name = "kepler"
        self.input_names = ['a']
        
        # NASA Trainingsdaten (Planeten unseres Sonnensystems)
        # a: [Merkur, Venus, Erde, Mars, Jupiter, Saturn]
        self.data_a = [0.387, 0.723, 1.000, 1.524, 5.203, 9.537]
        self.data_t = [0.241, 0.615, 1.000, 1.881, 11.86, 29.46]
        
        self.train_data = list(zip(self.data_a, self.data_t))
    
    def create_primitive_set(self):
        """Create problem-specific primitive set"""
        from core.primitives import create_typed_primitive_set, add_math_primitives
        
        # Ein Input (a), ein Output (float)
        pset = create_typed_primitive_set([float], float, "KEPLER_MAIN")
        pset = add_math_primitives(pset)
        
        # Rename arguments
        pset.renameArguments(ARG0='a')
        
        return pset
    
    def evaluate(self, individual, toolbox):
        """
        Fitness function: Mean Squared Error (MSE)
        """
        func = toolbox.compile(expr=individual)
        
        error = 0
        for a, target in self.train_data:
            try:
                output = func(a)
                error += (output - target)**2
            except (OverflowError, ZeroDivisionError, ValueError):
                error += 1e9
        
        mse = error / len(self.train_data)
        return mse, len(individual)
    
    def validate_solution(self, individual, toolbox):
        """Test solution on unseen data (e.g. Uranus, Neptune)"""
        func = toolbox.compile(expr=individual)
        
        # Testdaten: Uranus (a=19.18, T=84.01), Neptun (a=30.07, T=164.8)
        test_cases = [
            (19.18, 84.01),
            (30.07, 164.8)
        ]
        
        results = []
        for a, expected in test_cases:
            try:
                output = func(a)
                error = abs(output - expected)
                relative_error = (error / expected) * 100 if expected != 0 else 0
                results.append({
                    'input': (a,),
                    'output': output,
                    'expected': expected,
                    'error': error,
                    'correct': relative_error < 5.0  # Erlaube 5% Abweichung bei physikalischen Daten
                })
            except Exception:
                results.append({
                    'input': (a,),
                    'output': 'ERROR',
                    'expected': expected,
                    'error': 1e9,
                    'correct': False
                })
        
        return results
    
    def __str__(self):
        return "KeplerProblem(target=a^1.5)"
