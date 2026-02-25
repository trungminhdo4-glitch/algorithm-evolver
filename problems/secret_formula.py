"""
Secret Formula problem: Evolve f(x, y) = x² + sin(y)
"""
import random
import math
from deap import gp

class SecretFormulaProblem:
    """
    Defines the Secret Formula problem: evolve f(x, y) = x² + sin(y)
    """
    
    def __init__(self):
        self.name = "secret_formula"
        self.input_names = ['x', 'y']
        self.target_func = lambda x, y: x**2 + math.sin(y)
        
        # Trainingsdaten generieren (50 zufällige Punkte aus [-10, 10])
        random.seed(42)
        noise_level = 0.5  # Wie stark zittern die Messwerte?
        self.train_data = []
        for _ in range(50):
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            true_value = self.target_func(x, y)
            # Wir addieren zufälliges Rauschen (Gaussian Noise)
            noisy_value = true_value + random.gauss(0, noise_level)
            self.train_data.append(((x, y), noisy_value))
    
    def create_primitive_set(self):
        """Create problem-specific primitive set"""
        from core.primitives import create_typed_primitive_set, add_math_primitives
        
        pset = create_typed_primitive_set([float, float], float, "SF_MAIN")
        pset = add_math_primitives(pset)
        
        # Rename arguments for readability
        pset.renameArguments(ARG0='x', ARG1='y')
        
        return pset
    
    def evaluate(self, individual, toolbox):
        """
        Fitness function: Mean Squared Error (MSE)
        """
        func = toolbox.compile(expr=individual)
        
        error = 0
        for (x, y), target in self.train_data:
            try:
                output = func(x, y)
                error += (output - target)**2
            except (OverflowError, ZeroDivisionError, ValueError):
                error += 1e9
        
        mse = error / len(self.train_data)
        return mse, len(individual)
    
    def validate_solution(self, individual, toolbox):
        """Test solution on unseen test data"""
        func = toolbox.compile(expr=individual)
        
        # Unabhängiger Testsatz
        test_cases = [
            (3.0, 0.0, 9.0),      # 3² + sin(0) = 9
            (0.0, math.pi/2, 1.0), # 0² + sin(π/2) = 1
            (-2.0, -math.pi/2, 3.0),# (-2)² + sin(-π/2) = 4 - 1 = 3
            (1.0, 1.0, 1.0 + math.sin(1.0))
        ]
        
        results = []
        for x, y, expected in test_cases:
            try:
                output = func(x, y)
                error = abs(output - expected)
                results.append({
                    'input': (x, y),
                    'output': output,
                    'expected': expected,
                    'error': error,
                    'correct': error < 0.01
                })
            except Exception:
                results.append({
                    'input': (x, y),
                    'output': 'ERROR',
                    'expected': expected,
                    'error': 1e9,
                    'correct': False
                })
        
        return results
    
    def __str__(self):
        return "SecretFormula(target=x² + sin(y))"
