"""
Symbolic Regression problem configuration.
Separates problem definition from evolutionary engine.
"""
import random
from deap import gp  # ADD THIS IMPORT

class SymbolicRegressionProblem:
    """
    Defines the symbolic regression problem: evolve f(x,y) = x² + 2y
    
    Why a separate class? Each problem has:
    1. Its own fitness function
    2. Specific primitive requirements
    3. Unique validation logic
    """
    
    def __init__(self):
        self.name = "symbolic_regression"
        self.input_names = ['x', 'y']
        self.target_func = lambda x, y: x**2 + 2*y
    
    def create_primitive_set(self):
        """Create problem-specific primitive set"""
        from core.primitives import create_typed_primitive_set
        
        pset = create_typed_primitive_set([float, float], float, "SR_MAIN")
        
        # Rename arguments for readability
        pset.renameArguments(ARG0='x', ARG1='y')
        
        return pset
    
    def evaluate(self, individual, toolbox):
        """
        Fitness function for symbolic regression.
        
        Why separate from the engine? Fitness is problem-specific,
        while evolution operators are generic.
        """
        func = toolbox.compile(expr=individual)
        
        # Test points - comprehensive but manageable
        test_points = [
            (x, y) for x in [-2, -1, 0, 1, 2] 
            for y in [-2, -1, 0, 1, 2]
        ]
        
        error = 0
        for x, y in test_points:
            try:
                target = self.target_func(x, y)
                output = func(x, y)
                error += (output - target)**2
            except (OverflowError, ZeroDivisionError):
                # Heavy penalty for invalid operations
                error += 1e9
        
        return error, len(individual)
    
    def validate_solution(self, individual, toolbox):
        """Test solution on unseen data"""
        func = toolbox.compile(expr=individual)
        
        test_cases = [
            (3, 4, 17),    # x² + 2y = 9 + 8 = 17
            (0, 5, 10),    # 0 + 10 = 10
            (-2, 3, 10),   # 4 + 6 = 10
            (1.5, 2.5, 7.25)  # 2.25 + 5 = 7.25
        ]
        
        results = []
        for x, y, expected in test_cases:
            output = func(x, y)
            error = abs(output - expected)
            results.append({
                'input': (x, y),
                'output': output,
                'expected': expected,
                'error': error,
                'correct': error < 0.01
            })
        
        return results
    
    def __str__(self):
        return f"SymbolicRegression(target={self.target_func.__name__})"