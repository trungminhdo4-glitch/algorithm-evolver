"""
Max of Three problem: evolve max(x, y, z)
"""
import operator
from deap import gp

class MaxOfThreeProblem:
    def __init__(self):
        self.name = "max_of_three"
        self.input_names = ['x', 'y', 'z']
    
    def create_primitive_set(self):
        from core.primitives import create_typed_primitive_set, add_comparison_primitives
        
        pset = create_typed_primitive_set([float, float, float], float, "MAX3_MAIN")
        pset = add_comparison_primitives(pset)
        
        # Add the two-argument max function (building block for max of three)
        pset.addPrimitive(max, [float, float], float)
        
        # FIX: Add boolean terminals (True, False) for comparison operations
        pset.addTerminal(True, bool)
        pset.addTerminal(False, bool)
        
        # Rename arguments
        pset.renameArguments(ARG0='x', ARG1='y', ARG2='z')
        
        return pset
    
    def evaluate(self, individual, toolbox):
        func = toolbox.compile(expr=individual)
        
        # Test cases covering different scenarios
        test_cases = [
            (1, 2, 3, 3),    # z is max
            (3, 2, 1, 3),    # x is max  
            (2, 3, 1, 3),    # y is max
            (5, 5, 1, 5),    # tie (x and y)
            (0, 0, 0, 0),    # all equal
            (-1, -2, -3, -1),# negative values
            (10, -5, 0, 10)  # mix of positive and negative
        ]
        
        error = 0
        for x, y, z, target in test_cases:
            try:
                output = func(x, y, z)
                error += (output - target)**2
            except (OverflowError, ZeroDivisionError):
                error += 1e9
        
        return error, len(individual)
    
    def validate_solution(self, individual, toolbox):
        func = toolbox.compile(expr=individual)
        
        test_cases = [
            (7, 8, 9, 9),      # z is max
            (10, 5, 5, 10),    # x is max, tie for others
            (1.5, 2.5, 2.0, 2.5),  # y is max, decimals
            (-10, -20, -15, -10)   # negative
        ]
        
        results = []
        for x, y, z, expected in test_cases:
            output = func(x, y, z)
            error = abs(output - expected)
            results.append({
                'input': (x, y, z),
                'output': output,
                'expected': expected,
                'error': error,
                'correct': error < 0.01
            })
        
        return results
    
    def __str__(self):
        return "MaxOfThree(target=max(x, y, z))"