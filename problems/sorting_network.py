"""
Sorting Network problem: Evolve a network that sorts 3 numbers.
Returns sorted tuple (min, mid, max)
"""
import operator
from deap import gp

class SortingNetworkProblem:
    def __init__(self):
        self.name = "sorting_network"
        self.input_names = ['a', 'b', 'c']
    
    def create_primitive_set(self):
        from core.primitives import create_typed_primitive_set, add_comparison_primitives
        
        # We return a TUPLE of 3 floats (sorted order)
        pset = create_typed_primitive_set([float, float, float], tuple, "SORT_MAIN")
        pset = add_comparison_primitives(pset)
        
        # Add tuple construction primitives
        def make_tuple3(x, y, z):
            return (x, y, z)
        pset.addPrimitive(make_tuple3, [float, float, float], tuple)
        
        # Add swap operation (the core of sorting networks)
        def compare_swap(x, y):
            return (min(x, y), max(x, y))
        pset.addPrimitive(compare_swap, [float, float], tuple)
        
        # Add tuple access operations
        def get_first(t):
            return t[0]
        def get_second(t):
            return t[1]
        def get_third(t):
            return t[2]
            
        pset.addPrimitive(get_first, [tuple], float)
        pset.addPrimitive(get_second, [tuple], float)
        pset.addPrimitive(get_third, [tuple], float)
        
        # Rename arguments
        pset.renameArguments(ARG0='a', ARG1='b', ARG2='c')
        
        return pset
    
    def evaluate(self, individual, toolbox):
        func = toolbox.compile(expr=individual)
        
        # Test cases: different permutations of 3 numbers
        test_cases = [
            (1, 2, 3),    # Already sorted
            (3, 2, 1),    # Reverse sorted
            (2, 1, 3),    # First two swapped
            (3, 1, 2),    # Complex permutation
            (5, 5, 1),    # With duplicates
            (0, 0, 0),    # All equal
            (-1, -3, -2), # Negative numbers
            (10, -5, 0)   # Mixed positive/negative
        ]
        
        error = 0
        for a, b, c in test_cases:
            try:
                output = func(a, b, c)
                # Output should be a tuple of 3 sorted numbers
                if not isinstance(output, tuple) or len(output) != 3:
                    error += 1e6
                    continue
                    
                # Calculate error: sum of squared differences from sorted order
                target = tuple(sorted([a, b, c]))
                for out_val, target_val in zip(output, target):
                    error += (out_val - target_val)**2
            except (TypeError, IndexError, OverflowError):
                error += 1e9
        
        return error, len(individual)
    
    def validate_solution(self, individual, toolbox):
        func = toolbox.compile(expr=individual)
        
        test_cases = [
            (7, 3, 5, (3, 5, 7)),
            (1, 1, 2, (1, 1, 2)),
            (9, 4, 6, (4, 6, 9)),
            (-10, 0, -5, (-10, -5, 0))
        ]
        
        results = []
        for a, b, c, expected in test_cases:
            try:
                output = func(a, b, c)
                if isinstance(output, tuple) and len(output) == 3:
                    error = sum((o - e)**2 for o, e in zip(output, expected))
                    results.append({
                        'input': (a, b, c),
                        'output': output,
                        'expected': expected,
                        'error': error,
                        'correct': error < 0.01
                    })
                else:
                    results.append({
                        'input': (a, b, c),
                        'output': output,
                        'expected': expected,
                        'error': 1e6,
                        'correct': False
                    })
            except Exception:
                results.append({
                    'input': (a, b, c),
                    'output': 'ERROR',
                    'expected': expected,
                    'error': 1e9,
                    'correct': False
                })
        
        return results
    
    def __str__(self):
        return "SortingNetwork(target=sort(a, b, c))"