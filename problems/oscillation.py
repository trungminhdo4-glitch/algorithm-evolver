"""
Damped Oscillation problem: Discover A * exp(-delta * t) * cos(omega * t + phi)
"""
import math
import random
import numpy as np
from deap import gp
from core.physics import Dimension, DimensionalChecker

class OscillationProblem:
    def __init__(self):
        self.name = "oscillation"
        self.input_names = ['t']
        
        # Parameters (Target: 10 * exp(-0.1 * t) * cos(2 * t))
        self.A = 10.0
        self.delta = 0.1
        self.omega = 2.0
        self.target_func = lambda t: self.A * math.exp(-self.delta * t) * math.cos(self.omega * t)
        
        # Generation with noise
        random.seed(42)
        self.train_data = []
        self.sigma = 0.5
        for _ in range(100):
            t = random.uniform(0, 20)
            target = self.target_func(t)
            noisy_target = target + random.gauss(0, self.sigma)
            self.train_data.append(((t,), noisy_target))
            
        # Units
        self.pset_units = {
            't': Dimension(0, 0, 1), # Time [T]
            'rand_float': Dimension(0, 0, 0)
        }
        self.target_unit = Dimension(0, 0, 0) # Amplitude / dimensionless here
        
    def create_primitive_set(self):
        from core.primitives import create_typed_primitive_set, add_math_primitives
        pset = create_typed_primitive_set([float], float, "OSCILLATION_MAIN")
        pset = add_math_primitives(pset)
        pset.renameArguments(ARG0='t')
        return pset
        
    def evaluate(self, individual, toolbox):
        from core.simplification import ProgramSimplifier
        import sympy as sp
        
        checker = DimensionalChecker(self.pset_units)
        _, consistent = checker.check_tree(individual)
        
        dim_penalty = 1.0 if consistent else 1e6
        
        func = toolbox.compile(expr=individual)
        chi_sq = 0
        for (t,), target in self.train_data:
            try:
                output = func(t)
                chi_sq += ((output - target)**2) / (self.sigma**2)
            except Exception:
                chi_sq += 1e9
                
        accuracy = (chi_sq / len(self.train_data)) * dim_penalty
        
        try:
            simplifier = ProgramSimplifier()
            simplified_str = simplifier.simplify_individual(individual, self.create_primitive_set())
            expr = sp.sympify(simplified_str)
            complexity = sp.count_ops(expr) + len(expr.free_symbols) + 1
        except Exception:
            complexity = len(individual)
            
        return accuracy, float(complexity)

if __name__ == "__main__":
    p = OscillationProblem()
    print(f"Oscillation problem initialized with {len(p.train_data)} points.")
