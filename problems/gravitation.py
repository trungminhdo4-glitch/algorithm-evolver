"""
Newton's Law of Gravitation: Discover F = G * (m1 * m2) / r^2
"""
import math
import random
import numpy as np
from deap import gp
from core.physics import Dimension

class GravitationProblem:
    """
    Defines Newton's Law of Gravitation problem.
    m1, m2: Mass [M]
    r: Distance [L]
    Target: F = G * (m1 * m2) / r^2
    """
    
    def __init__(self, G=1.0):
        self.name = "gravitation"
        self.input_names = ['m1', 'm2', 'r']
        self.G = G
        self.target_func = lambda m1, m2, r: self.G * (m1 * m2) / (r**2)
        
        # Trainingsdaten generieren mit 1% Gaußschem Rauschen
        random.seed(42)
        np.random.seed(42)
        self.train_data = []
        self.sigma_percent = 0.01
        
        for _ in range(100):
            m1 = random.uniform(1, 100)
            m2 = random.uniform(1, 100)
            r = random.uniform(1, 10)
            target = self.target_func(m1, m2, r)
            # Add 1% noise
            noise = np.random.normal(0, self.sigma_percent * target)
            noisy_target = target + noise
            self.train_data.append(((m1, m2, r), noisy_target))

        # Dimensions-Definitionen [Mass, Length, Time, Amount, Temp]
        self.pset_units = {
            'm1': Dimension(1, 0, 0, 0, 0),      # [M]
            'm2': Dimension(1, 0, 0, 0, 0),      # [M]
            'r': Dimension(0, 1, 0, 0, 0),       # [L]
            'rand_float': Dimension(0, 0, 0, 0, 0), # Dimensionless
            'rand_dimensional': Dimension(-1, 3, -2, 0, 0) # Units of G: [L^3 M^-1 T^-2]
        }
        # F = M * L / T^2
        self.target_unit = Dimension(1, 1, -2, 0, 0) # Force [M L T^-2]
    
    def create_primitive_set(self):
        """Create problem-specific primitive set (Power Law style)"""
        from core.primitives import create_power_law_primitive_set
        
        # Drei Inputs (m1, m2, r), ein Output (float)
        pset = create_power_law_primitive_set([float, float, float], float, "GRAVITATION_POWER")
        
        # Rename arguments
        pset.renameArguments(ARG0='m1', ARG1='m2', ARG2='r')
        
        return pset
    
    def evaluate(self, individual, toolbox):
        """
        Fitness function: MSE and Symbolic Complexity with Dimensional Penalty.
        """
        from core.simplification import ProgramSimplifier
        from core.physics import DimensionalChecker
        import sympy as sp
        
        # 1. Dimensional Check
        checker = DimensionalChecker(self.pset_units)
        final_unit, consistent = checker.check_tree(individual)
        
        dim_penalty = 1.0
        if not consistent:
            dim_penalty = 1e9 # Severe penalty for physics nonsense
        elif final_unit != self.target_unit:
            dim_penalty = 1e6 # Penalty for wrong dimension
            
        # 2. Compile and calculate MSE
        func = toolbox.compile(expr=individual)
        
        mse = 0
        for (m1, m2, r), target in self.train_data:
            try:
                output = func(m1, m2, r)
                # MSE
                mse += (output - target)**2
            except (OverflowError, ZeroDivisionError, ValueError):
                mse += 1e12
        
        mse /= len(self.train_data)
        
        # Total cost for accuracy
        accuracy_score = mse * dim_penalty
        
        # 3. Symbolic Complexity
        try:
            simplifier = ProgramSimplifier()
            # Reuse names for simplification
            pset = self.create_primitive_set()
            simplified_str = simplifier.simplify_individual(individual, pset)
            
            if "Error" in simplified_str:
                complexity = len(individual)
            else:
                expr = sp.sympify(simplified_str)
                complexity = sp.count_ops(expr) + len(expr.free_symbols) + 1
        except Exception:
            complexity = len(individual)
            
        return accuracy_score, float(complexity)

    def __str__(self):
        return f"GravitationProblem(target=G * m1 * m2 / r^2, G={self.G})"
