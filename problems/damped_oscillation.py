"""
Damped Harmonic Oscillation problem: Discover x(t) = A * exp(-delta * t) * cos(omega * t)
"""
import math
import random
import numpy as np
from deap import gp

class DampedOscillationProblem:
    """
    Defines Damped Harmonic Oscillation problem.
    t: Time [T]
    x: Displacement [L]
    Target: x(t) = 1.0 * exp(-0.5 * t) * cos(2 * pi * t)
    """
    
    def __init__(self):
        self.name = "damped_oscillation"
        self.input_names = ['t']
        self.A = 1.0
        self.delta = 0.5
        self.omega = 2 * math.pi
        self.sigma = 0.01
        
        # Generation control
        self.current_generation = 0
        
        # Trainingsdaten generieren
        random.seed(42)
        self.t_values = np.linspace(0, 4 * math.pi, 100)
        self.x_values = self.A * np.exp(-self.delta * self.t_values) * np.cos(self.omega * self.t_values)
        
        # Add noise
        self.x_noisy = self.x_values + np.random.normal(0, self.sigma, size=self.x_values.shape)
        
        # Trainingsdaten als Liste von Tuples
        self.train_data = [((t,), x) for t, x in zip(self.t_values, self.x_noisy)]

        # Dimensions-Definitionen
        from core.physics import Dimension
        self.pset_units = {
            't': Dimension(0, 0, 1),           # Time [T]
            'rand_inv_time': Dimension(0, 0, -1), # [1/T]
            'rand_length': Dimension(0, 1, 0),   # [L]
            'rand_float': Dimension(0, 0, 0)     # Dimensionless
        }
        self.target_unit = Dimension(0, 1, 0)  # Displacement [L]
    
    def create_primitive_set(self):
        """Create problem-specific primitive set"""
        from core.primitives import create_typed_primitive_set, add_transcendental_primitives
        
        # Ein Input (t), ein Output (float)
        pset = create_typed_primitive_set([float], float, "OSCILLATION")
        pset = add_transcendental_primitives(pset)
        
        # Rename arguments
        pset.renameArguments(ARG0='t')
        
        return pset
    
    def evaluate(self, individual, toolbox=None):
        """
        Fitness function: MSE with Dimensional Penalty and Curriculum Learning.
        """
        # If toolbox is not provided, we might need a workaround or ensure it's provided.
        # In DEAP's map(evaluate, individuals), evaluate is called as evaluate(individual).
        # We need to make sure 'toolbox' is either not needed or handled.
        # However, EvolutionaryEngine.toolbox is usually what's passed.
        from core.physics import DimensionalChecker
        
        # 1. Dimensional Check
        checker = DimensionalChecker(self.pset_units)
        final_unit, consistent = checker.check_tree(individual)
        
        dim_penalty = 1.0
        if not consistent:
            dim_penalty = 1e9 # Severe penalty for physics nonsense
        elif final_unit != self.target_unit:
            dim_penalty = 1e3 # Softened penalty for wrong dimension (e.g. dimensionless instead of L)
            
        # 2. Curriculum Learning
        # Early generations: only evaluate on first part of the signal [0, 1.0]
        # Late generations: evaluate on whole range
        if self.current_generation <= 20:
             # Filter data for t <= 1.0
             eval_data = [(inp, target) for inp, target in self.train_data if inp[0] <= 1.0]
        else:
             eval_data = self.train_data

        # 3. Compile and calculate MSE
        func = toolbox.compile(expr=individual)
        
        mse = 0
        for (t,), target in eval_data:
            try:
                output = func(t)
                mse += (output - target)**2
            except (OverflowError, ZeroDivisionError, ValueError):
                mse += 1e9
        
        mse /= len(eval_data)
        
        # Total cost for accuracy
        accuracy_score = mse * dim_penalty
        
        # 4. Symbolic Complexity
        complexity = len(individual)
            
        return accuracy_score, float(complexity)
    
    def __str__(self):
        return f"DampedOscillationProblem(target=1.0 * exp(-0.5*t) * cos(6.28*t))"
