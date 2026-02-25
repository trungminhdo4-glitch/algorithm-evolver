"""
Projectile Motion problem: Discover distance = (v^2 * sin(2*angle)) / 9.81
"""
import math
import random
from deap import gp

class ProjectileProblem:
    """
    Defines Projectile Motion problem.
    v: Initial velocity (0-100 m/s)
    angle: Launch angle (0-90 degrees in radians)
    distance = (v**2 * math.sin(2 * angle)) / 9.81
    """
    
    def __init__(self):
        self.name = "projectile"
        self.input_names = ['v', 'angle']
        self.gravity = 9.81
        self.target_func = lambda v, angle: (v**2 * math.sin(2 * angle)) / self.gravity
        
        # Trainingsdaten generieren mit RAUSCHEN (Module 4)
        random.seed(42)
        self.train_data = []
        self.sigma = 2.0  # Fehlerbalken (2 Meter Standardabweichung)
        for _ in range(50):
            v = random.uniform(5, 100)
            angle = random.uniform(0.1, math.pi/2 - 0.1)
            target = self.target_func(v, angle)
            # Add Gaussian noise
            noisy_target = target + random.gauss(0, self.sigma)
            self.train_data.append(((v, angle), noisy_target))

        # Dimensions-Definitionen (Module 2)
        from core.physics import Dimension
        self.pset_units = {
            'v': Dimension(0, 1, -1),      # [L/T]
            'angle': Dimension(0, 0, 0),    # dimensionless
            'rand_float': Dimension(0, 0, 0) # Assumed dimensionless for now
        }
        self.target_unit = Dimension(0, 1, 0) # Distance [L]
    
    def create_primitive_set(self):
        """Create problem-specific primitive set"""
        from core.primitives import create_typed_primitive_set, add_math_primitives
        
        # Zwei Inputs (v, angle), ein Output (float)
        pset = create_typed_primitive_set([float, float], float, "PROJECTILE_MAIN")
        pset = add_math_primitives(pset)
        
        # Rename arguments
        pset.renameArguments(ARG0='v', ARG1='angle')
        
        return pset
    
    def evaluate(self, individual, toolbox):
        """
        Fitness function: Chi-Squared and Symbolic Complexity.
        Includes Dimensional Penalty.
        """
        from core.simplification import ProgramSimplifier
        from core.physics import DimensionalChecker
        import sympy as sp
        
        # 1. Dimensional Check (Module 2)
        checker = DimensionalChecker(self.pset_units)
        final_unit, consistent = checker.check_tree(individual)
        
        dim_penalty = 1.0
        if not consistent:
            dim_penalty = 1e6 # Severe penalty for physics nonsense
        elif final_unit != self.target_unit:
            dim_penalty = 1e4 # Penalty for wrong dimension (e.g. area instead of length)

        # 2. Compile and calculate Chi-Squared (Module 4)
        func = toolbox.compile(expr=individual)
        
        chi_sq = 0
        for (v, angle), target in self.train_data:
            try:
                output = func(v, angle)
                # Chi^2 = sum((obs - pred)^2 / sigma^2)
                chi_sq += ((output - target)**2) / (self.sigma**2)
            except (OverflowError, ZeroDivisionError, ValueError):
                chi_sq += 1e9
        
        # Total cost for accuracy
        accuracy_score = (chi_sq / len(self.train_data)) * dim_penalty
        
        # 3. Symbolic Complexity (with SymPy - Module 3 Deep Clean)
        try:
            simplifier = ProgramSimplifier()
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
    
    def validate_solution(self, individual, toolbox):
        """Test solution on unseen manual cases"""
        func = toolbox.compile(expr=individual)
        
        # Testdaten: Verschiedene v und Winkel
        test_cases = [
            (50.0, math.pi/4, 254.84), # 45 Grad
            (80.0, math.pi/6, 565.17), # 30 Grad
            (10.0, math.pi/3, 8.83)    # 60 Grad
        ]
        
        results = []
        for v, angle, expected in test_cases:
            try:
                output = func(v, angle)
                error = abs(output - expected)
                results.append({
                    'input': (v, angle),
                    'output': output,
                    'expected': expected,
                    'error': error,
                    'correct': error < 1.0  # Erlaube moderate Abweichung
                })
            except Exception:
                results.append({
                    'input': (v, angle),
                    'output': 'ERROR',
                    'expected': expected,
                    'error': 1e9,
                    'correct': False
                })
        
        return results
    
    def __str__(self):
        return f"ProjectileProblem(target=(v**2 * sin(2*angle)) / {self.gravity})"
