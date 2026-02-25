import math
import random
import numpy as np
from deap import gp
from core.physics import Dimension, DimensionalChecker

class IdealGasProblem:
    def __init__(self):
        self.name = "ideal_gas"
        self.input_names = ['P', 'V', 'n', 'R']
        self.R_physical = 8.314 # J/(mol*K)
        
        # P in Pascal, V in m^3, n in mol
        self.target_func = lambda P, V, n: (P * V) / (n * self.R_physical)
        
        random.seed(42)
        raw_inputs = []
        raw_targets = []
        for _ in range(150):
            P = random.uniform(1e5, 1e6)
            V = random.uniform(0.01, 1.0)
            n = random.uniform(0.1, 10)
            T = self.target_func(P, V, n)
            raw_inputs.append([P, V, n])
            raw_targets.append(T)
            
        # --- Proportional Normalization ---
        self.inputs_max = np.max(raw_inputs, axis=0) # [Pmax, Vmax, nmax]
        self.target_max = np.max(raw_targets)
        
        # Scaling factor such that T' = P'V' / (n'R') works exactly
        # R' = (nmax * R_phys * Tmax) / (Pmax * Vmax)
        self.R_scaled_value = (self.inputs_max[2] * self.R_physical * self.target_max) / (self.inputs_max[0] * self.inputs_max[1])
        
        def scale(val, vmax):
            return val / vmax
            
        self.train_data = []
        self.sigma = 0.001 # Even lower noise for clarity
        for i in range(len(raw_inputs)):
            scaled_in = [
                scale(raw_inputs[i][0], self.inputs_max[0]),
                scale(raw_inputs[i][1], self.inputs_max[1]),
                scale(raw_inputs[i][2], self.inputs_max[2]),
                self.R_scaled_value # R is constant but carries the scaling
            ]
            scaled_target = scale(raw_targets[i], self.target_max)
            self.train_data.append((tuple(scaled_in), scaled_target + random.gauss(0, self.sigma)))

        # --- Dimensions (5 Slots: M, L, T, N, Theta) ---
        self.pset_units = {
            'ARG0': Dimension(1, -1, -2, 0, 0), # P
            'ARG1': Dimension(0, 3, 0, 0, 0),   # V
            'ARG2': Dimension(0, 0, 0, 1, 0),   # n
            'ARG3': Dimension(1, 2, -2, -1, -1),# R
            'one': Dimension(0, 0, 0, 0, 0)
        }
        self.target_unit = Dimension(0, 0, 0, 0, 1) # Target Temperature [Theta]
        
    def create_primitive_set(self):
        from core.primitives import create_power_law_primitive_set
        # Power Law: Only mul and div. 4 inputs now!
        pset = create_power_law_primitive_set([float, float, float, float], float, "GAS_POWER")
        pset.renameArguments(ARG0='P', ARG1='V', ARG2='n', ARG3='R')
        
        pset.addTerminal(1.0, float, name='one')
        
        return pset
        
    def evaluate(self, individual, toolbox, generation=0):
        from core.physics import DimensionalChecker
        
        # 1. Dimensional Check
        checker = DimensionalChecker(self.pset_units)
        final_unit, consistent = checker.check_tree(individual)
        
        warmup_factor = min(1.0, generation / 100.0)
        
        # Euclidean Dimensional Penalty
        dim_penalty = 1.0
        if not consistent:
            dim_penalty = 1.0 + (warmup_factor * 1e12)
        elif final_unit != self.target_unit:
            # Euclidean distance in unit space
            dist_sq = sum((a - b)**2 for a, b in zip(final_unit.units, self.target_unit.units))
            dist = math.sqrt(dist_sq)
            # Continues feedback: even small distance penalized
            dim_penalty = 1.0 + (warmup_factor * (10**(2 + dist)))
        else:
            # Physical Reward: MASSIVE boost for correct dimension
            dim_penalty = 0.001 
            
        func = toolbox.compile(expr=individual)
        chi_sq = 0
        for inputs, target in self.train_data:
            try:
                # inputs now has 4 elements
                pred = func(*inputs)
                if np.isnan(pred) or np.isinf(pred) or abs(pred) > 1e20:
                    chi_sq += 1e12
                else:
                    chi_sq += ((pred - target)**2) / (self.sigma**2)
            except Exception:
                chi_sq += 1e12
                
        # Total cost
        accuracy_score = (chi_sq / len(self.train_data)) * dim_penalty
        
        return accuracy_score, float(len(individual))
