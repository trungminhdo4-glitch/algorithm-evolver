import numpy as np
from problems.ideal_gas import IdealGasProblem

problem = IdealGasProblem()
train_data = problem.train_data

def target_formula(P, V, n):
    # This is what we hope to find: T' = P' * V' / (n' * scaling_factor)
    # Actually, as derived: T' = (P' * V' / n') * (Pmax * Vmax / (nmax * R * Tmax))
    # In my code: self.R_scaled_value = (nmax * R * Tmax) / (Pmax * Vmax)
    # So T' = (P' * V' / n') / self.R_scaled_value
    # Or T' = (P' * V' / n') * (1/self.R_scaled_value)
    
    # Let's test if T' = P' * V' / (n' * R_const_scaled)
    # where R_const_scaled is the terminal I added.
    try:
        return (P * V) / (n * problem.R_scaled_value)
    except:
        return 0

chi_sq = 0
for inputs, target in train_data:
    pred = target_formula(*inputs)
    chi_sq += ((pred - target)**2) / (problem.sigma**2)

avg_chi_sq = chi_sq / len(train_data)
print(f"Target Formula Chi2: {avg_chi_sq:.4e}")
print(f"R_scaled_value: {problem.R_scaled_value}")
