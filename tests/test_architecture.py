import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problems.symbolic_regression import SymbolicRegressionProblem
from core.evolution import EvolutionaryEngine

# Quick test
problem = SymbolicRegressionProblem()
pset = problem.create_primitive_set()
print(f"✅ Primitive set created with {len(pset.primitives)} primitives")

engine = EvolutionaryEngine(
    pset=pset,
    evaluate_func=lambda ind, tbx: problem.evaluate(ind, tbx),
    population_size=10,  # Small for testing
    max_height=5
)

print("✅ EvolutionaryEngine initialized successfully")