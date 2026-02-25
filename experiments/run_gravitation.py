"""
Runner for Newton's Law of Gravitation discovery.
"""
import sys
import os
from deap import tools

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.gravitation import GravitationProblem
from core.evolution import EvolutionaryEngine
from core.simplification import ProgramSimplifier
from utils.fine_tune import refine_with_scipy

def run_gravitation_discovery(population_size=1000, generations=100, verbose=True):
    """Run Newton's Law discovery"""
    print("=" * 60)
    print("NEWTON'S LAW OF GRAVITATION - SCIENTIFIC DISCOVERY")
    print("Target: F = G * m1 * m2 / r^2")
    print("Strategy: Power Law PSet + NSGA-II + Dimensions")
    print("=" * 60)
    
    # 1. Setup problem
    problem = GravitationProblem(G=1.0)
    pset = problem.create_primitive_set()
    
    # 2. Setup engine
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,
        population_size=population_size,
        max_height=6
    )
    
    # Register NSGA2
    engine.toolbox.register("select", tools.selNSGA2)
    
    # 3. Run evolution
    print(f"\n🚀 Starting discovery...")
    print(f"   Population: {population_size}")
    print(f"   Generations: {generations}")
    
    population, log = engine.run_nsga2(
        generations=generations,
        seed=42,
        verbose=verbose
    )
    
    # 4. Extract Pareto Front
    print("\n📊 Extracting Pareto Front candidates...")
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    simplifier = ProgramSimplifier()
    candidates = []
    
    for ind in pareto_front[:10]:
        simplified = simplifier.simplify_individual(ind, pset)
        fitness = ind.fitness.values[0]
        complexity = ind.fitness.values[1]
        candidates.append((ind, simplified, fitness, complexity))
        
    candidates.sort(key=lambda x: x[2]) # Sort by MSE
    
    print("\nTop 5 Candidates (Simplified):")
    for i, (_, s, fit, comp) in enumerate(candidates[:5]):
        print(f" {i+1}: {s} (MSE: {fit:.4e}, Comp: {comp})")
        
    # 5. SciPy Fine-Tuning
    best_ind, best_formula, _, _ = candidates[0]
    
    print(f"\n🔧 Applying SciPy Fine-Tuning to best candidate:")
    print(f"   Input formula: {best_formula}")
    
    refined_formula, constants = refine_with_scipy(
        best_formula,
        [X for X, _ in problem.train_data],
        [y for _, y in problem.train_data],
        input_names=problem.input_names
    )
    
    print(f"   Refined formula: {refined_formula}")
    print(f"   Constants found: {constants}")
    
    # 6. Save results
    os.makedirs("results", exist_ok=True)
    with open("results/gravitation_pareto.txt", "w") as f:
        f.write("EXPERIMENT: Newton's Law of Gravitation\n")
        f.write(f"Target:      F = 1.0 * m1 * m2 / r^2\n")
        f.write(f"Discovered:  {refined_formula}\n")
        f.write(f"Constants:   {constants}\n\n")
        f.write("Pareto Front Candidates:\n")
        for i, (_, s, fit, comp) in enumerate(candidates):
            f.write(f"{i+1}: {s} (MSE: {fit:.4e}, Comp: {comp})\n")
    
    print(f"\n✅ Results saved to results/gravitation_pareto.txt")
    return refined_formula

if __name__ == "__main__":
    run_gravitation_discovery(population_size=1000, generations=100)
