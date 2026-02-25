"""
Scientific Discovery Runner: Damped Harmonic Oscillation with NSGA-II.
"""
import sys
import os
import math
import numpy as np
from deap import tools, base

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.damped_oscillation import DampedOscillationProblem
from core.evolution import EvolutionaryEngine
from core.simplification import ProgramSimplifier
from utils.analysis import ResultAnalyzer
import multiprocessing
import multiprocessing
from utils.fine_tune import refine_with_scipy

def evaluate_individual(individual, problem, toolbox):
    """Global function for multiprocessing support."""
    return problem.evaluate(individual, toolbox)

def run_oscillation_discovery(population_size=2000, generations=200, verbose=True):
    """Run Damped Harmonic Oscillation discovery"""
    # ... (rest of the setup is the same)
    print("=" * 60)
    print("DAMPED HARMONIC OSCILLATION - SCIENTIFIC DISCOVERY")
    print("Target: x(t) = A * exp(-delta * t) * cos(omega * t)")
    print("Strategy: NSGA-II + Curriculum Learning + SciPy Fine-Tuning")
    print("=" * 60)
    
    # 1. Configure the problem
    problem = DampedOscillationProblem()
    
    # 2. Create primitive set
    pset = problem.create_primitive_set()
    
    # 3. Create evolutionary engine
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,
        population_size=population_size,
        cxpb=0.8,
        mutpb=0.2,
        tournsize=5,
        max_height=7
    )
    
    # Explicitly register selNSGA2 for Pareto selection
    # Explicitly register evaluation with problem and toolbox context
    engine.toolbox.register("evaluate", evaluate_individual, problem=problem, toolbox=engine.toolbox)
    engine.toolbox.register("select", tools.selNSGA2)
    
    print(f"\n🚀 Starting discovery...")
    print(f"   Population: {population_size}")
    print(f"   Generations total: {generations}")
    
    # 4. Multi-Phase Run for Curriculum Learning
    population = engine.toolbox.population(n=population_size)
    
    for gen in range(1, generations + 1):
        problem.current_generation = gen
        
        if gen == 1:
            print(f"--- Phase 1: Focus on Initial Rise (t <= 1.0) ---")
        if gen == 21:
            print(f"--- Phase 2: Full Domain Discovery ---")
            
        population, log = engine.run_nsga2(
            generations=1,
            seed=42 + gen,
            verbose=False,
            population=population
        )
        
        if gen % 10 == 0 or gen == 1:
            # Get best by fitness
            best_ind = tools.selBest(population, 1)[0]
            # Count valid individuals (MSE < 1e6)
            valid_count = sum(1 for ind in population if ind.fitness.values[0] < 1e6)
            print(f"Gen {gen:3}: Best Fit = {best_ind.fitness.values[0]:.4e}, Valid = {valid_count}/{population_size}, Size = {len(best_ind)}")

    # 5. Pareto Front Analysis
    print(f"\n📊 Analyzing Pareto Front...")
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    pareto_front.sort(key=lambda ind: ind.fitness.values[1])
    
    simplifier = ProgramSimplifier()
    
    print(f"\n✨ Top Candidates:")
    candidates = []
    for i, ind in enumerate(pareto_front[:5]): # Show top 5
        mse = ind.fitness.values[0]
        complexity = ind.fitness.values[1]
        simplified = simplifier.simplify_individual(ind, pset)
        print(f"{i+1}: MSE={mse:.4e}, Size={complexity}, Formula: {simplified}")
        candidates.append((ind, simplified))
    
    # 6. SciPy Fine-Tuning on the most promising candidate
    # We look for something that has exp and cos
    best_candidate_idx = -1
    for i, (ind, simplified) in enumerate(candidates):
        if "exp" in simplified and "cos" in simplified:
            best_candidate_idx = i
            break
            
    if best_candidate_idx >= 0:
        best_ind, best_formula = candidates[best_candidate_idx]
    else:
        # Search the ENTIRE population for anything with exp and cos
        print("\n🔍 Structure not in Top Pareto. Searching entire population...")
        found_any = False
        for ind in population:
            s = simplifier.simplify_individual(ind, pset)
            if "exp" in s and "cos" in s:
                best_ind = ind
                best_formula = s
                found_any = True
                break
        if not found_any:
            print("❌ No individual with 'exp' and 'cos' found.")
            best_ind, best_formula = candidates[0]
        else:
            print(f"✅ Found candidate in population: {best_formula}")
    print(f"\n🔧 Applying SciPy Fine-Tuning to candidate {best_candidate_idx+1}:")
    print(f"   Input formula: {best_formula}")
    
    # Prepare data for fine-tuning
    X_data = [d[0] for d in problem.train_data]
    y_data = [d[1] for d in problem.train_data]
    
    refined_formula, constants = refine_with_scipy(best_formula, X_data, y_data, input_names=['t'])
    
    print(f"   Refined Formula: {refined_formula}")
    print(f"   Discovered Constants: {constants}")
    
    # 7. Validation
    # Compare with ground truth: delta=0.5, omega=6.28
    print(f"\n🎯 Discovery Comparison:")
    ground_truth = "x(t) = 1.0 * exp(-0.5 * t) * cos(6.283 * t)"
    print(f"   Target:   {ground_truth}")
    print(f"   Discovered: {refined_formula}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/oscillation_discovery.txt", "w", encoding="utf-8") as f:
        f.write("EXPERIMENT: Damped Harmonic Oscillation\n")
        f.write(f"Target:      {ground_truth}\n")
        f.write(f"Discovered:  {refined_formula}\n")
        f.write(f"Constants:   {constants}\n")
        f.write("\nPareto Front Candidates:\n")
        for i, (ind, formula) in enumerate(candidates):
            f.write(f"{i+1}: {formula} (MSE: {ind.fitness.values[0]:.4e})\n")
    
    return refined_formula

if __name__ == "__main__":
    run_oscillation_discovery()
