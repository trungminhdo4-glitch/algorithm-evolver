"""
Experiment Runner for Damped Harmonic Oscillation.
Discovers x(t) = A * exp(-delta * t) * cos(omega * t + phi).
"""
import sys
import os
from deap import tools
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.damped_oscillation import DampedOscillationProblem
from core.evolution import EvolutionaryEngine
from core.simplification import ProgramSimplifier
from utils.fine_tune import refine_with_scipy
from utils.publication_export import generate_latex_report

def evaluate_individual(individual, problem, toolbox):
    """Global function for multiprocessing support."""
    return problem.evaluate(individual, toolbox)

def run_oscillation_discovery():
    print("=" * 60)
    print("SCIENTIFIC DISCOVERY ENGINE - DAMPED OSCILLATION")
    print("Target: x(t) = A * exp(-delta * t) * cos(omega * t + phi)")
    print("Strategy: NSGA-II + Transcendental Primitives + Fine-Tuning")
    print("=" * 60)
    
    # 1. Setup problem
    problem = DampedOscillationProblem()
    pset = problem.create_primitive_set()
    
    # 2. Setup engine
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=None,
        population_size=1000,
        cxpb=0.6,
        mutpb=0.4,
        max_height=8
    )
    
    # Register evaluation with context
    engine.toolbox.register("evaluate", evaluate_individual, problem=problem, toolbox=engine.toolbox)
    engine.toolbox.register("select", tools.selNSGA2)
    
    # 3. Run evolution
    print("\nStarting evolution...")
    with multiprocessing.Pool() as pool:
        engine.toolbox.register("map", pool.map)
        
        population = engine.toolbox.population(n=engine.population_size)
        generations = 500
        
        for gen in range(0, generations, 10):
            problem.current_generation = gen
            population, log = engine.run_nsga2(generations=10, seed=42, population=population)
            best_fit = tools.selBest(population, 1)[0].fitness.values[0]
            print(f"Gen {gen:3d} | Best Fitness: {best_fit:.4e}")
    
    # 4. Pareto extraction & Fine-Tuning
    print("\nExtracting Pareto Front and Fine-Tuning...")
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    unique_formulas = set()
    diverse_results = []
    
    # Sort by fitness
    pareto_front.sort(key=lambda ind: ind.fitness.values[0])
    
    simplifier = ProgramSimplifier()
    
    for ind in pareto_front:
        fitness = ind.fitness.values[0]
        complexity = ind.fitness.values[1]
        
        # Simplify
        raw_simplified = simplifier.simplify_individual(ind, pset)
        
        if raw_simplified in unique_formulas:
            continue
        unique_formulas.add(raw_simplified)
        
        # Hybrid Step: Fine-Tuning numerical constants
        refined_formula, constants = refine_with_scipy(
            raw_simplified, 
            [X for X, _ in problem.train_data], 
            [y for _, y in problem.train_data],
            input_names=['t']
        )
        
        print(f"Comp: {complexity:<5.1f} | MSE: {fitness:<15.4e} | Formula: {refined_formula}")
        
        diverse_results.append({
            'complexity': complexity,
            'mse': fitness,
            'simplified': refined_formula
        })
        
        if len(diverse_results) >= 15:
            break
            
    # 5. Export results
    diverse_results.sort(key=lambda x: x['complexity'])
    report_path = generate_latex_report(diverse_results, save_path="results/damped_oscillation_report.tex")
    print(f"\nDiscovery report generated: {report_path}")
    
    return diverse_results

if __name__ == "__main__":
    run_oscillation_discovery()
