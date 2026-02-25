"""
Scientific Discovery Runner: Projectile Motion with NSGA-II Pareto Optimization.
"""
import sys
import os
import math
from deap import tools, base

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.projectile import ProjectileProblem
from core.evolution import EvolutionaryEngine
from core.simplification import ProgramSimplifier
from utils.analysis import ResultAnalyzer

def run_projectile_nsga2(population_size=500, generations=500, verbose=True):
    """Run Projectile Motion discovery with NSGA-II for scientific discovery"""
    print("=" * 60)
    print("PROJECTILE MOTION - SCIENTIFIC DISCOVERY (NSGA-II)")
    print("Target: Find distance = (v² * sin(2*angle)) / 9.81")
    print("Strategy: Pareto Optimization (Accuracy vs. Symbolic Complexity)")
    print("=" * 60)
    
    # 1. Configure the problem
    problem = ProjectileProblem()
    
    # 2. Create primitive set
    pset = problem.create_primitive_set()
    
    # 3. Create evolutionary engine
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,
        population_size=population_size,
        cxpb=0.8, # Higher crossover for complex discovery
        mutpb=0.2,
        tournsize=5,
        max_height=7 # Sufficient depth for v² * sin(2*a) / g
    )
    
    # Explicitly register selNSGA2 for Pareto selection
    engine.toolbox.register("select", tools.selNSGA2)
    
    print(f"\n🚀 Starting NSGA-II discovery...")
    print(f"   Population: {population_size}")
    print(f"   Generations: {generations}")
    print(f"   Max Tree Height: 7")
    
    # 4. Run NSGA-II evolution
    population, log = engine.run_nsga2(
        generations=generations,
        seed=42,
        verbose=verbose
    )
    
    # 5. Pareto Front Analysis
    print(f"\n📊 Analyzing Pareto Front (Accuracy vs. Complexity)...")
    
    # Extract non-dominated individuals (Pareto Front)
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    # Sort Pareto front by complexity (ascending)
    pareto_front.sort(key=lambda ind: ind.fitness.values[1])
    
    print(f"\n✨ Pareto Front Results:")
    print(f"{'#':<3} | {'MSE':<12} | {'Complexity':<10} | {'Formula (Simplified)'}")
    print("-" * 100)
    
    simplifier = ProgramSimplifier()
    
    results_list = []
    for i, ind in enumerate(pareto_front):
        mse = ind.fitness.values[0]
        complexity = ind.fitness.values[1]
        
        # Simplify for display
        simplified = simplifier.simplify_individual(ind, pset)
        
        print(f"{i+1:<3} | {mse:<12.4f} | {complexity:<10.1f} | {simplified}")
        results_list.append({
            'index': i+1,
            'mse': mse,
            'complexity': complexity,
            'formula': str(ind),
            'simplified': simplified
        })
    
    # 6. Select "Best Balance" Solution (manually or via a heuristic)
    # Usually the solution with MSE < threshold and lowest complexity
    best_scientific = None
    for res in results_list:
        if res['mse'] < 50.0: # Arbitrary "good enough" accuracy
            best_scientific = res
            break
    
    if not best_scientific:
        best_scientific = results_list[-1] # Fallback to most accurate
        
    print(f"\n🏆 Best Scientific Candidate (Index {best_scientific['index']}):")
    print(f"   Simplified: {best_scientific['simplified']}")
    print(f"   LaTeX: {ResultAnalyzer.to_latex(best_scientific['simplified'])}")
    
    # 7. Visualization
    print(f"\n🎨 Generating Pareto plots...")
    ResultAnalyzer.plot_fitness_progress(log, "NSGA-II Fitness Progress", 
                                       save_path="results/nsga2_fitness.png")
    ResultAnalyzer.plot_tree_size_progress(log, "NSGA-II Complexity Evolution", 
                                         save_path="results/nsga2_complexity.png")
    
    # Save text result
    with open("results/projectile_nsga2.txt", "w", encoding="utf-8") as f:
        f.write("Experiment: Projectile Scientific Discovery (NSGA-II)\n")
        f.write("Pareto Front:\n")
        f.write(f"{'MSE':<12} | {'Complexity':<10} | {'Formula'}\n")
        for res in results_list:
            f.write(f"{res['mse']:<12.4f} | {res['complexity']:<10.1f} | {res['simplified']}\n")
            
    return {
        'pareto_front': results_list,
        'best': best_scientific
    }

if __name__ == "__main__":
    run_projectile_nsga2()
