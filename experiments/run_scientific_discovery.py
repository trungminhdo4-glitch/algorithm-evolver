"""
Final Scientific Discovery Runner: Ideal Gas Law.
Validates the full Engine stack: NSGA-II + Dimensions + Chi^2 + Fine-Tuning.
"""
import sys
import os
from deap import tools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.ideal_gas import IdealGasProblem
from core.evolution import EvolutionaryEngine
from core.simplification import ProgramSimplifier
from utils.fine_tune import refine_with_scipy
from utils.publication_export import generate_latex_report

def run_scientific_gas_discovery():
    print("=" * 60)
    print("SCIENTIFIC DISCOVERY ENGINE - IDEAL GAS LAW")
    print("Target: T = (P * V) / (n * R)")
    print("Strategy: NSGA-II + Dimensional Analysis + Chi^2")
    print("=" * 60)
    
    # 1. Setup problem
    problem = IdealGasProblem()
    pset = problem.create_primitive_set()
    
    # 2. Setup engine
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=None,
        population_size=2000,
        cxpb=0.6,
        mutpb=0.4,
        max_height=10
    )
    
    # Custom evaluation wrapper to pass generation
    def wrapped_eval(individual):
        if not hasattr(wrapped_eval, "gen"):
            wrapped_eval.gen = 0
        return problem.evaluate(individual, engine.toolbox, generation=wrapped_eval.gen)

    engine.toolbox.register("evaluate", wrapped_eval)
    engine.toolbox.register("select", tools.selNSGA2)
    
    # 3. Run evolution with generation tracking
    print("\nStarting evolution with Dimensional Reward...")
    
    population = engine.toolbox.population(n=engine.population_size)
    for gen in range(0, 520, 40):
        wrapped_eval.gen = gen
        population, log = engine.run_nsga2(generations=40, seed=42, population=population)
        best_acc = tools.selBest(population, 1)[0].fitness.values[0]
        print(f"Gen {gen:3d} | Best Acc: {best_acc:.4e} | Pop Size: {len(population)}")
    
    # 4. Pareto extraction & Fine-Tuning
    print("\nExtracting Pareto Front and Fine-Tuning Constants...")
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    unique_formulas = set()
    diverse_results = []
    
    # Sort by accuracy first to find the best ones, then we'll filter
    pareto_front.sort(key=lambda ind: ind.fitness.values[0])
    
    simplifier = ProgramSimplifier()
    
    for ind in pareto_front:
        mse = ind.fitness.values[0]
        complexity = ind.fitness.values[1]
        
        # Initial simplification
        raw_simplified = simplifier.simplify_individual(ind, pset)
        
        # Filter duplicates or very similar
        if raw_simplified in unique_formulas:
            continue
        unique_formulas.add(raw_simplified)
        
        # Hybrid Step: MODUL 1 (Fine-Tuning)
        refined_formula, constants = refine_with_scipy(
            raw_simplified, 
            [X for X, _ in problem.train_data], 
            [y for _, y in problem.train_data],
            input_names=['P', 'V', 'n', 'R']
        )
        
        print(f"Comp: {complexity:<5.1f} | Chi2: {mse:<15.4e} | Formula: {refined_formula}")
        
        diverse_results.append({
            'complexity': complexity,
            'mse': mse,
            'simplified': refined_formula
        })
        
        if len(diverse_results) >= 20:
            break
            
    # 5. Export results: MODUL 5
    diverse_results.sort(key=lambda x: x['complexity']) # Re-sort for report
    report_path = generate_latex_report(diverse_results, save_path="results/scientific_gas_report.tex")
    print(f"\nScientific report generated: {report_path}")
    
    return diverse_results

if __name__ == "__main__":
    run_scientific_gas_discovery()
