"""
Universal CSV Discovery Runner: Discover formulas from any CSV file.
"""
import sys
import os
import argparse
from deap import tools

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.csv_problem import CsvProblem
from core.evolution import EvolutionaryEngine
from core.simplification import ProgramSimplifier
from utils.fine_tune import refine_with_scipy

def run_csv_experiment(file_path, target_col, population_size=2000, generations=200, units_dict=None):
    print("=" * 60)
    print(f"SYMBOLIC DISCOVERY FROM CSV: {os.path.basename(file_path)}")
    print(f"Target: {target_col}")
    print("=" * 60)
    
    # 1. Load problem
    problem = CsvProblem(file_path, target_col, units_dict=units_dict)
    print(f"Loaded {len(problem.train_data)} data points.")
    print(f"Inputs: {problem.input_names}")
    print(f"Inferred Units: {problem.target_unit}")
    
    pset = problem.create_primitive_set(style="power")
    
    # 2. Setup Engine
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,
        population_size=population_size,
        max_height=6
    )
    
    engine.toolbox.register("select", tools.selNSGA2)
    
    # 3. Run Evolution
    print("\n🚀 Starting discovery...")
    population, log = engine.run_nsga2(
        generations=generations,
        seed=42,
        verbose=True
    )
    
    # Analyze final population for structures
    print("\n📊 Final Status Check:")
    print(f"Total evaluated: {problem._eval_count}")
    print(f"Consistent: {problem._consistent_count}")
    print(f"Correct Dim: {problem._target_dim_count}")
    
    # 4. Results & Fine-Tuning
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    simplifier = ProgramSimplifier()
    
    print("\n📊 Pareto Front Candidates:")
    candidates = []
    for ind in pareto_front[:10]:
        s = simplifier.simplify_individual(ind, pset)
        fitness = ind.fitness.values[0]
        complexity = ind.fitness.values[1]
        candidates.append((ind, s, fitness, complexity))
        print(f" - {s} (Score: {fitness:.4e}, Comp: {complexity})")
        
    candidates.sort(key=lambda x: x[2]) # Sort by Score (Accuracy * Penalty)
    
    # Fine-tune the best
    best_ind, best_str, _, _ = candidates[0]
    print(f"\n🔧 Fine-tuning best candidate: {best_str}")
    
    refined_formula, constants = refine_with_scipy(
        best_str,
        [X for X, _ in problem.train_data],
        [y for _, y in problem.train_data],
        input_names=problem.input_names
    )
    
    print(f"✅ Refined Formula: {refined_formula}")
    print(f"✅ Constants: {constants}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    out_file = f"results/csv_discovery_{problem.name}.txt"
    with open(out_file, "w") as f:
        f.write(f"EXPERIMENT: {problem.name}\n")
        f.write(f"Source:     {file_path}\n")
        f.write(f"Discovered: {refined_formula}\n")
        f.write(f"Constants:  {constants}\n\n")
        f.write("Pareto Front:\n")
        for i, (_, s, fit, comp) in enumerate(candidates):
            f.write(f"{i+1}: {s} (Score: {fit:.4e}, Comp: {comp})\n")
        
    print(f"\n💾 Results saved to {out_file}")
    return refined_formula

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to CSV")
    parser.add_argument("--target", type=str, required=True, help="Target column")
    parser.add_argument("--pop", type=int, default=2000)
    parser.add_argument("--gen", type=int, default=200)
    args = parser.parse_args()
    
    run_csv_experiment(args.file, args.target, args.pop, args.gen)
