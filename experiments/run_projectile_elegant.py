"""
Elegant Experiment runner for Projectile Motion: Discover simple laws.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.projectile import ProjectileProblem
from core.evolution import EvolutionaryEngine
from utils.analysis import ResultAnalyzer
from core.simplification import ProgramSimplifier

def run_projectile_elegant(population_size=500, generations=100, verbose=True):
    """Run Projectile Motion discovery with Parsimony Pressure for elegant solutions"""
    print("=" * 60)
    print("PROJECTILE MOTION - ELEGANT DISCOVERY (RADICAL CURE)")
    print("Target: Find distance = (v² * sin(2*angle)) / 9.81")
    print("Strategy: Aggressive Parsimony Pressure (Penalty: 5.0 per node)")
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
        cxpb=0.7,
        mutpb=0.2,
        tournsize=5,
        max_height=5 # Strict limit for radical elegance
    )
    
    print(f"\n🚀 Starting final optimized discovery...")
    print(f"   Population: {population_size}")
    print(f"   Generations: {generations}")
    
    # 4. Run evolution
    population, log, hall_of_fame = engine.run(
        generations=generations,
        seed=42,
        verbose=verbose
    )
    
    # 5. Analyze results
    best_individual = hall_of_fame[0]
    print(f"\n🎉 RADICAL DISCOVERY COMPLETE")
    print(f"\n✨ Best evolved formula (Size: {len(best_individual)} nodes):")
    print(f"   {str(best_individual)}")
    
    # 5b. Symbolic Simplification
    print(f"\n🧹 Simplifying formula with SymPy...")
    simplifier = ProgramSimplifier()
    simplified_formula = simplifier.simplify_individual(best_individual, pset)
    print(f"   Simplified: {simplified_formula}")
    
    # 5c. LaTeX Export
    latex_formula = ResultAnalyzer.to_latex(simplified_formula)
    print(f"\n📝 LaTeX Format:")
    print(f"   {latex_formula}")
    
    print(f"\n📊 Fitness (MSE, Size): {best_individual.fitness.values}")
    
    # 6. Validation on test cases
    print(f"\n🔍 Validation on test cases:")
    validation_results = problem.validate_solution(best_individual, engine.toolbox)
    
    correct_count = 0
    for i, result in enumerate(validation_results):
        status = "✓" if result['correct'] else "✗"
        if result['correct']: correct_count += 1
        output_str = f"{result['output']:.2f}" if isinstance(result['output'], (int, float)) else str(result['output'])
        print(f"   Test {i+1}: v={result['input'][0]}, angle={result['input'][1]:.2f} "
              f"-> d={output_str} (expected {result['expected']:.2f}) {status}")
    
    success_rate = (correct_count / len(validation_results)) * 100
    print(f"\n📈 Success rate: {success_rate:.1f}%")
    
    # 7. Analysis & Visualization
    print(f"\n📊 Generating analysis plots...")
    ResultAnalyzer.plot_fitness_progress(log, "Fitness Progress (Radical Parsimony)", 
                                       save_path="results/radical_fitness.png")
    ResultAnalyzer.plot_tree_size_progress(log, "Tree Size Evolution (Radical Parsimony)", 
                                         save_path="results/radical_size.png")
    ResultAnalyzer.plot_tree(best_individual, f"Radical Tree (Nodes: {len(best_individual)})", 
                           save_path="results/radical_tree.png")
    
    # Save text result
    with open("results/projectile_radical.txt", "w", encoding="utf-8") as f:
        f.write("Experiment: Projectile Radical Cure\n")
        f.write(f"Weights: MSE (1.0), Size (5.0)\n")
        f.write(f"Best Formula: {str(best_individual)}\n")
        f.write(f"Simplified: {simplified_formula}\n")
        f.write(f"LaTeX: {latex_formula}\n")
        f.write(f"Size: {len(best_individual)} nodes\n")
        f.write(f"MSE: {best_individual.fitness.values[0]}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")
    
    return {
        'best_individual': best_individual,
        'simplified': simplified_formula,
        'latex': latex_formula,
        'log': log,
        'success_rate': success_rate
    }

if __name__ == "__main__":
    run_projectile_elegant()
