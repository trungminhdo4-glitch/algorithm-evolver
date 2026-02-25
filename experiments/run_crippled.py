"""
Experiment runner for the "Crippled Tools" test.
Solving x² + sin(y) without multiplication.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.secret_formula import SecretFormulaProblem
from core.evolution import EvolutionaryEngine
from utils.analysis import ResultAnalyzer

def run_crippled_experiment(population_size=1000, generations=200, verbose=True):
    """Run Crippled Tools experiment"""
    print("=" * 60)
    print("CRIPPLED TOOLS EXPERIMENT")
    print("Target: f(x, y) = x² + sin(y) WITHOUT multiplication")
    print("=" * 60)
    
    # 1. Configure the problem
    problem = SecretFormulaProblem()
    print(f"\n📋 Problem: {problem}")
    
    # 2. Create primitive set
    pset = problem.create_primitive_set()
    print(f"🔧 Primitive set created with {len(pset.primitives)} primitive types")
    print(f"   (Multiplication removed, exp/log available)")
    
    # 3. Create evolutionary engine
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,
        population_size=population_size,
        cxpb=0.6,
        mutpb=0.3,
        tournsize=7,  # Slightly higher tournament size for harder problem
        max_height=12
    )
    
    print(f"\n🚀 Starting crippled evolution...")
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
    print(f"\n🎉 EVOLUTION COMPLETE")
    print(f"\n✨ Best evolved algorithm (Workaround?):")
    print(f"   {str(best_individual)}")
    print(f"\n📊 Fitness (MSE): {best_individual.fitness.values[0]}")
    
    # 6. Validate on unseen data
    print(f"\n🔍 Validation on unseen data:")
    validation_results = problem.validate_solution(best_individual, engine.toolbox)
    
    for i, result in enumerate(validation_results):
        status = "✓" if result['correct'] else "✗"
        output_str = f"{result['output']:.4f}" if isinstance(result['output'], (int, float)) else str(result['output'])
        print(f"   Test {i+1}: f{result['input']} = {output_str} "
              f"(expected {result['expected']:.4f}) {status}")
    
    # 7. Analysis & Visualization
    print(f"\n📊 Generating analysis plots...")
    ResultAnalyzer.plot_fitness_progress(log, f"Fitness Progress - Crippled Tools", 
                                       save_path=f"results/crippled_fitness.png")
    ResultAnalyzer.plot_tree(best_individual, f"Best Tree - Crippled Tools", 
                           save_path=f"results/crippled_tree.png")
    
    # Save detailed result
    os.makedirs("results", exist_ok=True)
    with open("results/crippled_formula.txt", "w", encoding="utf-8") as f:
        f.write("Experiment: Crippled Tools (No Multiplication)\n")
        f.write(f"Population: {population_size}, Generations: {generations}\n")
        f.write(f"Best Algorithm: {str(best_individual)}\n")
        f.write(f"Fitness: {best_individual.fitness.values[0]}\n")
    
    return {
        'problem': problem,
        'best_individual': best_individual,
        'fitness': best_individual.fitness.values[0],
        'validation': validation_results,
        'log': log
    }

if __name__ == "__main__":
    run_crippled_experiment()
