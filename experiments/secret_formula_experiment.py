"""
Experiment definition for the Secret Formula problem.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.secret_formula import SecretFormulaProblem
from core.evolution import EvolutionaryEngine
from utils.analysis import ResultAnalyzer

def run_secret_formula_experiment(population_size=200, generations=50, verbose=True):
    """Run complete Secret Formula experiment"""
    print("=" * 60)
    print("SECRET FORMULA EXPERIMENT")
    print("Target: f(x, y) = x² + sin(y)")
    print("=" * 60)
    
    # 1. Configure the problem
    problem = SecretFormulaProblem()
    print(f"\n📋 Problem: {problem}")
    
    # 2. Create primitive set
    pset = problem.create_primitive_set()
    print(f"🔧 Primitive set created with {len(pset.primitives)} primitive types")
    
    # 3. Create evolutionary engine with correct parameters
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,
        population_size=population_size,
        cxpb=0.6,
        mutpb=0.3,
        tournsize=5,
        max_height=12  # Wie angefordert
    )
    
    print(f"\n🚀 Starting evolution...")
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
    print(f"\n✨ Best evolved algorithm:")
    print(f"   {str(best_individual)}")
    print(f"\n📊 Fitness (MSE): {best_individual.fitness.values[0]}")
    
    # 6. Validate on unseen data
    print(f"\n🔍 Validation on unseen data:")
    validation_results = problem.validate_solution(best_individual, engine.toolbox)
    
    for i, result in enumerate(validation_results):
        status = "✓" if result['correct'] else "✗"
        if result['output'] == 'ERROR':
            output_str = 'ERROR'
        else:
            output_str = f"{result['output']:.4f}"
        print(f"   Test {i+1}: f{result['input']} = {output_str} "
              f"(expected {result['expected']:.4f}) {status}")
    
    # 7. Calculate success rate
    correct = sum(1 for r in validation_results if r['correct'])
    success_rate = correct / len(validation_results) * 100
    print(f"\n📈 Success rate: {success_rate:.1f}% ({correct}/{len(validation_results)} tests)")
    
    # 8. Analysis & Visualization
    print(f"\n📊 Generating analysis plots...")
    ResultAnalyzer.plot_fitness_progress(log, f"Fitness Progress - {problem.name}", 
                                       save_path=f"results/{problem.name}_fitness.png")
    ResultAnalyzer.plot_tree(best_individual, f"Best Tree - {problem.name}", 
                           save_path=f"results/{problem.name}_tree.png")
    
    return {
        'problem': problem,
        'best_individual': best_individual,
        'fitness': best_individual.fitness.values[0],
        'validation': validation_results,
        'success_rate': success_rate,
        'log': log
    }

if __name__ == "__main__":
    results = run_secret_formula_experiment()
