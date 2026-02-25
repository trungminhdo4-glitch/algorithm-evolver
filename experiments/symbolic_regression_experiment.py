"""
Experiment definition for symbolic regression.
Connects problem configuration with evolutionary engine.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.symbolic_regression import SymbolicRegressionProblem
from core.evolution import EvolutionaryEngine
from utils.analysis import ResultAnalyzer

def run_symbolic_regression_experiment(population_size=100, generations=40, verbose=True):
    """Run complete symbolic regression experiment"""
    print("=" * 60)
    print("SYMBOLIC REGRESSION EXPERIMENT")
    print("Target: f(x, y) = x² + 2y")
    print("=" * 60)
    
    # 1. Configure the problem
    problem = SymbolicRegressionProblem()
    print(f"\n📋 Problem: {problem}")
    
    # 2. Create primitive set
    pset = problem.create_primitive_set()
    print(f"🔧 Primitive set created with {len(pset.primitives)} primitive types")
    
    # 3. Create evolutionary engine with correct parameters
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,  # Pass the bound method directly
        population_size=population_size,  # Use the parameter
        cxpb=0.6,
        mutpb=0.3,
        tournsize=5,
        max_height=10
    )
    
    print(f"\n🚀 Starting evolution...")
    print(f"   Population: {population_size}")
    print(f"   Generations: {generations}")
    print(f"   Crossover: {engine.cxpb}, Mutation: {engine.mutpb}")
    
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
        print(f"   Test {i+1}: f{result['input']} = {result['output']:.4f} "
              f"(expected {result['expected']}) {status}")
    
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
    # When run directly, use default parameters
    results = run_symbolic_regression_experiment()