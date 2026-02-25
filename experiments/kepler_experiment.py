"""
Experiment runner for Kepler's Law problem.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.kepler import KeplerProblem
from core.evolution import EvolutionaryEngine
from utils.analysis import ResultAnalyzer

def run_kepler_experiment(population_size=200, generations=100, verbose=True):
    """Run Kepler's Law discovery experiment"""
    print("=" * 60)
    print("KEPLER'S LAW EXPERIMENT (Physics Discovery)")
    print("Target: Find T = a^1.5 from NASA data")
    print("=" * 60)
    
    # 1. Configure the problem
    problem = KeplerProblem()
    print(f"\n📋 Problem: {problem}")
    
    # 2. Create primitive set
    pset = problem.create_primitive_set()
    print(f"🔧 Primitive set created with {len(pset.primitives)} primitive types")
    
    # 3. Create evolutionary engine
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,
        population_size=population_size,
        cxpb=0.7,
        mutpb=0.2,
        tournsize=5,
        max_height=10
    )
    
    print(f"\n🚀 Starting physics discovery...")
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
    print(f"\n🎉 DISCOVERY COMPLETE")
    print(f"\n✨ Best evolved law:")
    print(f"   {str(best_individual)}")
    print(f"\n📊 Fitness (MSE): {best_individual.fitness.values[0]}")
    
    # 6. Validate on unseen outer planets
    print(f"\n🔍 Validation on outer planets (Uranus, Neptune):")
    validation_results = problem.validate_solution(best_individual, engine.toolbox)
    
    for i, result in enumerate(validation_results):
        status = "✓" if result['correct'] else "✗"
        if result['output'] == 'ERROR':
            output_str = 'ERROR'
        else:
            output_str = f"{result['output']:.2f}"
        print(f"   Test {i+1}: a={result['input'][0]} -> T={output_str} "
              f"(expected {result['expected']:.2f}) {status}")
    
    # 7. Calculate success rate
    correct = sum(1 for r in validation_results if r['correct'])
    success_rate = correct / len(validation_results) * 100
    
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
    results = run_kepler_experiment()
