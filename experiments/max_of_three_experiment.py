"""
Experiment for max-of-three problem.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.max_of_three import MaxOfThreeProblem
from core.evolution import EvolutionaryEngine

def run_max_of_three_experiment(population_size=100, generations=60, verbose=True):
    print("=" * 60)
    print("MAX OF THREE EXPERIMENT")
    print("Target: max(x, y, z)")
    print("=" * 60)
    
    problem = MaxOfThreeProblem()
    print(f"\n📋 Problem: {problem}")
    
    pset = problem.create_primitive_set()
    print(f"🔧 Primitive set created with {len(pset.primitives)} primitive types")
    
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,
        population_size=population_size,
        cxpb=0.6,
        mutpb=0.3,
        tournsize=5,
        max_height=12  # Slightly higher for more complex logic
    )
    
    print(f"\n🚀 Starting evolution...")
    print(f"   Population: {population_size}")
    print(f"   Generations: {generations}")
    print(f"   Crossover: {engine.cxpb}, Mutation: {engine.mutpb}")
    
    population, log, hall_of_fame = engine.run(
        generations=generations,
        seed=42,
        verbose=verbose
    )
    
    best_individual = hall_of_fame[0]
    print(f"\n🎉 EVOLUTION COMPLETE")
    print(f"\n✨ Best evolved algorithm:")
    print(f"   {str(best_individual)}")
    print(f"\n📊 Fitness (MSE): {best_individual.fitness.values[0]}")
    
    print(f"\n🔍 Validation on unseen data:")
    validation_results = problem.validate_solution(best_individual, engine.toolbox)
    
    for i, result in enumerate(validation_results):
        status = "✓" if result['correct'] else "✗"
        print(f"   Test {i+1}: max{result['input']} = {result['output']:.4f} "
              f"(expected {result['expected']}) {status}")
    
    correct = sum(1 for r in validation_results if r['correct'])
    success_rate = correct / len(validation_results) * 100
    print(f"\n📈 Success rate: {success_rate:.1f}% ({correct}/{len(validation_results)} tests)")
    
    return {
        'problem': problem,
        'best_individual': best_individual,
        'fitness': best_individual.fitness.values[0],
        'validation': validation_results,
        'success_rate': success_rate
    }

if __name__ == "__main__":
    results = run_max_of_three_experiment()