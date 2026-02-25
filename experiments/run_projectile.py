"""
Experiment runner for Projectile Motion problem.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.projectile import ProjectileProblem
from core.evolution import EvolutionaryEngine
from utils.analysis import ResultAnalyzer

def run_projectile_experiment(population_size=1000, generations=100, verbose=True):
    """Run Projectile Motion discovery experiment"""
    print("=" * 60)
    print("PROJECTILE MOTION EXPERIMENT (Ballistics)")
    print("Target: Find distance = (v² * sin(2*angle)) / 9.81")
    print("=" * 60)
    
    # 1. Configure the problem
    problem = ProjectileProblem()
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
        max_height=12
    )
    
    print(f"\n🚀 Starting ballistic discovery...")
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
    print(f"\n✨ Best evolved formula:")
    print(f"   {str(best_individual)}")
    print(f"\n📊 Fitness (MSE): {best_individual.fitness.values[0]}")
    
    # 6. Validate on unseen cases
    print(f"\n🔍 Validation on test cases:")
    validation_results = problem.validate_solution(best_individual, engine.toolbox)
    
    for i, result in enumerate(validation_results):
        status = "✓" if result['correct'] else "✗"
        output_str = f"{result['output']:.2f}" if isinstance(result['output'], (int, float)) else str(result['output'])
        print(f"   Test {i+1}: v={result['input'][0]}, angle={result['input'][1]:.2f} "
              f"-> d={output_str} (expected {result['expected']:.2f}) {status}")
    
    # 7. Analysis & Visualization
    print(f"\n📊 Generating analysis plots...")
    ResultAnalyzer.plot_fitness_progress(log, f"Fitness Progress - Projectile Motion", 
                                       save_path=f"results/projectile_fitness.png")
    ResultAnalyzer.plot_tree(best_individual, f"Best Tree - Projectile Motion", 
                           save_path=f"results/projectile_tree.png")
    
    # 8. Calculate success rate
    correct = sum(1 for r in validation_results if r['correct'])
    success_rate = (correct / len(validation_results)) * 100
    print(f"\n📈 Success rate: {success_rate:.1f}% ({correct}/{len(validation_results)} tests)")
    
    return {
        'problem': problem,
        'best_individual': best_individual,
        'fitness': best_individual.fitness.values[0],
        'validation': validation_results,
        'success_rate': success_rate,
        'log': log
    }

if __name__ == "__main__":
    run_projectile_experiment()
