"""
Experiment for sorting network problem.
"""
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problems.sorting_network import SortingNetworkProblem
from core.evolution import EvolutionaryEngine

# Try to import analysis tools (optional)
try:
    from utils.analysis import ResultAnalyzer
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False
    print("⚠️  Analysis module not found. Install with: pip install matplotlib networkx")

def run_sorting_network_experiment(population_size=150, generations=80, verbose=True, analyze=True):
    """
    Run sorting network evolution experiment.
    
    Args:
        population_size: Number of individuals in population
        generations: Number of generations to evolve
        verbose: Print detailed progress
        analyze: Generate analysis plots and reports
        
    Returns:
        Dictionary with experiment results
    """
    print("=" * 60)
    print("SORTING NETWORK EXPERIMENT")
    print("Target: Sort 3 numbers (return tuple in ascending order)")
    print("=" * 60)
    
    problem = SortingNetworkProblem()
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
        max_height=15  # Higher for more complex networks
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
        print(f"   Test {i+1}: sort{result['input']} = {result['output']} "
              f"(expected {result['expected']}) {status}")
    
    correct = sum(1 for r in validation_results if r['correct'])
    success_rate = correct / len(validation_results) * 100
    print(f"\n📈 Success rate: {success_rate:.1f}% ({correct}/{len(validation_results)} tests)")
    
    # Analyze the sorting network
    if correct == len(validation_results):
        print(f"\n🔬 Network Analysis:")
        func = engine.toolbox.compile(expr=best_individual)
        print(f"   Example: sort(3, 1, 2) = {func(3, 1, 2)}")
        print(f"   Example: sort(9, 0, 5) = {func(9, 0, 5)}")
    
    # Optional: Run detailed analysis
    if analyze and HAS_ANALYSIS:
        print(f"\n📊 PERFORMING DETAILED ANALYSIS...")
        _perform_detailed_analysis(problem, best_individual, engine.toolbox, log, 
                                   population_size, generations, success_rate, validation_results)
    elif analyze and not HAS_ANALYSIS:
        print(f"\n⚠️  Analysis tools not available. Install matplotlib and networkx.")
        print(f"   pip install matplotlib networkx")
    
    # Return results dictionary (compatible with run_experiment.py)
    results = {
        'problem': problem,
        'best_individual': best_individual,
        'fitness': best_individual.fitness.values[0],
        'validation': validation_results,
        'success_rate': success_rate,
        'log': log,  # Added for analysis
        'engine': engine,  # Added for analysis
        'toolbox': engine.toolbox  # Added for analysis
    }
    
    return results

def _perform_detailed_analysis(problem, best_individual, toolbox, log, 
                               population_size, generations, success_rate, validation_results):
    """
    Perform detailed analysis of the experiment results.
    """
    try:
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = f"analysis/sorting_network_pop{population_size}_gen{generations}_{timestamp}"
        os.makedirs(analysis_dir, exist_ok=True)
        
        print(f"   📁 Analysis directory: {analysis_dir}")
        
        # 1. Analyze the best algorithm
        print(f"   🔍 Analyzing best algorithm structure...")
        analysis_results = ResultAnalyzer.analyze_algorithm(best_individual, toolbox)
        
        if analysis_results:
            print(f"   ✅ Tree Info: {analysis_results['tree_info']['total_nodes']} nodes, "
                  f"depth {analysis_results['tree_info']['tree_depth']}")
            print(f"   ✅ Test Performance: {analysis_results['summary']['success_rate']:.1f}% success rate")
        
        # 2. Generate fitness progress plot
        print(f"   📈 Generating fitness progress plot...")
        fitness_plot_path = os.path.join(analysis_dir, "fitness_progress.png")
        ResultAnalyzer.plot_fitness_progress(
            log, 
            f"Sorting Network Fitness - Pop{population_size}, Gen{generations}",
            fitness_plot_path
        )
        
        # 3. Generate tree size evolution plot
        print(f"   📊 Generating tree size evolution plot...")
        size_plot_path = os.path.join(analysis_dir, "tree_size_evolution.png")
        ResultAnalyzer.plot_tree_size_progress(
            log,
            f"Tree Size Evolution - Pop{population_size}, Gen{generations}",
            size_plot_path
        )
        
        # 4. Generate tree visualization (if not too large)
        tree_size = len(best_individual)
        if tree_size < 100:  # Only visualize if tree is reasonably sized
            print(f"   🌳 Generating tree visualization ({tree_size} nodes)...")
            tree_plot_path = os.path.join(analysis_dir, "best_tree.png")
            ResultAnalyzer.plot_tree(
                best_individual,
                f"Best Sorting Network (Fitness: {best_individual.fitness.values[0]:.4f})",
                tree_plot_path,
                max_nodes=100
            )
        else:
            print(f"   ⚠️  Tree too large for visualization ({tree_size} nodes)")
        
        # 5. Generate comprehensive report
        print(f"   📄 Generating comprehensive report...")
        results_dict = {
            'problem': str(problem),
            'best_individual': str(best_individual),
            'fitness': best_individual.fitness.values[0],
            'validation': validation_results,
            'success_rate': success_rate,
            'population_size': population_size,
            'generations': generations,
            'tree_info': {
                'total_nodes': len(best_individual),
                'tree_depth': best_individual.height
            } if analysis_results else {}
        }
        
        report_dir = ResultAnalyzer.generate_report(
            "sorting_network",
            results_dict,
            log,
            analysis_dir
        )
        
        # 6. Save raw data for reproducibility
        print(f"   💾 Saving raw experiment data...")
        raw_data_path = os.path.join(analysis_dir, "experiment_data.json")
        _save_experiment_data(raw_data_path, problem, best_individual, log, results_dict)
        
        print(f"\n✅ Analysis complete! Results saved to: {analysis_dir}")
        
        # 7. Print analysis summary
        _print_analysis_summary(analysis_results, tree_size, best_individual.fitness.values[0])
        
    except Exception as e:
        print(f"   ❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def _save_experiment_data(filepath, problem, best_individual, log, results_dict):
    """Save experiment data for reproducibility."""
    try:
        # Prepare data for JSON serialization
        data = {
            'experiment': 'sorting_network',
            'timestamp': datetime.now().isoformat(),
            'problem': str(problem),
            'best_algorithm': str(best_individual),
            'results': results_dict,
            'evolution_data': {
                'generations': log.select('gen') if log else [],
                'fitness_min': log.chapters['fitness'].select('min') if log and hasattr(log, 'chapters') and 'fitness' in log.chapters else [],
                'fitness_avg': log.chapters['fitness'].select('avg') if log and hasattr(log, 'chapters') and 'fitness' in log.chapters else [],
                'size_avg': log.chapters['size'].select('avg') if log and hasattr(log, 'chapters') and 'size' in log.chapters else []
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
            
        print(f"      Raw data saved: {filepath}")
    except Exception as e:
        print(f"      ⚠️ Could not save raw data: {e}")

def _print_analysis_summary(analysis_results, tree_size, fitness):
    """Print a concise analysis summary."""
    if analysis_results:
        print(f"\n📋 ANALYSIS SUMMARY:")
        print(f"   ──────────────────────────────")
        print(f"   Tree Complexity: {tree_size} nodes")
        print(f"   Tree Depth: {analysis_results['tree_info']['tree_depth']}")
        print(f"   Final Fitness: {fitness:.6f}")
        print(f"   Test Success: {analysis_results['summary']['success_rate']:.1f}%")
        print(f"   Average Error: {analysis_results['summary']['average_error']:.6f}")
        
        # Identify algorithm type
        algorithm_str = str(analysis_results['tree_info']['expression'])
        if 'compare_swap' in algorithm_str:
            print(f"   Algorithm Type: Compare-Swap Network")
        elif 'if_then_else' in algorithm_str:
            print(f"   Algorithm Type: Conditional Logic")
        else:
            print(f"   Algorithm Type: Custom")
        
        # Check for common patterns
        if algorithm_str.count('compare_swap') >= 2:
            print(f"   Network Depth: {algorithm_str.count('compare_swap')} compare-swap operations")
        
        print(f"   ──────────────────────────────")

# Quick analysis function without full ResultAnalyzer
def quick_analyze_algorithm(individual, toolbox, test_cases=None):
    """
    Quick analysis without requiring full analysis module.
    """
    try:
        func = toolbox.compile(expr=individual)
        
        if test_cases is None:
            test_cases = [
                (1, 2, 3), (3, 2, 1), (2, 1, 3),
                (1, 1, 2), (-1, -3, 0), (5, 3, 5)
            ]
        
        results = {
            'total_nodes': len(individual),
            'tree_depth': individual.height,
            'expression': str(individual),
            'test_results': [],
            'correct_count': 0,
            'total_tests': len(test_cases)
        }
        
        for test_input in test_cases:
            try:
                output = func(*test_input)
                expected = tuple(sorted(test_input))
                
                if isinstance(output, tuple) and len(output) == 3:
                    error = sum((o - e)**2 for o, e in zip(output, expected))
                    correct = error < 0.001
                    if correct:
                        results['correct_count'] += 1
                else:
                    error = float('inf')
                    correct = False
                
                results['test_results'].append({
                    'input': test_input,
                    'output': output,
                    'expected': expected,
                    'error': error,
                    'correct': correct
                })
                
            except Exception as e:
                results['test_results'].append({
                    'input': test_input,
                    'output': f'ERROR: {e}',
                    'expected': tuple(sorted(test_input)),
                    'error': float('inf'),
                    'correct': False
                })
        
        results['success_rate'] = (results['correct_count'] / results['total_tests']) * 100
        return results
        
    except Exception as e:
        print(f"❌ Error in quick analysis: {e}")
        return None

if __name__ == "__main__":
    # When run directly, use default parameters
    print("🧬 Running Sorting Network Experiment (Standalone Mode)")
    print("-" * 60)
    
    results = run_sorting_network_experiment(
        population_size=150,
        generations=80,
        verbose=True,
        analyze=True
    )
    
    print(f"\n✅ Experiment completed!")
    
    # Quick analysis if full analysis not available
    if not HAS_ANALYSIS:
        print(f"\n📊 Quick Analysis (without visualization):")
        quick_results = quick_analyze_algorithm(
            results['best_individual'],
            results['toolbox']
        )
        
        if quick_results:
            print(f"   Tree Size: {quick_results['total_nodes']} nodes")
            print(f"   Tree Depth: {quick_results['tree_depth']}")
            print(f"   Success Rate: {quick_results['success_rate']:.1f}%")
            print(f"   Correct Tests: {quick_results['correct_count']}/{quick_results['total_tests']}")