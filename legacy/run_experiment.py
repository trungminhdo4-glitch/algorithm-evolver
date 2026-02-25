"""
Main entry point for running experiments.
Provides a clean interface to run different experiments.
"""
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(
        description="Run Genetic Programming Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py symbolic_regression --population 100 --generations 40
  python run_experiment.py max_of_three --population 150 --generations 60 --verbose
  python run_experiment.py sorting_network --population 200 --generations 80 --verbose
        """
    )
    parser.add_argument('experiment', 
                       choices=['symbolic_regression', 'max_of_three', 'sorting_network', 'secret_formula', 'kepler', 'projectile'],
                       help='Experiment to run')
    parser.add_argument('--generations', type=int, default=40,
                       help='Number of generations (default: 40)')
    parser.add_argument('--population', type=int, default=100,
                       help='Population size (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress')
    
    args = parser.parse_args()
    
    print(f"🧬 Algorithm Evolver - Genetic Programming System")
    print(f"Experiment: {args.experiment}")
    print(f"Generations: {args.generations}, Population: {args.population}")
    print(f"Random seed: {args.seed}")
    print("-" * 60)
    
    if args.experiment == 'symbolic_regression':
        try:
            from experiments.symbolic_regression_experiment import run_symbolic_regression_experiment
            results = run_symbolic_regression_experiment(
                population_size=args.population,
                generations=args.generations,
                verbose=args.verbose
            )
        except ImportError as e:
            print(f"\n❌ Error: Symbolic Regression experiment not properly set up")
            print(f"   Error details: {e}")
            print("\n🔧 Check that these files exist:")
            print("   1. problems/symbolic_regression.py")
            print("   2. experiments/symbolic_regression_experiment.py")
            sys.exit(1)
            
    elif args.experiment == 'max_of_three':
        try:
            from experiments.max_of_three_experiment import run_max_of_three_experiment
            results = run_max_of_three_experiment(
                population_size=args.population,
                generations=args.generations,
                verbose=args.verbose
            )
        except ImportError as e:
            print(f"\n❌ Error: Max of Three experiment not properly set up")
            print(f"   Error details: {e}")
            print("\n🔧 Create these files:")
            print("   1. problems/max_of_three.py")
            print("   2. experiments/max_of_three_experiment.py")
            sys.exit(1)
            
    elif args.experiment == 'sorting_network':
        try:
            from experiments.sorting_network_experiment import run_sorting_network_experiment
            results = run_sorting_network_experiment(
                population_size=args.population,
                generations=args.generations,
                verbose=args.verbose
            )
        except ImportError as e:
            print(f"\n❌ Error: Sorting Network experiment not properly set up")
            print(f"   Error details: {e}")
            print("\n🔧 Create these files:")
            print("   1. problems/sorting_network.py")
            print("   2. experiments/sorting_network_experiment.py")
            sys.exit(1)
            
    elif args.experiment == 'secret_formula':
        try:
            from experiments.secret_formula_experiment import run_secret_formula_experiment
            results = run_secret_formula_experiment(
                population_size=args.population,
                generations=args.generations,
                verbose=args.verbose
            )
        except ImportError as e:
            print(f"\n❌ Error: Secret Formula experiment not properly set up")
            print(f"   Error details: {e}")
            print("\n🔧 Check that these files exist:")
            print("   1. problems/secret_formula.py")
            print("   2. experiments/secret_formula_experiment.py")
            sys.exit(1)
    elif args.experiment == 'kepler':
        try:
            from experiments.kepler_experiment import run_kepler_experiment
            results = run_kepler_experiment(
                population_size=args.population,
                generations=args.generations,
                verbose=args.verbose
            )
        except ImportError as e:
            print(f"\n❌ Error: Kepler experiment not properly set up")
            print(f"   Error details: {e}")
            print("\n🔧 Check that these files exist:")
            print("   1. problems/kepler.py")
            print("   2. experiments/kepler_experiment.py")
            sys.exit(1)
    elif args.experiment == 'projectile':
        try:
            from experiments.run_projectile import run_projectile_experiment
            results = run_projectile_experiment(
                population_size=args.population,
                generations=args.generations,
                verbose=args.verbose
            )
        except ImportError as e:
            print(f"\n❌ Error: Projectile experiment not properly set up")
            print(f"   Error details: {e}")
            sys.exit(1)
    else:
        print(f"Error: Unknown experiment '{args.experiment}'")
        sys.exit(1)
    
    print(f"\n✅ Experiment completed successfully!")
    
    # Save results to file
    if args.experiment == 'symbolic_regression':
        results_file = f"results/symbolic_regression_pop{args.population}_gen{args.generations}.txt"
    elif args.experiment == 'max_of_three':
        results_file = f"results/max_of_three_pop{args.population}_gen{args.generations}.txt"
    elif args.experiment == 'secret_formula':
        results_file = f"results/secret_formula_pop{args.population}_gen{args.generations}.txt"
    elif args.experiment == 'kepler':
        results_file = f"results/kepler_pop{args.population}_gen{args.generations}.txt"
    elif args.experiment == 'projectile':
        results_file = f"results/projectile_pop{args.population}_gen{args.generations}.txt"
    else:  # sorting_network
        results_file = f"results/sorting_network_pop{args.population}_gen{args.generations}.txt"
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save results with UTF-8 encoding
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Population: {args.population}, Generations: {args.generations}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Best Algorithm: {results['best_individual']}\n")
        f.write(f"Fitness: {results['fitness']}\n")
        f.write(f"Success Rate: {results['success_rate']:.1f}%\n")
        f.write("\nValidation Results:\n")
        for i, val in enumerate(results['validation']):
            status = "✓" if val['correct'] else "✗"
            # Handle different output types (float, tuple, etc.)
            if isinstance(val['output'], tuple):
                output_str = str(val['output'])
            else:
                output_str = f"{val['output']:.4f}"
                
            if isinstance(val['expected'], tuple):
                expected_str = str(val['expected'])
            else:
                expected_str = f"{val['expected']:.4f}"
                
            f.write(f"  Test {i+1}: {val['input']} -> {output_str} (expected {expected_str}) {status}\n")
    
    print(f"📁 Results saved to: {results_file}")
    
    # Print summary of all available experiments
    print(f"\n📊 Available Experiments:")
    print(f"   symbolic_regression  - Evolve f(x,y) = x² + 2y")
    print(f"   max_of_three         - Evolve max(x,y,z)")
    print(f"   sorting_network      - Evolve sorting network for 3 numbers")
    print(f"   secret_formula       - Evolve f(x,y) = x² + sin(y)")
    print(f"   kepler               - Discover Kepler's Third Law (T = a^1.5)")
    print(f"   projectile           - Discover Projectile Motion Law (d = v²*sin(2*angle)/g)")

if __name__ == "__main__":
    main()