"""
Scientific Discovery Engine - Master Demo CLI
"""
import os
import sys
import multiprocessing

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    print("=" * 70)
    print("      🔬 SCIENTIFIC DISCOVERY ENGINE: UNLOCKING PHYSICAL LAWS 🔬")
    print("=" * 70)
    print(" (c) 2026 Powered by Genetic Programming & Hybrid Optimization")
    print("=" * 70)

def main():
    clear_screen()
    print_banner()
    
    print("\nPlease choose an experiment to run:")
    print(" [1] Kepler's Law          - Planetary motion discovery (T \u221d a\u00b3\u00b2)")
    print(" [2] Ballistic Motion      - Classical mechanics / Projectile distance")
    print(" [3] Ideal Gas Law         - Thermodynamics (P*V = n*R*T)")
    print(" [4] Damped Oscillation    - Complex dynamics & Dimensional Analysis")
    print(" [5] Newton's Law          - Gravitational discovery (F \u221d m1*m2/r\u00b2)")
    print(" [6] NASA Airfoil Discovery - Real Data: Noise law discovery (P \u221d U\u2075)")
    print(" [0] Exit")
    
    choice = input("\n> ")
    
    if choice == '0':
        print("Goodbye!")
        return

    use_mp = input("\nEnable Multiprocessing? (y/n) [Default: n]: ").lower() == 'y'
    
    if use_mp:
        print("🚀 Multiprocessing enabled. Utilizing all CPU cores.")
    
    print("\n" + "-" * 40)
    
    try:
        if choice == '1':
            from experiments.kepler_experiment import run_kepler_experiment
            run_kepler_experiment(population_size=300, generations=100)
            print(f"\n\u2705 Result saved to: results/kepler_fitness.png")
            
        elif choice == '2':
            from experiments.run_projectile_nsga2 import run_projectile_nsga2
            run_projectile_nsga2(population_size=500, generations=100)
            print(f"\n\u2705 Results saved to: results/projectile_nsga2.txt")
            
        elif choice == '3':
            from experiments.run_scientific_discovery import run_scientific_gas_discovery
            run_scientific_gas_discovery()
            print(f"\n\u2705 LaTeX Report saved to: results/scientific_gas_report.tex")
            
        elif choice == '4':
            from experiments.run_oscillation import run_oscillation_discovery
            run_oscillation_discovery(population_size=1000, generations=150)
            print(f"\n\u2705 Discovery log saved to: results/oscillation_discovery.txt")
            
        elif choice == '5':
            from experiments.run_gravitation import run_gravitation_discovery
            run_gravitation_discovery(population_size=500, generations=50)
            print(f"\n\u2705 Discovery log saved to: results/gravitation_pareto.txt")
            
        elif choice == '6':
            from utils.data_fetcher import fetch_nasa_airfoil
            from experiments.run_nasa_airfoil import run_nasa_airfoil
            print("🔍 Ensuring NASA data is available...")
            fetch_nasa_airfoil()
            print("🚀 Starting NASA Airfoil Discovery...")
            run_nasa_airfoil()
            print(f"\n\u2705 Discovery results saved to: results/nasa_airfoil_results.txt")
            
        else:
            print("Invalid choice.")
            return

    except Exception as e:
        print(f"\n\u274c An error occurred: {e}")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to return to menu...")
    main()

if __name__ == "__main__":
    main()
