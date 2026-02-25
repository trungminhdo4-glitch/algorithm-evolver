from problems.damped_oscillation import DampedOscillationProblem
from core.evolution import EvolutionaryEngine
from deap import tools

def test_minimal_evolution():
    print("Testing minimal evolution...")
    prob = DampedOscillationProblem()
    pset = prob.create_primitive_set()
    
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=prob.evaluate,
        population_size=10
    )
    
    # Register NSGA2
    engine.toolbox.register("select", tools.selNSGA2)
    
    print("Running 2 generations...")
    try:
        population, log = engine.run_nsga2(generations=2)
        print("Success!")
        best_ind = tools.selBest(population, 1)[0]
        print(f"Best Fit: {best_ind.fitness.values}")
    except Exception as e:
        print(f"Error during evolution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_evolution()
