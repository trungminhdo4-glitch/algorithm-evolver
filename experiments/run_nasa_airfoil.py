import sys
import os
import multiprocessing
import pandas as pd
from deap import gp, tools, base
import operator
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.csv_problem import CsvLoaderProblem
from core.evolution import EvolutionaryEngine
from core.primitives import protected_div, protected_exp, protected_log, protected_pow
from core.physics import Dimension

def nasa_transformation(y):
    return 10**(y / 10.0)

def log_uniform_const():
    # Very large range to match Pressure Ratio magnitude (10^14)
    import random
    sign = 1 if random.random() > 0.5 else -1
    exponent = random.uniform(-15, 15)
    return sign * (10 ** exponent)

def run_nasa_airfoil():
    print("=" * 60)
    print("NASA AIRFOIL DISCOVERY (LEVEL 2)")
    print("Goal: Discover Pressure Ratio law (P ∝ U^5)")
    print("=" * 60)

    file_path = "data/nasa_airfoil.csv"
    target_column = "SSPL"

    # Unit Map Definition
    units = {
        "Frequency": [0, 0, -1, 0, 0],       # Hz (1/T)
        "AngleOfAttack": [0, 0, 0, 0, 0],    # Grad (Dimensionlos)
        "ChordLength": [0, 1, 0, 0, 0],      # Meter (L)
        "FreeStreamVelocity": [0, 1, -1, 0, 0], # m/s (L/T)
        "SuctionSideDisplacement": [0, 1, 0, 0, 0], # Meter (L)
        "SSPL": [0, 0, 0, 0, 0] # Nach Transformation: Dimensionloses Druckverhältnis
    }

    # Transformation Logic: L_dB -> P_ratio
    # Formel: y_new = 10^(y_old / 10)
    # Use named function instead of lambda for picklability
    transformation = nasa_transformation

    # 1. Initialize Problem
    problem = CsvLoaderProblem(
        file_path=file_path,
        target_column=target_column,
        unit_map=units,
        transformation=transformation
    )

    # 2. Primitive Set
    pset = gp.PrimitiveSetTyped("NASA", [float]*5, float)
    
    # Operators
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protected_div, [float, float], float)
    pset.addPrimitive(protected_pow, [float, float], float, name="pow")
    pset.addPrimitive(protected_log, [float], float, name="log")
    pset.addPrimitive(protected_exp, [float], float, name="exp")
    
    # Ephemerals
    pset.addEphemeralConstant("rand_dimensional", log_uniform_const, float)

    # Map ARG names to columns for readability
    kwargs = {f"ARG{i}": col for i, col in enumerate(problem.input_columns)}
    pset.renameArguments(**kwargs)

    # 3. Setup Engine
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem.evaluate,
        population_size=1000,
        max_height=8
    )

    # NSGA-II selection
    engine.toolbox.register("select", tools.selNSGA2)

    # Multiprocessing
    pool = multiprocessing.Pool()
    engine.toolbox.register("map", pool.map)

    # 4. Run Evolution
    print("\n🚀 Starting discovery (100 generations)...")
    population, log = engine.run_nsga2(
        generations=100,
        seed=42,
        verbose=True
    )

    # 5. Pareto Front Analysis
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    print("\n📊 Pareto Front Candidates (Pressure Formula):")
    results = []
    for ind in pareto_front[:10]:
        formula = str(ind)
        fitness = ind.fitness.values[0]
        complexity = ind.fitness.values[1]
        results.append((formula, fitness, complexity))
        print(f" - {formula} (MSE: {fitness:.4e}, complexity: {complexity})")

    # Save Results
    os.makedirs("results", exist_ok=True)
    with open("results/nasa_airfoil_results.txt", "w") as f:
        f.write("NASA Airfoil Discovery Pareto Front\n")
        f.write("Units: dB -> P_ratio (10^(dB/10))\n\n")
        for i, (form, fit, comp) in enumerate(results):
            f.write(f"{i+1}: {form}\n   MSE: {fit:.4e}, Complexity: {comp}\n\n")

    pool.close()
    pool.join()
    return results

if __name__ == "__main__":
    run_nasa_airfoil()
