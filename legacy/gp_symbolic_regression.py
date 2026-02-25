import random
import operator
import numpy as np
from functools import partial
from deap import base, creator, tools, gp, algorithms

# DEFINE THE TARGET: We want to evolve an algorithm that matches f(x, y) = x^2 + 2y
def evaluate_algorithm(individual, toolbox):
    """
    The fitness function. It evaluates how good a candidate algorithm is.
    Lower returned error = better fitness.
    """
    # Compile the tree into an executable function
    func = toolbox.compile(expr=individual)
    # Test points: a simple grid of input values
    test_points = [(x, y) for x in [-2, -1, 0, 1, 2] for y in [-2, -1, 0, 1, 2]]
    error = 0
    for x, y in test_points:
        try:
            # Calculate squared difference from target
            error += (func(x, y) - (x**2 + 2*y))**2
        except (OverflowError, ZeroDivisionError):
            # Penalize invalid operations heavily
            error += 1e9
    return error,

# Create the set of primitive operations (the building blocks for our algorithms)
pset = gp.PrimitiveSetTyped("MAIN", [float, float], float)  # Takes 2 floats, returns 1 float

# Add basic arithmetic as primitives
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
def protected_div(a, b):
    """Division that returns 1 if denominator is near zero."""
    return 1.0 if abs(b) < 1e-9 else a / b
pset.addPrimitive(protected_div, [float, float], float)

# Add the input terminals and random constants
pset.renameArguments(ARG0='x', ARG1='y')
pset.addEphemeralConstant("rand_const", partial(random.uniform, -1, 1), float)

# Define the blueprint for an individual (a candidate algorithm) and its fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Single objective, minimize
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
# Method to generate a random expression tree
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
# Method to create an Individual from an expression
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# Method to create a population of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Register our evaluation function from Step 1
toolbox.register("evaluate", evaluate_algorithm, toolbox=toolbox)
# Register the compile function to turn trees into code
toolbox.register("compile", gp.compile, pset=pset)

# --- REGISTER EVOLUTIONARY OPERATORS ---
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Apply height limits to the ACTUAL OPERATORS (not expr generators)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

# Log statistics
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

def main():
    random.seed(42)  # For reproducibility
    pop = toolbox.population(n=100)  # Create 100 random algorithms
    hof = tools.HallOfFame(1)  # Keep the single best ever found

    # Parameters: 40 generations, 60% crossover prob, 30% mutation prob
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3,
                                   ngen=40, stats=stats, halloffame=hof,
                                   verbose=True)

    # Print the champion algorithm
    best_ind = hof[0]
    print(f"\n✨ Best evolved algorithm:\n{str(best_ind)}")
    print(f"\n🧮 Best fitness (error): {best_ind.fitness.values[0]}")

    # Optional: Test it on a new point
    best_func = toolbox.compile(expr=best_ind)
    print(f"\n🔍 Test: For x=3, y=4.")
    print(f"    Target (x²+2y) = {3**2 + 2*4}")
    print(f"    Evolved algorithm outputs: {best_func(3, 4)}")
    return pop, log, hof

if __name__ == "__main__":
    main()