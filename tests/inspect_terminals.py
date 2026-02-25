from problems.damped_oscillation import DampedOscillationProblem
from deap import gp

def inspect_terminals():
    prob = DampedOscillationProblem()
    pset = prob.create_primitive_set()
    
    print(f"Terminals for float:")
    for t in pset.terminals[float]:
        print(f"  Terminal: {t}")
        print(f"    Name: {t.name}")
        print(f"    Class: {t.__class__.__name__}")
        print(f"    Class Bases: {t.__class__.__bases__}")

if __name__ == "__main__":
    inspect_terminals()
