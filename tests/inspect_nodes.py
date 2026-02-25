import sys
import os
from deap import gp

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.csv_problem import CsvProblem

def inspect_deap_nodes():
    file_path = "data/gravity_test.csv"
    target_col = "Force [N]"
    
    prob = CsvProblem(file_path, target_col)
    pset = prob.create_primitive_set(style="power")
    
    # Create an individual with one terminal
    ind = gp.PrimitiveTree([pset.terminals[float][0]])
    print(f"Individual: {ind}")
    terminal = ind[0]
    print(f"Terminal Name: {terminal.name}")
    print(f"Terminal Value (if hasattr): {getattr(terminal, 'value', 'N/A')}")
    print(f"Terminal Class: {terminal.__class__.__name__}")
    
    # Try an ephemeral
    ephemeral_gen = [t for t in pset.terminals[float] if hasattr(t, 'name') and 'rand' in t.name]
    if ephemeral_gen:
        # DEAP ephemerals are functions in the pset, but in the tree they are Terminal instances
        # We need to generate one
        expr = gp.genFull(pset, min_=0, max_=0)
        ind2 = gp.PrimitiveTree(expr)
        print(f"\nIndividual 2: {ind2}")
        node = ind2[0]
        print(f"Node Name: {node.name}")
        print(f"Node Class: {node.__class__.__name__}")

if __name__ == "__main__":
    inspect_deap_nodes()
