from deap import gp
import random
from functools import partial
import operator

def test_ephemerals():
    pset = gp.PrimitiveSetTyped("TEST", [float], float)
    # Add a primitive so genFull has something to work with if height > 0
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addEphemeralConstant("rand_inv_time", partial(random.uniform, -10, 10), float)
    
    # Try to generate a single terminal
    expr = gp.genFull(pset, min_=0, max_=0)
    node = expr[0]
    
    print(f"Node: {node}")
    print(f"  Name: {node.name}")
    print(f"  Class: {node.__class__.__name__}")
    print(f"  Bases: {node.__class__.__bases__}")
    
    # Check if we can find the name 'rand_inv_time' in the class or object
    # DEAP ephemerals usually have the name in the class's __name__ if done correctly,
    # but addEphemeralConstant might mangle it.
    
    # Let's look at pset.terminals
    print(f"Pset Terminals: {pset.terminals}")
    for t_type, terminals in pset.terminals.items():
        for t in terminals:
            print(f"  Terminal in pset: {t}, Name: {t.name}, Class: {t.__class__.__name__}")

if __name__ == "__main__":
    test_ephemerals()
