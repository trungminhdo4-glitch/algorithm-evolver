from deap import gp
import operator
import math
from core.physics import Dimension, DimensionalChecker
from problems.damped_oscillation import DampedOscillationProblem

def test_checker():
    prob = DampedOscillationProblem()
    pset = prob.create_primitive_set()
    
    # Manually create a tree: rand_length * exp(rand_inv_time * t) * cos(rand_inv_time * t)
    # This is complex to build manually with gp.PrimitiveTree.from_string but let's try.
    
    # We need to make sure the names match what's in the pset
    # t is ARG0
    # rand_inv_time, rand_length are ephemerals
    
    # Let's use a simpler one first: mul(rand_inv_time, t)
    # To get actual objects, we can look into pset
    
    checker = DimensionalChecker(prob.pset_units)
    
    # Simplified test: exp(mul(rand_inv_time, t))
    # We can use symbolic individuals if we are careful
    
    def get_term(name):
        for t in pset.terminals[float]:
            if t.name == name: return t
        return None

    mul_prim = pset.primitives[float][2] # mul
    exp_prim = [p for p in pset.primitives[float] if p.name == 'exp'][0]
    t_term = get_term('t')
    
    # Find ephemeral classes
    inv_time_class = None
    for t in pset.terminals[float]:
        if 'rand_inv_time' in t.__class__.__name__:
            inv_time_class = t.__class__
            break
            
    if not inv_time_class:
        print("Could not find rand_inv_time class")
        return

    # Create an instance
    inv_time_inst = inv_time_class(-0.5)
    
    # Tree: exp(mul(inv_time, t))
    # In prefix notation: exp, mul, inv_time, t
    ind = [exp_prim, mul_prim, inv_time_inst, t_term]
    
    unit, consistent = checker.check_tree(ind)
    print(f"Tree: exp(mul(rand_inv_time, t))")
    print(f"  Unit: {unit}")
    print(f"  Consistent: {consistent}")
    
    # Target unit test: mul(rand_length, exp(...))
    length_class = None
    for t in pset.terminals[float]:
        if 'rand_length' in t.__class__.__name__:
            length_class = t.__class__
            break
    
    length_inst = length_class(1.0)
    # mul, length, exp, mul, inv_time, t
    ind2 = [mul_prim, length_inst, exp_prim, mul_prim, inv_time_inst, t_term]
    unit2, consistent2 = checker.check_tree(ind2)
    print(f"Tree: mul(rand_length, exp(mul(rand_inv_time, t)))")
    print(f"  Unit: {unit2}")
    print(f"  Consistent: {consistent2}")
    print(f"  Matches target unit (L): {unit2 == prob.target_unit}")

if __name__ == "__main__":
    test_checker()
