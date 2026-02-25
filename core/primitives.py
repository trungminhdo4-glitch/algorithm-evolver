"""
Common primitive operations reusable across problems
"""
import operator
import math
import random
from functools import partial
from deap import gp

def protected_log(x):
    """Logarithmus, der bei x<=0 0 zurückgibt."""
    if x <= 1e-9:
        return 0.0
    return math.log(x)

def protected_sqrt(x):
    """Quadratwurzel, die den Absolutbetrag verwendet."""
    return math.sqrt(abs(x))

def protected_div(a, b):
    return 1.0 if abs(b) < 1e-9 else a / b

def protected_exp(x):
    try:
        # Clip to avoid overflow
        return math.exp(max(-100, min(100, x)))
    except OverflowError:
        return 1e10

def if_then_else(condition, out1, out2):
    return out1 if condition else out2

def create_typed_primitive_set(input_types, output_type, name="MAIN"):
    if not isinstance(input_types, list):
        input_types = [input_types]
    
    pset = gp.PrimitiveSetTyped(name, input_types, output_type)
    
    pset.addPrimitive(operator.add, [output_type, output_type], output_type)
    pset.addPrimitive(operator.sub, [output_type, output_type], output_type)
    pset.addPrimitive(operator.mul, [output_type, output_type], output_type)
    
    pset.addPrimitive(protected_div, [output_type, output_type], output_type)
    
    if name != "OSCILLATION":
        pset.addEphemeralConstant("rand_float", 
                                  partial(random.uniform, -10, 10), 
                                  output_type)
    
    # Optional dimensional ephemerals
    pset.addEphemeralConstant("rand_inv_time", 
                              partial(random.uniform, -10, 10), 
                              output_type)
    pset.addEphemeralConstant("rand_length", 
                              partial(random.uniform, -10, 10), 
                              output_type)
    
    return pset

def add_transcendental_primitives(pset):
    """Adds sin, cos, and protected exp to the primitive set."""
            
    pset.addPrimitive(math.sin, [float], float)
    pset.addPrimitive(math.cos, [float], float)
    pset.addPrimitive(protected_exp, [float], float, name="exp")
    return pset

def create_power_law_primitive_set(input_types, output_type, name="POWER"):
    """Creates a primitive set with ONLY multiplication and division."""
    if not isinstance(input_types, list):
        input_types = [input_types]
    
    pset = gp.PrimitiveSetTyped(name, input_types, output_type)
    
    pset.addPrimitive(operator.mul, [output_type, output_type], output_type)
    
    pset.addPrimitive(protected_div, [output_type, output_type], output_type)
    
    pset.addEphemeralConstant("rand_dimensional", 
                              partial(random.uniform, -10, 10), 
                              output_type)
    
    return pset

def add_comparison_primitives(pset):
    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.gt, [float, float], bool)
    pset.addPrimitive(operator.eq, [float, float], bool)
    
    pset.addPrimitive(operator.and_, [bool, bool], bool)
    pset.addPrimitive(operator.or_, [bool, bool], bool)
    pset.addPrimitive(operator.not_, [bool], bool)
    
    pset.addPrimitive(if_then_else, [bool, float, float], float)
    
    pset.addTerminal(True, bool)
    pset.addTerminal(False, bool)
    
    return pset

def add_math_primitives(pset):
    pset.addPrimitive(math.sin, [float], float)
    pset.addPrimitive(math.cos, [float], float)
    pset.addPrimitive(math.exp, [float], float)
    pset.addPrimitive(protected_log, [float], float)
    pset.addPrimitive(protected_sqrt, [float], float)
    return pset

def add_algebra_primitives(pset):
    """Adds only the basic algebraic operators. No transcendental functions."""
    # The basic arithmetic operators (+, -, *, /) are already added in create_typed_primitive_set
    pset.addPrimitive(abs, [float], float)
    return pset