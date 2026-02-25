import sympy as sp
import math

class ProgramSimplifier:
    """
    Algebraic simplification of Genetic Programming trees using SymPy.
    """
    def __init__(self):
        # Mapping from GP primitive names to SymPy functions/operators
        self.primitive_map = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'protected_div': lambda x, y: x / y,
            'sin': sp.sin,
            'cos': sp.cos,
            'exp': sp.exp,
            'protected_log': sp.log,
            'protected_sqrt': sp.sqrt,
            'abs': sp.Abs,
            'neg': lambda x: -x
        }
    
    def simplify_individual(self, individual, pset):
        """
        Simplifies a DEAP individual using SymPy.
        """
        try:
            # 1. Convert to SymPy expression
            # Use a local dict for variables from the pset
            locals_dict = {name: sp.Symbol(name) for name in pset.arguments}
            
            # Use a helper to evaluate the tree string as a sympy expression
            expr_str = str(individual)
            
            # Sympy can parse strings but we need to handle the prefix notation of DEAP
            # Easiest way is to define functions with the GP names that return Sympy expressions
            sympy_expr = eval(expr_str, {"__builtins__": None}, {**self.primitive_map, **locals_dict})
            
            # 2. Simplify
            simplified_expr = sp.simplify(sympy_expr)
            
            # 3. Return as string
            return str(simplified_expr)
        except Exception as e:
            # Fallback to original string if simplification fails
            return f"Error simplifying: {e} (Original: {str(individual)})"

    @staticmethod
    def to_latex(expr_str):
        """Converts a simplified expression string to LaTeX"""
        try:
            expr = sp.sympify(expr_str)
            return sp.latex(expr)
        except:
            return expr_str