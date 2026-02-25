from deap import gp

class Dimension:
    def __init__(self, m=0, l=0, t=0, n=0, theta=0):
        self.units = (m, l, t, n, theta)
        
    def __add__(self, other):
        return Dimension(*(a + b for a, b in zip(self.units, other.units)))
        
    def __sub__(self, other):
        return Dimension(*(a - b for a, b in zip(self.units, other.units)))
        
    def __eq__(self, other):
        if other is None: return False
        try:
            return all(abs(a - b) < 1e-9 for a, b in zip(self.units, other.units))
        except Exception:
            return False
        
    def is_dimensionless(self):
        return all(abs(u) < 1e-9 for u in self.units)
        
    def __repr__(self):
        return f"Dim(M{self.units[0]} L{self.units[1]} T{self.units[2]} N{self.units[3]} Th{self.units[4]})"

class DimensionalChecker:
    """
    Checks if a GP tree is physically consistent.
    5 Slots: (Mass, Length, Time, Amount, Temperature)
    """
    def __init__(self, pset_units):
        self.pset_units = pset_units
        
    def get_node_units(self, node, children_units):
        name = node.name
        
        # 1. Addition / Subtraction (Units must be identical)
        if name in ['add', 'sub']:
            if len(children_units) == 2 and children_units[0] == children_units[1]:
                return children_units[0]
            return None
            
        # 2. Multiplication
        if name == 'mul':
            if len(children_units) == 2 and children_units[0] and children_units[1]:
                return children_units[0] + children_units[1]
            return None
            
        # 3. Division
        if name in ['protected_div', 'div']:
            if len(children_units) == 2 and children_units[0] and children_units[1]:
                return children_units[0] - children_units[1]
            return None
            
        # 4. Transcendental functions (Arg MUST be dimensionless)
        if name in ['sin', 'cos', 'exp', 'protected_log', 'log']:
            if children_units[0] and children_units[0].is_dimensionless():
                return Dimension(0, 0, 0, 0, 0)
            return None
            
        if name == 'protected_sqrt' or name == 'sqrt':
            if children_units[0]:
                return Dimension(*(u/2 for u in children_units[0].units))
            return None
            
        return None

    def check_tree(self, individual):
        """
        Recursively determines the units of the entire tree.
        Returns a tuple (Dimension, is_consistent).
        """
        # Dictionary to store units of subtrees
        # Since individuals are list of primitives, we can traverse backwards (postfix)
        stack = []
        consistent = True
        
        for node in reversed(individual):
            if isinstance(node, gp.Terminal):
                # Get unit from pset_units. 
                # For regular terminals, node.name is 'ARG0', 'P', etc.
                # For ephemerals, node.name is the value string, but the class name is 'rand_L' etc.
                unit = self.pset_units.get(node.name)
                if unit is None:
                    unit = self.pset_units.get(node.__class__.__name__, Dimension(0, 0, 0, 0, 0))
                stack.append(unit)
            else:
                # Primitive
                args = [stack.pop() for _ in range(node.arity)]
                if not consistent or any(u is None for u in args):
                    consistent = False
                    stack.append(None)
                    continue
                
                node_unit = self.get_node_units(node, args)
                if node_unit is None:
                    consistent = False
                stack.append(node_unit)
                
        return stack[0], consistent
