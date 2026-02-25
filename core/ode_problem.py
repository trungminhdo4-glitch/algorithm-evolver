import numpy as np
from scipy.integrate import odeint
from core.physics import Dimension, DimensionalChecker
from core.simplification import ProgramSimplifier
import sympy as sp

class OdeProblem:
    """
    Evaluates fitness by integrating the discovered ODE and matching the trajectory.
    Target: dy/dt = f(y, t)
    """
    
    def __init__(self, t, y_data, initial_state, name="ode_discovery", input_units=None, target_unit=None):
        """
        Args:
            t (array): Time steps.
            y_data (array): Observed state trajectory (can be multi-dimensional).
            initial_state (array/float): y(t0).
            input_units (dict): Units for state variables.
            target_unit (Dimension): Expected unit for dy/dt.
        """
        self.t = np.array(t)
        self.y_data = np.array(y_data)
        self.initial_state = initial_state
        self.name = name
        self.input_units = input_units or {}
        self.target_unit = target_unit
        
        # Determine number of state variables
        if len(self.y_data.shape) == 1:
            self.num_states = 1
        else:
            self.num_states = self.y_data.shape[1]

    def evaluate(self, individual, func):
        """
        Fitness Logic:
        1. Dimensional consistency.
        2. Trajectory Matching (Integration).
        3. Complexity.
        """
        # 1. Dimensional Check
        checker = DimensionalChecker(self.input_units)
        final_unit, consistent = checker.check_tree(individual)
        
        dim_penalty = 1.0
        if not consistent:
            dim_penalty = 1e6
        elif self.target_unit and final_unit != self.target_unit:
            dim_penalty = 1e3

        # 2. Trajectory Matching (Integration)
        # func is already compiled by the EvolutionaryEngine wrapper
        
        # Wrapper for odeint: dy/dt = f(y, t)
        def ode_wrapper(y, t):
            try:
                # Individual is expected to take (y, t) as arguments
                # For multi-state, we might need to unpack y
                if self.num_states == 1:
                    return func(y[0], t)
                else:
                    return func(*y, t)
            except (OverflowError, ZeroDivisionError, ValueError):
                return 0.0

        try:
            # Integrate the ODE
            y_hat = odeint(ode_wrapper, self.initial_state, self.t)
            
            # MSE between integrated and observed
            mse = np.mean((self.y_data - y_hat.flatten() if self.num_states==1 else y_hat)**2)
            
            if np.isnan(mse) or np.isinf(mse):
                mse = 1e12
        except Exception:
            mse = 1e12

        accuracy_score = mse * dim_penalty

        # 3. Complexity
        try:
            simplifier = ProgramSimplifier()
            # Note: We assume a pset naming convention where inputs are y, t or x1, x2...
            # This is a bit tricky since we don't have the pset here, 
            # so we fallback to length if simplification fails.
            complexity = len(individual)
        except Exception:
            complexity = len(individual)

        return accuracy_score, float(complexity)

    def __str__(self):
        return f"OdeProblem({self.name}, states={self.num_states})"
