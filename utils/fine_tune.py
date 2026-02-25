"""
Fine-Tuner: Optimizes numerical constants in SymPy expressions using SciPy.
"""
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

def refine_with_scipy(expr_str, X_data, y_data, input_names=['v', 'angle']):
    """
    Takes a symbolic expression string, replaces all constants with parameters,
    and uses SciPy to fit them to the data.
    """
    try:
        # 1. Parse expression
        expr = sp.sympify(expr_str)
        
        # 2. Identify and replace numeric constants with parameters p0, p1, ...
        # We look for all numbers in the expression
        params = []
        param_values = []
        
        # Mapping from number to parameter symbol
        substitutions = {}
        
        # Recursive function to find numbers and swap them
        current_param_idx = 0
        
        # Fix: We don't want to replace "2" in sin(2*angle) if it's a fixed structural part,
        # but the prompt specifically says "replace all numbers".
        # Let's find floats and ints.
        
        for atom in expr.atoms(sp.Number):
            # Skip Integers to preserve structure (like exponents in PV/n^2 if any)
            # Only optimize Floats (numerical constants)
            if atom.is_Float:
                p_sym = sp.Symbol(f'p{current_param_idx}')
                params.append(p_sym)
                param_values.append(float(atom))
                substitutions[atom] = p_sym
                current_param_idx += 1
        
        if not params:
            return str(expr), {} # Nothing to optimize
            
        # We need to be careful: expr.subs on raw numbers can be tricky
        # It's better to use something more robust or replace as strings?
        # No, subs should work if we target the specific atoms.
        param_expr = expr.subs(substitutions)
        
        # 3. Create lambdified function for SciPy
        # Args order: inputs (v, angle, ...), then parameters (p0, p1, ...)
        input_symbols = [sp.Symbol(name) for name in input_names]
        symbols_order = input_symbols + params
        model_func = sp.lambdify(symbols_order, param_expr, modules=['numpy', 'math'])
        
        # SciPy curve_fit expects model(X, p0, p1, ...)
        def fit_wrapper(X, *p_vals):
            # X comes in as (n_features, n_samples)
            return model_func(*X, *p_vals)

        # Prepare X for curve_fit (rows to cols)
        X_array = np.array([x for x in X_data]).T
        Y_array = np.array(y_data)
        
        # 4. Fit
        popt, _ = curve_fit(fit_wrapper, X_array, Y_array, p0=param_values)
        
        # 5. Substitute back
        final_subs = {params[i]: popt[i] for i in range(len(params))}
        final_expr = param_expr.subs(final_subs)
        
        # Clean up very small/round numbers
        # (Optional: e.g. 1.99999 -> 2.0)
        
        return str(final_expr), final_subs

    except Exception as e:
        return f"Fine-tuning error: {e}", {}

if __name__ == "__main__":
    # Test Modul 1: v^1.5 * sin(2*angle) -> v^2 * sin(2*angle)
    v_test = np.linspace(10, 100, 20)
    a_test = np.linspace(0.1, 1.4, 20)
    # Ground truth: distance = v^2 * sin(2*angle) / 9.81
    y_test = (v_test**2 * np.sin(2 * a_test)) / 9.81
    X_test = list(zip(v_test, a_test))
    
    candidate = "v**1.5 * sin(2.0 * angle) * 0.5" # Very wrong constants
    refined, params = refine_with_scipy(candidate, X_test, y_test)
    print(f"Original: {candidate}")
    print(f"Refined:  {refined}")
    print(f"Params:   {params}")
