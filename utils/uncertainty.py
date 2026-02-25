import numpy as np
import emcee
import re
import sympy as sp
from scipy.optimize import minimize

def estimate_uncertainty(formula_str, X, y, input_names=None):
    """
    Estimates uncertainty of parameters in a formula using MCMC.
    
    Args:
        formula_str (str): Discovered formula (e.g., 'p0 * x * y').
        X (list of tuples/arrays): Input data points.
        y (array): Target values.
        input_names (list): Names of inputs (e.g., ['x', 'y']).
        
    Returns:
        results (dict): Map of parameter names to (mean, std).
    """
    # 1. Identify parameters (p0, p1, ...)
    params = re.findall(r'p\d+', formula_str)
    params = sorted(list(set(params)))
    
    if not params:
        return {}

    # 2. Convert formula to a numerical function
    X_data = np.array(X)
    y_data = np.array(y)
    
    if input_names is None:
        # Infer from ARGn style if possible
        args = re.findall(r'ARG\d+', formula_str)
        input_names = sorted(list(set(args)))

    def model(p_values, X):
        # Substitute parameters and inputs
        local_expr = formula_str
        for i, p_name in enumerate(params):
            local_expr = local_expr.replace(p_name, str(p_values[i]))
        
        # This is slow, but safe for arbitrary strings. 
        # In a real system, we'd compile this with lambdify.
        context = {name: X[:, i] for i, name in enumerate(input_names)}
        context['np'] = np
        try:
            return eval(local_expr, {"__builtins__": None}, context)
        except Exception:
            return np.zeros_like(y_data)

    # 3. Define Log-Likelihood
    def log_likelihood(p_values, X, y):
        y_model = model(p_values, X)
        sigma2 = np.var(y - y_model) + 1e-9
        return -0.5 * np.sum((y - y_model) ** 2 / sigma2 + np.log(sigma2))

    def log_prior(p_values):
        # Simple flat prior (unbounded for now)
        return 0.0

    def log_probability(p_values, X, y):
        lp = log_prior(p_values)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(p_values, X, y)

    # 4. Initialize MCMC
    n_params = len(params)
    n_walkers = 32
    n_steps = 500
    
    # Starting point: Use curve_fit or minimize to get a decent guess
    p0 = np.ones(n_params)
    
    # Add small random noise for walkers
    pos = p0 + 1e-4 * np.random.randn(n_walkers, n_params)
    
    # 5. Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability, args=(X_data, y_data))
    sampler.run_mcmc(pos, n_steps, progress=False)
    
    # 6. Analyze samples
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    
    results = {}
    for i, p_name in enumerate(params):
        mcmc_mean = np.mean(flat_samples[:, i])
        mcmc_std = np.std(flat_samples[:, i])
        results[p_name] = (mcmc_mean, mcmc_std)
        
    return results
