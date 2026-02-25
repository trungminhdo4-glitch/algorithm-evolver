import numpy as np
import os
import sys
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ode_problem import OdeProblem
from core.evolution import EvolutionaryEngine
from core.primitives import create_power_law_primitive_set, create_typed_primitive_set
from utils.differentiation import compute_derivative
from utils.uncertainty import estimate_uncertainty
from deap import tools

def simulate_lotka_volterra(alpha=1.1, beta=0.4, delta=0.1, gamma=0.4, x0=10, y0=5, t_max=50, num_steps=500, noise=0.01):
    """Simulates the predator-prey system."""
    t = np.linspace(0, t_max, num_steps)
    
    def system(state, t):
        x, y = state
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return [dxdt, dydt]
    
    sol = odeint(system, [x0, y0], t)
    x, y = sol[:, 0], sol[:, 1]
    
    # Add noise
    x_noise = x + np.random.normal(0, noise * np.mean(x), size=x.shape)
    y_noise = y + np.random.normal(0, noise * np.mean(y), size=y.shape)
    
    return t, x_noise, y_noise

def run_predator_prey_experiment(population_size=1000, generations=100):
    print("=" * 60)
    print("🚀 DISCOVERING DYNAMICAL SYSTEMS: LOTKA-VOLTERRA")
    print("=" * 60)
    
    # 1. Simulate Data
    t, x_obs, y_obs = simulate_lotka_volterra()
    
    # 2. Pre-process: Compute smoothed derivatives
    # We use these for initial "static" regression to seed the search or as target
    dx_dt = compute_derivative(t, x_obs)
    dy_dt = compute_derivative(t, y_obs)
    
    # 3. Setup Discovery for dx/dt = f(x, y, t)
    # We'll first try to find dxdt
    from core.physics import Dimension
    input_units = {
        'ARG0': Dimension(0, 0, 0, 0, 0), # x
        'ARG1': Dimension(0, 0, 0, 0, 0), # y
        'ARG2': Dimension(0, 0, 0, 0, 0), # t
        'rand_float': Dimension(0, 0, 0, 0, 0),
        'rand_dimensional': Dimension(0, 0, -1, 0, 0) # Units of 1/Time
    }
    target_unit = Dimension(0, 0, -1, 0, 0) # Rate [1/T]
    
    problem_x = OdeProblem(t, x_obs, initial_state=10.0, name="dxdt", input_units=input_units, target_unit=target_unit)
    
    # PSet with x, y as inputs
    pset = create_typed_primitive_set([float, float, float], float, name="ODE")
    pset.renameArguments(ARG0='x', ARG1='y', ARG2='t')
    
    engine = EvolutionaryEngine(
        pset=pset,
        evaluate_func=problem_x.evaluate,
        population_size=population_size,
        max_height=5
    )
    
    print("\n🔍 Discovering equation for dx/dt (Prey)...")
    pop_x, log_x = engine.run_nsga2(generations=generations)
    
    # 4. Results & Uncertainty
    best_ind = tools.selBest(pop_x, 1)[0]
    from core.simplification import ProgramSimplifier
    simplifier = ProgramSimplifier()
    best_str = simplifier.simplify_individual(best_ind, pset)
    
    print(f"\n✅ Best Discovery for dx/dt: {best_str}")
    
    print("\n⚖️ Estimating Uncertainty with MCMC...")
    # For MCMC we need static data points (x, y) as inputs and dx_dt as target
    X_mcmc = list(zip(x_obs, y_obs, t))
    uncertainty = estimate_uncertainty(best_str, X_mcmc, dx_dt, input_names=['x', 'y', 't'])
    
    print("\n📊 Final discovery Report (dx/dt):")
    for param, (mean, std) in uncertainty.items():
        print(f"  {param}: {mean:.4f} ± {std:.4f}")
    
    # Save visualization
    plt.figure(figsize=(10, 5))
    plt.plot(t, x_obs, 'ro', alpha=0.3, label='Observed (Prey)')
    plt.plot(t, y_obs, 'bo', alpha=0.3, label='Observed (Predator)')
    plt.legend()
    plt.title("Lotka-Volterra Time Series")
    plt.savefig("results/predator_prey_trajectory.png")
    print(f"\n💾 Results saved to results/predator_prey_trajectory.png")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_predator_prey_experiment(population_size=1000, generations=100)
