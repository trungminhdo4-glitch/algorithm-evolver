# V2.0 Roadmap: Dynamical Discovery

## Overview
V2.0 shifts the Scientific Discovery Engine from static symbolic regression to dynamical system identification. Instead of fitting $y = f(x)$, we now discover the underlying differential equations $\dot{y} = f(y, t)$.

## Mathematical Foundations

### 1. Numerical Differentiation
To discover an ODE from time-series data $y(t)$, we need to estimate the state derivatives $\dot{y}$.
Raw numerical differentiation is extremely sensitive to noise. We use **Savitzky-Golay filters** to smooth the trajectory before computing the gradient:
$$y_{smooth} = \text{Savgol}(y, \text{window}, \text{poly})$$
$$\dot{y} \approx \frac{dy_{smooth}}{dt}$$

### 2. Trajectory Matching (Fitness Evaluation)
A candidate formula $f(y, t)$ is evaluated not just by its fit to $\dot{y}$, but by its ability to reproduce the entire trajectory.
We use **Numerical Integration** (Runge-Kutta methods via `odeint`) to simulate the system starting from $y(t_0)$:
$$\hat{y}(t) = y(t_0) + \int_{t_0}^t f(\hat{y}, \tau) d\tau$$
The fitness is the Mean Squared Error (MSE) between the integrated trajectory $\hat{y}$ and the observed data $y$:
$$\text{Fitness} = \frac{1}{N} \sum_{i=0}^N (y(t_i) - \hat{y}(t_i))^2$$

### 3. Bayesian Uncertainty Estimation
Static fine-tuning provides point estimates of constants. V2.0 introduces **Markov Chain Monte Carlo (MCMC)** sampling using the `emcee` library.
This allows us to quantify the uncertainty of physical constants (e.g., growth rates, decay constants):
$$k = \bar{k} \pm \sigma_k$$

## Implementation Status
- [x] Numerical Differentiator (`utils/differentiation.py`)
- [x] ODE Problem & Evaluator (`core/ode_problem.py`)
- [x] Bayesian Uncertainty (`utils/uncertainty.py`)
- [x] Lotka-Volterra Experiment (`experiments/run_predator_prey.py`)
- [x] Demo Integration (Option [8])
