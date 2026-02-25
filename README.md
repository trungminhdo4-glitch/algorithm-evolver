# 🔬 Scientific Discovery Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![DEAP](https://img.shields.io/badge/Genetic_Programming-DEAP-orange.svg)
![SciPy](https://img.shields.io/badge/Optimization-SciPy-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

The **Scientific Discovery Engine** is a powerful symbolic regression framework designed to rediscover fundamental physical laws from empirical data. Unlike standard machine learning, it prioritizes **interpretability**, **physical consistency**, and **structural elegance**.

## 🌟 Key Features

*   **Physics-Informed Discovery:** Enforces strict dimensional consistency across Mass, Length, Time, and more. Penalizes "physics nonsense" to guide evolution toward plausible laws.
*   **Hybrid Optimization:** Combines Genetic Programming (for structural discovery) with SciPy's `curve_fit` (for precise constant refinement).
*   **High Performance:** Fully parallelized evaluation logic to utilize modern multi-core architectures.
*   **Dimensional Constants:** Supports automatic discovery of unit-bearing constants (e.g., Gravitational constant, Gas constant).
*   **Automated Reporting:** Generates publication-ready LaTeX reports covering the Pareto front of discovered formulas.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/scientific-discovery-engine.git
cd scientific-discovery-engine
pip install -r requirements.txt
```

### Run the Demo
Experience the engine in action through the interactive CLI:
```bash
python demo.py
```

## 📊 Discovery Showcase

The engine has successfully rediscovered the following laws:

| Experiment | Discovered Law | Physical Domain |
| :--- | :--- | :--- |
| **Kepler's 3rd Law** | $T^2 \propto a^3$ | Orbital Mechanics |
| **Ballistic Motion** | $d = \frac{v^2 \sin(2\theta)}{g}$ | Classical Mechanics |
| **Ideal Gas Law** | $PV = nRT$ | Thermodynamics |
| **Damped Oscillation** | $x(t) = A e^{-\delta t} \cos(\omega t)$ | Dynamic Systems |

## 🏗 Architecture

The engine is built on a modular stack:
- **`core/`**: Evolutionary logic, physics kernels, and program simplification.
- **`problems/`**: Physical problem definitions and synthetic data generators.
- **`utils/`**: Fine-tuning modules and LaTeX exporters.

---
*Developed for the intersection of Artificial Intelligence and Physical Sciences.*
