"""
Visualization utilities for Genetic Programming results.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_fitness_progress(log, title="Fitness Progress"):
    """
    Plot fitness progress across generations.
    
    Why visualize? Helps understand:
    1. Convergence speed
    2. Stagnation issues
    3. Effectiveness of parameters
    """
    gen = log.select('gen')
    fit_mins = log.chapters['fitness'].select('min')
    fit_avgs = log.chapters['fitness'].select('avg')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot fitness
    ax1.plot(gen, fit_mins, 'b-', label="Minimum Fitness", linewidth=2)
    ax1.plot(gen, fit_avgs, 'r-', label="Average Fitness", linewidth=2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (Lower is Better)", color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_tree_size_progress(log, title="Tree Size Evolution"):
    """Plot tree size (complexity) across generations"""
    gen = log.select('gen')
    size_avgs = log.chapters['size'].select('avg')
    size_mins = log.chapters['size'].select('min')
    size_maxs = log.chapters['size'].select('max')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(gen, size_mins, size_maxs, alpha=0.2, color='blue')
    ax.plot(gen, size_avgs, 'b-', label="Average Size", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Tree Size (Nodes)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.title(title)
    plt.tight_layout()
    return fig