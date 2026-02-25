"""
Utilities package for Genetic Programming analysis and visualization.
"""

# Import analysis module
try:
    from .analysis import (
        ResultAnalyzer,
        plot_fitness,
        plot_tree_visualization,
        create_analysis_report
    )
    
    __all__ = [
        'ResultAnalyzer',
        'plot_fitness',
        'plot_tree_visualization',
        'create_analysis_report'
    ]
    
except ImportError:
    # Analysis module not available
    __all__ = []