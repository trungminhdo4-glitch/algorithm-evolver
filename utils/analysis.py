"""
Analysis and visualization utilities for Genetic Programming results.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("⚠️  NetworkX not installed. Tree visualization disabled.")
    print("   Install with: pip install networkx")

try:
    from deap import gp
    HAS_DEAP_GP = True
except ImportError:
    HAS_DEAP_GP = False
    print("⚠️  DEAP GP module not available. Tree visualization disabled.")

class ResultAnalyzer:
    """
    Analyzes and visualizes GP experiment results.
    """
    
    @staticmethod
    def plot_fitness_progress(log, title="Fitness Progress", save_path=None):
        """
        Plot fitness progress across generations.
        
        Args:
            log: DEAP statistics log
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if log is None or len(log) == 0:
            print("❌ No log data available for plotting")
            return None
            
        try:
            gen = log.select("gen")
            # Extract fitness values (supporting multi-objective)
            fit_mins = log.chapters["fitness"].select("min")
            fit_avgs = log.chapters["fitness"].select("avg")
            fit_maxs = log.chapters["fitness"].select("max")
            
            # If fitness is multi-objective (tuple/list), take the first objective (typically MSE)
            def extract_first(fit_list):
                if len(fit_list) > 0 and isinstance(fit_list[0], (tuple, list, np.ndarray)):
                    return [f[0] for f in fit_list]
                return fit_list
                
            fit_mins = extract_first(fit_mins)
            fit_avgs = extract_first(fit_avgs)
            fit_maxs = extract_first(fit_maxs)
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot fitness statistics
            ax1.plot(gen, fit_mins, 'b-', label="Best Fitness (MSE)", linewidth=2, alpha=0.8)
            ax1.plot(gen, fit_avgs, 'g-', label="Average Fitness (MSE)", linewidth=1, alpha=0.6)
            ax1.plot(gen, fit_maxs, 'r-', label="Worst Fitness (MSE)", linewidth=1, alpha=0.4)
            
            # Fill between min and avg for visual clarity
            ax1.fill_between(gen, fit_mins, fit_avgs, alpha=0.2, color='blue')
            
            ax1.set_xlabel("Generation", fontsize=12)
            ax1.set_ylabel("Fitness (Lower is Better)", fontsize=12, color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper right')
            ax1.set_title(title, fontsize=14, fontweight='bold')
            
            # Add annotation for final fitness
            if len(fit_mins) > 0:
                final_fitness = fit_mins[-1]
                ax1.annotate(f'Final: {final_fitness:.4f}', 
                           xy=(gen[-1], final_fitness),
                           xytext=(gen[-1] - len(gen)/10, final_fitness * 1.1),
                           arrowprops=dict(arrowstyle='->', color='blue'),
                           fontsize=10)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Fitness plot saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Error plotting fitness: {e}")
            return None
    
    @staticmethod
    def plot_tree_size_progress(log, title="Tree Size Evolution", save_path=None):
        """
        Plot tree size (complexity) across generations.
        
        Args:
            log: DEAP statistics log
            title: Plot title
            save_path: Optional path to save the figure
        """
        if log is None or len(log) == 0:
            print("❌ No log data available for plotting")
            return None
            
        try:
            gen = log.select("gen")
            size_avgs = log.chapters["size"].select("avg")
            size_mins = log.chapters["size"].select("min")
            size_maxs = log.chapters["size"].select("max")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot size statistics
            ax.plot(gen, size_avgs, 'b-', label="Average Size", linewidth=2)
            ax.fill_between(gen, size_mins, size_maxs, alpha=0.2, color='blue')
            
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel("Tree Size (Nodes)", fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left')
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Add annotation for final size
            if len(size_avgs) > 0:
                final_size = size_avgs[-1]
                ax.annotate(f'Final avg: {final_size:.1f} nodes', 
                          xy=(gen[-1], final_size),
                          xytext=(gen[-1] - len(gen)/10, final_size * 1.1),
                          arrowprops=dict(arrowstyle='->', color='blue'),
                          fontsize=10)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Tree size plot saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Error plotting tree size: {e}")
            return None
    
    @staticmethod
    def plot_tree(individual, title="GP Tree Visualization", save_path=None, max_nodes=50):
        """
        Visualize a GP tree structure.
        
        Args:
            individual: DEAP individual (PrimitiveTree)
            title: Plot title
            save_path: Optional path to save the figure
            max_nodes: Maximum nodes to display (for large trees)
            
        Returns:
            matplotlib Figure object or None
        """
        if not HAS_NETWORKX or not HAS_DEAP_GP:
            print("⚠️  Tree visualization requires NetworkX and DEAP")
            return None
            
        try:
            # Extract tree structure
            nodes, edges, labels = gp.graph(individual)
            
            # Check if tree is too large
            if len(nodes) > max_nodes:
                print(f"⚠️  Tree has {len(nodes)} nodes (>{max_nodes}). Visualization may be cluttered.")
                print(f"   Consider increasing max_nodes or analyzing a simpler tree.")
                
            # Create graph
            graph = nx.Graph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)
            
            # Calculate layout (try Graphviz first, then fallback to spring)
            try:
                pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
            except (ImportError, Exception):
                print("⚠️  Graphviz layout failed. Falling back to spring layout.")
                pos = nx.spring_layout(graph)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Draw nodes
            node_colors = []
            node_sizes = []
            for node in nodes:
                label_str = str(labels.get(node, str(node)))
                # Color code by node type
                if 'ARG' in label_str or label_str in ['a', 'b', 'c', 'x', 'y', 'z']:
                    node_colors.append('lightgreen')  # Inputs
                elif label_str.replace('.', '').replace('-', '').isdigit():
                    node_colors.append('lightyellow')  # Constants
                elif '(' in label_str and ')' in label_str:
                    node_colors.append('lightblue')   # Functions
                else:
                    node_colors.append('lightgray')
                
                # Size based on label length
                node_sizes.append(2000 + len(label_str) * 100)
            
            nx.draw(graph, pos, 
                    labels=labels, 
                    with_labels=True,
                    node_color=node_colors,
                    node_size=node_sizes,
                    font_size=9,
                    font_weight='bold',
                    edge_color='gray',
                    width=1.5,
                    ax=ax)
            
            # Add statistics
            stats_text = f"Total nodes: {len(nodes)}\nTree depth: {individual.height}"
            plt.text(0.02, 0.98, stats_text, 
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Tree visualization saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Error visualizing tree: {e}")
            return None
    
    @staticmethod
    def plot_convergence_comparison(logs_dict, title="Convergence Comparison", save_path=None):
        """
        Compare convergence across multiple experiments.
        
        Args:
            logs_dict: Dictionary of {experiment_name: log}
            title: Plot title
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(logs_dict)))
        
        for i, (exp_name, log) in enumerate(logs_dict.items()):
            if log and len(log) > 0:
                gen = log.select("gen")
                fit_mins = log.chapters["fitness"].select("min")
                ax.plot(gen, fit_mins, 
                       color=colors[i], 
                       label=exp_name, 
                       linewidth=2,
                       alpha=0.8)
        
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Best Fitness", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yscale('log')  # Log scale for better comparison
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Convergence comparison saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def analyze_algorithm(individual, toolbox, test_cases=None):
        """
        Analyze the performance and structure of a GP individual.
        
        Args:
            individual: GP individual
            toolbox: DEAP toolbox with compile function
            test_cases: List of test inputs or None for default
            
        Returns:
            Dictionary with analysis results
        """
        try:
            func = toolbox.compile(expr=individual)
            
            # Default test cases if not provided
            if test_cases is None:
                test_cases = [
                    (1, 2, 3), (3, 2, 1), (2, 1, 3),
                    (1, 1, 2), (-1, -3, 0), (5, 3, 5)
                ]
            
            results = {
                'tree_info': {
                    'total_nodes': len(individual),
                    'tree_depth': individual.height,
                    'expression': str(individual)
                },
                'performance': [],
                'summary': {}
            }
            
            # Test on each case
            correct_count = 0
            total_error = 0
            
            for test_input in test_cases:
                try:
                    output = func(*test_input)
                    
                    # Calculate expected output (sorted tuple for sorting network)
                    expected = tuple(sorted(test_input))
                    
                    # Check if output is valid
                    if isinstance(output, tuple) and len(output) == 3:
                        error = sum((o - e)**2 for o, e in zip(output, expected))
                        correct = error < 0.001
                        if correct:
                            correct_count += 1
                        total_error += error
                    else:
                        error = float('inf')
                        correct = False
                    
                    results['performance'].append({
                        'input': test_input,
                        'output': output,
                        'expected': expected,
                        'error': error,
                        'correct': correct
                    })
                    
                except Exception as e:
                    results['performance'].append({
                        'input': test_input,
                        'output': f'ERROR: {e}',
                        'expected': tuple(sorted(test_input)),
                        'error': float('inf'),
                        'correct': False
                    })
            
            # Calculate summary statistics
            results['summary'] = {
                'success_rate': (correct_count / len(test_cases)) * 100 if test_cases else 0,
                'average_error': total_error / len(test_cases) if test_cases else float('inf'),
                'correct_count': correct_count,
                'total_tests': len(test_cases)
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error analyzing algorithm: {e}")
            return None
    
    @staticmethod
    def generate_report(experiment_name, results, log, output_dir="analysis_reports"):
        """
        Generate a comprehensive analysis report.
        
        Args:
            experiment_name: Name of the experiment
            results: Results dictionary from experiment
            log: DEAP statistics log
            output_dir: Directory to save reports
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate plots
            fitness_plot_path = os.path.join(report_dir, "fitness_progress.png")
            size_plot_path = os.path.join(report_dir, "tree_size_evolution.png")
            tree_plot_path = os.path.join(report_dir, "best_tree.png")
            
            ResultAnalyzer.plot_fitness_progress(log, 
                                               f"Fitness Progress - {experiment_name}",
                                               fitness_plot_path)
            
            ResultAnalyzer.plot_tree_size_progress(log,
                                                 f"Tree Size Evolution - {experiment_name}",
                                                 size_plot_path)
            
            # Generate text report
            report_path = os.path.join(report_dir, "report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"GP EXPERIMENT ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Experiment: {experiment_name}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 40 + "\n\n")
                
                f.write("📊 PERFORMANCE SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Success Rate: {results.get('success_rate', 0):.1f}%\n")
                f.write(f"Best Fitness: {results.get('fitness', 'N/A')}\n")
                f.write(f"Validation Tests: {len(results.get('validation', []))}\n\n")
                
                f.write("🧬 BEST ALGORITHM\n")
                f.write("-" * 40 + "\n")
                best_ind = results.get('best_individual', 'N/A')
                f.write(f"Expression: {str(best_ind)}\n")
                f.write(f"Tree Size: {len(best_ind) if hasattr(best_ind, '__len__') else 'N/A'} nodes\n")
                f.write(f"Tree Depth: {best_ind.height if hasattr(best_ind, 'height') else 'N/A'}\n\n")
                
                f.write("✅ VALIDATION RESULTS\n")
                f.write("-" * 40 + "\n")
                for i, val in enumerate(results.get('validation', [])):
                    status = "PASS" if val.get('correct', False) else "FAIL"
                    f.write(f"Test {i+1}: {val.get('input', 'N/A')} → {val.get('output', 'N/A')} "
                           f"(expected {val.get('expected', 'N/A')}) [{status}]\n")
            
            print(f"✅ Analysis report saved to: {report_dir}")
            print(f"   - Fitness plot: {fitness_plot_path}")
            print(f"   - Size plot: {size_plot_path}")
            print(f"   - Text report: {report_path}")
            
            # Save raw data as JSON
            data_path = os.path.join(report_dir, "raw_data.json")
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'experiment_name': experiment_name,
                    'timestamp': timestamp,
                    'results': results,
                    'log_data': {
                        'generations': log.select('gen') if log else [],
                        'fitness_min': log.chapters['fitness'].select('min') if log else [],
                        'fitness_avg': log.chapters['fitness'].select('avg') if log else [],
                        'size_avg': log.chapters['size'].select('avg') if log else []
                    }
                }, f, default=str, indent=2)
            
            return report_dir
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return None

    @staticmethod
    def to_latex(expr_str):
        """
        Converts a simplified expression string to LaTeX.
        """
        try:
            import sympy as sp
            # Extract actual formula if it contains error messages or metadata
            if "Original:" in expr_str:
                expr_str = expr_str.split("Original:")[1].strip(" )")
            
            expr = sp.sympify(expr_str)
            return sp.latex(expr)
        except Exception as e:
            return f"LaTeX error: {e}"


# Convenience functions (alternative to class methods)
def plot_fitness(log, title="Fitness Progress", save_path=None):
    """Convenience wrapper for fitness plotting"""
    return ResultAnalyzer.plot_fitness_progress(log, title, save_path)

def plot_tree_visualization(individual, title="GP Tree", save_path=None):
    """Convenience wrapper for tree visualization"""
    return ResultAnalyzer.plot_tree(individual, title, save_path)

def create_analysis_report(experiment_name, results, log, output_dir="analysis"):
    """Convenience wrapper for report generation"""
    return ResultAnalyzer.generate_report(experiment_name, results, log, output_dir)


# Example usage
if __name__ == "__main__":
    print("🧬 GP Analysis Module - Example Usage")
    print("=" * 40)
    
    # Example of how to use this module
    print("\nAvailable functions:")
    print("1. ResultAnalyzer.plot_fitness_progress(log, title, save_path)")
    print("2. ResultAnalyzer.plot_tree_size_progress(log, title, save_path)")
    print("3. ResultAnalyzer.plot_tree(individual, title, save_path)")
    print("4. ResultAnalyzer.analyze_algorithm(individual, toolbox, test_cases)")
    print("5. ResultAnalyzer.generate_report(exp_name, results, log, output_dir)")
    
    print("\nConvenience wrappers:")
    print("  plot_fitness(log, title, save_path)")
    print("  plot_tree_visualization(individual, title, save_path)")
    print("  create_analysis_report(exp_name, results, log, output_dir)")