# Datei: utils/advanced_visualization.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AdvancedVisualizer:
    @staticmethod
    def plot_3d_function(individual, toolbox, x_range=(-5,5), y_range=(-5,5)):
        """3D Plot der evolvierten Funktion"""
        func = toolbox.compile(expr=individual)
        
        x = np.linspace(*x_range, 50)
        y = np.linspace(*y_range, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = func(X[i,j], Y[i,j])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X,Y)')
        plt.title('Evolved Function')
        plt.show()
    
    @staticmethod
    def plot_convergence_dashboard(logbook):
        """Dashboard mit allen Metriken"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Fitness Entwicklung
        axes[0,0].plot(logbook.select("min"))
        axes[0,0].set_title('Best Fitness per Generation')
        axes[0,0].set_ylabel('Error')
        axes[0,0].set_xlabel('Generation')
        axes[0,0].grid(True)
        
        # Populations-Diversität
        axes[0,1].plot(logbook.select("size"))
        axes[0,1].set_title('Population Size Distribution')
        axes[0,1].set_ylabel('Size')
        axes[0,1].set_xlabel('Generation')
        
        # Erfolgsrate über Zeit
        # ... weitere Plots
        
        plt.tight_layout()
        plt.show()