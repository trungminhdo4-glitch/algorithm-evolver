import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.physics import Dimension
from experiments.run_csv_discovery import run_csv_experiment

def verify_csv_gravitation():
    file_path = "data/gravity_test.csv"
    target_col = "Force [N]"
    
    # Define units for the gravitational constant to make discovery possible
    units_dict = {
        'rand_dimensional': Dimension(-1, 3, -2, 0, 0) # Units of G
    }
    
    print("Running verification of Universal CSV Loader on Gravitation Data...")
    run_csv_experiment(
        file_path, 
        target_col, 
        population_size=1000, 
        generations=100, 
        units_dict=units_dict
    )

if __name__ == "__main__":
    verify_csv_gravitation()
