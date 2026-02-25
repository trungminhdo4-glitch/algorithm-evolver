import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.csv_problem import CsvProblem
from core.physics import Dimension

def debug_csv_problem():
    file_path = "data/gravity_test.csv"
    target_col = "Force [N]"
    
    # Matching the verification script
    units_dict = {
        'rand_dimensional': Dimension(-1, 3, -2, 0, 0) # Units of G
    }
    
    prob = CsvProblem(file_path, target_col, units_dict=units_dict)
    print(f"Target Column: {prob.target_column}")
    print(f"Sanitized Target: {prob.sanitized_target}")
    print(f"Target Unit: {prob.target_unit}")
    
    for name, unit in prob.pset_input_units.items():
        print(f"Input '{name}' Unit: {unit}")

if __name__ == "__main__":
    debug_csv_problem()
