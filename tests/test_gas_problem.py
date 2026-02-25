from problems.ideal_gas import IdealGasProblem
import numpy as np

print("Testing IdealGasProblem instantiation...")
try:
    problem = IdealGasProblem()
    pset = problem.create_primitive_set()
    print("Problem and PSet created successfully.")
    
    print("Terminal names in pset:")
    for t in pset.terminals[float]:
        print(f" - {t.name}")
        
    print("Inputs max:", problem.inputs_max)
    print("Train data sample:", problem.train_data[0])
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
