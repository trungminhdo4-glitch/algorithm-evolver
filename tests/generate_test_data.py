import pandas as pd
import numpy as np
import random

def generate_gravity_data():
    G = 1.0
    data = []
    
    for _ in range(100):
        m1 = random.uniform(1, 100)
        m2 = random.uniform(1, 100)
        r = random.uniform(1, 10)
        F = G * (m1 * m2) / (r**2)
        
        # 1% Noise
        noise = np.random.normal(0, 0.01 * F)
        noisy_F = F + noise
        
        data.append({
            'm1 [kg]': m1,
            'm2 [kg]': m2,
            'r [m]': r,
            'Force [N]': noisy_F
        })
        
    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/gravity_test.csv", index=False)
    print("Generated data/gravity_test.csv")

if __name__ == "__main__":
    import os
    generate_gravity_data()
