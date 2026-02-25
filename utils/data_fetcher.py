import pandas as pd
import os
import requests

def fetch_nasa_airfoil():
    """
    Fetches the NASA Airfoil Self-Noise dataset from UCI Machine Learning Repository,
    cleans it, and saves it as a CSV.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
    columns = ["Frequency", "AngleOfAttack", "ChordLength", "FreeStreamVelocity", "SuctionSideDisplacement", "SSPL"]
    
    print(f"Fetching data from {url}...")
    try:
        # Load the file directly from the URL
        df = pd.read_csv(url, sep='\t', header=None, names=columns)
        
        # Ensure data directory exists
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        csv_path = os.path.join(data_dir, "nasa_airfoil.csv")
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved cleaned data to {csv_path}")
        print(f"Shape: {df.shape}")
        return csv_path
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    fetch_nasa_airfoil()
