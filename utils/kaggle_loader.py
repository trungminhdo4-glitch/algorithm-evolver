import os
import zipfile
import pandas as pd
import re

def check_kaggle_auth():
    """Checks if kaggle.json exists in the expected location."""
    kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_path):
        print("\n" + "!" * 60)
        print("WARNING: Kaggle API Token (kaggle.json) not found!")
        print(f"Please place your kaggle.json in: {kaggle_path}")
        print("How to get it:")
        print("1. Go to kaggle.com -> Settings -> Create New AI Token.")
        print("2. Download the json and move it to the location above.")
        print("!" * 60 + "\n")
        return False
    return True

def download_kaggle_dataset(dataset_id, output_filename=None):
    """
    Downloads a Kaggle dataset, unzips it, and finds the CSV/DAT file.
    Args:
        dataset_id: z.B. 'fedesoriano/airfoil-self-noise-dataset'
        output_filename: Name for the final CSV (optional)
    Returns:
        Path to the local CSV.
    """
    if not check_kaggle_auth():
        return None

    try:
        import kaggle
    except ImportError:
        print("Kaggle library not installed. Please run 'pip install kaggle'.")
        return None

    # Determine paths
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_name = dataset_id.split('/')[-1]
    download_path = os.path.join(data_dir, dataset_name)
    os.makedirs(download_path, exist_ok=True)

    print(f"🚀 Downloading dataset '{dataset_id}'...")
    kaggle.api.dataset_download_files(dataset_id, path=download_path, unzip=True)

    # Find the data file (csv or dat)
    files = []
    for root, dirs, filenames in os.walk(download_path):
        for f in filenames:
            if f.endswith('.csv') or f.endswith('.dat'):
                files.append(os.path.join(root, f))

    if not files:
        print(f"❌ No CSV or DAT files found in dataset {dataset_id}")
        return None

    # Take the first one or the biggest one? Usually there is only one relevant.
    source_file = files[0]
    
    if output_filename:
        target_file = os.path.join(data_dir, f"{output_filename}.csv")
    else:
        target_file = os.path.join(data_dir, f"{dataset_name}.csv")

    # Load and save to ensure CSV format and clean renaming
    if source_file.endswith('.dat'):
        # Airfoil dataset is often tab-separated .dat
        df = pd.read_csv(source_file, sep='\t', header=None)
        # If no header, we might need to handle this later
    else:
        df = pd.read_csv(source_file)

    df.to_csv(target_file, index=False)
    print(f"✅ Dataset saved to: {target_file}")
    
    return target_file

def inject_units_interactively(file_path):
    """
    Interactive CLI to display headers and inject [unit] tags.
    """
    df = pd.read_csv(file_path)
    print("\n" + "=" * 60)
    print(f"UNIT HEADER INJECTION for: {os.path.basename(file_path)}")
    print("=" * 60)
    print("Preview of data (first 5 rows):")
    print(df.head())
    print("-" * 60)

    inject = input("Do you want to inject physical units into the headers? (y/n): ").lower()
    if inject != 'y':
        return file_path

    new_columns = []
    for col in df.columns:
        print(f"\nCurrent column: '{col}'")
        unit = input(f"Enter unit for '{col}' (e.g., 'kg', 'm/s') or press Enter for dimensionless: ").strip()
        if unit:
            # Clean unit name (remove brackets if user added them)
            unit = unit.replace('[', '').replace(']', '')
            new_columns.append(f"{col} [{unit}]")
        else:
            new_columns.append(col)

    df.columns = new_columns
    df.to_csv(file_path, index=False)
    print(f"\n✅ Updated headers saved to {file_path}")
    print(f"New headers: {list(df.columns)}")
    
    return file_path
