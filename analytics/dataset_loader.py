"""
Module 2: Dataset Loader
Fetches the official Seaborn Titanic dataset and saves it to analytics/titanic.csv.
Ensures 100% reproducible offline execution.
"""

import os
import requests
import io
import pandas as pd

LOCAL_TITANIC_CSV = "analytics/titanic.csv"
DATA_URL = "https://cdn.jsdelivr.net/gh/mwaskom/seaborn-data/titanic.csv"

def get_or_load_titanic(csv_path: str = LOCAL_TITANIC_CSV) -> pd.DataFrame:
    """
    Loads Titanic from local CSV if present, otherwise fetches the seaborn dataset
    via CDN and persists to disk.
    """
    if os.path.exists(csv_path):
        print(f"Loading cached Titanic dataset from {csv_path}...")
        return pd.read_csv(csv_path)

    print(f"Fetching Seaborn Titanic dataset from {DATA_URL}...")
    try:
        resp = requests.get(DATA_URL, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"CDN fetch failed ({e}), attempting seaborn fallback...")
        import seaborn as sns
        df = sns.load_dataset("titanic")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved Titanic dataset to {csv_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df

if __name__ == "__main__":
    get_or_load_titanic()
