import json
import os
import pandas as pd

def load_product_database(filename: str, dummy_data=None) -> pd.DataFrame:
    """
    Loads product data from a JSON file and returns as a DataFrame.
    If the file does not exist, creates and saves dummy data if provided.
    """
    if os.path.exists(filename):
        print(f"Loading product database from {filename}...")
        with open(filename, 'r') as f:
            product_data = json.load(f)
    else:
        print(f"File {filename} not found. Creating a dummy product database and saving it.")
        if dummy_data is None:
            raise ValueError("No dummy data provided and file does not exist.")
        with open(filename, 'w') as f:
            json.dump(dummy_data, f, indent=2)
        product_data = dummy_data
    return pd.json_normalize(product_data)

def add_combined_features(product_database: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'combined_features' column to the product DataFrame for NLP-based retrieval.
    """
    product_database['combined_features'] = product_database.apply(
        lambda row: f"{row['name']} {row['description']} {' '.join(row['effects'])} {' '.join(row['ingredients'])} {row['type']}",
        axis=1
    )
    return product_database