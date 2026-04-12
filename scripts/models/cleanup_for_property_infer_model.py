import pandas as pd
import re
import os

# --- CONSTANTS ---
RAW_CSV_PATH = "data/csv/watches_classified.csv"
CLEAN_CSV_PATH = "data/csv/watch_records_cleaned.csv"
IMAGE_ROOT_DIR = "data/images/classified/face"

# --- CORE LOGIC ---

def generate_cleaned_dataset(input_path, output_path):
    """
    Orchestrates the cleaning process to produce a model-ready CSV.
    """
    raw_dataframe = pd.read_csv(input_path)
    
    cleaned_dataframe = select_relevant_columns(raw_dataframe)
    cleaned_dataframe = apply_data_transformations(cleaned_dataframe)
    cleaned_dataframe = filter_missing_images(cleaned_dataframe, IMAGE_ROOT_DIR)

    cleaned_dataframe.to_csv(output_path, index=False)
    print(f"Success: Cleaned data saved to {output_path}")

def select_relevant_columns(df):
    """
    Drops metadata and keeps only visual/price attributes.
    """
    required_columns = [
        "Unnamed: 0", "Brand", "Case material", "Dial", 
        "Price", "face_image_name"
    ]
    return df[required_columns].copy()

def apply_data_transformations(df):
    """
    Applies cleaning logic to individual columns.
    """
    df["Price"] = df["Price"].apply(extract_numeric_price)
    df = df.dropna(subset=["Price", "Brand", "Case material"])
    return df

def extract_numeric_price(price_string):
    """
    Converts '$1,800 [Negotiable]' or '€2.500' to a float value.
    Returns None if the price cannot be parsed.
    """
    if pd.isna(price_string):
        return None
        
    # Remove all non-numeric characters except the decimal point
    clean_string = re.sub(r'[^\d.]', '', price_string.replace(',', ''))
    
    try:
        return float(clean_string)
    except ValueError:
        return None

def filter_missing_images(df, image_dir):
    """
    Ensures every row in the CSV actually has a corresponding file on disk.
    """
    def check_file_exists(row):
        file_path = os.path.join(image_dir, f"{row['Unnamed: 0']}_{row['face_image_name']}")
        return os.path.exists(file_path)

    valid_mask = df.apply(check_file_exists, axis=1)
    print(f"Dropped {len(df) - valid_mask.sum()} rows due to missing image files.")
    
    return df[valid_mask]

# --- EXECUTION ---

if __name__ == "__main__":
    generate_cleaned_dataset(RAW_CSV_PATH, CLEAN_CSV_PATH)