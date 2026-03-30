import pandas as pd
import sqlite3
import os

# --- CONFIGURATION ---
CSV_INPUT = "processing/full.csv"
DB_PATH = "progress.db"
CSV_OUTPUT = "marking/watches_classified.csv"

def get_db_marks(db_path):
    """Fetch all face marks from the SQLite database."""
    with sqlite3.connect(db_path) as connection:
        query = "SELECT watch_url, face_image_index FROM face_marks"
        return pd.read_sql_query(query, connection)

def get_face_filename(row):
    """
    Determines the specific image filename based on the saved index.
    Returns None if skipped or no image is found.
    """
    directory = row.get('image_directory')
    index = row.get('face_image_index')

    # Skip if no mark exists or if the watch was marked as 'skipped' (-1)
    if pd.isna(index) or index < 0:
        return None

    if not os.path.isdir(str(directory)):
        return "Directory Not Found"

    # Get sorted files to match the UI's display logic
    try:
        images = sorted([
            file for file in os.listdir(directory) 
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if 0 <= int(index) < len(images):
            return images[int(index)]
    except Exception:
        return "Error Accessing Files"

    return "Index Out of Bounds"

def generate_classified_csv():
    """Merges CSV data with database marks and exports the result."""
    # 1. Load Data
    original_df = pd.read_csv(CSV_INPUT)
    marks_df = get_db_marks(DB_PATH)

    # 2. Merge on the unique 'url' identifier
    merged_df = pd.merge(
        original_df, 
        marks_df, 
        left_on='url', 
        right_on='watch_url', 
        how='left'
    )

    # 3. Create the new column 'face_image_name'
    print("Processing image indices...")
    merged_df['face_image_name'] = merged_df.apply(get_face_filename, axis=1)

    # 4. Clean up and Save
    # Drop the extra join column from SQLite
    if 'watch_url' in merged_df.columns:
        merged_df = merged_df.drop(columns=['watch_url'])

    merged_df.to_csv(CSV_OUTPUT, index=False)
    print(f"Successfully generated: {CSV_OUTPUT}")

if __name__ == "__main__":
    if os.path.exists(CSV_INPUT) and os.path.exists(DB_PATH):
        generate_classified_csv()
    else:
        print("Error: Ensure both watches.csv and progress.db exist in the directory.")