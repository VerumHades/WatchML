import os
import sqlite3
import pandas as pd
import shutil
from PIL import Image, ImageDraw
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_PATH = "data/csv/full.csv"
DB_PATH = "data/db/zoom_mark_progress.db"
SOURCE_ROOT = r"data/images/flattened"
OUTPUT_ROOT = r"data/images/organized_dataset"

# Color for the area outside the watch face (R, G, B)
FILL_COLOR = (0, 0, 0) 

# Map DB integers to folder names
CATEGORY_MAP = {
    0: "no_face",
    1: "face",
    2: "misoriented"
}

def create_directory_structure():
    """Initializes the output folders including a subfolder for masked faces."""
    for folder in CATEGORY_MAP.values():
        os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)
    
    # Dedicated folder for the cut-out/masked versions
    os.makedirs(os.path.join(OUTPUT_ROOT, "face_masked"), exist_ok=True)
    print(f"Directory structure initialized at: {OUTPUT_ROOT}")

REPAIR_681_OFFSET = False # Set to True to fix old labels during processing
FIXED_SIZE = 681         # The size of your current images

def apply_face_mask(source_path, target_path, nx, ny, nr):
    with Image.open(source_path).convert("RGB") as img:
        w, h = img.size # Works for 681x681 or any other size
        
        # ADJUSTMENT LOGIC:
        # If data was saved 'the old way' (relative to 800x600 canvas):
        if REPAIR_681_OFFSET:
            # Reconstruct the old 100px offset math
            # old_nx = (pixel_x) / 800
            # true_nx = (pixel_x - 100) / 600
            pixel_x = nx * 800
            pixel_y = ny * 600
            pixel_r = nr * 800
            
            # Recalculate relative to the 600px square inside the 800x600 canvas
            nx = pixel_x / 600
            ny = pixel_y / 600
            nr = pixel_r / 600

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        cx, cy = nx * w, ny * h
        r = nr * w
        
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        
        result = Image.new("RGB", (w, h), FILL_COLOR)
        result.paste(img, (0, 0), mask=mask)
        result.save(target_path, quality=95)

def organize_and_mask():
    """Main execution loop to sort and process the dataset."""
    df_watches = pd.read_csv(CSV_PATH)
    
    with sqlite3.connect(DB_PATH) as conn:
        df_labels = pd.read_sql_query("SELECT * FROM watch_labels", conn)
    
    create_directory_structure()
    url_to_index = {row['url']: idx for idx, row in df_watches.iterrows()}

    print("Processing images and applying masks...")
    for _, label_row in tqdm(df_labels.iterrows(), total=len(df_labels)):
        watch_url = label_row['watch_url']
        filename = label_row['image_filename']
        status = label_row['is_face']

        if watch_url not in url_to_index:
            continue

        # Resolve Source Path
        watch_idx = url_to_index[watch_url]
        watch_data = df_watches.iloc[watch_idx]
        rel_dir = watch_data['image_directory'].replace("processing/images/", "").replace("processing\\images\\", "")
        source_path = os.path.join(SOURCE_ROOT, rel_dir, filename)

        if not os.path.isfile(source_path):
            continue

        # Metadata for filename linking
        new_filename = f"{watch_idx}__{filename}"
        category_folder = CATEGORY_MAP.get(status, "unknown")
        
        # 1. Standard Copy (All categories)
        standard_target = os.path.join(OUTPUT_ROOT, category_folder, new_filename)
        shutil.copy2(source_path, standard_target)

        # 2. Masking (Faces only)
        if status == 1:
            masked_target = os.path.join(OUTPUT_ROOT, "face_masked", new_filename)
            try:
                apply_face_mask(
                    source_path, 
                    masked_target, 
                    label_row['center_x'], 
                    label_row['center_y'], 
                    label_row['radius']
                )
            except Exception as e:
                print(f"Error processing {new_filename}: {e}")

if __name__ == "__main__":
    organize_and_mask()
    print(f"\nOrganization complete. Check {OUTPUT_ROOT}/face_masked for cut-outs.")