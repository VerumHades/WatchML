import os
import shutil
import pandas as pd

# --- CONSTANTS ---
SOURCE_CSV_PATH = "marking/watches_classified.csv"
BASE_IMAGE_DIRECTORY_COLUMN = "image_directory"
FACE_INDEX_COLUMN = "face_image_index"

DESTINATION_ROOT = "data/classified"
CLASS_YES_DIR = os.path.join(DESTINATION_ROOT, "face_images")
CLASS_NO_DIR = os.path.join(DESTINATION_ROOT, "not_face_images")


def ensure_directory_exists(directory_path):
    """
    Creates the directory and any necessary parent directories.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def get_all_images_in_folder(folder_path):
    """
    Returns a sorted list of image files found in the specified directory.
    """
    if not os.path.exists(folder_path):
        return []
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    return sorted(files)


def migrate_file(source_folder, filename, target_folder, row_index):
    """
    Copies a file with a unique row-based prefix to the target folder.
    """
    source_path = os.path.join(source_folder, filename)
    unique_name = f"{row_index}_{filename}"
    destination_path = os.path.join(target_folder, unique_name)
    
    try:
        shutil.copy2(source_path, destination_path)
        return True
    except Exception:
        return False


def process_row(row, row_index):
    """
    Separates the face image from the rest of the images in the directory.
    """
    directory = str(row[BASE_IMAGE_DIRECTORY_COLUMN]).strip()
    face_idx = row[FACE_INDEX_COLUMN]

    # Skip if directory is missing or classification didn't happen
    if pd.isna(directory) or pd.isna(face_idx) or face_idx == -1 or pd.isna(row["face_image_name"]):
        return 0, 0

    face_idx = int(face_idx)
    all_images = get_all_images_in_folder(directory)
    
    if not all_images:
        return 0, 0

    yes_count = 0
    no_count = 0

    for current_idx, filename in enumerate(all_images):
        # If this index matches the face_index, it goes to YES
        if current_idx == face_idx and face_idx != -1:
            if migrate_file(directory, filename, CLASS_YES_DIR, row_index):
                yes_count += 1
        # Otherwise (or if face_index is -1), it goes to NO
        else:
            if migrate_file(directory, filename, CLASS_NO_DIR, row_index):
                no_count += 1

    return yes_count, no_count


def process_image_consolidation():
    """
    Coordinates the extraction of faces vs non-faces based on CSV indices.
    """
    data_frame = pd.read_csv(SOURCE_CSV_PATH)
    
    ensure_directory_exists(CLASS_YES_DIR)
    ensure_directory_exists(CLASS_NO_DIR)

    total_yes = 0
    total_no = 0

    for index, row in data_frame.iterrows():
        yes_inc, no_inc = process_row(row, index)
        total_yes += yes_inc
        total_no += no_inc

    print_final_report(total_yes, total_no, len(data_frame))


def print_final_report(yes, no, rows):
    """
    Displays the final count of sorted images.
    """
    print("=" * 35)
    print(f"CLASSIFICATION REPORT")
    print("-" * 35)
    print(f"Face Images (Yes):  {yes}")
    print(f"Other Images (No):  {no}")
    print(f"Total Rows Parsed:  {rows}")
    print(f"Output: {os.path.abspath(DESTINATION_ROOT)}")
    print("=" * 35)


if __name__ == "__main__":
    try:
        process_image_consolidation()
    except Exception as error:
        print(f"Critical error: {error}")