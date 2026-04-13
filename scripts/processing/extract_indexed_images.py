import os
import csv
import shutil

SOURCE_BASE_DIRECTORY = "data/images/flattened"
OUTPUT_DIRECTORY = "data/images/indexed"
INPUT_CSV_FILE = "data/csv/full.csv"

def run_image_extraction_pipeline(csv_file_path):
    """
    Initializes the extraction process by preparing the output 
    environment and calculating the total rows for progress tracking.
    """
    create_output_directory(OUTPUT_DIRECTORY)
    
    total_records = count_csv_rows(csv_file_path)
    process_csv_rows_with_progress(csv_file_path, total_records)

def count_csv_rows(csv_file_path):
    """
    Performs a preliminary scan of the file to determine the total 
    number of records for the progress bar.
    """
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        return sum(1 for line in csv_file) - 1

def create_output_directory(directory_path):
    """
    Ensures the destination directory exists to prevent 
    file system errors during copying.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def process_csv_rows_with_progress(csv_file_path, total_records):
    """
    Reads the CSV file and updates a manual progress bar in the 
    console as it iterates through records.
    """
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for real_index, row_data in enumerate(reader):
            handle_row_image_transfer(real_index, row_data)
            display_progress_bar(real_index + 1, total_records)
    print("\nProcessing Complete.")

def display_progress_bar(iteration, total):
    """
    Calculates the completion percentage and renders a visual 
    bar directly in the terminal output.
    """
    bar_length = 40
    progress = iteration / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\rProgress: |{bar}| {progress:.1%}', end='\r')

def handle_row_image_transfer(current_index, row_data):
    """
    Extracts the image directory from the row and initiates 
    the file transfer for that specific entry.
    """
    raw_directory_path = row_data.get('image_directory', '')
    source_folder = resolve_source_path(raw_directory_path)
    
    migrate_files_to_output(current_index, source_folder)

def resolve_source_path(raw_path):
    """
    Cleans the raw path from the CSV to extract the final 
    folder name and merges it with the source base.
    """
    clean_folder_name = os.path.basename(os.path.normpath(raw_path))
    return os.path.join(SOURCE_BASE_DIRECTORY, clean_folder_name)

def migrate_files_to_output(index_prefix, source_path):
    """
    Scans the resolved source directory and copies all files 
    found within to the central output folder.
    """
    if not os.path.isdir(source_path):
        return

    for file_name in os.listdir(source_path):
        finalize_file_copy(index_prefix, file_name, source_path)

def finalize_file_copy(prefix, file_name, source_folder):
    """
    Performs the physical copy of the file while renaming 
    it with the zero-based row index prefix.
    """
    full_source_file_path = os.path.join(source_folder, file_name)
    
    if os.path.isfile(full_source_file_path):
        prefixed_name = f"{prefix}_{file_name}"
        destination_path = os.path.join(OUTPUT_DIRECTORY, prefixed_name)
        shutil.copy2(full_source_file_path, destination_path)

if __name__ == "__main__":
    run_image_extraction_pipeline(INPUT_CSV_FILE)