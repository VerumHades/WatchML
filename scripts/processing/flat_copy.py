import os
import shutil

def make_flat_copy_from_subdirectories(
        source_path = "data/images/scraped",
        destination_path = "data/images/flattened"):
    """
    Copies all the contents of subdirectories of a directory into one flat destination directory.

    :param source_path: _description_, defaults to "data/images/scraped"
    :type source_path: str, optional
    :param destination_path: _description_, defaults to "data/images/flattened"
    :type destination_path: str, optional
    """
    
    if not os.path.exists(source_path):
        print("Invalid source path")
        exit()

    os.makedirs(destination_path, exist_ok=True)

    for brand_dir in os.listdir(source_path):
        if not os.path.isdir():
            continue

        shutil.copytree(
            os.path.join(source_path, brand_dir), 
            destination_path, 
            dirs_exist_ok=True
        )
if __name__ == "__main__":
    make_flat_copy_from_subdirectories()