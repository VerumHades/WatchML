import os
import shutil

SOURCE_PATH = "data/images"
DESTINATION_PATH = "processing/images"

if __name__ == "__main__":
    if not os.path.exists(SOURCE_PATH):
        print("Invalid source path")
        exit()

    os.makedirs(DESTINATION_PATH, exist_ok=True)

    for brand_dir in os.listdir(SOURCE_PATH):
        shutil.copytree(
            os.path.join(SOURCE_PATH, brand_dir), 
            DESTINATION_PATH, 
            dirs_exist_ok=True
        )