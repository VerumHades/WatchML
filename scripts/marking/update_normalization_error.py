import sqlite3

# --- CONFIGURATION ---
DB_PATH = "data/db/zoom_mark_progress.db"

def repair_normalized_coordinates():
    """
    Translates coordinates from 800x600 Canvas-space to 600x600 Image-space.
    Assumes 681x681 images were displayed as 600x600 centered at X=400.
    """
    try:
        connection = sqlite3.connect(DB_PATH)
        cursor = connection.cursor()

        # 1. Calculate the pixel values from the old normalized format (relative to 800x600)
        # 2. Subtract the 100px horizontal offset ( (800-600)/2 )
        # 3. Divide by the actual displayed image size (600px)
        
        sql_update = """
            UPDATE watch_labels
            SET 
                center_x = (center_x * 800) / 600,
                center_y = (center_y * 600) / 600,
                radius   = (radius * 800) / 600
        """

        cursor.execute(sql_update)
        connection.commit()
        
        rows_affected = cursor.rowcount
        print(f"Successfully repaired {rows_affected} labels.")
        print("Your coordinates are now relative to the image pixels (0.0 to 1.0).")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if connection:
            connection.close()

if __name__ == "__main__":
    # WARNING: Run this only ONCE. 
    # Running it multiple times will shift coordinates further.
    confirm = input("This will permanently shift DB coordinates. Proceed? (y/n): ")
    if confirm.lower() == 'y':
        repair_normalized_coordinates()
    else:
        print("Operation cancelled.")