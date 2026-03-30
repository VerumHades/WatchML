import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import sqlite3
import os

# --- CONFIGURATION ---
CSV_PATH = "processing/full.csv"
DB_PATH = "progress.db"
IMAGE_SIZE = (600, 600)

# KEYBINDS
KEY_NEXT = "d"
KEY_PREV = "a"
KEY_MARK = "f"
KEY_SKIP = "s"

class WatchDataManager:
    """Handles CSV loading, SQLite persistence, and progress filtering."""
    
    def __init__(self, csv_path, db_path):
        self.df = pd.read_csv(csv_path)
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Creates the tracking table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_marks (
                    watch_url TEXT PRIMARY KEY,
                    face_image_index INTEGER
                )
            """)

    def get_marked_urls(self):
        """Returns a set of all watch URLs already processed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT watch_url FROM face_marks")
            return {row[0] for row in cursor.fetchall()}

    def mark_face(self, watch_url, image_index):
        """Saves the chosen image index for the watch URL."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO face_marks VALUES (?, ?)",
                (watch_url, image_index)
            )

    def skip_watch(self, watch_url):
        """Marks a watch as processed with a sentinel value (-1) to indicate skipped."""
        self.mark_face(watch_url, -1)

    def get_watch_data(self, index):
        return self.df.iloc[index].to_dict()

    def get_total_count(self):
        return len(self.df)


class WatchReviewApp:
    """UI Application for reviewing watch images and marking faces."""

    def __init__(self, root, manager):
        self.root = root
        self.manager = manager
        
        # Start at the first unmarked watch
        self.current_watch_index = self._get_first_unmarked_index()
        self.current_image_index = 0
        self.auto_next_var = tk.BooleanVar(value=True)
        
        self._setup_ui()
        self._bind_keys()
        self._load_current_record()

    def _get_first_unmarked_index(self):
        """Finds the first index in the dataframe not present in the DB."""
        marked_urls = self.manager.get_marked_urls()
        for i, row in self.manager.df.iterrows():
            if row['url'] not in marked_urls:
                return i
        return 0

    def _setup_ui(self):
        self.root.title("Watch Face Labeler")
        
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(main_container)
        self.image_label.pack(side=tk.LEFT, padx=10)

        self.details_text = tk.Text(main_container, width=40, height=30)
        self.details_text.pack(side=tk.RIGHT, fill=tk.Y)

        bottom_bar = ttk.Frame(self.root, padding="5")
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(bottom_bar, text=f"Last ({KEY_PREV})", command=self.show_previous_image).pack(side=tk.LEFT)
        ttk.Button(bottom_bar, text=f"Next ({KEY_NEXT})", command=self.show_next_image).pack(side=tk.LEFT)
        
        ttk.Button(bottom_bar, text=f"Mark Face ({KEY_MARK})", command=self.mark_as_face).pack(side=tk.RIGHT)
        ttk.Button(bottom_bar, text=f"Skip Watch ({KEY_SKIP})", command=self.skip_entire_watch).pack(side=tk.RIGHT, padx=5)

        ttk.Checkbutton(bottom_bar, text="Auto-next Watch", variable=self.auto_next_var).pack(side=tk.RIGHT, padx=10)

    def _bind_keys(self):
        self.root.bind(f"<{KEY_PREV}>", lambda e: self.show_previous_image())
        self.root.bind(f"<{KEY_NEXT}>", lambda e: self.show_next_image())
        self.root.bind(f"<{KEY_MARK}>", lambda e: self.mark_as_face())
        self.root.bind(f"<{KEY_SKIP}>", lambda e: self.skip_entire_watch())

    def _load_current_record(self):
        if self.current_watch_index >= self.manager.get_total_count():
            self.details_text.insert(tk.END, "\n\nALL WATCHES PROCESSED!")
            return
            
        self.data = self.manager.get_watch_data(self.current_watch_index)
        self.current_image_index = 0
        self._update_display()

    def _update_display(self):
        self._refresh_text_details()
        self._refresh_image()

    def _refresh_text_details(self):
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, f"Watch {self.current_watch_index + 1} of {self.manager.get_total_count()}\n\n")
        for key, value in self.data.items():
            if key != "image_directory":
                self.details_text.insert(tk.END, f"{key}: {value}\n")

    def _refresh_image(self):
        directory = self.data.get("image_directory", "")
        if not os.path.exists(directory):
            self.image_label.config(image='', text="Directory Not Found")
            return

        images = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not images:
            self.image_label.config(image='', text="No Images Found")
            return

        image_path = os.path.join(directory, images[self.current_image_index % len(images)])
        img = Image.open(image_path)
        img.thumbnail(IMAGE_SIZE)
        
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img)

    def show_next_image(self):
        self.current_image_index += 1
        self._refresh_image()

    def show_previous_image(self):
        self.current_image_index -= 1
        self._refresh_image()

    def skip_entire_watch(self):
        """Records the watch as skipped and moves to the next."""
        self.manager.skip_watch(self.data["url"])
        self._move_to_next_watch()

    def mark_as_face(self):
        self.manager.mark_face(self.data["url"], self.current_image_index)
        if self.auto_next_var.get():
            self._move_to_next_watch()

    def _move_to_next_watch(self):
        if self.current_watch_index < self.manager.get_total_count() - 1:
            self.current_watch_index += 1
            self._load_current_record()

if __name__ == "__main__":
    root = tk.Tk()
    manager = WatchDataManager(CSV_PATH, DB_PATH)
    app = WatchReviewApp(root, manager)
    root.mainloop()