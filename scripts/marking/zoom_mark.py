import tkinter as tk
from tkinter import ttk
import sqlite3
import pandas as pd
import os
from PIL import Image, ImageTk

# --- CONFIGURATION ---
CSV_PATH = "data/csv/full.csv"
DB_PATH = "data/db/zoom_mark_progress.db"
ROOT_DATA_DIR = r"data/images/flattened"
DEFAULT_RADIUS = 50
THUMBNAIL_SIZE = (100, 75)

STATUS_COLORS = {0: "red", 1: "green", 2: "orange", None: "gray"}

class WatchDataManager:
    """Handles all SQLite and CSV data operations."""

    def __init__(self, csv_path, db_path):
        self.data_frame = pd.read_csv(csv_path)
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Creates the labeling table with geometric and classification data."""
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS watch_labels (
                    watch_url TEXT,
                    image_filename TEXT,
                    is_face INTEGER,
                    center_x REAL,
                    center_y REAL,
                    radius REAL,
                    PRIMARY KEY (watch_url, image_filename)
                )
            """)

    def save_label(self, url, filename, status, x, y, r):
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO watch_labels VALUES (?, ?, ?, ?, ?, ?)",
                (url, filename, status, x, y, r)
            )

    def get_label_data(self, url, filename):
        """Retrieves existing coordinates and status for a specific image."""
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT is_face, center_x, center_y, radius FROM watch_labels "
                "WHERE watch_url = ? AND image_filename = ?", (url, filename)
            )
            return cursor.fetchone()

    def get_labels_map(self, url):
        """Returns a dict of {filename: status} for a watch folder."""
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "SELECT image_filename, is_face FROM watch_labels WHERE watch_url = ?", (url,)
            )
            return {row[0]: row[1] for row in cursor.fetchall()}

class GeometryEditor(tk.Canvas):
    """
    Adaptive interactive canvas for arbitrary image sizes.
    Ensures coordinates are normalized to actual image pixels, not the canvas.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, cursor="cross", bg="#1e1e1e", **kwargs)
        self.raw_image = None
        self.tk_image = None
        
        # Geometry tracking (relative to the 800x600 canvas)
        self.img_offset_x = 0
        self.img_offset_y = 0
        self.img_display_w = 1
        self.img_display_h = 1
        
        self.circle_x, self.circle_y = 400, 300
        self.radius = DEFAULT_RADIUS
        self._setup_bindings()

    def _setup_bindings(self):
        """Sets up mouse interactions for moving and resizing the circle."""
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<MouseWheel>", self._on_mouse_scroll)
        # Support for Linux/Other mouse wheel variants
        self.bind("<Button-4>", lambda e: self._on_mouse_scroll(e, 1))
        self.bind("<Button-5>", lambda e: self._on_mouse_scroll(e, -1))

    def load_image(self, path, existing_data=None):
        """
        Loads the image and maps existing normalized data back to canvas pixels.
        """
        self.raw_image = Image.open(path)
        
        # We must refresh first to calculate offsets/scale before rehydrating
        self.refresh()
        
        if existing_data:
            _, nx, ny, nr = existing_data
            self._rehydrate_coordinates(nx, ny, nr)
            self.refresh()

    def refresh(self):
        """Calculates scaling and centering, then renders the image and circle."""
        if not self.raw_image:
            return

        canvas_w = self.winfo_width() if self.winfo_width() > 1 else 800
        canvas_h = self.winfo_height() if self.winfo_height() > 1 else 600

        # Calculate scale to fit while maintaining aspect ratio
        scale = min(canvas_w / self.raw_image.width, canvas_h / self.raw_image.height)
        
        self.img_display_w = int(self.raw_image.width * scale)
        self.img_display_h = int(self.raw_image.height * scale)
        
        # Calculate padding to center the image (Letterboxing)
        self.img_offset_x = (canvas_w - self.img_display_w) // 2
        self.img_offset_y = (canvas_h - self.img_display_h) // 2
        
        resized = self.raw_image.resize(
            (self.img_display_w, self.img_display_h), 
            Image.Resampling.LANCZOS
        )
        
        self.tk_image = ImageTk.PhotoImage(resized)
        self.delete("all")
        
        # Draw image centered
        self.create_image(
            self.img_offset_x, 
            self.img_offset_y, 
            anchor="nw", 
            image=self.tk_image
        )
        
        self._draw_circle()

    def _draw_circle(self):
        """Renders the cyan circle overlay."""
        x, y, r = self.circle_x, self.circle_y, self.radius
        self.create_oval(x-r, y-r, x+r, y+r, outline="cyan", width=3)

    def _rehydrate_coordinates(self, nx, ny, nr):
        """Translates 0.0-1.0 image-space values back into canvas-space pixels."""
        self.circle_x = (nx * self.img_display_w) + self.img_offset_x
        self.circle_y = (ny * self.img_display_h) + self.img_offset_y
        self.radius = nr * self.img_display_w

    def get_normalized(self):
        """
        Maps circle center and radius to 0.0-1.0 relative to the IMAGE boundaries.
        0.0 = Top/Left edge of image, 1.0 = Bottom/Right edge of image.
        """
        # Protect against division by zero if refresh hasn't run
        w = max(1, self.img_display_w)
        h = max(1, self.img_display_h)

        nx = (self.circle_x - self.img_offset_x) / w
        ny = (self.circle_y - self.img_offset_y) / h
        nr = self.radius / w
        
        return (nx, ny, nr)

    def _on_mouse_drag(self, event):
        """Updates circle position based on mouse motion."""
        self.circle_x, self.circle_y = event.x, event.y
        self.refresh()

    def _on_mouse_scroll(self, event, manual_delta=None):
        """Resizes circle radius."""
        if manual_delta:
            delta = manual_delta
        else:
            delta = 1 if event.delta > 0 else -1
            
        self.radius = max(5, self.radius + (delta * 4))
        self.refresh()

class LabelingDialog(tk.Toplevel):
    """Popup window for re-classifying from the dashboard."""

    def __init__(self, parent, image_path, url, filename, manager, on_save):
        super().__init__(parent)
        self.title(f"Edit: {filename}")
        self.editor = GeometryEditor(self, width=800, height=600)
        self.editor.pack()
        
        data = manager.get_label_data(url, filename)
        self.editor.load_image(image_path, data)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=10)
        
        for code, label in [(0, "No Face"), (1, "Face"), (2, "Orient")]:
            ttk.Button(btn_frame, text=label, 
                       command=lambda c=code: self._save(url, filename, c, manager, on_save)).pack(side=tk.LEFT, expand=True)

    def _save(self, url, fname, code, manager, callback):
        nx, ny, nr = self.editor.get_normalized()
        manager.save_label(url, fname, code, nx, ny, nr)
        callback()
        self.destroy()

class InspectionDashboard(tk.Toplevel):
    """
    Paginated dashboard that synchronizes with the main application's 
    current marking progress.
    """

    def __init__(self, parent, manager, current_watch_idx, on_global_refresh):
        super().__init__(parent)
        self.manager = manager
        self.on_global_refresh = on_global_refresh
        
        # State Management
        self.page_size = 10
        self.current_page = current_watch_idx // self.page_size
        self.cached_photos = [] 

        self.title("Watch Inspection Dashboard")
        self.geometry("1200x900")
        self._build_ui()
        self.render_page()

    def _build_ui(self):
        """Creates the fixed header and the scrollable content area."""
        # --- Fixed Navigation Header ---
        nav_frame = ttk.Frame(self, padding=10, relief="raised")
        nav_frame.pack(side="top", fill="both")

        ttk.Button(nav_frame, text="<< Previous", command=self._prev_page).pack(side="left", padx=5)
        self.page_label = ttk.Label(nav_frame, text="Page 1", font=("Arial", 10, "bold"))
        self.page_label.pack(side="left", padx=20)
        ttk.Button(nav_frame, text="Next >>", command=self._next_page).pack(side="left", padx=5)

        ttk.Button(nav_frame, text="Jump to My Progress", command=self._jump_to_current).pack(side="right", padx=10)

        # --- Scrollable Content Area ---
        self.canvas = tk.Canvas(self, bg="#121212")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

    def render_page(self):
        """Clears current view and renders the requested slice of watches."""
        # Clear existing rows and memory
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.cached_photos = [] 

        total_watches = len(self.manager.data_frame)
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, total_watches)
        
        # Update Page Indicator
        total_pages = (total_watches // self.page_size) + 1
        self.page_label.config(text=f"Page {self.current_page + 1} of {total_pages} (Watches {start_idx}-{end_idx})")

        # Render rows
        batch = self.manager.data_frame.iloc[start_idx:end_idx]
        for _, row in batch.iterrows():
            self._create_watch_row(row)
            
        # Reset scroll to top on page change
        self.canvas.yview_moveto(0)

    def _create_watch_row(self, row):
        """Standard row builder for a single watch."""
        container = ttk.Frame(self.scroll_frame, padding=10)
        container.pack(fill="x", pady=2)
        
        info = f"{row['Brand']}\n{row.get('Model', 'N/A')}"
        ttk.Label(container, text=info, width=20).pack(side="left")

        img_strip = ttk.Frame(container)
        img_strip.pack(side="left", fill="x", expand=True)

        raw_path = row['image_directory'].replace("processing/images/", "").replace("processing\\images\\", "")
        abs_path = os.path.join(ROOT_DATA_DIR, raw_path)

        if os.path.exists(abs_path):
            labels = self.manager.get_labels_map(row['url'])
            # Render up to 15 thumbnails per watch for performance
            image_files = sorted([f for f in os.listdir(abs_path) if f.lower().endswith(('.jpg', '.png'))])[:15]
            
            for fname in image_files:
                self._add_thumbnail(img_strip, abs_path, fname, row['url'], labels.get(fname))

    def _add_thumbnail(self, parent, path, fname, url, status):
        full_path = os.path.join(path, fname)
        try:
            img = Image.open(full_path)
            img.thumbnail(THUMBNAIL_SIZE)
            photo = ImageTk.PhotoImage(img)
            self.cached_photos.append(photo)

            btn = tk.Button(
                parent, image=photo, 
                bg=STATUS_COLORS.get(status, "gray"), 
                bd=2, relief="flat",
                command=lambda: LabelingDialog(self, full_path, url, fname, self.manager, self._on_edit_done)
            )
            btn.pack(side="left", padx=1)
        except: pass

    # --- Navigation Logic ---

    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.render_page()

    def _next_page(self):
        max_page = len(self.manager.data_frame) // self.page_size
        if self.current_page < max_page:
            self.current_page += 1
            self.render_page()

    def _jump_to_current(self):
        """Calculates page based on where the user is in MainApp."""
        # We fetch the current index from the parent app
        self.current_page = self.master_app.watch_idx // self.page_size
        self.render_page()

    def _on_edit_done(self):
        """Visual refresh of the current page to update border colors."""
        self.render_page()
        self.on_global_refresh()


class MainApp:
    """
    Main Sequential Mode: Entry point for image-by-image labeling.
    Controls the flow of watches and coordinates with the Inspection Dashboard.
    """

    def __init__(self, root):
        self.root = root
        self.manager = WatchDataManager(CSV_PATH, DB_PATH)
        
        # State: track current watch, current image in folder, and the work queue
        self.watch_idx = 0
        self.img_idx = 0
        self.queue = []
        
        self._setup_ui()
        self._setup_keybinds()
        self._next()

    def _setup_ui(self):
        """Initializes the main canvas and control layout."""
        self.editor = GeometryEditor(self.root, width=800, height=600)
        self.editor.pack(fill=tk.BOTH, expand=True)
        
        controls = ttk.Frame(self.root)
        controls.pack(fill=tk.X, pady=10)
        
        # Action Buttons
        ttk.Button(controls, text="No Face (S)", 
                   command=lambda: self._finalize(0)).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Orient (W)", 
                   command=lambda: self._finalize(2)).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Dashboard (D)", 
                   command=self._open_dashboard).pack(side=tk.LEFT, padx=20)
        ttk.Button(controls, text="Confirm Face (F)", 
                   command=lambda: self._finalize(1)).pack(side=tk.RIGHT, padx=5)
        
        self.status = ttk.Label(controls, text="", font=("Consolas", 10))
        self.status.pack(side=tk.BOTTOM, pady=5)

    def _setup_keybinds(self):
        """Maps physical keys to classification actions."""
        self.root.bind("f", lambda e: self._finalize(1))
        self.root.bind("s", lambda e: self._finalize(0))
        self.root.bind("w", lambda e: self._finalize(2))
        self.root.bind("d", lambda e: self._open_dashboard())

    def _next(self):
        """Moves to the next unlabeled image across all watches."""
        while self.watch_idx < len(self.manager.data_frame):
            row = self.manager.data_frame.iloc[self.watch_idx]
            
            if not self.queue:
                self._prepare_watch_queue(row)
            
            if self.img_idx < len(self.queue):
                self._load_current_image(row)
                return

            # Reset for next watch
            self.watch_idx += 1
            self.img_idx = 0
            self.queue = []
            
        self.status.config(text="ALL DATA LABELED!")

    def _prepare_watch_queue(self, row):
        """Filters the watch directory for images not yet in the database."""
        path_fragment = row['image_directory'].replace("processing/images/", "").replace("processing\\images\\", "")
        abs_path = os.path.join(ROOT_DATA_DIR, path_fragment)
        
        if os.path.exists(abs_path):
            labeled_map = self.manager.get_labels_map(row['url'])
            all_files = sorted(os.listdir(abs_path))
            self.queue = [f for f in all_files if f not in labeled_map]

    def _load_current_image(self, row):
        """Calculates path and updates the editor and status bar."""
        filename = self.queue[self.img_idx]
        path_fragment = row['image_directory'].replace("processing/images/", "").replace("processing\\images\\", "")
        full_path = os.path.join(ROOT_DATA_DIR, path_fragment, filename)
        
        self.editor.load_image(full_path)
        self.status.config(text=f"Watch Index: {self.watch_idx} | Brand: {row['Brand']} | File: {filename}")

    def _finalize(self, code):
        """Saves geometry and classification, then advances."""
        if not self.queue:
            return

        row = self.manager.data_frame.iloc[self.watch_idx]
        filename = self.queue[self.img_idx]
        
        # Get normalized coordinates (nx, ny, nr) from editor
        nx, ny, nr = self.editor.get_normalized()
        
        self.manager.save_label(row['url'], filename, code, nx, ny, nr)
        
        self.img_idx += 1
        self._next()

    def _open_dashboard(self):
        """
        Launches the Dashboard, passing the current watch index 
        for immediate synchronization.
        """
        dashboard = InspectionDashboard(
            parent=self.root, 
            manager=self.manager, 
            current_watch_idx=self.watch_idx, 
            on_global_refresh=self._next
        )
        # Link back to self so dashboard can call MainApp methods if needed
        dashboard.master_app = self

if __name__ == "__main__":
    rt = tk.Tk(); MainApp(rt); rt.mainloop()