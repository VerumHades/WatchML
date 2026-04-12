import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from torchvision import transforms
import pickle
import os

# --- 1. MODEL LOADER ---
# Note: Ensure BrandConditionedNet is defined in your environment or imported
from scripts.models.train_infer_model import BrandConditionedNet 

class WatchExpertUI:
    def __init__(self, root, model_path, encoder_path):
        self.root = root
        self.root.title("Watch Expert: Model Inference Lab")
        
        # Load Resources
        with open(encoder_path, 'rb') as f:
            self.encoders = pickle.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BrandConditionedNet(
            num_brands=len(self.encoders['brand'].classes_),
            num_models=len(self.encoders['core_model'].classes_)
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # State Variables
        self.raw_image = None
        self.display_image = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.crop_coords = None

        self.setup_ui()

    def setup_ui(self):
        # Top Bar: Brand Selection
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        tk.Label(top_frame, text="Select Brand:").pack(side=tk.LEFT, padx=5)
        self.brand_var = tk.StringVar(self.root)
        brands = sorted(self.encoders['brand'].classes_.tolist())
        self.brand_var.set(brands[0])
        self.brand_menu = tk.OptionMenu(top_frame, self.brand_var, *brands)
        self.brand_menu.pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="RUN INFERENCE", bg="green", fg="white", command=self.infer).pack(side=tk.LEFT, padx=10)

        # Canvas for Drawing Crop
        self.canvas = tk.Canvas(self.root, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        
        self.raw_image = Image.open(path).convert("RGB")
        # Resize for display only
        width, height = self.raw_image.size
        ratio = min(800/width, 600/height)
        new_size = (int(width*ratio), int(height*ratio))
        
        self.display_image = self.raw_image.resize(new_size, Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red', width=2)

    def on_move_press(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        # Calculate real coordinates relative to original raw image
        ratio_w = self.raw_image.size[0] / self.display_image.size[0]
        ratio_h = self.raw_image.size[1] / self.display_image.size[1]
        
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        self.crop_coords = (x1*ratio_w, y1*ratio_h, x2*ratio_w, y2*ratio_h)

    def infer(self):
        if self.raw_image is None or self.crop_coords is None:
            messagebox.showwarning("Warning", "Please load an image and select a crop area first!")
            return

        # 1. Prepare Crop
        cropped = self.raw_image.crop(self.crop_coords)
        
        # 2. Transform (Square up and Normalize)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(cropped).unsqueeze(0).to(self.device)

        # 3. Get Brand ID
        selected_brand = self.brand_var.get()
        brand_id = self.encoders['brand'].transform([selected_brand])[0]
        brand_tensor = torch.tensor([brand_id], dtype=torch.long).to(self.device)

        # 4. Predict
        with torch.no_grad():
            output = self.model(img_tensor, brand_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(probabilities, 0)

        model_name = self.encoders['core_model'].inverse_transform([pred.item()])[0]
        
        # Display Result
        messagebox.showinfo("Result", f"Brand: {selected_brand}\nPredicted Model: {model_name}\nConfidence: {conf.item()*100:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    # Update these paths to your actual files
    app = WatchExpertUI(
        root, 
        model_path='models/expert_watch_model.pth', 
        encoder_path='models/expert_encoders.pkl'
    )
    root.mainloop()