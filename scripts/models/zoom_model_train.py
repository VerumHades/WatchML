import os
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0005  # Slightly lower for multi-task stability
DB_PATH = "data/db/zoom_mark_progress.db"
TRAIN_DATA_PATH = "data/images/organized_dataset"
MIN_RADIUS_FOR_FINDING = 0.15  # 15% of image width

def get_train_transform():
    """
    Standardizes image size and applies augmentations to prevent 
    overfitting on small datasets.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        # Watches can be sideways or flipped; these help the model generalize
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30), 
        # Handles different lighting conditions in watch photography
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # Standard normalization for MobileNet pre-trained weights
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class MultiTaskWatchDataset(Dataset):
    """
    Combines directory-based classification with DB-based geometry.
    Filters for 'finding' based on radius size.
    """
    def __init__(self, root_dir, db_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_map = {"no_face": 0, "face": 1, "misoriented": 2}
        
        self._load_data(db_path)

    def _load_data(self, db_path):
        """Cross-references folders with SQL coordinates."""
        with sqlite3.connect(db_path) as conn:
            for folder, class_id in self.class_map.items():
                folder_path = os.path.join(self.root_dir, folder)
                if not os.path.exists(folder_path): continue
                
                for fname in os.listdir(folder_path):
                    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                    
                    cursor = conn.execute(
                        "SELECT center_x, center_y, radius FROM watch_labels WHERE image_filename = ?", 
                        (fname,)
                    )
                    geo = cursor.fetchone()
                    
                    self.samples.append({
                        "path": os.path.join(folder_path, fname),
                        "label": class_id,
                        "geo": torch.tensor(geo if geo else [0.5, 0.5, 0.0], dtype=torch.float32)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        return image, sample["label"], sample["geo"]

class WatchMultiTaskModel(nn.Module):
    """
    MobileNetV3 with two heads: 
    1. Classifier (3 classes)
    2. Regressor (x, y, radius)
    """
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights="DEFAULT")
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        num_features = backbone.classifier[0].in_features
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.Hardswish(),
            nn.Linear(128, 3)
        )
        
        # Regression Head (Finding)
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid() # Keeps coordinates between 0 and 1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x), self.regressor(x)

def save_checkpoint(model, epoch, loss, folder="models/face_zoom_regressor"):
    """
    Saves a model checkpoint with metadata in the filename.
    Creates the directory if it doesn't exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    # Format: watch_model_ep05_loss0.1598.pth
    filename = f"watch_model_ep{epoch+1:02d}_loss{loss:.4f}.pth"
    path = os.path.join(folder, filename)
    
    torch.save(model.state_dict(), path)
    print(f" ---> Checkpoint saved: {filename}")

def train():
    """Trains the model with an interrupt-safe save mechanism."""
    dataset = MultiTaskWatchDataset(TRAIN_DATA_PATH, DB_PATH, transform=get_train_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = WatchMultiTaskModel().to(DEVICE)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_geo = nn.MSELoss(reduction='none') 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"--- Training on {DEVICE} | Checkpoints will save every epoch ---")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for imgs, labels, geos in loader:
            imgs, labels, geos = imgs.to(DEVICE), labels.to(DEVICE), geos.to(DEVICE)
            
            optimizer.zero_grad()
            out_cls, out_geo = model(imgs)
            
            # Loss Logic
            loss_cls = criterion_cls(out_cls, labels)
            geo_mask = (labels == 1) & (geos[:, 2] > MIN_RADIUS_FOR_FINDING)
            loss_geo = criterion_geo(out_geo[geo_mask], geos[geo_mask]).mean() if geo_mask.any() else 0.0
            
            total_batch_loss = loss_cls + (loss_geo * 10.0)
            total_batch_loss.backward()
            optimizer.step()
            
            epoch_loss += total_batch_loss.item()

        # Calculate final average loss for the epoch
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}", end="")
        
        # Save every single epoch automatically
        save_checkpoint(model, epoch, avg_loss)

    print("--- Training finished successfully ---")

if __name__ == "__main__":
    train()