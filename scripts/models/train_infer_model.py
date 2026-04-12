import os
import torch
import pandas as pd
import numpy as np
import pickle
import cv2
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. COMPUTER VISION UTILITIES ---

def calculate_texture_density(image_tensor):
    """
    Extracts Canny edges and calculates the ratio of 'detail' pixels.
    Acts as a proxy for watch condition/wear.
    """
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    
    white_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    return torch.tensor([white_pixels / total_pixels], dtype=torch.float32)

def apply_circular_mask(image_tensor, radius=1.05):
    _, h, w = image_tensor.shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    mask = (torch.sqrt(x**2 + y**2) <= radius).float() 
    return image_tensor * mask

# --- 2. DATASET ENGINE ---

class WatchValueDataset(Dataset):
    def __init__(self, dataframe, encoders, transform=None):
        self.df = dataframe
        self.transform = transform
        self.encoders = encoders
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        # Image Processing
        image = Image.open(row['full_path']).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        image = apply_circular_mask(image)
        
        # Calculate Scuff Score (Condition Proxy)
        scuff_score = calculate_texture_density(image)
        
        # Categorical Metadata
        meta_labels = torch.tensor([
            row['brand_label'],
            row['core_model_label'],
            row['bezel_material_label'],
            row['case_material_label'],
            row['dial_label']
        ], dtype=torch.long)

        # Target: Log Price
        target_price = torch.tensor([np.log1p(row['price'])], dtype=torch.float32)
            
        return {
            'image': image,
            'scuff_score': scuff_score,
            'meta': meta_labels
        }, target_price

# --- 3. ARCHITECTURE (The "Luxury-Aware" Net) ---

class WatchValuationNet(nn.Module):
    def __init__(self, n_categories):
        super().__init__()
        
        # Vision backbone (ResNet18)
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Identity()
        
        # High-Capacity Embeddings (Crucial for luxury brands)
        # We use 64 for Brand/Model to ensure Patek != Tudor
        self.brand_emb = nn.Embedding(n_categories[0], 64)
        self.model_emb = nn.Embedding(n_categories[1], 64)
        self.other_embs = nn.ModuleList([nn.Embedding(n, 16) for n in n_categories[2:]])
        
        # Fusion Head (Visual + Scuff Score + Metadata)
        total_meta_dim = 64 + 64 + (len(n_categories[2:]) * 16) + 1 # +1 for scuff_score
        
        self.regressor = nn.Sequential(
            nn.Linear(512 + total_meta_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, scuff_score, meta):
        x_vis = self.backbone(image)
        
        # Metadata Stream
        e_brand = self.brand_emb(meta[:, 0])
        e_model = self.model_emb(meta[:, 1])
        e_others = torch.cat([emb(meta[:, i+2]) for i, emb in enumerate(self.other_embs)], dim=1)
        
        # Concatenate everything
        x_combined = torch.cat((x_vis, e_brand, e_model, e_others, scuff_score), dim=1)
        return self.regressor(x_combined)

# --- 4. TRAINING ORCHESTRATOR ---

def prepare_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    all_files = [f for f in os.listdir(image_dir) if '_image_' in f]
    file_map = {f.split('_image_')[0]: f for f in all_files}
    
    df['full_path'] = df.iloc[:, 1].astype(str).map(file_map).apply(
        lambda x: os.path.join(image_dir, x) if pd.notnull(x) else None
    )
    df = df.dropna(subset=['full_path', 'price']).copy()

    encoders = {}
    cat_cols = ['brand', 'core_model', 'bezel_material', 'case_material', 'dial']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna('Unknown').astype(str)
        df[f'{col}_label'] = le.fit_transform(df[col])
        encoders[col] = le
        
    return df, encoders

def run_valuation_training(csv_path, image_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, encoders = prepare_data(csv_path, image_dir)
    train_df, _ = train_test_split(df, test_size=0.1, random_state=42)

    n_categories = [len(encoders[c].classes_) for c in ['brand', 'core_model', 'bezel_material', 'case_material', 'dial']]
    
    loader = DataLoader(
        WatchValueDataset(train_df, encoders, get_transforms()), 
        batch_size=16, shuffle=True
    )

    model = WatchValuationNet(n_categories).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.HuberLoss() # Better than MSE for handling those $500k outliers

    for epoch in range(25):
        model.train()
        epoch_loss = 0
        for batch, target in loader:
            img, scuff, meta = batch['image'].to(device), batch['scuff_score'].to(device), batch['meta'].to(device)
            
            optimizer.zero_grad()
            output = model(img, scuff, meta)
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/len(loader):.5f}")

    # Save everything
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/haggle_model.pth")
    with open('models/expert_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    run_valuation_training('data/csv/face_inference_clean.csv', 'data/images/classified/face_images')