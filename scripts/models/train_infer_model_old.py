import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch.nn.functional as F

class VisualizationService:
    """
    Provides tools to inspect the model's internal attention.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def get_heatmap(self, image_tensor, target_head='brand'):
        """
        Generates a Grad-CAM heatmap for a specific output head.
        """
        self.model.eval()
        # Hook into the last convolutional layer of ResNet
        target_layer = self.model.backbone.layer4[-1]
        handler = target_layer.register_backward_hook(lambda m, i, o: self.save_gradient(o[0]))
        
        # Forward pass
        output = self.model(image_tensor.unsqueeze(0))
        target_index = output[target_head].argmax()
        
        # Backward pass
        self.model.zero_grad()
        output[target_head][0, target_index].backward()
        
        # Process heatmap
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.mean(weights * target_layer.weight, dim=1).squeeze().detach()
        heatmap = np.maximum(heatmap.numpy(), 0)
        heatmap /= np.max(heatmap)
        
        handler.remove()
        return heatmap
        
# --- 1. DATASET ENGINE ---

class WatchDataset(Dataset):
    """
    Bridge between the CSV labels and the physical image files.
    """
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row['full_path']).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        labels = {
            'brand': torch.tensor(row['brand_label'], dtype=torch.long),
            'core_model': torch.tensor(row['core_model_label'], dtype=torch.long),
            'model_variant': torch.tensor(row['model_variant_label'], dtype=torch.long)
        }
        return image, labels

# --- 2. ARCHITECTURE ---

class MultiOutputWatchNet(nn.Module):
    """
    A Multi-Head CNN using a ResNet-50 backbone.
    """
    def __init__(self, num_brands, num_models, num_variants):
        super(MultiOutputWatchNet, self).__init__()
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        input_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer with a dummy
        self.backbone.fc = nn.Identity()

        # Define 3 distinct heads
        self.brand_head = nn.Linear(input_features, num_brands)
        self.model_head = nn.Linear(input_features, num_models)
        self.variant_head = nn.Linear(input_features, num_variants)

    def forward(self, x):
        features = self.backbone(x)
        return {
            'brand': self.brand_head(features),
            'core_model': self.model_head(features),
            'model_variant': self.variant_head(features)
        }

# --- 3. UTILITY FUNCTIONS ---

def get_transforms():
    """
    Standardizes 681x681 images to 224x224 with ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

def prepare_data(csv_path, image_dir):
    """
    Refactored data preparation using index-to-file mapping.
    """
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"The CSV file at {csv_path} is empty.")

    # 1. Map Images using the Row Index
    all_files = [f for f in os.listdir(image_dir) if '_image_' in f]
    file_map = {f.split('_image_')[0]: f for f in all_files}
    
    # We map the index to the filename, then convert to Series to use .apply()
    df['full_path'] = pd.Series(df.index.astype(str).map(file_map)).apply(
        lambda filename: os.path.join(image_dir, filename) if pd.notnull(filename) else None
    ).values # Use .values to ensure the Series aligns back to the DataFrame rows
    
    # 2. Filter and Validate
    df = df.dropna(subset=['full_path']).copy()

    if df.empty:
        raise ValueError("Zero images matched using the row index as ID.")

    # 3. Encode Labels
    encoders = {}
    for col in ['brand', 'core_model', 'model_variant']:
        le = LabelEncoder()
        df[f'{col}_label'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    return df, encoders

# --- 4. TRAINING LOOP ---
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

def save_model(model, epoch):
    """
    Saves the model state dictionary using a timestamped directory for version control.
    """
    # Create a unique string for the current run (e.g., 2026-04-12_17-15)
    run_directory = f"models/face_inference/run_{timestamp}"
    
    os.makedirs(run_directory, exist_ok=True)
    
    file_path = os.path.join(run_directory, f"watch_model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), file_path)
    
    print(f"Checkpoint saved to: {file_path}")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Executes a training pass and tracks performance for all three watch attributes.
    """
    model.train()
    
    # Initialize trackers
    total_samples = 0
    running_loss = 0.0
    correct = {'brand': 0, 'core_model': 0, 'model_variant': 0}

    for images, labels in loader:
        images = images.to(device)
        optimizer.zero_grad()
        
        # 1. Forward Pass
        outputs = model(images)
        
        # 2. Map labels to device (Matching your specific keys)
        targets = {k: v.to(device) for k, v in labels.items()}
        
        # 3. Multi-Head Loss Calculation
        loss_brand = criterion(outputs['brand'], targets['brand'])
        loss_model = criterion(outputs['core_model'], targets['core_model'])
        loss_variant = criterion(outputs['model_variant'], targets['model_variant'])
        
        # 4. Backprop
        total_loss = (1.0 * loss_brand) + (1.0 * loss_model) + (1.0 * loss_variant)
        total_loss.backward()
        optimizer.step()

        # --- Statistics Collection ---
        running_loss += total_loss.item()
        total_samples += images.size(0)

        for head in ['brand', 'core_model', 'model_variant']:
            _, predicted = torch.max(outputs[head], 1)
            correct[head] += (predicted == targets[head]).sum().item()

    # Calculate Epoch Averages
    avg_loss = running_loss / len(loader)
    acc = {h: (correct[h] / total_samples) * 100 for h in correct}

    # Print Dashboard
    print(f"\n" + "="*30)
    print(f" EPOCH STATISTICS")
    print(f"-"*30)
    print(f" Avg Total Loss: {avg_loss:.4f}")
    print(f" Brand Accuracy: {acc['brand']:>6.2f}%")
    print(f" Model Accuracy: {acc['core_model']:>6.2f}%")
    print(f" Variant Acc:    {acc['model_variant']:>6.2f}%")
    print("="*30 + "\n")

    save_model(model, epoch)

    return avg_loss, acc

# --- 5. ORCHESTRATOR ---

def run_pytorch_training(csv_path, image_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Prep
    df, encoders = prepare_data(csv_path, image_dir)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Loaders
    train_ds = WatchDataset(train_df, transform=get_transforms())
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    # core_model Setup
    core_model = MultiOutputWatchNet(
        len(encoders['brand'].classes_),
        len(encoders['core_model'].classes_),
        len(encoders['model_variant'].classes_)
    ).to(device)

    #for param in core_model.backbone.parameters():
    #    param.requires_grad = False
    
    optimizer = optim.Adam(core_model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print(f"Training on {device}...")
    for epoch in range(20):
        avg_loss, accuracy = train_one_epoch(core_model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step(avg_loss)


run_pytorch_training('data/csv/face_inference_clean.csv', 'data/images/normalized_dials')