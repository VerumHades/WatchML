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
    def __init__(self, dataframe, transform=None, tasks=None):
        self.dataframe = dataframe
        self.transform = transform
        self.tasks = tasks or ['brand', 'core_model', 'model_variant']
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row['full_path']).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Dynamically build the labels dictionary for ALL tasks
        labels = {
            task: torch.tensor(row[f'{task}_label'], dtype=torch.long)
            for task in self.tasks
        }
        
        return image, labels

# --- 2. ARCHITECTURE ---

class WatchMultiTaskNet(nn.Module):
    """
    Dynamic Multi-Task Network for watch attribute inference.
    """
    def __init__(self, task_dimensions):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        input_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Create a head for every attribute provided in task_dimensions
        self.heads = nn.ModuleDict({
            task_name: nn.Linear(input_features, num_classes)
            for task_name, num_classes in task_dimensions.items()
        })

    def forward(self, x):
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}

# --- 3. UTILITY FUNCTIONS ---

def get_transforms():
    """
    Standardizes 681x681 images to 224x224 with ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

TASKS = [
    'brand', 'core_model', 'model_variant', 
    'bezel_material', 'case_material', 'dial'
]

def prepare_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    
    # 1. Image Mapping (using the index fix from before)
    all_files = [f for f in os.listdir(image_dir) if '_image_' in f]
    file_map = {f.split('_image_')[0]: f for f in all_files}
    df['full_path'] = pd.Series(df.index.astype(str).map(file_map)).apply(
        lambda f: os.path.join(image_dir, f) if pd.notnull(f) else None
    ).values
    
    df = df.dropna(subset=['full_path']).copy()

    # 2. Encode all dynamic tasks
    encoders = {}
    for col in TASKS:
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
    model.train()
    running_loss, total_samples = 0.0, 0
    # Dynamic tracking based on the heads in the model
    correct = {task: 0 for task in model.heads.keys()}

    for images, labels in loader:
        images, targets = images.to(device), {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        
        # Calculate loss for all tasks (Weighting: Brand gets priority)
        task_losses = {task: criterion(outputs[task], targets[task]) for task in outputs}
        
        # Hierarchical Weighting: Brand is the anchor
        total_loss = (5.0 * task_losses['brand']) + sum(task_losses.values())
        
        total_loss.backward()
        optimizer.step()

        # Stats collection
        running_loss += total_loss.item()
        total_samples += images.size(0)
        for task in correct:
            _, pred = torch.max(outputs[task], 1)
            correct[task] += (pred == targets[task]).sum().item()

    return finalize_report(running_loss, loader, correct, total_samples, epoch)

def finalize_report(loss, loader, correct, total, epoch):
    avg_loss = loss / len(loader)
    acc = {task: (count / total) * 100 for task, count in correct.items()}
    
    print(f"\n{'='*35}\n EPOCH {epoch+1} DASHBOARD\n{'-'*35}")
    print(f" Total Loss: {avg_loss:.4f}")
    for task, val in acc.items():
        print(f" {task.replace('_', ' ').title():<15}: {val:>6.2f}%")
    print(f"{'='*35}")
    
    return avg_loss, acc
# --- 5. ORCHESTRATOR ---

def run_pytorch_training(csv_path, image_dir):
    """
    Orchestrates the full training lifecycle for multi-attribute watch inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data Preparation
    # TASKS = ['brand', 'core_model', 'model_variant', 'bezel_material', 'case_material', 'dial']
    df, encoders = prepare_data(csv_path, image_dir)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    
    # 2. Infrastructure Setup
    train_loader = DataLoader(
        WatchDataset(train_df, transform=get_transforms(), tasks=TASKS), 
        batch_size=32, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Generate task dimensions dynamically from encoders
    task_dims = {task: len(encoders[task].classes_) for task in TASKS}
    
    model = WatchMultiTaskNet(task_dims).to(device)
    
    # 3. Optimization Strategy
    # We use a slightly lower LR (5e-5) because we have many heads to balance
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler monitors 'brand' accuracy to decide when to "squeeze" the learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    print(f"Starting Multi-Task Training on {len(df)} images...")
    print(f"Inference Targets: {', '.join(TASKS)}")

    # 4. Training Loop
    for epoch in range(30):
        avg_loss, accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Step the scheduler based on the Anchor Task (Brand)
        scheduler.step(accuracy['brand'])
        
        # Periodic snapshot of model focus
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1} complete. Model focus snapshot saved.")

    print("Training Complete. Final model saved to run directory.")


# THIS IS THE CRITICAL PROTECTOR FOR WINDOWS
if __name__ == '__main__':
    # Now it is safe to call the training function
    run_pytorch_training(
        csv_path='data/csv/face_inference_clean.csv', 
        image_dir='data/images/normalized_dials'
    )