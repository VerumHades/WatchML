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

class DatasetAnalyzer:
    """
    Provides a comprehensive statistical breakdown of the watch dataset.
    """
    def __init__(self, dataframe, tasks):
        self.dataframe = dataframe
        self.tasks = tasks

    def print_full_rundown(self):
        """
        Executes the full suite of data diagnostics and prints to console.
        """
        self._print_header("DATASET DIAGNOSTIC RUNDOWN")
        self._print_general_stats()
        self._print_price_distribution()
        self._print_categorical_cardinality()
        self._print_footer()

    def _print_header(self, title):
        print(f"\n{'='*50}\n {title}\n{'='*50}")

    def _print_footer(self):
        print(f"{'='*50}\n")

    def _print_general_stats(self):
        total_samples = len(self.dataframe)
        print(f"Total Samples    : {total_samples}")
        print(f"Memory Usage     : {self.dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"{'-'*50}")

    def _print_price_distribution(self):
        """
        Analyzes the distribution of the target variable (price).
        """
        prices = self.dataframe['price']
        percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        stats = prices.describe(percentiles=percentiles)

        print("PRICE DISTRIBUTION")
        print(f" Mean   : {stats['mean']:,.2f}")
        print(f" Std Dev: {stats['std']:,.2f}")
        print(f" Min    : {stats['min']:,.2f}")
        
        for p in percentiles:
            label = f"{int(p*100)}th"
            print(f" {label:<7}: {stats[f'{int(p*100)}%']:,.2f}")
            
        print(f" Max    : {stats['max']:,.2f}")
        print(f"{'-'*50}")

    def _print_categorical_cardinality(self):
        """
        Reports on the unique classes and density of the classification tasks.
        """
        print("CATEGORICAL OVERVIEW (Tasks)")
        print(f"{'Task Name':<20} | {'Unique':<8} | {'Top Category'}")
        print(f"{'-'*48}")
        
        for task in self.tasks:
            unique_count = self.dataframe[task].nunique()
            top_val = self.dataframe[task].mode()[0]
            print(f"{task.replace('_', ' ').title():<20} | {unique_count:<8} | {top_val}")

# --- 1. DATASET ENGINE ---

class WatchDataset(Dataset):
    """
    Bridge between CSV labels, physical images, and price targets.
    """
    def __init__(self, dataframe, transform=None, tasks=None):
        self.dataframe = dataframe
        self.transform = transform
        self.tasks = tasks or ['brand', 'core_model', 'model_variant']
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = self._load_and_transform(row['full_path'])
        
        labels = {
            task: torch.tensor(int(row[f'{task}_label']), dtype=torch.long)
            for task in self.tasks
        }
        
        # Log-transform price for numerical stability
        price = torch.tensor([np.log1p(row['price'])], dtype=torch.float32)
        
        return image, labels, price

    def _load_and_transform(self, path):
        image = Image.open(path).convert("RGB")
        return self.transform(image) if self.transform else image

# --- 2. HIERARCHICAL ARCHITECTURE ---

class WatchHierarchicalNet(nn.Module):
    """
    Infers price ONLY from the predicted attribute logits.
    """
    def __init__(self, task_dimensions):
        super().__init__()
        self.backbone = self._setup_backbone()
        visual_features = 2048

        # Classification Heads
        self.heads = nn.ModuleDict({
            task: nn.Linear(visual_features, num_classes)
            for task, num_classes in task_dimensions.items()
        })

        # Price Head: Input is the sum of all attribute logits
        logit_input_dim = sum(task_dimensions.values())
        self.price_head = nn.Sequential(
            nn.Linear(logit_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def _setup_backbone(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        return resnet

    def forward(self, x):
        visual_emb = self.backbone(x)
        
        # Generate attribute logits
        attribute_logits = {task: head(visual_emb) for task, head in self.heads.items()}
        
        # Concatenate logits for price inference
        price_input = torch.cat([logits for logits in attribute_logits.values()], dim=1)
        price_prediction = self.price_head(price_input)
        
        return attribute_logits, price_prediction

# --- 3. UTILITIES ---

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def prepare_data(csv_path, image_dir, tasks):
    df = pd.read_csv(csv_path)
    
    files = [f for f in os.listdir(image_dir) if '_image_' in f]
    file_map = {f.split('_image_')[0]: f for f in files}
    df['full_path'] = pd.Series(df.index.astype(str).map(file_map)).apply(
        lambda f: os.path.join(image_dir, f) if pd.notnull(f) else None
    ).values
    
    df = df.dropna(subset=['full_path', 'price']).copy()

    encoders = {}
    for col in tasks:
        le = LabelEncoder()
        df[f'{col}_label'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    return df, encoders

# --- 4. TRAINING & LOGGING ---

def train_one_epoch(model, loader, optimizer, criteria, device, epoch):
    model.train()
    running_loss, running_mse, total_samples = 0.0, 0.0, 0
    correct = {task: 0 for task in model.heads.keys()}

    for images, labels, prices in loader:
        images, prices = images.to(device), prices.to(device)
        targets = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad(set_to_none=True)
        attr_out, price_out = model(images)
        
        clf_loss = sum(criteria['clf'](attr_out[t], targets[t]) for t in attr_out)
        reg_loss = criteria['reg'](price_out, prices)
        
        (clf_loss + reg_loss).backward()
        optimizer.step()

        # Track total loss and specific price MSE
        running_loss += (clf_loss + reg_loss).item()
        running_mse += reg_loss.item()
        
        total_samples += images.size(0)
        for task in correct:
            correct[task] += (attr_out[task].argmax(1) == targets[task]).sum().item()

    return finalize_report(running_loss, loader, correct, total_samples, epoch, running_mse)

def finalize_report(loss, loader, correct, total, epoch, running_mse):
    """
    Renders the training dashboard with classification accuracy and price error.
    """
    avg_loss = loss / len(loader)
    # Calculate Root Mean Squared Error from the log-space MSE
    avg_rmse = np.sqrt(running_mse / len(loader))
    # Approximate percentage error: (exp(rmse) - 1) * 100
    avg_price_error_pct = (np.exp(avg_rmse) - 1) * 100
    
    acc = {task: (count / total) * 100 for task, count in correct.items()}
    
    print(f"\n{'='*40}\n EPOCH {epoch+1} DASHBOARD\n{'-'*40}")
    print(f" Total Loss      : {avg_loss:.4f}")
    print(f" Price RMSE (Log): {avg_rmse:.4f}")
    print(f" Est. Price Error: ±{avg_price_error_pct:.2f}%")
    print(f"{'-'*40}")
    
    for task, val in acc.items():
        display_name = task.replace('_', ' ').title()
        print(f" {display_name:<15}: {val:>6.2f}%")
    print(f"{'='*40}")
    
    return avg_loss, acc

# --- 5. MAIN ---

def run_pytorch_training(csv_path, image_dir):
    tasks = ['brand', 'core_model', 'model_variant', 'bezel_material', 'case_material', 'dial']
    df, encoders = prepare_data(csv_path, image_dir, tasks)
    train_df, _ = train_test_split(df, test_size=0.15, random_state=42)
    
    # Analyze data before splitting
    analyzer = DatasetAnalyzer(df, tasks)
    analyzer.print_full_rundown()

    loader = DataLoader(
        WatchDataset(train_df, transform=get_transforms(), tasks=tasks), 
        batch_size=32, shuffle=True, num_workers=4
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_dims = {t: len(encoders[t].classes_) for t in tasks}
    model = WatchHierarchicalNet(task_dims).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criteria = {'clf': nn.CrossEntropyLoss(), 'reg': nn.MSELoss()}

    print(f"Starting Hierarchical Training on {len(train_df)} images...")
    for epoch in range(30):
        train_one_epoch(model, loader, optimizer, criteria, device, epoch)

if __name__ == '__main__':
    run_pytorch_training('data/csv/face_inference_clean.csv', 'data/images/normalized_dials_full')