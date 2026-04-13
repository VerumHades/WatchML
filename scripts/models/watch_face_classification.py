import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# --- CONSTANTS ---
DATA_ROOT = "data/images/classified"
MODEL_SAVE_PATH = "watch_face_model.pth"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
PATIENCE = 2  # Number of epochs to wait for improvement before stopping


def get_device():
    """
    Returns the CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    
    print("CUDA not found. Falling back to CPU.")
    return torch.device("cpu")


def get_data_transforms():
    """
    Standard transforms with slight rotation to prevent overfitting.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),  # Helps model generalize
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def prepare_data_loader(data_path):
    """
    Optimized loader using multiple CPU cores to feed the GPU.
    """
    dataset = datasets.ImageFolder(data_path, transform=get_data_transforms())
    return DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )


def build_binary_model(device):
    """
    Modifies ResNet18, unfreezes the final layer block for fine-tuning, 
    and enables CUDNN benchmark.
    """
    torch.backends.cudnn.benchmark = True 
    model = models.resnet18(weights='DEFAULT')
    
    # Freeze all layers initially
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    # Unfreeze the last residual block (layer4) to allow specialized learning
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
        
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)

def run_training_step(model, batch, optimizer, criterion, device):
    """
    Executes a single forward and backward pass for one batch.
    """
    images, labels = batch[0].to(device), batch[1].to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_model(model, train_loader, device):
    """
    Trains with Early Stopping to prevent performance decay.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            epoch_loss += run_training_step(model, batch, optimizer, criterion, device)
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

        # --- EARLY STOPPING LOGIC ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            epochs_without_improvement += 1
            print(f"No improvement. Patience: {epochs_without_improvement}/{PATIENCE}")
            
        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered. Reverting to best model weights.")
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
            break
    
    return model


def predict_single_image(model, image_path, class_names, device):
    """
    Predicts the class of an individual image file using the GPU.
    """
    model.eval()
    transform = get_data_transforms()
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_index = torch.max(output, 1)
        
    return class_names[predicted_index.item()]


def scan_for_missed_faces(model, folder_to_scan, class_names, device):
    """
    Iterates through a folder to find images identified as 'class_yes'.
    """
    print(f"\nScanning {folder_to_scan} for missed faces...")
    for filename in os.listdir(folder_to_scan):
        path = os.path.join(folder_to_scan, filename)
        
        if os.path.isfile(path):
            prediction = predict_single_image(model, path, class_names, device)
            if prediction == 'class_yes':
                print(f"Potential Face Detected: {filename}")


def main():
    """
    Main execution flow for training and scanning.
    """
    device = get_device()
    
    loader = prepare_data_loader(DATA_ROOT)
    class_names = loader.dataset.classes
    
    classifier = build_binary_model(device)
    classifier = train_model(classifier, loader, device)
    
    print(f"Final training complete. Best model is saved at {MODEL_SAVE_PATH}")
    scan_for_missed_faces(classifier, os.path.join(DATA_ROOT, "class_no"), class_names, device)


if __name__ == "__main__":
    main()