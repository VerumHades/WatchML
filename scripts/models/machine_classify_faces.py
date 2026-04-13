import os
import shutil
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm  # Standard for progress bars

# --- CONSTANTS ---
MODEL_PATH = "watch_face_model.pth"
SOURCE_DIR = "data/images/indexed"
OUTPUT_ROOT = "data/images/machine_classified_faces"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_inference_transforms():
    """
    Returns standard transforms for inference.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_trained_model(model_path, device):
    """
    Initializes ResNet18 and loads the saved weights.
    """
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def ensure_output_dirs(root_path):
    """
    Creates the necessary output subdirectories.
    """
    for folder in ["face", "not_face"]:
        os.makedirs(os.path.join(root_path, folder), exist_ok=True)


def classify_and_move(model, source_folder, output_root, device):
    """
    Scans source folder with progress tracking and collects statistics.
    """
    transform = get_inference_transforms()
    class_map = {0: "face", 1: "not_face"}
    
    # Initialize statistics
    stats = {"face": 0, "not_face": 0, "errors": 0}
    
    # Collect all valid image paths first to initialize the progress bar
    all_files = []
    for root, _, files in os.walk(source_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(root, filename))

    total_images = len(all_files)
    print(f"Found {total_images} images. Starting classification on {device}...")

    # Wrap the loop in tqdm for a progress bar
    for file_path in tqdm(all_files, desc="Classifying", unit="img"):
        try:
            image = Image.open(file_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_idx = torch.max(output, 1)
                label = class_map[predicted_idx.item()]

            # Move/Copy logic
            dest_folder = os.path.join(output_root, label)
            shutil.copy2(file_path, os.path.join(dest_folder, os.path.basename(file_path)))
            
            stats[label] += 1
            
        except Exception as error:
            stats["errors"] += 1
            # We don't print errors inside the loop to avoid breaking the progress bar
    
    return stats


def print_statistics(stats):
    """
    Prints a clean summary of the classification results.
    """
    total = stats["face"] + stats["not_face"] + stats["errors"]
    
    print("\n" + "="*30)
    print("   CLASSIFICATION SUMMARY")
    print("="*30)
    print(f"Total Processed: {total}")
    print(f"Watch Faces:     {stats['face']}")
    print(f"Not Watch Faces: {stats['not_face']}")
    print(f"Errors:          {stats['errors']}")
    
    if total > 0:
        face_percent = (stats["face"] / (stats["face"] + stats["not_face"])) * 100
        print(f"Face Ratio:      {face_percent:.1f}%")
    print("="*30)


def main():
    """
    Main execution flow.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find {MODEL_PATH}.")
        return

    ensure_output_dirs(OUTPUT_ROOT)
    model = load_trained_model(MODEL_PATH, DEVICE)
    
    results = classify_and_move(model, SOURCE_DIR, OUTPUT_ROOT, DEVICE)
    print_statistics(results)


if __name__ == "__main__":
    main()