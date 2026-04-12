import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_DIR = "models/face_zoom_regressor"
RESULTS_CSV = "evaluation_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from zoom_model_train import MultiTaskWatchDataset, TRAIN_DATA_PATH, DB_PATH, get_train_transform, WatchMultiTaskModel, MIN_RADIUS_FOR_FINDING


def run_model_evaluation():
    """
    Evaluates all checkpoints in the directory, printing individual 
    breakdowns followed by a final summary table.
    """
    # 1. Initialize Dataset & Models
    dataset = MultiTaskWatchDataset(TRAIN_DATA_PATH, DB_PATH, transform=get_train_transform())
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")])
    summary_list = []

    print(f"--- Starting Evaluation of {len(model_files)} Checkpoints ---\n")

    for model_name in model_files:
        metrics = _evaluate_single_checkpoint(model_name, loader, len(dataset))
        
        # Print Individual Breakdown
        print(f"RESULTS FOR: {model_name}")
        print(metrics)
        print("-" * 40)
        
        summary_list.append(metrics)

    # 2. Final Overall Summary
    _print_overall_summary(summary_list)

def _evaluate_single_checkpoint(model_name, loader, total_samples):
    """
    Loads one specific model and runs it through the full dataset 
    to gather precise error metrics.
    """
    model = WatchMultiTaskModel().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_name), map_location=DEVICE))
    model.eval()

    correct_cls = 0
    dist_errors = []
    rad_errors = []

    with torch.no_grad():
        for imgs, labels, geos in loader:
            imgs, labels, geos = imgs.to(DEVICE), labels.to(DEVICE), geos.to(DEVICE)
            out_cls, out_geo = model(imgs)
            
            # Classification Logic
            _, pred_cls = torch.max(out_cls, 1)
            if pred_cls == labels:
                correct_cls += 1
            
            # Regression Logic (Only for faces meeting size threshold)
            if labels == 1 and geos[0, 2] > MIN_RADIUS_FOR_FINDING:
                p_x, p_y, p_r = out_geo[0]
                t_x, t_y, t_r = geos[0]
                
                # Center point Euclidean distance
                distance = torch.sqrt((p_x - t_x)**2 + (p_y - t_y)**2).item()
                dist_errors.append(distance)
                rad_errors.append(torch.abs(p_r - t_r).item())

    # Return structured dictionary for pprint
    return {
        "model_file": model_name,
        "accuracy_pct": round((correct_cls / total_samples) * 100, 2),
        "avg_center_error": round(sum(dist_errors)/len(dist_errors), 5) if dist_errors else 0.0,
        "avg_radius_error": round(sum(rad_errors)/len(rad_errors), 5) if rad_errors else 0.0,
        "valid_faces_evaluated": len(dist_errors)
    }

def _print_overall_summary(summary_list):
    """Prints a sorted table of all model performances."""
    df = pd.DataFrame(summary_list)
    # Sort by accuracy first, then lowest center error
    df = df.sort_values(by=["accuracy_pct", "avg_center_error"], ascending=[False, True])
    
    print("\n" + "="*60)
    print("FINAL OVERALL SUMMARY (Sorted by Best Performance)")
    print("="*60)
    print(df.to_string(index=False))
    
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nReport exported to: {RESULTS_CSV}")

if __name__ == "__main__":
    run_model_evaluation()