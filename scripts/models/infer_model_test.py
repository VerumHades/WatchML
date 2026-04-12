import os
import torch
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import DataLoader

# --- 1. UTILITIES ---

def get_safe_labels(df, encoders):
    """
    Encodes categorical columns. Maps unseen labels to 'Unknown' 
    to prevent crash during audit.
    """
    cat_cols = ['brand', 'core_model', 'bezel_material', 'case_material', 'dial']
    for col in cat_cols:
        le = encoders[col]
        class_to_idx = {cls: i for i, cls in enumerate(le.classes_)}
        unknown_idx = class_to_idx.get('Unknown', 0)
        
        df[f'{col}_label'] = df[col].fillna('Unknown').astype(str).apply(
            lambda x: class_to_idx.get(x, unknown_idx)
        )
    return df

def map_image_paths(df, image_dir):
    """
    Reconstructs the full path for images based on the ID column.
    """
    all_files = [f for f in os.listdir(image_dir) if '_image_' in f]
    file_map = {f.split('_image_')[0]: f for f in all_files}
    
    # Assumes the first column is the ID used in filenames
    id_col = df.columns[0]
    df['full_path'] = df[id_col].astype(str).map(file_map).apply(
        lambda x: os.path.join(image_dir, x) if pd.notnull(x) else None
    )
    return df.dropna(subset=['full_path', 'price']).copy()

# --- 2. AUDIT ENGINE ---

def run_valuation_audit(csv_path, image_dir, model_path, encoder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Metadata Support
    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)
    
    # Prepare Dataframe
    df = pd.read_csv(csv_path)
    df = map_image_paths(df, image_dir)
    df = get_safe_labels(df, encoders)

    # Import classes from your reworked training file
    from train_infer_model import WatchValueDataset, WatchValuationNet, get_transforms
        
    dataset = WatchValueDataset(df, encoders, get_transforms())
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize Model (New Architecture)
    n_categories = [len(encoders[c].classes_) for c in ['brand', 'core_model', 'bezel_material', 'case_material', 'dial']]
    model = WatchValuationNet(n_categories).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    print(f"Auditing Valuation on {len(df)} images...")
    
    with torch.no_grad():
        for i, (batch, target_log_price) in enumerate(loader):
            img = batch['image'].to(device)
            scuff = batch['scuff_score'].to(device) # Single float now
            meta = batch['meta'].to(device)
            
            # Predict
            log_pred = model(img, scuff, meta)
            
            # Inverse log transform: exp(x) - 1
            pred_price = np.expm1(log_pred.item())
            actual_price = np.expm1(target_log_price.item())
            
            results.append({
                'model': df.iloc[i]['core_model'],
                'actual': actual_price,
                'predicted': pred_price,
                'spread': pred_price - actual_price
            })

    print_final_report(pd.DataFrame(results))

def print_final_report(audit_df):
    """
    Calculates and displays the arbitrage metrics.
    """
    audit_df['error_pct'] = (abs(audit_df['spread']) / audit_df['actual']) * 100
    
    print("\n" + "$" * 40)
    print(" VALUATION ENGINE AUDIT REPORT")
    print("$" * 40)
    print(f" Total Audited:        {len(audit_df)} watches")
    print(f" Avg. Pricing Error:   {audit_df['error_pct'].mean():.2f}%")
    print(f" Mean Absolute Error:  ${abs(audit_df['spread']).mean():.2f}")
    print("-" * 40)
    
    # Top 3 Haggle Deals (Underpriced by market)
    top_deals = audit_df.sort_values(by='spread', ascending=False).head(3)
    print(" TOP 3 HAGGLE OPPORTUNITIES (Underpriced):")
    for _, row in top_deals.iterrows():
        print(f" - {row['model']}: Listed ${row['actual']:.0f} | Predicted ${row['predicted']:.0f} (+${row['spread']:.0f})")
        
    print("-" * 40)
    # Top 3 Overpriced
    overpriced = audit_df.sort_values(by='spread', ascending=True).head(3)
    print(" TOP 3 OVERPRICED WATCHES:")
    for _, row in overpriced.iterrows():
        print(f" - {row['model']}: Listed ${row['actual']:.0f} | Predicted ${row['predicted']:.0f} (-${abs(row['spread']):.0f})")
    print("$" * 40 + "\n")

if __name__ == "__main__":
    run_valuation_audit(
        csv_path='data/csv/face_inference_clean.csv', 
        image_dir='data/images/classified/face_images',
        model_path='models/haggle_model.pth',
        encoder_path='models/expert_encoders.pkl'
    )