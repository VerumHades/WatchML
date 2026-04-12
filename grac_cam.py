import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. WRAPPER FOR MULTI-INPUT GRAD-CAM ---

class ModelWrapper(torch.nn.Module):
    """
    Grad-CAM expects a model that takes a single tensor.
    This wrapper 'freezes' the brand_id so Grad-CAM can process the image.
    """
    def __init__(self, model, brand_id):
        super().__init__()
        self.model = model
        self.brand_id = brand_id

    def forward(self, x):
        return self.model(x, self.brand_id)

# --- 2. THE VISUALIZATION ENGINE ---

def explain_watch_prediction(model_path, image_path, brand_name, encoders):
    """
    Loads the model, runs inference, and generates the Grad-CAM heatmap.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Data
    brand_idx = encoders['brand'].transform([brand_name])[0]
    brand_tensor = torch.tensor([brand_idx]).to(device)
    
    # Load Image
    raw_image = Image.open(image_path).convert("RGB")
    rgb_img = np.float32(raw_image.resize((448, 448))) / 255
    
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # Wrap model for Grad-CAM compatibility
    wrapped_model = ModelWrapper(model, brand_tensor).to(device).eval()
    target_layers = [wrapped_model.model.backbone.layer4[-1]]

    # Generate CAM
    with GradCAM(model=wrapped_model, target_layers=target_layers) as cam:
        # Get prediction to see what the model thinks
        output = wrapped_model(input_tensor)
        prediction_idx = torch.argmax(output, dim=1).item()
        predicted_model = encoders['core_model'].inverse_transform([prediction_idx])[0]
        
        print(f"Input Brand: {brand_name}")
        print(f"Predicted Model: {predicted_model}")

        # Target the top prediction for the heatmap
        targets = [ClassifierOutputTarget(prediction_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Overlay heatmap on image
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Save or Show
        cv2.imwrite("grad_cam_result.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        print("Heatmap saved as grad_cam_result.jpg")

# --- 3. EXECUTION ---

if __name__ == "__main__":
    import pickle
    from scripts.models.train_infer_model import BrandConditionedNet # Ensure you import your class

    # Load Encoders
    with open('models/expert_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    # Initialize and Load Model
    model = BrandConditionedNet(
        num_brands=len(encoders['brand'].classes_),
        num_models=len(encoders['core_model'].classes_)
    )
    model.load_state_dict(torch.load("models/expert_watch_model.pth"))
    model.eval()

    # TEST ON A PROBLEM IMAGE
    explain_watch_prediction(
        model_path="models/expert_watch_model.pth",
        image_path="C:/Users/filah/Downloads/b.jpg", 
        brand_name="rolex", 
        encoders=encoders
    )