import os
import torch
import tkinter as tkinter_root
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
from torchvision import transforms

from scripts.models.train_infer_model_old import WatchMultiTaskNet, TASKS, prepare_data


class GradCamEngine:
    """
    Proper Grad-CAM implementation for multi-head models.
    """
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

    def _forward_hook(self, module, input_value, output_value):
        self.activations = output_value

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, target_head):
        self.model.eval()

        target_layer = self.model.backbone.layer4[-1]

        forward_handle = target_layer.register_forward_hook(self._forward_hook)
        backward_handle = target_layer.register_backward_hook(self._backward_hook)

        outputs = self.model(image_tensor.unsqueeze(0))
        target_index = torch.argmax(outputs[target_head], dim=1)

        self.model.zero_grad()
        outputs[target_head][0, target_index].backward()

        pooled_gradients = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted = self.activations * pooled_gradients
        heatmap = torch.mean(weighted, dim=1).squeeze()

        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)

        forward_handle.remove()
        backward_handle.remove()

        return heatmap.detach().cpu().numpy()


class ModelInferenceService:
    """
    Handles model loading, inference, decoding, and Grad-CAM.
    """
    def __init__(self, task_dimensions, encoders, device):
        self.device = device
        self.encoders = encoders
        self.model = WatchMultiTaskNet(task_dimensions).to(device)
        self.grad_cam_engine = GradCamEngine(self.model)

    def load_weights(self, model_path):
        state_dictionary = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dictionary)
        self.model.eval()

    def predict_top_k(self, image_tensor, top_k=3):
        with torch.no_grad():
            outputs = self.model(image_tensor.unsqueeze(0).to(self.device))

        results = {}

        for task, output in outputs.items():
            probabilities = torch.softmax(output, dim=1)
            top_probabilities, top_indices = torch.topk(probabilities, top_k)

            decoded = []
            for index, probability in zip(top_indices[0], top_probabilities[0]):
                label = self.encoders[task].inverse_transform([index.item()])[0]
                decoded.append((label, probability.item()))

            results[task] = decoded

        return results

    def generate_heatmap(self, image_tensor, task_name):
        return self.grad_cam_engine.generate(
            image_tensor.to(self.device),
            task_name
        )


def overlay_heatmap_on_image(pil_image, heatmap):
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(pil_image.size)
    heatmap_array = np.array(heatmap_resized)

    if heatmap_array.ndim == 2:
        heatmap_array = np.stack([heatmap_array] * 3, axis=-1)

    original_array = np.array(pil_image)

    overlay = (0.6 * original_array + 0.4 * heatmap_array).astype(np.uint8)
    return Image.fromarray(overlay)


class ModelTesterUI:
    """
    UI for model inspection with Grad-CAM.
    """
    def __init__(self, root_window, model_service, transform_pipeline):
        self.root_window = root_window
        self.model_service = model_service
        self.transform_pipeline = transform_pipeline

        self.selected_image_tensor = None
        self.original_pil_image = None
        self.current_display_image = None

        self._build_layout()

    def _build_layout(self):
        self.root_window.title("Watch Model Inspector")

        self.model_selector = ttk.Combobox(self.root_window, width=90)
        self.model_selector.pack(pady=5)

        tkinter_root.Button(
            self.root_window,
            text="Load Model",
            command=self._handle_model_load
        ).pack()

        self.task_selector = ttk.Combobox(
            self.root_window,
            values=TASKS,
            width=30
        )
        self.task_selector.set(TASKS[0])
        self.task_selector.pack(pady=5)

        tkinter_root.Button(
            self.root_window,
            text="Select Image",
            command=self._handle_image_selection
        ).pack(pady=5)

        self.image_label = tkinter_root.Label(self.root_window)
        self.image_label.pack()

        tkinter_root.Button(
            self.root_window,
            text="Run Inference",
            command=self._handle_prediction
        ).pack(pady=5)

        tkinter_root.Button(
            self.root_window,
            text="Show Heatmap",
            command=self._handle_heatmap
        ).pack(pady=5)

        tkinter_root.Button(
            self.root_window,
            text="Reset Image",
            command=self._reset_image
        ).pack(pady=5)

        self.output_label = tkinter_root.Label(self.root_window, justify="left")
        self.output_label.pack()

        self._populate_model_dropdown()

    def _populate_model_dropdown(self):
        model_paths = []
        for root_directory, _, files in os.walk("models/face_inference"):
            for file_name in files:
                if file_name.endswith(".pth"):
                    model_paths.append(os.path.join(root_directory, file_name))

        self.model_selector["values"] = model_paths

    def _handle_model_load(self):
        path = self.model_selector.get()
        if path:
            self.model_service.load_weights(path)
            self.output_label.config(text=f"Loaded:\n{path}")

    def _handle_image_selection(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        pil_image = Image.open(file_path).convert("RGB")
        self.original_pil_image = pil_image

        self._display_image(pil_image)
        self.selected_image_tensor = self.transform_pipeline(pil_image)

    def _display_image(self, pil_image):
        resized = pil_image.resize((224, 224))
        tk_image = ImageTk.PhotoImage(resized)

        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image
        self.current_display_image = pil_image

    def _handle_prediction(self):
        if self.selected_image_tensor is None:
            return

        predictions = self.model_service.predict_top_k(self.selected_image_tensor)

        formatted_lines = []
        for task, values in predictions.items():
            formatted_lines.append(f"{task}:")
            for label, confidence in values:
                formatted_lines.append(f"  - {label} ({confidence:.2f})")

        self.output_label.config(text="\n".join(formatted_lines))

    def _handle_heatmap(self):
        if self.selected_image_tensor is None:
            return

        selected_task = self.task_selector.get()

        heatmap = self.model_service.generate_heatmap(
            self.selected_image_tensor,
            selected_task
        )

        overlay = overlay_heatmap_on_image(
            self.original_pil_image,
            heatmap
        )

        self._display_image(overlay)

    def _reset_image(self):
        if self.original_pil_image is not None:
            self._display_image(self.original_pil_image)


def build_transform_pipeline():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def build_task_dimensions(encoders):
    return {task: len(encoders[task].classes_) for task in TASKS}


def launch_application():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataframe, encoders = prepare_data(
        csv_path="data/csv/full_clean.csv",
        image_dir="data/images/normalized_dials_full"
    )

    task_dimensions = build_task_dimensions(encoders)
    model_service = ModelInferenceService(task_dimensions, encoders, device)

    transform_pipeline = build_transform_pipeline()

    root_window = tkinter_root.Tk()
    ModelTesterUI(root_window, model_service, transform_pipeline)
    root_window.mainloop()


if __name__ == "__main__":
    launch_application()