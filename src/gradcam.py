import torch
import cv2
import numpy as np
import timm
from torchvision import transforms
from PIL import Image

MODEL_PATH = "D:/DeepfakeGuard/models/xception_best.pth"
IMG_SIZE = 380
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# Choose last conv layer for GradCAM
target_layer = model.blocks[-1]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        loss = output[:, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze().cpu().numpy()

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def apply_heatmap(img_path, save_path="gradcam_result.jpg"):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(tensor)

    img_cv = cv2.cvtColor(np.array(img.resize((IMG_SIZE, IMG_SIZE))), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)

    cv2.imwrite(save_path, overlay)
    print(f"✅ GradCAM saved at {save_path}")

if __name__ == "__main__":
    test_img = "D:/DeepfakeGuard/data_faces_split/test/fake/000_003_2.jpg"
    apply_heatmap(test_img)
