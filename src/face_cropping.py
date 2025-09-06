import os
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch

# ==============================
# Config
# ==============================
INPUT_DIRS = [
    "D:/DeepfakeGuard/Processed Outputs/real",
    "D:/DeepfakeGuard/Processed Outputs/fake"
]
OUTPUT_BASE = "data_faces"  # Cropped faces will be saved here
IMAGE_SIZE = (224, 224)

os.makedirs(OUTPUT_BASE, exist_ok=True)

# ==============================
# Face Detector
# ==============================
mtcnn = MTCNN(keep_all=False, device="cuda:0" if torch.cuda.is_available() else "cpu")

# ==============================
# Cropping Function
# ==============================
def crop_and_save_faces(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(input_dir), desc=f"Cropping {input_dir}"):
        try:
            img_path = os.path.join(input_dir, img_name)
            img = Image.open(img_path).convert("RGB")

            # Detect faces
            box, prob = mtcnn.detect(img)

            if box is not None and prob[0] is not None and prob[0] > 0.90:
                # Clip bounding box to image size
                x1, y1, x2, y2 = map(int, box[0])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img.width, x2); y2 = min(img.height, y2)

                face = img.crop((x1, y1, x2, y2))
                face = face.resize(IMAGE_SIZE)
            else:
                # ⚠️ Fallback: save resized full image
                face = img.resize(IMAGE_SIZE)

            # Save cropped face
            face.save(os.path.join(output_dir, img_name))

        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")


# ==============================
# Main Loop
# ==============================
if __name__ == "__main__":
    for folder in INPUT_DIRS:
        label = os.path.basename(folder)
        output_dir = os.path.join(OUTPUT_BASE, label)
        crop_and_save_faces(folder, output_dir)

    print(f"✅ Cropped faces saved to {OUTPUT_BASE}/real and {OUTPUT_BASE}/fake")
