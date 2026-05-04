import os
import torch
import torch.nn as nn
import shutil
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_m
from tqdm import tqdm

# config
MODEL_PATH = "/data/CIS2/model_maya/efficient_net_model/best_effnet_v2m_collision.pth"

# Updated to use the JHU-collision-CAO1 dataset directory
INPUT_FOLDER = "/data/CIS2/JHU-collision-CAO1_annotated/20260227-174116-812174/bad"
OUTPUT_DIR = "./effnet_v2m_collision_inference_bad"

GOOD_DIR = os.path.join(OUTPUT_DIR, "good")
BAD_DIR = os.path.join(OUTPUT_DIR, "bad")

# size to 224 for image
IMG_SIZE = 224 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ceate output dirs
os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(BAD_DIR, exist_ok=True)

# model effnet v2m
# Initialize the base V2-M architecture
model = efficientnet_v2_m(weights=None)

# Adjust the classifier for 2 classes (Good vs Bad)
# EfficientNet-V2 models use 'classifier[1]' for the final linear layer
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

# Load state dict
# Note: If your checkpoint is a full dictionary, use checkpoint['model_state_dict']
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

print(f" Loaded EfficientNet-V2-M from {MODEL_PATH}")

# transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# load images
image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print(f" Found {len(image_files)} images")

# inference
with torch.no_grad():
    for fname in tqdm(image_files, desc="Running Collision Inference"):
        path = os.path.join(INPUT_FOLDER, fname)

        try:
            img = Image.open(path).convert("RGB")
        except:
            print(f" Skipping unreadable: {fname}")
            continue

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        # Class mapping: 0 = good, 1 = bad
        if pred == 1:
            dest = os.path.join(BAD_DIR, fname)
        else:
            dest = os.path.join(GOOD_DIR, fname)

        shutil.copy(path, dest)

print("\n Done!")
print(f"Good frames moved to → {GOOD_DIR}")
print(f"Bad frames moved to → {BAD_DIR}")
