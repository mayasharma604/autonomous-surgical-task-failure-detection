import os
import torch
import torch.nn as nn
import shutil
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_m
from tqdm import tqdm

# config
MODEL_PATH = "/data/CIS2/model_maya/efficient_net_model/checkpoints_effnet/best_model.pth"
INPUT_FOLDER = "/data/virtuoso_cao_demo/tissue_2/1_retract/20251229-143557-916908/endoscope"
OUTPUT_DIR = "./inference_output"

GOOD_DIR = os.path.join(OUTPUT_DIR, "good")
BAD_DIR = os.path.join(OUTPUT_DIR, "bad")

IMG_SIZE = 224 # Check if your training used 480 (V2-M default) or 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create output directories
os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(BAD_DIR, exist_ok=True)

# model definition
class SqueezeTime(nn.Module):
    def forward(self, x):
        return x.squeeze(1)

# initialize EfficientNet-V2-M
base_model = efficientnet_v2_m(weights=None)

# adjust first layer for 6-channel input (RGB + Mask/Seg)
# For V2-M, this is features[0][0]
base_model.features[0][0] = nn.Conv2d(
    6, 24, kernel_size=3, stride=2, padding=1, bias=False
)

# adjust classifier for binary output (1 neuron)
base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, 1)

# Wrap in Sequential if your training used SqueezeTime
model = nn.Sequential(
    SqueezeTime(),
    base_model
).to(DEVICE)

# load checkpoint
print(f" Loading EfficientNet-V2-M from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# Check if model was saved as 'model_state_dict' or raw
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Fix "module." or Sequential "1." prefixes if necessary
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # If saved within the Sequential(SqueezeTime, base_model) wrapper:
    name = k.replace('1.', '') if k.startswith('1.') else k
    # If saved with DataParallel:
    name = name.replace('module.', '')
    new_state_dict[name] = v

base_model.load_state_dict(new_state_dict)
model.eval()
print(" Weights loaded successfully.")

# transforms and data
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Uncomment if used in training
])

image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(".png")
])

# inference loop
with torch.no_grad():
    for fname in tqdm(image_files, desc="Processing Tension"):
        path = os.path.join(INPUT_FOLDER, fname)

        img = Image.open(path).convert("RGB")
        img = transform(img)

        # Create the 6-channel input [6, H, W]
        # (Assuming you concatenated the same image or a mask during training)
        img6 = torch.cat([img, img], dim=0) 

        # Add Batch and Time dimensions -> [1, 1, 6, H, W]
        img_input = img6.unsqueeze(0).unsqueeze(0).to(DEVICE)

        output = model(img_input)
        prob = torch.sigmoid(output).item()

        # Classification decision
        if prob > 0.5:
            dest = os.path.join(BAD_DIR, fname)
        else:
            dest = os.path.join(GOOD_DIR, fname)

        shutil.copy(path, dest)

print(f"\n Done! Check results in {OUTPUT_DIR}")
