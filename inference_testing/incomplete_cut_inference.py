import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

# config
MODEL_PATH = "/data/CIS2/model_maya/efficient_net_model/best_v2m_incomplete_cut_regression.pth"

# Root for all phases (1_retract, 2_collision, 3_resect, etc.)
ROOT_INPUT_FOLDER = "/data/virtuoso_cao_demo/tissue_2" 
OUTPUT_CSV = "./incomplete_cut_endoscope_only_weights.csv"

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model v2m
model = models.efficientnet_v2_m(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid() 
)

model = model.to(DEVICE)

# Load state dict
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '')
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

print(f" Loaded EfficientNet-V2-M Regression model")

# transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# filtered folder inference
all_results = []

for root, dirs, files in os.walk(ROOT_INPUT_FOLDER):
    
    if os.path.basename(root) != "endoscope":
        continue

    image_files = sorted([f for f in files if f.lower().endswith((".png", ".jpg"))])
    
    if not image_files:
        continue
    
    # Captures the phase name (e.g., 3_resect) by looking back up the path
    phase_label = os.path.basename(os.path.dirname(os.path.dirname(root)))
    print(f"\n Found Endoscope Data in: {phase_label}")

    with torch.no_grad():
        for fname in tqdm(image_files, desc=f"Weights for {phase_label}"):
            path = os.path.join(root, fname)
            
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                output = model(img_tensor)
                weight = output.item() 
                
                all_results.append({
                    "phase": phase_label,
                    "folder_path": root,
                    "filename": fname,
                    "predicted_weight": round(weight, 4)
                })
                
            except Exception as e:
                print(f" Error in {path}: {e}")

# export
if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Done! Processed all endoscope folders across phases.")
    print(f" CSV saved: {OUTPUT_CSV}")
    
    # Quick view to ensure it's pulling from the right spots
    print("\nPreview:")
    print(df[['phase', 'filename', 'predicted_weight']].head(10).to_string(index=False))
else:
    print(" No endoscope folders found in the tree.")
