import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

# =========================
# Config
# =========================
MODEL_PATH = "/data/CIS2/model_maya/efficient_net_model/best_v2m_resect_regression.pth"
TRIAL_BASE_PATH = "/data/virtuoso_cao_demo/tissue_2/"

# We will grab the first available trial from Phase 2 as our target
START_PHASE = "2_resect_start"
RESECT_PHASES = ["2_resect_start", "3_resect", "4_resect_home", "5_retract_home"]

OUTPUT_CSV = "tissue2_full_sequence_inference.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Model Setup
# =========================
def get_regression_model():
    model = models.efficientnet_v2_m(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(model.classifier[1].in_features, 1),
        nn.Sigmoid()
    )
    return model

model = get_regression_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================
# SMART FOLDER MATCHING
# =========================
# 1. Get the list of trials in Phase 2
p2_path = os.path.join(TRIAL_BASE_PATH, START_PHASE)
available_trials = sorted([d for d in os.listdir(p2_path) if os.path.isdir(os.path.join(p2_path, d))])

if not available_trials:
    print(f"❌ No trials found in {p2_path}")
    exit()

# Pick the first trial and its date prefix (e.g., 20251229-143558)
target_trial_full = available_trials[0]
target_prefix = "-".join(target_trial_full.split("-")[:2]) 

print(f"🎯 Target Trial Prefix: {target_prefix} (Full: {target_trial_full})")

results = []

# 2. Loop through phases and find the folder that matches that prefix
for phase in RESECT_PHASES:
    phase_dir = os.path.join(TRIAL_BASE_PATH, phase)
    if not os.path.exists(phase_dir):
        continue
    
    # Find a folder in this phase that starts with our target prefix
    match = [d for d in os.listdir(phase_dir) if d.startswith(target_prefix)]
    
    if not match:
        print(f"➖ Phase {phase}: No matching trial prefix found. Skipping.")
        continue
    
    actual_trial_folder = match[0]
    img_dir = os.path.join(phase_dir, actual_trial_folder, "endoscope")
    
    if not os.path.exists(img_dir):
        continue

    image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])
    print(f"📦 Processing {phase} | Folder: {actual_trial_folder} | Frames: {len(image_files)}")

    with torch.no_grad():
        for fname in tqdm(image_files, desc=phase):
            path = os.path.join(img_dir, fname)
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                prediction = model(img_tensor).item()

                results.append({
                    "phase": phase,
                    "actual_folder": actual_trial_folder,
                    "filename": fname,
                    "predicted_progress": round(prediction, 4)
                })
            except:
                continue

# =========================
# Save
# =========================
if results:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Done! CSV saved to {OUTPUT_CSV}")
    print("\n--- Summary ---")
    print(df.groupby('phase')['predicted_progress'].mean())
else:
    print("❌ No matching sequence found across folders.")
