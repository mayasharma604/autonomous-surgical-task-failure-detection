import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import shutil

from rae import ResidualAutoencoder

# paths/config
model_path = "./rae_best.pth"
good_folder = "/data/CIS2/JHU-collision-CAO1_annotated/20260227-175258-402169/good"
bad_folder  = "/data/CIS2/JHU-collision-CAO1_annotated/20260227-175258-402169/bad"

all_dirs = [good_folder, bad_folder]

normal_dir = "./all_normal_frames"
anomaly_dir = "./all_anomaly_frames"
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(anomaly_dir, exist_ok=True)

# device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResidualAutoencoder().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

threshold = checkpoint["threshold"]
val_mean  = checkpoint["val_mean"]
val_std   = checkpoint["val_std"]
print(f"Loaded threshold: {threshold:.6f}")


# transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# load images
images = []
paths = []

for d in all_dirs:
    for root, _, files in os.walk(d):
        for f in sorted(files):
            if f.endswith(".png"):
                path = os.path.join(root, f)
                paths.append(path)
                img = Image.open(path).convert("RGB")
                images.append(transform(img))

images_tensor = torch.stack(images).to(device)
print(f"Loaded {len(images_tensor)} images")


def compute_confidence(error, threshold, val_mean, val_std):
    is_collision = error >= threshold
    z = (error - val_mean) / (val_std + 1e-8)
    collision_confidence = torch.sigmoid(torch.tensor(z)).item()
    if is_collision:
        confidence = collision_confidence
    else:
        confidence = 1 - collision_confidence
    return bool(is_collision), round(confidence * 100, 1)

# inference
with torch.no_grad():
    recon = model(images_tensor)
    errors = torch.mean((images_tensor - recon) ** 2, dim=(1, 2, 3))

# determine anomalies vs normal
for i, path in enumerate(paths):
    is_collision, confidence = compute_confidence(
        errors[i].item(), threshold, val_mean, val_std
    )
    label = "COLLISION" if is_collision else "normal"
    print(f"{label} ({confidence}%) — {os.path.basename(path)}")
    target_dir = anomaly_dir if is_collision else normal_dir
    shutil.copy(path, target_dir)

print(f"Saved {sum(errors >= threshold).item()} anomalous frames to {anomaly_dir}")
print(f"Saved {sum(errors < threshold).item()} normal frames to {normal_dir}")
