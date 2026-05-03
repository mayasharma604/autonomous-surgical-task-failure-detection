import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import shutil

from rae import ResidualAutoencoder

# -------------------------------
# Paths
# -------------------------------
model_path = "./rae_best.pth"
good_folder = "/data/CIS2/JHU-collision-CAO1_annotated/20260227-175258-402169/good"
bad_folder  = "/data/CIS2/JHU-collision-CAO1_annotated/20260227-175258-402169/bad"

all_dirs = [good_folder, bad_folder]

normal_dir = "./all_normal_frames"
anomaly_dir = "./all_anomaly_frames"
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(anomaly_dir, exist_ok=True)

# -------------------------------
# Device and model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResidualAutoencoder().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

threshold = checkpoint["threshold"]
val_mean  = checkpoint["val_mean"]
val_std   = checkpoint["val_std"]
print(f"Loaded threshold: {threshold:.6f}")


# -------------------------------
# Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -------------------------------
# Load images
# -------------------------------
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

# -------------------------------
# Inference
# -------------------------------
with torch.no_grad():
    recon = model(images_tensor)
    errors = torch.mean((images_tensor - recon) ** 2, dim=(1, 2, 3))

# -------------------------------
# Determine anomalous vs normal
# -------------------------------
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

'''
import torch
import torch.nn.functional as F
from rae import ResidualAutoencoder  # your model class
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from shutil import copyfile

# -----------------------
# 1️⃣ Settings
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

good_folder = "/home/ves/CIS2/JHU-collision-CAO1_annotated/20260227-175258-402169/good"
bad_folder = "/home/ves/CIS2/JHU-collision-CAO1_annotated/20260227-175258-402169/bad"

model_path = "rae_model.pth"
top_N = 5  # number of top anomalies to save

# -----------------------
# 2️⃣ Load model
# -----------------------
model = ResidualAutoencoder().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------
# 3️⃣ Prepare image loader
# -----------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def load_images_from_folder(folder, label):
    image_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".png"):
                image_paths.append((os.path.join(root, f), label))
    return image_paths

# Load good and bad frames
good_images = load_images_from_folder(good_folder, label="good")
bad_images = load_images_from_folder(bad_folder, label="bad")
all_images = good_images + bad_images

print(f"Loaded {len(all_images)} images ({len(good_images)} good, {len(bad_images)} bad)")

# -----------------------
# 4️⃣ Compute anomaly scores
# -----------------------
anomaly_scores = []

with torch.no_grad():
    for path, label in all_images:
        img = transform(Image.open(path)).unsqueeze(0).to(device)
        recon = model(img)
        error = F.mse_loss(recon, img, reduction='mean').item()
        anomaly_scores.append((error, path, label))

# Sort by highest anomaly
anomaly_scores.sort(reverse=True, key=lambda x: x[0])

# -----------------------
# 5️⃣ Print top anomalies
# -----------------------
print(f"Top {top_N} anomaly frames:")
for score, path, label in anomaly_scores[:top_N]:
    print(f"{score:.5f} - {label} - {path}")

# -----------------------
# 6️⃣ Save anomaly scores plot
# -----------------------
scores = [s for s, _, _ in anomaly_scores]
labels = [l for _, _, l in anomaly_scores]
plt.figure(figsize=(12,4))
plt.scatter(range(len(scores)), scores, c=['green' if l=='good' else 'red' for l in labels], label='Frames')
plt.title("Anomaly scores over frames")
plt.xlabel("Frame index")
plt.ylabel("Reconstruction error (MSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig("anomaly_scores.png")
print("Saved anomaly scores plot as anomaly_scores.png")

# -----------------------
# 7️⃣ Save top anomalous frames
# -----------------------
os.makedirs("top_anomalies", exist_ok=True)
for i, (score, path, label) in enumerate(anomaly_scores[:top_N]):
    filename = os.path.basename(path)
    new_path = os.path.join("top_anomalies", f"{i+1}_{score:.5f}_{label}_{filename}")
    copyfile(path, new_path)

print(f"Saved top {top_N} anomalous frames in ./top_anomalies/")
'''