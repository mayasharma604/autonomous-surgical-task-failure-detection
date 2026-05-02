import sys
sys.path = [p for p in sys.path if "/opt/ros" not in p]

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.metrics import precision_score, recall_score
import pandas as pd
from PIL import Image
import os
import numpy as np

# =========================
# CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../annotations/incomplete_cut_labels2.csv")
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_resnet50_trial_split.pth"

# =========================
# 1. DATASET CLASS
# =========================
class SurgicalProgressDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV not found at {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        print(f"📊 Dataset Loaded: {len(self.df)} total frames.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_path = self.df.iloc[idx]['relative_path']
        actual_path = os.path.join(BASE_DIR, "..", raw_path)
        image = Image.open(actual_path).convert("RGB")
        label = float(self.df.iloc[idx]['progress'])
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# =========================
# 2. TRIAL-BASED SPLIT LOGIC
# =========================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = SurgicalProgressDataset(CSV_PATH, transform=data_transforms)

# Get unique trial IDs (unique_key handles same trial IDs across different phases)
unique_trials = full_dataset.df['unique_key'].unique()
np.random.seed(42) # For consistent research results
np.random.shuffle(unique_trials)

# Split Trials 80/20
train_trial_count = int(0.8 * len(unique_trials))
train_trial_ids = unique_trials[:train_trial_count]
val_trial_ids = unique_trials[train_trial_count:]

# Map Trial IDs back to row indices
train_indices = full_dataset.df[full_dataset.df['unique_key'].isin(train_trial_ids)].index.tolist()
val_indices = full_dataset.df[full_dataset.df['unique_key'].isin(val_trial_ids)].index.tolist()

train_set = Subset(full_dataset, train_indices)
val_set = Subset(full_dataset, val_indices)

print(f"✅ Trial-Based Split Summary:")
print(f"   Train: {len(train_trial_ids)} trials | {len(train_indices)} frames")
print(f"   Val:   {len(val_trial_ids)} trials | {len(val_indices)} frames")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# =========================
# 3. MODEL SETUP
# =========================
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid() 
)
model = model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# 4. TRAINING LOOP
# =========================
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss, all_train_preds, all_train_labels = 0.0, [], []
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        all_train_preds.extend(outputs.detach().cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

    model.eval()
    val_loss, all_val_preds, all_val_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            outputs = model(images)
            v_loss = criterion(outputs, labels)
            val_loss += v_loss.item()
            all_val_preds.extend(outputs.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    # Metrics
    train_p, train_l = np.array(all_train_preds).flatten(), np.array(all_train_labels).flatten()
    val_p, val_l = np.array(all_val_preds).flatten(), np.array(all_val_labels).flatten()
    
    train_acc = (np.abs(train_p - train_l) < 0.1).mean() * 100
    val_acc = (np.abs(val_p - val_l) < 0.1).mean() * 100
    
    val_preds_bin = (val_p > 0.5).astype(int)
    val_labels_bin = (val_l > 0.5).astype(int)
    precision = precision_score(val_labels_bin, val_preds_bin, zero_division=0)
    recall = recall_score(val_labels_bin, val_preds_bin, zero_division=0)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"  Loss -> Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
    print(f"  Acc  -> Train (±0.1): {train_acc:.2f}% | Val (±0.1): {val_acc:.2f}%")
    print(f"  Binary -> Precision: {precision:.4f} | Recall: {recall:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ⭐ Best Model Saved (Strict Trial Split)")
