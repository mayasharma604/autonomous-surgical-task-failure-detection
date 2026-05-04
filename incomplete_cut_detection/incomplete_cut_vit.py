import sys
# Shield against ROS path conflicts
sys.path = [p for p in sys.path if "/opt/ros" not in p]

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
BATCH_SIZE = 16 
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 35 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_vit_incomplete_cut_regression.pth"

# =========================
# 1. DATASET CLASS
# =========================
class SurgicalProgressDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        print(f"📊 ViT Dataset Loaded: {len(self.df)} images.")

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
# 2. AUGMENTATIONS
# =========================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = SurgicalProgressDataset(CSV_PATH, transform=data_transforms)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# =========================
# 3. MODEL SETUP (ViT Split)
# =========================
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

# Adjust the head for regression
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid() 
)
model = model.to(DEVICE)
criterion = nn.MSELoss()

# =========================
# 4. SHARED EPOCH FUNCTION
# =========================
def run_epoch(optimizer, loader, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
        with torch.no_grad():
            for i in range(5): # Check 5 random validation samples
                img, label = val_set[i]
                img = img.unsqueeze(0).to(DEVICE)
                pred = model(img).item()
                print(f"Sample {i} | Real Progress: {label:.4f} | Predicted: {pred:.4f}")
    
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.set_grad_enabled(is_train):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
            if is_train:
                optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            
            if is_train:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy().flatten())
            all_labels.extend(lbls.cpu().numpy().flatten())
            
    avg_loss = running_loss / len(loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (np.abs(all_preds - all_labels) < 0.1).mean() * 100
    bin_preds = (all_preds > 0.80).astype(int) 
    bin_labels = (all_labels > 0.80).astype(int)
    prec = precision_score(bin_labels, bin_preds, zero_division=0)
    rec = recall_score(bin_labels, bin_preds, zero_division=0)
    
    return avg_loss, acc, prec, rec

# =========================
# 5. EXECUTION
# =========================
best_val_loss = float('inf')
total_eps = PHASE1_EPOCHS + PHASE2_EPOCHS

# --- PHASE 1 ---
print("\n PHASE 1: Training Head Only (ViT Backbone Frozen)")
for name, param in model.named_parameters():
    if "heads" not in name:
        param.requires_grad = False

optimizer = optim.AdamW(model.heads.parameters(), lr=PHASE1_LR)

for epoch in range(PHASE1_EPOCHS):
    # Fixed the argument order here:
    t_loss, t_acc, t_p, t_r = run_epoch(optimizer, train_loader, is_train=True)
    v_loss, v_acc, v_p, v_r = run_epoch(None, val_loader, is_train=False) # Optimizer is None
    
    print(f"\nEpoch [{epoch+1}/{total_eps}]")
    print(f"  Training Accuracy (±0.1): {t_acc:.2f}%")
    print(f"  Validation Accuracy (±0.1): {v_acc:.2f}%")
    print(f"  Validation Precision: {v_p:.4f} | Validation Recall: {v_r:.4f}")
    
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

# --- PHASE 2 ---
print("\nPHASE 2: Full Fine-Tuning (Transformers Unfrozen)")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=0.01)

for epoch in range(PHASE2_EPOCHS):
    curr_ep = epoch + PHASE1_EPOCHS + 1
    t_loss, t_acc, t_p, t_r = run_epoch(optimizer, train_loader, is_train=True)
    v_loss, v_acc, v_p, v_r = run_epoch(optimizer, val_loader, is_train=False)
    
    print(f"\nEpoch [{curr_ep}/{total_eps}]")
    print(f"  Training Accuracy (±0.1): {t_acc:.2f}%")
    print(f"  Validation Accuracy (±0.1): {v_acc:.2f}%")
    print(f"  Validation Precision: {v_p:.4f} | Validation Recall: {v_r:.4f}")
    
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(" Best Model Saved!")
