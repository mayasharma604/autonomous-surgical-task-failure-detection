import sys
sys.path = [p for p in sys.path if "/opt/ros" not in p]

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_score, recall_score
import os
import numpy as np

# =========================
# CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../annotations/incomplete_cut_labels2.csv")
BATCH_SIZE = 32
PHASE1_LR = 1e-3  
PHASE2_LR = 1e-5  
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_b0_incomplete_cut_regression.pth"

# =========================
# 1. DATASET CLASS
# =========================
class SurgicalProgressDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        print(f"📊 Dataset Loaded: {len(self.df)} images.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['relative_path']
        if not img_path.startswith("/"):
            img_path = os.path.join(BASE_DIR, "..", img_path)
        
        image = Image.open(img_path).convert("RGB")
        label = float(self.df.iloc[idx]['progress'])
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# =========================
# 2. AUGMENTATIONS & LOADING
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
# 3. MODEL SETUP
# =========================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid() 
)
model = model.to(DEVICE)
criterion = nn.MSELoss()

# =========================
# 4. SHARED TRAINING FUNCTION
# =========================
def run_epoch(epoch, optimizer, loader, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
    
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

    # Accuracy: Pred is within 10% of True Progress
    acc = (np.abs(all_preds - all_labels) < 0.1).mean() * 100
    
    # Binary Metrics (Threshold at 0.80 to define "Incomplete/Complete Cut")
    threshold = 0.80
    bin_preds = (all_preds > threshold).astype(int)
    bin_labels = (all_labels > threshold).astype(int)
    
    prec = precision_score(bin_labels, bin_preds, zero_division=0)
    rec = recall_score(bin_labels, bin_preds, zero_division=0)
    
    return avg_loss, acc, prec, rec

# =========================
# 5. EXECUTION
# =========================
best_val_loss = float('inf')

# --- PHASE 1 ---
print("\n🧊 PHASE 1: Backbone Frozen (5 Epochs)")
for param in model.features.parameters():
    param.requires_grad = False
optimizer = optim.Adam(model.classifier.parameters(), lr=PHASE1_LR)

for epoch in range(PHASE1_EPOCHS):
    t_loss, t_acc, t_p, t_r = run_epoch(epoch, optimizer, train_loader, is_train=True)
    v_loss, v_acc, v_p, v_r = run_epoch(epoch, optimizer, val_loader, is_train=False)
    
    print(f"\nEpoch [{epoch+1}/{PHASE1_EPOCHS+PHASE2_EPOCHS}]")
    print(f"  Train -> Loss: {t_loss:.6f} | Acc: {t_acc:.2f}% | Prec: {t_p:.4f} | Rec: {t_r:.4f}")
    print(f"  Val   -> Loss: {v_loss:.6f} | Acc: {v_acc:.2f}% | Prec: {v_p:.4f} | Rec: {v_r:.4f}")
    
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

# --- PHASE 2 ---
print("\n🔥 PHASE 2: Full Fine-Tuning (25 Epochs)")
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=PHASE2_LR)

for epoch in range(PHASE2_EPOCHS):
    real_epoch = epoch + PHASE1_EPOCHS + 1
    t_loss, t_acc, t_p, t_r = run_epoch(epoch, optimizer, train_loader, is_train=True)
    v_loss, v_acc, v_p, v_r = run_epoch(epoch, optimizer, val_loader, is_train=False)
    
    print(f"\nEpoch [{real_epoch}/{PHASE1_EPOCHS+PHASE2_EPOCHS}]")
    print(f"  Train -> Loss: {t_loss:.6f} | Acc: {t_acc:.2f}% | Prec: {t_p:.4f} | Rec: {t_r:.4f}")
    print(f"  Val   -> Loss: {v_loss:.6f} | Acc: {v_acc:.2f}% | Prec: {v_p:.4f} | Rec: {v_r:.4f}")
    
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("  ⭐ Best Model Saved (Val Loss Improved)")
