# tension_train_resnet_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Shield against ROS path conflicts
sys.path = [p for p in sys.path if "/opt/ros" not in p]
sys.path.append("/data/tensionData/training/shared")
import tension_training as tt
class SqueezeTime(nn.Module):
    def forward(self, x):
        return x.squeeze(1) # Removes the dimension at index 1 (Time dimension for 1-frame windows)

# =========================
# Config
# =========================
WORKSPACE_ROOT = '/data/tensionData'
WINDOW_SIZE = 1        
PREDICT_FRAMES = 1
BATCH_SIZE = 64        
NUM_WORKERS = 12       
LEARNING_RATE = 1e-4
IMG_SIZE = 224
EPOCHS = 20 # Increased slightly for Trial Split convergence
DEVICE = tt.setup_device(seed=42)
MODEL_SAVE_PATH = "best_resnet18_tension_trial_split.pth"

# =========================
# Prepare masks & clips (Trial-Based Split)
# =========================
mask_lookup = tt.make_mask_lookup(WORKSPACE_ROOT)
seg_mask_lookup = tt.make_seg_mask_lookup(WORKSPACE_ROOT)

clips = tt.index_all_clips(WORKSPACE_ROOT, mask_lookup, min_clip_len=WINDOW_SIZE)

# CRITICAL: This performs the split by Trial ID, not by individual frames
train_clips, val_clips = tt.split_clips_by_trial(clips)

train_windows = tt.build_windows(train_clips, WINDOW_SIZE, PREDICT_FRAMES)
val_windows = tt.build_windows(val_clips, WINDOW_SIZE, PREDICT_FRAMES)

# Balance to ensure the model doesn't just guess "No Tension"
train_windows = tt.balance_windows(train_windows, PREDICT_FRAMES)
val_windows = tt.balance_windows(val_windows, PREDICT_FRAMES)

# =========================
# Loaders
# =========================
train_loader, val_loader, train_ds, val_ds = tt.make_loaders(
    train_windows, val_windows,
    BATCH_SIZE, NUM_WORKERS,
    IMG_SIZE, PREDICT_FRAMES,
    mask_lookup,
    seg_mask_lookup=seg_mask_lookup,
    dual_stream=True
)

# =========================
# Model
# =========================
from torchvision.models import resnet18

base_model = resnet18(weights=None) 
# Adjust for 6-channel input (RGB + Mask/Seg)
base_model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
base_model.fc = nn.Linear(base_model.fc.in_features, 1)

model = nn.Sequential(
    SqueezeTime(),
    base_model
).to(DEVICE)

# =========================
# Optimizer & Loss
# =========================
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
criterion = nn.BCEWithLogitsLoss() # Better for binary precision/recall tasks

# =========================
# Custom Training Loop for CIS Metrics
# =========================
best_val_loss = float('inf')

print(f"\n🚀 Starting Tension Training (Trial Split)")
print(f"Train Windows: {len(train_windows)} | Val Windows: {len(val_windows)}\n")

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    train_loss = 0.0
    correct_t = 0
    total_t = 0
    
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).float()
        optimizer.zero_grad()
        
        # In dual-stream/tt setup, labels might need unsqueezing
        if lbls.dim() == 1: lbls = lbls.unsqueeze(1)
            
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_t += (preds == lbls).sum().item()
        total_t += lbls.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = (correct_t / total_t) * 100

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).float()
            if lbls.dim() == 1: lbls = lbls.unsqueeze(1)
            
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            val_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Calculate CIS Metrics
    val_acc = (all_preds == all_labels).mean() * 100
    val_prec = precision_score(all_labels, all_preds, zero_division=0)
    val_rec = recall_score(all_labels, all_preds, zero_division=0)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"  Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f}")

    # Save Best Model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(base_model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ⭐ Best Model Saved to {MODEL_SAVE_PATH}")
    
    scheduler.step(avg_val_loss)
