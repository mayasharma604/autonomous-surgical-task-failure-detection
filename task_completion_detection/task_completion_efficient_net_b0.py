import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Shield against ROS path conflicts on the server
sys.path = [p for p in sys.path if "/opt/ros" not in p]

# =========================
# CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../annotations/task_completion_labels2.csv")
BATCH_SIZE = 32  
PHASE1_EPOCHS = 15 
PHASE2_EPOCHS = 40
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_task_classifier_b0.pth"

# =========================
# 1. DATASET (Updated for Classification)
# =========================
class ProgressRegressionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # We now know your column is named 'phase'
        self.target_col = 'phase'

        if self.target_col not in self.df.columns:
            raise KeyError(f"❌ Column '{self.target_col}' not found. Available: {self.df.columns.tolist()}")

        self.classes = sorted(self.df[self.target_col].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        print(f"📊 Success! Training for {len(self.classes)} phases.")
        print(f"📂 Phases identified: {self.classes}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['relative_path']
        if not img_path.startswith("/"):
            img_path = os.path.join(BASE_DIR, "..", img_path)
        
        image = Image.open(img_path).convert("RGB")
        
        # Get the label from the 'phase' column
        phase_name = self.df.iloc[idx][self.target_col]
        label = self.class_to_idx[phase_name]
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
# Transforms
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), 
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize Dataset
full_dataset = ProgressRegressionDataset(CSV_PATH, transform=data_transforms)
num_tasks = len(full_dataset.classes)

# Trial-Based Split
unique_trials = np.array(full_dataset.df['unique_key'].unique())
np.random.seed(42) 
np.random.shuffle(unique_trials)

train_count = int(0.8 * len(unique_trials))
train_trial_ids = unique_trials[:train_count]
val_trial_ids = unique_trials[train_count:]

train_indices = full_dataset.df[full_dataset.df['unique_key'].isin(train_trial_ids)].index.tolist()
val_indices = full_dataset.df[full_dataset.df['unique_key'].isin(val_trial_ids)].index.tolist()

train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# =========================
# 2. MODEL SETUP (EfficientNet Classifier)
# =========================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Linear(512, num_tasks) # Outputs one score per task
)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

# =========================
# 3. TRAINING FUNCTION
# =========================
def run_epoch(optimizer, loader, is_train=True):
    if is_train: model.train()
    else: model.eval()
    
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.set_grad_enabled(is_train):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            if is_train: optimizer.zero_grad()
            
            outs = model(imgs)
            loss = criterion(outs, lbls)
            
            if is_train:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            _, preds = torch.max(outs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, acc, prec, rec

# =========================
# 4. EXECUTION (Optimizers + Loop)
# =========================
best_val_loss = float('inf')
total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS

# Define Optimizers initially
# PHASE 1: Head Only
for param in model.features.parameters():
    param.requires_grad = False
opt1 = optim.AdamW(model.classifier.parameters(), lr=PHASE1_LR)

# PHASE 2: (Will be redefined/updated when unfreezing)
opt2 = optim.AdamW(model.parameters(), lr=PHASE2_LR)

print(f"\n🚀 Starting Training on {DEVICE}")
print(f"📊 Targets: {num_tasks} phases | Split: {len(train_trial_ids)} Train / {len(val_trial_ids)} Val trials")

for epoch in range(total_epochs):
    # Determine which optimizer and mode to use
    if epoch < PHASE1_EPOCHS:
        current_opt = opt1
        phase_label = "PHASE 1 (Head)"
    else:
        # Unfreeze backbone on the first iteration of Phase 2
        if epoch == PHASE1_EPOCHS:
            print("\n🔥 Unfreezing backbone for Fine-Tuning...")
            for param in model.parameters():
                param.requires_grad = True
            # Refresh opt2 to include all model parameters now that they are unfrozen
            opt2 = optim.AdamW(model.parameters(), lr=PHASE2_LR)
            
        current_opt = opt2
        phase_label = "PHASE 2 (Full)"

    # Run Training & Validation
    train_loss, train_acc, _, _ = run_epoch(current_opt, train_loader, is_train=True)
    val_loss, val_acc, val_prec, val_rec = run_epoch(None, val_loader, is_train=False)

    # Print Metrics
    print(f"[{phase_label}] Epoch {epoch+1}/{total_epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}%")
    print(f"  Precision:  {val_prec:.4f}   | Recall: {val_rec:.4f}")

    # Save Best Model based on Validation Loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': current_opt.state_dict(),
            'val_loss': val_loss,
            'classes': full_dataset.classes
        }, MODEL_SAVE_PATH)
        print(f"  ⭐ [SAVED] New best validation loss at Epoch {epoch+1}")
    
    print("-" * 30)
