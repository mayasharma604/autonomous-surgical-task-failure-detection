import sys
# Clean ROS paths
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

# config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../annotations/task_completion_labels2.csv")
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_resnet18_task_completion.pth"

# dataset class
class TaskCompletionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV not found: {csv_file}")
            
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Numerical progress column from your CSV
        self.target_col = 'progress' 

        print(f" Dataset: {len(self.df)} samples.")
        print(f"   Targeting binary classes from '{self.target_col}' column.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_path = self.df.iloc[idx]['relative_path']
        img_path = os.path.join(BASE_DIR, "..", raw_path)
        
        image = Image.open(img_path).convert("RGB")
        
        # Binary Thresholding
        # Completed (Target) -> 0 | In-progress -> 1
        val = self.df.iloc[idx][self.target_col]
        label = 0 if val >= 0.95 else 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# augmentation and loading
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = TaskCompletionDataset(CSV_PATH, transform=data_transforms)
train_size = int(0.8 * len(full_dataset))
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset)-train_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# model set up
# Using ResNet-18 for a good balance of speed and accuracy
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # Binary output
model = model.to(DEVICE)

# Class weights to help with the "Completed" class rarity
# We'll use 12.0 here as a slightly more conservative start than the 15.0 used in B0
weights = torch.tensor([12.0, 1.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

#training loop
best_val_loss = float('inf')

print(f" Training ResNet-18 on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # val
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            v_loss = criterion(outputs, labels)
            val_loss += v_loss.item()
            
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    
    precision = precision_score(all_labels, all_preds, pos_label=0, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"  Target (Completed) -> Precision: {precision:.4f} | Recall: {recall:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   Best Model Saved")

print("\nTraining Complete.")
