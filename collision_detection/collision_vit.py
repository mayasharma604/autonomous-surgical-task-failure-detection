import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score

# -----------------------
# 1. Dataset Class Definition
# -----------------------
class NestedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.classes = ['good', 'bad']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Assuming structure: root_dir/Trial_Name/good/image.jpg
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Directory not found: {root_dir}")

        for trial_folder in os.listdir(root_dir):
            trial_path = os.path.join(root_dir, trial_folder)
            if not os.path.isdir(trial_path):
                continue
            
            for dirpath, _, filenames in os.walk(trial_path):
                folder_name = os.path.basename(dirpath).lower()
                if folder_name in self.classes:
                    label_idx = self.class_to_idx[folder_name]
                    for fname in filenames:
                        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                            self.data.append((
                                os.path.join(dirpath, fname), 
                                label_idx, 
                                trial_folder
                            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label, _ = self.data[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------
# 2. Configuration & Transforms
# -----------------------
data_dir = "/data/CIS2/JHU-collision-CAO1_annotated"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224
batch_size = 16
epochs = 50
model_save_path = "best_vit_collision_trial_split.pth"

train_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------
# 3. Data Split (Trial-Based)
# -----------------------
# Indexing trials first
full_ds_indexer = NestedImageDataset(root_dir=data_dir)
unique_trials = np.unique([d[2] for d in full_ds_indexer.data])
np.random.seed(42)
np.random.shuffle(unique_trials)

split_idx = max(1, int(0.8 * len(unique_trials)))
train_trial_names = unique_trials[:split_idx]
val_trial_names = unique_trials[split_idx:]

train_samples = [d for d in full_ds_indexer.data if d[2] in train_trial_names]
val_samples = [d for d in full_ds_indexer.data if d[2] in val_trial_names]

# Create specific Dataset instances for Train and Val
train_dataset = NestedImageDataset(root_dir=data_dir)
train_dataset.data = train_samples
train_dataset.transform = train_transform

val_dataset = NestedImageDataset(root_dir=data_dir)
val_dataset.data = val_samples
val_dataset.transform = val_transform

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"✅ Trial Split: {len(train_trial_names)} Train Trials | {len(val_trial_names)} Val Trials")
print(f"📊 Samples: {len(train_samples)} Train Images | {len(val_samples)} Val Images")

# -----------------------
# 4. Model Setup (ViT)
# -----------------------
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
model.heads.head = nn.Linear(model.heads.head.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# -----------------------
# 5. Training Loop
# -----------------------
best_val_loss = float('inf')

for epoch in range(epochs):
    # Training Phase
    model.train()
    running_loss, train_correct, train_total = 0.0, 0, 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(trainloader)
    train_acc = (train_correct / train_total) * 100

    # Validation Phase
    model.eval()
    val_loss, all_labels, all_preds = 0.0, [], []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_val_loss = val_loss / len(testloader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    val_acc = (all_labels == all_preds).mean() * 100
    val_prec = precision_score(all_labels, all_preds, zero_division=0)
    val_rec = recall_score(all_labels, all_preds, zero_division=0)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"  Precision:  {val_prec:.4f}     | Recall:    {val_rec:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"  ⭐ [SAVED] New best performance at Epoch {epoch+1}")
    
    scheduler.step(avg_val_loss)
