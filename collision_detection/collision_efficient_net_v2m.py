import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

# 1. trial-aware dataset (ensures unseen trials for val)
class TrialAwareCollisionDataset(Dataset):
    def __init__(self, root_dir, trial_list, transform=None):
        self.transform = transform
        self.samples = []
        
        for trial in trial_list:
            trial_path = os.path.join(root_dir, trial)
            for label_name in ['good', 'bad']:
                label_path = os.path.join(trial_path, label_name)
                if os.path.exists(label_path):
                    # mapping good=0, bad=1
                    label_idx = 0 if label_name == 'good' else 1 
                    for fname in os.listdir(label_path):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                            self.samples.append((os.path.join(label_path, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 2. data setup & split
data_dir = "/data/CIS2/JHU-collision-CAO1_annotated"
input_size = 384 # optimized for v2-m

train_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# split trials so we dont have same folders in train and val
all_trials = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
train_trials, val_trials = train_test_split(all_trials, test_size=0.2, random_state=42)

train_subset = TrialAwareCollisionDataset(data_dir, train_trials, transform=train_transform)
val_subset = TrialAwareCollisionDataset(data_dir, val_trials, transform=val_transform)

# using smaller batch size since v2-m uses a lot of memory
trainloader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=8)
valloader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=8)

print(f"Trial Split Complete for EfficientNet-V2-M.")
print(f"Training: {len(train_trials)} trials ({len(train_subset)} images)")
print(f"Val:      {len(val_trials)} trials ({len(val_subset)} images)")

# 3. efficientnet-v2-m model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_V2_M_Weights.DEFAULT
model = efficientnet_v2_m(weights=weights)

# change classifier for 2 classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# 4. training loop
epochs = 50
best_val_loss = float('inf')
early_stop_patience = 10
patience_counter = 0
model_save_path = "best_effnet_v2m_collision.pth"

for epoch in range(epochs):
    # training phase
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(trainloader)
    avg_train_acc = train_correct / train_total

    # validation phase
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_val_loss = val_loss / len(valloader)
    avg_val_acc = val_correct / val_total
    
    # metrics for the validation set
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    scheduler.step(avg_val_loss)

    # output metrics
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"  TRAIN | Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2%}")
    print(f"  VAL   | Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.2%}")
    print(f"  METRICS (Val) | Precision: {prec:.4f} | Recall: {rec:.4f}")

    # saving & early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), model_save_path)
        print("  --> [SAVED] Best model updated.")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"\n[EARLY STOP] No improvement for {early_stop_patience} epochs.")
            break
