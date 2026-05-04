import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 1. path configuration
BASE_DATA_DIR = "/data/CIS2/JHU-collision-CAO1_annotated"
CHECKPOINT_DIR = "./checkpoints_collision"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resnet-18 
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

# 3. trial-aware indexing
def get_collision_files(base_path):
    all_trials = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    train_trial_names, val_trial_names = train_test_split(all_trials, test_size=0.2, random_state=42)
    
    def collect_from_trials(trial_list):
        data = []
        for trial in trial_list:
            trial_path = os.path.join(base_path, trial)
            for label_name in ['good', 'bad']:
                label_path = os.path.join(trial_path, label_name)
                if os.path.exists(label_path):
                    # 0 is good, 1 is bad
                    label_idx = 0 if label_name == 'good' else 1
                    for img_name in os.listdir(label_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            data.append((os.path.join(label_path, img_name), label_idx))
        return data

    return collect_from_trials(train_trial_names), collect_from_trials(val_trial_names)

# 4. dataset & loaders
class CollisionDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
    def __len__(self): return len(self.image_list)
    def __getitem__(self, idx):
        img_path, label = self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

train_files, val_files = get_collision_files(BASE_DATA_DIR)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = CollisionDataset(train_files, transform=transform)
val_ds = CollisionDataset(val_files, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=8)

# 5. training loop
model = ResNet18().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float('inf')
patience = 5
counter = 0

print(f"Training on {len(train_files)} images | Validating on {len(val_files)} images...")

for epoch in range(20):
    # training phase
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_correct / train_total

    # validation phase
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().view(-1, 1)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels).item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_correct / val_total
    
    # get validation metrics
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # print metrics
    print(f"\nEpoch {epoch+1}")
    print(f"TRAIN | Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2%}")
    print(f"VAL   | Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.2%}")
    print(f"METRICS (Val) | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # save best and check early stop
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_collision_resnet.pth"))
        print(">> saved best model")
    else:
        counter += 1
        if counter >= patience:
            print(f"\nstopping early, no improvement for {patience} epochs.")
            break
