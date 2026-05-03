import os
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# =========================
# 1. Residual Block
# =========================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# =========================
# 2. Encoder
# =========================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResBlock(3, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 512, stride=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# =========================
# 3. Decoder
# =========================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.rb1 = ResBlock(256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.rb2 = ResBlock(128, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.rb3 = ResBlock(64, 64)
        self.up4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

    def forward(self, x):
        x = self.rb1(self.up1(x))
        x = self.rb2(self.up2(x))
        x = self.rb3(self.up3(x))
        x = self.up4(x)
        return x

# =========================
# 4. Residual Autoencoder
# =========================
class ResidualAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# =========================
# 5. Dataset
# =========================




class CollisionDataset(Dataset):
    def __init__(self, tissue_dirs, root_dir, transform=None):
        self.image_paths = []
        task_dirs = ["1_retract", "3_resect"]
        for tissue in tissue_dirs:
            for task in task_dirs:
                task_path = os.path.join(root_dir, tissue, task)
                if not os.path.exists(task_path):
                    continue
                for trial in os.listdir(task_path):
                    endoscope_path = os.path.join(task_path, trial, "endoscope")
                    if not os.path.exists(endoscope_path):
                        continue
                    for img_name in os.listdir(endoscope_path):
                        if img_name.endswith(".png"):
                            self.image_paths.append(os.path.join(endoscope_path, img_name))
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.image_paths[idx]

# =========================
# 6. Train / Validation Split
# =========================
def split_tissues(root_dir, train_ratio=0.7, val_ratio=0.15):
    tissues = [d for d in os.listdir(root_dir) if d.startswith("tissue_")]
    tissues = sorted(tissues)
    random.shuffle(tissues)
    n = len(tissues)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train = tissues[:train_end]
    val = tissues[train_end:val_end]
    test = tissues[val_end:]
    return train, val, test

# =========================
# 7. Training Loop
# =========================
def train(root_dir="/data/virtuoso_cao_demo", num_epochs=50, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_tissues, val_tissues, test_tissues = split_tissues(root_dir)
    print(f"Train tissues: {len(train_tissues)}, Val tissues: {len(val_tissues)}, Test tissues: {len(test_tissues)}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = CollisionDataset(train_tissues, root_dir, transform)
    val_dataset = CollisionDataset(val_tissues, root_dir, transform)
    test_dataset = CollisionDataset(test_tissues, root_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = ResidualAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    train_losses, val_losses = [], []

    # =========================
    # Training
    # =========================
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat = model(x)
                val_loss += criterion(x_hat, x).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "rae_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Plot loss curves
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    print("Saved loss curve as loss_curve.png")

    model.load_state_dict(torch.load("rae_best.pth"))
    print("Loaded best model for anomaly detection")

    model.eval()
    val_errors = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            recon = model(x)
            error = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
            val_errors.extend(error.cpu().numpy())

    val_errors = torch.tensor(val_errors)
    threshold = torch.quantile(val_errors, 0.95).item()
    print(f"Anomaly threshold (95th percentile): {threshold:.6f}")

    # ← REPLACE the old torch.save with this
    torch.save({
        "model_state": model.state_dict(),
        "threshold": threshold,
        "val_mean": val_errors.mean().item(),
        "val_std": val_errors.std().item()
    }, "rae_best.pth")
    print("Saved best model + threshold to rae_best.pth")

    # =========================
    # 8. Anomaly detection + visualization
    # =========================
    model.eval()
    anomaly_scores = []
    with torch.no_grad():
        for x, paths in test_loader:
            x = x.to(device)
            recon = model(x)
            error = torch.mean((x - recon) ** 2, dim=(1,2,3))
            anomaly_scores.append((error.item(), paths[0]))

    # Sort by error descending
    anomaly_scores.sort(reverse=True, key=lambda t: t[0])
    print("Top 5 anomaly frames:")
    for score, path in anomaly_scores[:5]:
        print(f"{score:.5f} - {path}")
        img_orig = Image.open(path)
        img_recon = transforms.ToPILImage()(recon.squeeze().cpu())
        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        axs[0].imshow(img_orig)
        axs[0].set_title("Original")
        axs[0].axis("off")
        axs[1].imshow(img_recon)
        axs[1].set_title(f"Reconstruction (err={score:.5f})")
        axs[1].axis("off")
        plt.show()

if __name__ == "__main__":
    train()

'''

import os
import random
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# =========================
# 1. Residual Block
# =========================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        return self.relu(out)


# =========================
# 2. Encoder
# =========================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = ResBlock(3, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 512, stride=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# =========================
# 3. Decoder
# =========================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.rb1 = ResBlock(256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.rb2 = ResBlock(128, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.rb3 = ResBlock(64, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

    def forward(self, x):
        x = self.rb1(self.up1(x))
        x = self.rb2(self.up2(x))
        x = self.rb3(self.up3(x))
        x = self.up4(x)
        return x


# =========================
# 4. Autoencoder
# =========================
class ResidualAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# =========================
# 5. Dataset Loader
# =========================
class RetractDataset(Dataset):
    def __init__(self, root_dir, num_tissues=30):
        self.image_paths = []
        
        tissues = [d for d in os.listdir(root_dir) if d.startswith("tissue_")]
        tissues = sorted(tissues)
        
        selected_tissues = random.sample(tissues, num_tissues)
        
        for tissue in selected_tissues:
            retract_path = os.path.join(root_dir, tissue, "1_retract")
            
            if not os.path.exists(retract_path):
                continue
            
            for trial in os.listdir(retract_path):
                endoscope_path = os.path.join(retract_path, trial, "endoscope")
                
                if not os.path.exists(endoscope_path):
                    continue
                
                for img_name in os.listdir(endoscope_path):
                    if img_name.endswith(".png"):
                        self.image_paths.append(
                            os.path.join(endoscope_path, img_name)
                        )

        print(f"Loaded {len(self.image_paths)} images")

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img


# =========================
# 6. Training Setup
# =========================
def train():

    root_dir = "/data/virtuoso_cao_demo"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ResidualAutoencoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    dataset = RetractDataset(root_dir, num_tissues=20)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    num_epochs = 10

    # =========================
    # Training Loop
    # =========================
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for x in dataloader:
            x = x.to(device)
            
            x_hat = model(x)
            loss = criterion(x_hat, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "rae_model.pth")
    print("Model saved as rae_model.pth")

    # =========================
    # 7. Anomaly Scoring Demo
    # =========================
    model.eval()

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            recon = model(x)
            
            error = torch.mean((x - recon)**2, dim=(1,2,3))
            print("Sample anomaly scores:", error[:5])
            break


# =========================
# 8. Run Script
# =========================
if __name__ == "__main__":
    train()

'''
