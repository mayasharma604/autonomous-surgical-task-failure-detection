import os
import random
import shutil
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

# residual block
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

# encoder
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

# decoder
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

# autoencoder
class ResidualAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

# dataset
class TensionDataset(Dataset):
    def __init__(self, tissue_dirs, root_dir, csv_dir, split="train", transform=None):
        self.samples = []

        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        valid_tissues = set([f"tissue_{i}" for i in range(1, 17)])

        for tissue in tissue_dirs:

            if tissue not in valid_tissues:
                continue

            csv_path = os.path.join(csv_dir, f"label_{tissue}.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                rel_path = row["filename"]
                label = row["label"]

                parts = rel_path.split("/")

                img_path = os.path.join(
                root_dir,
                tissue,
                parts[0],
                parts[1],     
                "endoscope",
                parts[2]
                )

                if not os.path.exists(img_path):
                    continue
                    
                if split == "train":
                    if label == "good":
                        self.samples.append((img_path, 0))
                else:
                    self.samples.append((img_path, 0 if label == "good" else 1))


        print(f"{split} dataset size: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label, path

# split tissues
def split_tissues(root_dir, train_ratio=0.7, val_ratio=0.15):
    tissues = [f"tissue_{i}" for i in range(1, 17)]
    random.shuffle(tissues)

    n = len(tissues)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return tissues[:train_end], tissues[train_end:val_end], tissues[val_end:]

# training
def train(
    root_dir="/data/virtuoso_cao_demo",
    csv_dir="/data/tensionData/TensionLabelsDuplicate",
    num_epochs=100,
    batch_size=8,
    lr=1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_tissues, val_tissues, test_tissues = split_tissues(root_dir)
    print("Train:", train_tissues)
    print("Val:", val_tissues)
    print("Test:", test_tissues)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = TensionDataset(train_tissues, root_dir, csv_dir, "train", transform)
    val_dataset   = TensionDataset(val_tissues, root_dir, csv_dir, "val", transform)
    test_dataset  = TensionDataset(test_tissues, root_dir, csv_dir, "test", transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ResidualAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    # train loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for x, _, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATION
        model.eval()
        val_errors = []
        val_labels = []

        with torch.no_grad():
            for x, labels, _ in val_loader:
                x = x.to(device)
                recon = model(x)
                err = torch.mean(torch.abs(x - recon), dim=(1,2,3))
                val_errors.extend(err.cpu().numpy())
                val_labels.extend(labels.numpy())

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

    torch.save(model.state_dict(), "rae_model.pth")
    print("Model saved.")

    # threshold selection
    good_errors = [e for e, l in zip(val_errors, val_labels) if l == 0]
    threshold = np.percentile(good_errors, 99)
    print("Anomaly threshold:", threshold)

    # test to folders and copy
    pred_good_dir = "predicted_good"
    pred_bad_dir  = "predicted_bad"
    os.makedirs(pred_good_dir, exist_ok=True)
    os.makedirs(pred_bad_dir,  exist_ok=True)

    model.eval()
    results = []

    with torch.no_grad():
        for x, label, path in test_loader:
            x = x.to(device)
            recon = model(x)

            err  = torch.mean(torch.abs(x - recon), dim=(1,2,3)).item()
            pred = 1 if err > threshold else 0

            results.append((err, pred, label.item(), path[0], recon.cpu()))

    # Copy frames into predicted_good / predicted_bad
    # Filename encodes tissue + original relative path so there are no collisions
    for err, pred, label, path, _ in results:
        # Build a flat filename: tissue_X__subfolder__frame.png
        parts   = path.split(os.sep)           # split full path into parts
        fname   = "__".join(parts[-5:])        # e.g. tissue_13__1_retract__session__endoscope__frame.png
        dst_dir = pred_bad_dir if pred == 1 else pred_good_dir
        dst     = os.path.join(dst_dir, fname)
        shutil.copy2(path, dst)

    print(f"\nFrames copied → '{pred_good_dir}/' and '{pred_bad_dir}/'")

    # Summary counts
    n_gt_good   = sum(1 for _, _, label, _, _ in results if label == 0)
    n_gt_bad    = sum(1 for _, _, label, _, _ in results if label == 1)
    n_pred_good = sum(1 for _, pred, _, _, _ in results if pred == 0)
    n_pred_bad  = sum(1 for _, pred, _, _, _ in results if pred == 1)

    print(f"\n{'':=<40}")
    print(f"  Ground Truth  — good: {n_gt_good:>5} | bad: {n_gt_bad:>5}")
    print(f"  Predicted     — good: {n_pred_good:>5} | bad: {n_pred_bad:>5}")
    print(f"{'':=<40}")

    # top 5 anomalies visual
    results.sort(reverse=True, key=lambda t: t[0])

    print("\nTop 5 anomalies:")
    for err, pred, label, path, recon in results[:5]:
        print(f"Err={err:.4f} | Pred={pred} | GT={label} | {path}")

        img_orig  = Image.open(path)
        img_recon = transforms.ToPILImage()(recon.squeeze())

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img_orig);  axs[0].set_title("Original");                  axs[0].axis("off")
        axs[1].imshow(img_recon); axs[1].set_title(f"Recon (err={err:.4f})");    axs[1].axis("off")
        plt.show()


# main
if __name__ == "__main__":
    train()
