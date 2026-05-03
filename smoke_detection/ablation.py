"""
Smoke Detection — Ablation Study
Backbones: ResNet-50 | ViT-B/16 | EfficientNet-B0
Labels:    bad = smoke present | good = no smoke

Results for all three models are saved to ./ablation_output/
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ──────────────────────────────────────────────
# CONFIGURATION  ← edit these paths before running
# ──────────────────────────────────────────────

# Artificially introduced smoke images for training (5 datasets)
TRAIN_SMOKE_DIR = "/data/CIS2/smoke_detection/smokeTrain"

# Training CSV — annotated frames from training tissues only
TRAIN_CSV = "/data/CIS2/smoke_detection/annotations.csv"

# Validation CSV — clean frames from held-out tissues (no smoke frames here)
VAL_CSV       = "/data/CIS2/smoke_detection/annotations_val.csv"

# Validation smoke folder — smoke frames for validation (separate from training)
VAL_SMOKE_DIR = "/data/CIS2/smoke_detection/smokeVal"

# ──────────────────────────────────────────────
# HYPER-PARAMETERS
# ──────────────────────────────────────────────

STAGE1_EPOCHS = 10          # backbone frozen
STAGE2_EPOCHS = 10          # full fine-tune  (total = 20)
BATCH_SIZE    = 16
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR       = Path("./ablation_output")

# Per-backbone settings  (img_size, lr_head, lr_full)
BACKBONE_CFG = {
    "resnet50":      {"img_size": 224, "lr_head": 1e-3, "lr_full": 1e-4},
    "vit_b16":       {"img_size": 224, "lr_head": 1e-3, "lr_full": 5e-5},
    "efficientnet_b0": {"img_size": 224, "lr_head": 1e-3, "lr_full": 1e-4},
}

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_smoke_images(folder: str) -> list:
    """Recursively collect all images from the smokeTrain folder."""
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
        paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    if not paths:
        print(f"  Warning: No smoke images found in {folder} -- check TRAIN_SMOKE_DIR.")
    return paths


def load_csv(csv_path: str) -> tuple:
    """
    Read an annotations CSV and return (smoke_paths, clean_paths).
    Skips rows whose file no longer exists on disk.
    """
    import csv as _csv

    smoke_paths, clean_paths = [], []
    missing = 0

    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            p = row["path"]
            l = int(row["label"])
            if not os.path.exists(p):
                missing += 1
                continue
            if l == 1:
                smoke_paths.append(p)
            else:
                clean_paths.append(p)

    if missing:
        print(f"  Warning: Skipped {missing} missing files in {csv_path}.")

    print(f"  {Path(csv_path).name}: {len(smoke_paths)} smoke | {len(clean_paths)} clean")
    return smoke_paths, clean_paths


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────

class SmokeDataset(Dataset):
    """label 1 = bad (smoke) | label 0 = good (no smoke)"""

    def __init__(self, smoke_paths, good_paths, transform=None):
        self.samples = (
            [(p, 1) for p in smoke_paths] +
            [(p, 0) for p in good_paths]
        )
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def build_transforms(train: bool, img_size: int) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ──────────────────────────────────────────────
# MODEL FACTORY
# ──────────────────────────────────────────────

def build_model(backbone_name: str) -> nn.Module:
    if backbone_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.fc.in_features, 2),
        )

    elif backbone_name == "vit_b16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.heads.head.in_features, 2),
        )

    elif backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 2),
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    return model


def freeze_backbone(model: nn.Module, backbone_name: str):
    """Freeze everything except the classification head."""
    head_keys = {
        "resnet50":        "fc",
        "vit_b16":         "heads",
        "efficientnet_b0": "classifier",
    }
    head = head_keys[backbone_name]
    for name, param in model.named_parameters():
        if not name.startswith(head):
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ──────────────────────────────────────────────
# EPOCH RUNNER
# ──────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train and optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n        = len(loader.dataset)
    avg_loss = total_loss / n
    prec     = precision_score(all_labels, all_preds, zero_division=0)
    rec      = recall_score(all_labels, all_preds, zero_division=0)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, prec, rec, acc, all_labels, all_preds


# ──────────────────────────────────────────────
# TRAIN ONE BACKBONE
# ──────────────────────────────────────────────

def train_backbone(backbone_name, train_smoke, val_smoke, train_clean, val_clean):
    cfg      = BACKBONE_CFG[backbone_name]
    img_size = cfg["img_size"]
    lr_head  = cfg["lr_head"]
    lr_full  = cfg["lr_full"]

    print(f"\n{'═'*72}")
    print(f"  BACKBONE: {backbone_name.upper()}")
    print(f"  img_size={img_size}  lr_head={lr_head}  lr_full={lr_full}  device={DEVICE}")
    print(f"{'═'*72}")
    print(f"  Train: {len(train_smoke)} smoke | {len(train_clean)} clean")
    print(f"  Val:   {len(val_smoke)} smoke | {len(val_clean)} clean")

    train_ds = SmokeDataset(train_smoke, train_clean, build_transforms(True,  img_size))
    val_ds   = SmokeDataset(val_smoke,   val_clean,   build_transforms(False, img_size))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(backbone_name).to(DEVICE)

    # Weighted loss for class imbalance
    n_bad, n_good = len(train_smoke), len(train_clean)
    total   = n_bad + n_good
    weights = torch.tensor([total / (2 * n_good), total / (2 * n_bad)], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = {k: [] for k in ["train_loss", "val_loss", "precision", "recall", "accuracy"]}
    final_labels, final_preds = [], []

    header = (f"\n{'Epoch':>5} {'Stage':>6} | {'Train Loss':>10} {'Val Loss':>10} | "
              f"{'Precision':>9} {'Recall':>7} {'Accuracy':>9}")
    print(header)
    print("-" * 72)

    def log_row(epoch, stage, tr, vl, p, r, a):
        print(f"{epoch:>5} {stage:>6} | {tr:>10.4f} {vl:>10.4f} | "
              f"{p:>9.4f} {r:>7.4f} {a:>9.4f}")

    # ── Stage 1: frozen backbone ─────────────
    freeze_backbone(model, backbone_name)
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr_head, weight_decay=WEIGHT_DECAY)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS)

    for epoch in range(1, STAGE1_EPOCHS + 1):
        tr_loss, _, _, _, _, _            = run_epoch(model, train_loader, criterion, opt1, train=True)
        vl_loss, prec, rec, acc, vl, vp  = run_epoch(model, val_loader, criterion, train=False)
        sch1.step()
        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["precision"].append(prec);     history["recall"].append(rec)
        history["accuracy"].append(acc)
        final_labels, final_preds = vl, vp
        log_row(epoch, "S1", tr_loss, vl_loss, prec, rec, acc)

    # ── Stage 2: full fine-tune ──────────────
    unfreeze_all(model)
    opt2 = optim.AdamW(model.parameters(), lr=lr_full, weight_decay=WEIGHT_DECAY)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS)

    for epoch in range(STAGE1_EPOCHS + 1, STAGE1_EPOCHS + STAGE2_EPOCHS + 1):
        tr_loss, _, _, _, _, _            = run_epoch(model, train_loader, criterion, opt2, train=True)
        vl_loss, prec, rec, acc, vl, vp  = run_epoch(model, val_loader, criterion, train=False)
        sch2.step()
        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["precision"].append(prec);     history["recall"].append(rec)
        history["accuracy"].append(acc)
        final_labels, final_preds = vl, vp
        log_row(epoch, "S2", tr_loss, vl_loss, prec, rec, acc)

    # Save checkpoint
    ckpt = OUT_DIR / f"{backbone_name}_smoke_detector.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"\n💾  Saved → {ckpt}")

    return history, final_labels, final_preds


# ──────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────

COLORS = {
    "resnet50":        "#4C72B0",
    "vit_b16":         "#DD8452",
    "efficientnet_b0": "#55A868",
}

DISPLAY_NAMES = {
    "resnet50":        "ResNet-50",
    "vit_b16":         "ViT-B/16",
    "efficientnet_b0": "EfficientNet-B0",
}


def save_individual_plots(backbone_name, history, val_labels, val_preds):
    bdir = OUT_DIR / backbone_name
    bdir.mkdir(parents=True, exist_ok=True)
    total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    epochs = range(1, total_epochs + 1)
    color  = COLORS[backbone_name]
    name   = DISPLAY_NAMES[backbone_name]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{name} — Training Summary", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color=color)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color=color, linestyle="--")
    axes[0].axvline(STAGE1_EPOCHS + 0.5, color="gray", linestyle=":", label="Stage 2")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(epochs, history["precision"], label="Precision", color=color)
    axes[1].plot(epochs, history["recall"],    label="Recall",    color=color, linestyle="--")
    axes[1].axvline(STAGE1_EPOCHS + 0.5, color="gray", linestyle=":")
    axes[1].set_title("Precision & Recall (Val)"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(epochs, history["accuracy"], color=color)
    axes[2].axvline(STAGE1_EPOCHS + 0.5, color="gray", linestyle=":", label="Stage 2")
    axes[2].set_title("Accuracy (Val)"); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(bdir / "training_curves.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm   = confusion_matrix(val_labels, val_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["good (no smoke)", "bad (smoke)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(bdir / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"  📊  {name} plots saved → {bdir}/")


def save_comparison_plot(all_histories, all_labels, all_preds):
    """Side-by-side comparison of all three backbones."""
    total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    epochs = range(1, total_epochs + 1)

    metrics = ["train_loss", "val_loss", "precision", "recall", "accuracy"]
    titles  = ["Train Loss", "Val Loss", "Precision (Val)", "Recall (Val)", "Accuracy (Val)"]

    fig = plt.figure(figsize=(24, 10))
    gs  = gridspec.GridSpec(2, 6, figure=fig, hspace=0.4, wspace=0.35)

    # Top row: 5 metric plots
    for col, (metric, title) in enumerate(zip(metrics, titles)):
        ax = fig.add_subplot(gs[0, col])
        for bname, hist in all_histories.items():
            ax.plot(epochs, hist[metric],
                    label=DISPLAY_NAMES[bname],
                    color=COLORS[bname],
                    linestyle="--" if "val" in metric else "-")
        ax.axvline(STAGE1_EPOCHS + 0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.legend(fontsize=8)

    # Bottom row: one confusion matrix per backbone (cols 1–3, centred)
    cm_cols = [1, 3, 5]
    for (bname, vl, vp), col in zip(
        [(b, all_labels[b], all_preds[b]) for b in all_histories],
        cm_cols
    ):
        ax  = fig.add_subplot(gs[1, col])
        cm  = confusion_matrix(vl, vp)
        disp = ConfusionMatrixDisplay(cm, display_labels=["good", "bad"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(DISPLAY_NAMES[bname], fontsize=11)

    fig.suptitle("Ablation Study — Backbone Comparison", fontsize=16, fontweight="bold", y=1.01)
    out_path = OUT_DIR / "ablation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n🏆  Ablation comparison chart saved → {out_path}")


def print_final_summary(all_histories, all_labels, all_preds):
    print(f"\n{'═'*72}")
    print("  ABLATION SUMMARY — Final Validation Metrics")
    print(f"{'═'*72}")
    print(f"  {'Backbone':<20} {'Precision':>9} {'Recall':>7} {'Accuracy':>9} {'Val Loss':>10}")
    print(f"  {'-'*58}")
    for bname, hist in all_histories.items():
        vl    = all_labels[bname]
        vp    = all_preds[bname]
        prec  = precision_score(vl, vp, zero_division=0)
        rec   = recall_score(vl, vp, zero_division=0)
        acc   = accuracy_score(vl, vp)
        vloss = hist["val_loss"][-1]
        print(f"  {DISPLAY_NAMES[bname]:<20} {prec:>9.4f} {rec:>7.4f} {acc:>9.4f} {vloss:>10.4f}")
    print(f"{'═'*72}\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🖥️  Device: {DEVICE}")

    # Collect paths once — reused across all backbones
    print("\n📂 Collecting image paths ...")

    # Smoke: artificially introduced training datasets
    smoke_train_dir = collect_smoke_images(TRAIN_SMOKE_DIR)
    assert smoke_train_dir, "No smoke images found -- check TRAIN_SMOKE_DIR."

    # Training set: smokeTrain dir + CSV-labelled frames (training tissues only)
    train_smoke_csv, train_clean = load_csv(TRAIN_CSV)
    train_smoke = smoke_train_dir + train_smoke_csv

    # Validation set: entirely separate tissues from their own CSV
    val_smoke             = collect_smoke_images(VAL_SMOKE_DIR)
    _, val_clean          = load_csv(VAL_CSV)   # VAL_CSV contains clean frames only

    assert train_clean, "No clean training images found -- check TRAIN_CSV."
    assert val_smoke, "No val smoke images found -- check VAL_SMOKE_DIR."
    assert val_clean, "No val clean images found -- check VAL_CSV."

    print(f"  Train: {len(train_smoke)} smoke | {len(train_clean)} clean")
    print(f"  Val:   {len(val_smoke)} smoke | {len(val_clean)} clean")

    all_histories, all_labels, all_preds = {}, {}, {}

    for backbone in BACKBONE_CFG:
        set_seed(SEED)  # same seed per backbone for fair comparison
        hist, vl, vp = train_backbone(
            backbone, train_smoke, val_smoke, train_clean, val_clean
        )
        all_histories[backbone] = hist
        all_labels[backbone]    = vl
        all_preds[backbone]     = vp
        save_individual_plots(backbone, hist, vl, vp)

    save_comparison_plot(all_histories, all_labels, all_preds)
    print_final_summary(all_histories, all_labels, all_preds)
    print("✅  Ablation study complete.\n")


if __name__ == "__main__":
    main()
