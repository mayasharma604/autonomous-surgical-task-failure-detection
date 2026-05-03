"""
Smoke Detection — Tissue Prediction Sorter
Feed any unseen tissue folder and the script will sort every frame
into predicted_clean/ and predicted_smoke/ subfolders.

Usage:
    python sort_tissue.py --tissue /path/to/tissue_folder --backbone resnet50
    python sort_tissue.py --tissue /path/to/tissue_folder --all
    python sort_tissue.py --tissue /path/to/tissue_folder --all --copy
"""

import os
import glob
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

TISSUE_DIR = ""          # ← leave empty, pass via --tissue at runtime

CHECKPOINTS = {
    "resnet50":        "./ablation_output/resnet50_smoke_detector.pth",
    "vit_b16":         "./ablation_output/vit_b16_smoke_detector.pth",
    "efficientnet_b0": "./ablation_output/efficientnet_b0_smoke_detector.pth",
    "efficientnetv2m": "./smoke_detector_output/smoke_detector_efficientnetv2m.pth",
}

# Glob patterns to search for images inside the tissue folder
# Covers both flat folders and the nested 3_resect/*/endoscope/ layout
IMAGE_PATTERNS = [
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.PNG",
    "**/*.JPG",
]

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_ROOT    = Path("./sorted_predictions")

DISPLAY_NAMES = {
    "resnet50":        "ResNet-50",
    "vit_b16":         "ViT-B/16",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnetv2m": "EfficientNetV2-M",
}

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────

def collect_images(tissue_dir: str) -> list:
    paths = []
    for pattern in IMAGE_PATTERNS:
        paths.extend(glob.glob(os.path.join(tissue_dir, pattern), recursive=True))
    # Deduplicate (different patterns can match same file)
    paths = sorted(set(paths))
    return paths


class TissueDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.paths     = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB")
        return self.transform(img), path


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────
# MODEL FACTORY
# ──────────────────────────────────────────────

def build_model(backbone_name: str) -> nn.Module:
    if backbone_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))

    elif backbone_name == "vit_b16":
        model = models.vit_b_16(weights=None)
        model.heads = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.heads.head.in_features, 2),
        )

    elif backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.2, inplace=True), nn.Linear(in_f, 2))

    elif backbone_name == "efficientnetv2m":
        model = models.efficientnet_v2_m(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3, inplace=True), nn.Linear(in_f, 2))

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    ckpt = CHECKPOINTS[backbone_name]
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


# ──────────────────────────────────────────────
# INFERENCE + SORTING
# ──────────────────────────────────────────────

def run_and_sort(backbone_name: str, loader: DataLoader,
                 out_dir: Path, use_copy: bool):
    """
    Runs inference and either symlinks or copies each image
    into predicted_clean/ or predicted_smoke/.
    Filenames get the confidence appended so you can sort by certainty.
    e.g.  frame_0042__conf97.3.png
    """
    clean_dir = out_dir / "predicted_clean"
    smoke_dir = out_dir / "predicted_smoke"
    clean_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    transfer = shutil.copy2 if use_copy else os.symlink

    n_clean = n_smoke = 0

    model = build_model(backbone_name)

    with torch.no_grad():
        for imgs, paths in loader:
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1).cpu()
            confs  = probs[:, 1].cpu()           # P(smoke)

            for path, pred, conf in zip(paths, preds, confs):
                src      = Path(path)
                conf_pct = conf.item() * 100
                stem     = src.stem
                suffix   = src.suffix

                # e.g.  frame_0001__smoke97.3.png  or  frame_0001__clean4.2.png
                if int(pred) == 1:
                    tag     = f"smoke{conf_pct:.1f}"
                    dst_dir = smoke_dir
                    n_smoke += 1
                else:
                    tag     = f"clean{100 - conf_pct:.1f}"
                    dst_dir = clean_dir
                    n_clean += 1

                dst = dst_dir / f"{stem}__{tag}{suffix}"

                # Avoid collisions if multiple subdirs have same filename
                counter = 1
                while dst.exists():
                    dst = dst_dir / f"{stem}__{tag}_{counter}{suffix}"
                    counter += 1

                if use_copy:
                    shutil.copy2(src, dst)
                else:
                    # Symlink needs absolute source path
                    os.symlink(src.resolve(), dst)

    return n_clean, n_smoke


# ──────────────────────────────────────────────
# SUMMARY PRINTOUT
# ──────────────────────────────────────────────

def print_result(backbone_name, n_clean, n_smoke, out_dir):
    total = n_clean + n_smoke
    name  = DISPLAY_NAMES.get(backbone_name, backbone_name)
    pct   = (n_smoke / total * 100) if total else 0
    print(f"  {name:<22}  "
          f"🟢 clean: {n_clean:>5}  🔴 smoke: {n_smoke:>5}  "
          f"({pct:.1f}% flagged as smoke)")
    print(f"  {'':22}  → {out_dir}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sort tissue frames into clean/smoke folders by model prediction"
    )
    parser.add_argument("--tissue",     type=str, default=TISSUE_DIR,
                        help="Path to tissue folder (searched recursively for images)")
    parser.add_argument("--backbone",   type=str, choices=list(CHECKPOINTS.keys()),
                        help="Single backbone to use")
    parser.add_argument("--all",        action="store_true",
                        help="Run all backbones (each gets its own output subfolder)")
    parser.add_argument("--copy",       action="store_true",
                        help="Copy images instead of symlinking (safer across drives)")
    parser.add_argument("--conf",       type=float, default=0.5,
                        help="Decision threshold for smoke (default 0.5)")
    return parser.parse_args()


def main():
    args = parse_args()

    tissue_dir = args.tissue.strip()
    if not tissue_dir:
        print("❌  No tissue path provided. Use --tissue /path/to/tissue_folder")
        return
    if not os.path.isdir(tissue_dir):
        print(f"❌  Directory not found: {tissue_dir}")
        return

    print(f"\n🖥️  Device  : {DEVICE}")
    print(f"📁  Tissue  : {tissue_dir}")
    print(f"🔁  Mode    : {'copy' if args.copy else 'symlink'}")
    print(f"📊  Threshold: P(smoke) > {args.conf:.2f}\n")

    # Collect images
    image_paths = collect_images(tissue_dir)
    if not image_paths:
        print("❌  No images found. Check the tissue path or IMAGE_PATTERNS in the script.")
        return
    print(f"  Found {len(image_paths)} images\n")

    dataset = TissueDataset(image_paths, get_transform())
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    # Tissue name for output folder
    tissue_name = Path(tissue_dir).name or "tissue"

    targets = list(CHECKPOINTS.keys()) if args.all else [args.backbone]

    print(f"{'─'*70}")
    for backbone_name in targets:
        ckpt = CHECKPOINTS.get(backbone_name, "")
        if not os.path.exists(ckpt):
            print(f"  ⚠️  Checkpoint not found, skipping: {backbone_name}  ({ckpt})")
            continue

        out_dir = OUT_ROOT / tissue_name / backbone_name
        n_clean, n_smoke = run_and_sort(backbone_name, loader, out_dir, args.copy)
        print_result(backbone_name, n_clean, n_smoke, out_dir)
        print()

    print(f"{'─'*70}")
    print("✅  Done.\n")
    print("Tip: filenames encode confidence, e.g.:")
    print("  frame_0001__smoke94.3.png  → predicted smoke, 94.3% confident")
    print("  frame_0002__clean99.1.png  → predicted clean, 99.1% confident\n")


if __name__ == "__main__":
    main()
