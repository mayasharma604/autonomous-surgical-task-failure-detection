"""
Smoke Detection — Misclassification Viewer
Saves grid images of every misclassified frame per backbone,
split by error type: false positives and false negatives.

Usage:
    # All backbones:
    python misclassifications.py --all

    # Single backbone:
    python misclassifications.py --backbone efficientnetv2m \
        --checkpoint ./smoke_detector_output/smoke_detector_efficientnetv2m.pth
"""

import os
import glob
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# config

VAL_SMOKE_DIR = "/data/CIS2/smoke_detection/smokeVal"
VAL_CSV       = "/data/CIS2/smoke_detection/annotations_val.csv"  # clean frames only

CHECKPOINTS = {
    "resnet50":        "./ablation_output/resnet50_smoke_detector.pth",
    "vit_b16":         "./ablation_output/vit_b16_smoke_detector.pth",
    "efficientnet_b0": "./ablation_output/efficientnet_b0_smoke_detector.pth",
    "efficientnetv2m": "./smoke_detector_output/smoke_detector_efficientnetv2m.pth",
}

IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_WORKERS   = 4
THUMB_SIZE    = 160           # px per thumbnail in the grid
MAX_PER_GRID  = 100           # cap grid size so files stay manageable
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR       = Path("./inference_output")

DISPLAY_NAMES = {
    "resnet50":        "ResNet-50",
    "vit_b16":         "ViT-B/16",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnetv2m": "EfficientNetV2-M",
}

# data helpers

def collect_smoke_images(folder: str) -> list:
    """Collect all images from a smoke folder recursively."""
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
        paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    if not paths:
        print(f"  Warning: No smoke images found in {folder}")
    return paths


def load_csv(csv_path: str) -> tuple:
    """
    Read an annotations CSV, return (smoke_paths, clean_paths).
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


class SmokeDataset(Dataset):
    """Returns (tensor, label, path) so we can trace misclassifications back to disk."""
    def __init__(self, smoke_paths, good_paths, transform):
        self.samples = [(p, 1) for p in smoke_paths] + [(p, 0) for p in good_paths]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long), path


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# model training

def build_model(backbone_name):
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

    model.load_state_dict(torch.load(CHECKPOINTS[backbone_name], map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


# inference for per image labels

def run_inference(model, loader):
    """
    Returns a list of dicts, one per image:
      path, true_label, pred_label, conf_smoke, correct
    """
    records = []
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1).cpu()
            confs  = probs[:, 1].cpu()         

            for path, true, pred, conf in zip(paths, labels, preds, confs):
                records.append({
                    "path":       path,
                    "true_label": int(true),
                    "pred_label": int(pred),
                    "conf_smoke": float(conf),
                    "correct":    int(true) == int(pred),
                })

    return records


# grid for misclassifications

def load_thumb(path):
    """Load image and resize to thumbnail. Returns PIL Image."""
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        return img
    except Exception:
        # Return a grey placeholder if image fails to load
        return Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), color=(180, 180, 180))


def build_grid(records, error_type, backbone_name, out_dir):
    """
    error_type: 'fp' (false positive: predicted smoke, actually clean)
                'fn' (false negative: predicted clean, actually smoke)
    """
    if error_type == "fp":
        subset = [r for r in records if r["pred_label"] == 1 and r["true_label"] == 0]
        title_str = "False Positives — predicted SMOKE, actually CLEAN"
        border_color = "#e74c3c"  
    else:
        subset = [r for r in records if r["pred_label"] == 0 and r["true_label"] == 1]
        title_str = "False Negatives — predicted CLEAN, actually SMOKE"
        border_color = "#e67e22" 

    name = DISPLAY_NAMES.get(backbone_name, backbone_name)
    total = len(subset)

    if total == 0:
        print(f"  No {error_type.upper()}s for {name}")
        return

    # Sort by confidence distance from decision boundary (most wrong first)
    if error_type == "fp":
        subset.sort(key=lambda r: r["conf_smoke"], reverse=True) 
    else:
        subset.sort(key=lambda r: r["conf_smoke"])              

    capped = subset[:MAX_PER_GRID]
    n      = len(capped)
    ncols  = min(10, n)
    nrows  = math.ceil(n / ncols)

    dpi       = 100
    cell_px   = THUMB_SIZE + 30      # thumb + space for confidence text
    fig_w     = ncols * cell_px / dpi + 1
    fig_h     = nrows * cell_px / dpi + 1.2   # extra for title

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w * dpi / 96, fig_h * dpi / 96),
                             dpi=dpi)
    axes = np.array(axes).reshape(-1)   # flatten for easy indexing

    fig.suptitle(
        f"{name}  |  {title_str}\n"
        f"{n} shown  (of {total} total)  —  sorted by model confidence",
        fontsize=10, fontweight="bold", y=1.01
    )

    for i, ax in enumerate(axes):
        if i < n:
            r     = capped[i]
            thumb = load_thumb(r["path"])
            ax.imshow(thumb)

            # Confidence label below image
            conf_pct = r["conf_smoke"] * 100
            ax.set_xlabel(f"P(smoke)={conf_pct:.1f}%", fontsize=6, labelpad=2)

            # Coloured border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)

            # Tissue ID from path if parseable
            try:
                tissue = [p for p in Path(r["path"]).parts if "tissue_" in p][0]
                ax.set_title(tissue, fontsize=5, pad=1)
            except IndexError:
                ax.set_title(Path(r["path"]).name[:20], fontsize=5, pad=1)

            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.axis("off")

    plt.tight_layout()
    fname = out_dir / f"misclassified_{error_type}.png"
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved {n}/{total} {error_type.upper()}s → {fname}")


# text logging

def save_misclassification_log(records, backbone_name, out_dir):
    fps = [r for r in records if r["pred_label"] == 1 and r["true_label"] == 0]
    fns = [r for r in records if r["pred_label"] == 0 and r["true_label"] == 1]

    log_path = out_dir / "misclassifications.txt"
    with open(log_path, "w") as f:
        name = DISPLAY_NAMES.get(backbone_name, backbone_name)
        f.write(f"Misclassification Log — {name}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"FALSE POSITIVES ({len(fps)}) — predicted SMOKE, actually CLEAN\n")
        f.write("-" * 70 + "\n")
        fps_sorted = sorted(fps, key=lambda r: r["conf_smoke"], reverse=True)
        for r in fps_sorted:
            f.write(f"  P(smoke)={r['conf_smoke']*100:5.1f}%  {r['path']}\n")

        f.write(f"\nFALSE NEGATIVES ({len(fns)}) — predicted CLEAN, actually SMOKE\n")
        f.write("-" * 70 + "\n")
        fns_sorted = sorted(fns, key=lambda r: r["conf_smoke"])
        for r in fns_sorted:
            f.write(f"  P(smoke)={r['conf_smoke']*100:5.1f}%  {r['path']}\n")

    print(f" Full path log → {log_path}")


# summary panel

def save_summary_panel(records, backbone_name, out_dir):
    """
    Single-page summary: counts, confidence histogram of errors,
    and breakdown by tissue.
    """
    name = DISPLAY_NAMES.get(backbone_name, backbone_name)
    fps  = [r for r in records if r["pred_label"] == 1 and r["true_label"] == 0]
    fns  = [r for r in records if r["pred_label"] == 0 and r["true_label"] == 1]
    total_errors = len(fps) + len(fns)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{name} — Misclassification Summary  "
                 f"({total_errors} errors / {len(records)} images)",
                 fontsize=13, fontweight="bold")

    # error type bar
    axes[0].bar(["False Positives\n(clean→smoke)", "False Negatives\n(smoke→clean)"],
                [len(fps), len(fns)],
                color=["#e74c3c", "#e67e22"], alpha=0.85, edgecolor="white")
    axes[0].set_title("Error Counts")
    axes[0].set_ylabel("Number of images")
    for i, v in enumerate([len(fps), len(fns)]):
        axes[0].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

    # confidence histogram of errors
    fp_confs = [r["conf_smoke"] for r in fps]
    fn_confs = [r["conf_smoke"] for r in fns]
    if fp_confs:
        axes[1].hist(fp_confs, bins=20, color="#e74c3c", alpha=0.7,
                     label=f"FP (n={len(fps)})")
    if fn_confs:
        axes[1].hist(fn_confs, bins=20, color="#e67e22", alpha=0.7,
                     label=f"FN (n={len(fns)})")
    axes[1].axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Confidence of Misclassified Images")
    axes[1].set_xlabel("P(smoke)"); axes[1].set_ylabel("Count")
    axes[1].legend()

    # errors per tissue
    tissue_errors = {}
    for r in fps + fns:
        try:
            tissue = [p for p in Path(r["path"]).parts if "tissue_" in p][0]
        except IndexError:
            tissue = "unknown"
        tissue_errors[tissue] = tissue_errors.get(tissue, 0) + 1

    if tissue_errors:
        tissues = sorted(tissue_errors.keys())
        counts  = [tissue_errors[t] for t in tissues]
        axes[2].barh(tissues, counts, color="#7f8c8d", alpha=0.8)
        axes[2].set_title("Errors per Tissue")
        axes[2].set_xlabel("Error count")
        axes[2].invert_yaxis()
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "No tissue info\nfound in paths",
                     ha="center", va="center", transform=axes[2].transAxes)

    plt.tight_layout()
    plt.savefig(out_dir / "misclassification_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Summary panel → {out_dir / 'misclassification_summary.png'}")


# main

def parse_args():
    parser = argparse.ArgumentParser(description="Smoke detector misclassification viewer")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all",      action="store_true",
                      help="Run all backbones in CHECKPOINTS")
    mode.add_argument("--backbone", type=str,
                      choices=list(CHECKPOINTS.keys()),
                      help="Single backbone to inspect")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (only with --backbone)")
    return parser.parse_args()


def process_backbone(backbone_name, val_loader):
    print(f"\n  {'─'*60}")
    print(f"  {DISPLAY_NAMES.get(backbone_name, backbone_name)}")
    print(f"  {'─'*60}")

    model   = build_model(backbone_name)
    records = run_inference(model, val_loader)

    errors  = [r for r in records if not r["correct"]]
    fps     = [r for r in records if r["pred_label"] == 1 and r["true_label"] == 0]
    fns     = [r for r in records if r["pred_label"] == 0 and r["true_label"] == 1]
    print(f"    Total errors : {len(errors)} / {len(records)}")
    print(f"    False positives (clean→smoke) : {len(fps)}")
    print(f"    False negatives (smoke→clean) : {len(fns)}")

    bdir = OUT_DIR / backbone_name
    bdir.mkdir(parents=True, exist_ok=True)

    build_grid(records, "fp", backbone_name, bdir)
    build_grid(records, "fn", backbone_name, bdir)
    save_summary_panel(records, backbone_name, bdir)
    save_misclassification_log(records, backbone_name, bdir)


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Device : {DEVICE}")

    # Override checkpoint if provided
    if args.backbone and args.checkpoint:
        CHECKPOINTS[args.backbone] = args.checkpoint

    # Build val loader
    print("\n Collecting validation image paths …")
    val_smoke          = collect_smoke_images(VAL_SMOKE_DIR)
    _, val_clean       = load_csv(VAL_CSV)
    assert val_smoke,  "No val smoke images — check VAL_SMOKE_DIR."
    assert val_clean,  "No val clean images — check VAL_CSV."
    print(f"  Val: {len(val_smoke)} smoke | {len(val_clean)} clean")

    val_ds     = SmokeDataset(val_smoke, val_clean, get_val_transform())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    targets = list(CHECKPOINTS.keys()) if args.all else [args.backbone]

    for backbone_name in targets:
        ckpt = CHECKPOINTS[backbone_name]
        if not os.path.exists(ckpt):
            print(f"\n    Checkpoint not found, skipping: {ckpt}")
            continue
        process_backbone(backbone_name, val_loader)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
 
