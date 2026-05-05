"""
Smoke Detection — Inference & Benchmarking Script
Measures: latency (ms), confidence, precision, recall, accuracy, AUC-ROC
Works with all three backbones: ResNet-50 | ViT-B/16 | EfficientNet-B0

Usage:
    # Run all checkpoints against the val set:
    python inference.py --all

    # Run a single checkpoint:
    python inference.py --backbone resnet50 --checkpoint ./ablation_output/resnet50_smoke_detector.pth

    # Run on a single image:
    python inference.py --backbone resnet50 --checkpoint ./ablation_output/resnet50_smoke_detector.pth --image /path/to/image.png
"""

import os
import glob
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

# config

VAL_SMOKE_DIR = "/data/CIS2/smoke_detection/smokeVal"
VAL_CSV       = "/data/CIS2/smoke_detection/annotations_val.csv"

CHECKPOINTS = {
    "resnet50":        "./ablation_output/resnet50_smoke_detector.pth",
    "vit_b16":         "./ablation_output/vit_b16_smoke_detector.pth",
    "efficientnet_b0": "./ablation_output/efficientnet_b0_smoke_detector.pth",
    "efficientnetv2m": "./smoke_detector_output/smoke_detector_efficientnetv2m.pth",
}

IMG_SIZE     = 224
BATCH_SIZE   = 32         
NUM_WORKERS  = 4
WARMUP_RUNS  = 10         
LATENCY_RUNS = 50         
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR      = Path("./inference_output")

DISPLAY_NAMES = {
    "resnet50":        "ResNet-50",
    "vit_b16":         "ViT-B/16",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnetv2m": "EfficientNetV2-M",
}

COLORS = {
    "resnet50":        "#4C72B0",
    "vit_b16":         "#DD8452",
    "efficientnet_b0": "#55A868",
    "efficientnetv2m": "#C44E52",
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
    """label 1 = bad (smoke) | label 0 = good (no smoke)"""
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
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# model

def build_model(backbone_name: str) -> nn.Module:
    if backbone_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.fc.in_features, 2),
        )
    elif backbone_name == "vit_b16":
        model = models.vit_b_16(weights=None)
        model.heads = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.heads.head.in_features, 2),
        )
    elif backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 2),
        )
    elif backbone_name == "efficientnetv2m":
        model = models.efficientnet_v2_m(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 2),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return model


def load_model(backbone_name: str, ckpt_path: str) -> nn.Module:
    model = build_model(backbone_name)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# latency benchmark

def benchmark_latency(model: nn.Module) -> dict:
    """
    Measures single-image latency by repeated forward passes.
    Uses CUDA events for GPU timing (more accurate than time.perf_counter on GPU).
    """
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(dummy)

    latencies_ms = []

    if DEVICE.type == "cuda":
        for _ in range(LATENCY_RUNS):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            with torch.no_grad():
                _ = model(dummy)
            end_evt.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_evt.elapsed_time(end_evt))
    else:
        for _ in range(LATENCY_RUNS):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms":   float(np.mean(latencies_ms)),
        "std_ms":    float(np.std(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "min_ms":    float(np.min(latencies_ms)),
        "max_ms":    float(np.max(latencies_ms)),
        "fps":       1000.0 / float(np.mean(latencies_ms)),
    }


# full dataset inference

def run_inference(model: nn.Module, loader: DataLoader):
    """
    Returns parallel lists of true labels, predicted labels,
    and confidence scores (probability of 'bad'/smoke class).
    """
    all_labels, all_preds, all_confs = [], [], []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)         
            smoke_conf = probs[:, 1].cpu().numpy()   
            preds = logits.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_confs.extend(smoke_conf)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_confs),
    )


# single image inference

def infer_single(model: nn.Module, image_path: str) -> dict:
    transform = get_val_transform()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Latency for this single image
    if DEVICE.type == "cuda":
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start_evt.record()
            logits = model(tensor)
            end_evt.record()
        torch.cuda.synchronize()
        latency_ms = start_evt.elapsed_time(end_evt)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000

    probs      = F.softmax(logits, dim=1)[0]
    pred_idx   = probs.argmax().item()
    label_map  = {0: "good (no smoke)", 1: "bad (smoke)"}

    return {
        "prediction":    label_map[pred_idx],
        "confidence":    float(probs[pred_idx]) * 100,
        "prob_smoke":    float(probs[1]) * 100,
        "prob_no_smoke": float(probs[0]) * 100,
        "latency_ms":    latency_ms,
    }


# plotting

def save_individual_report(backbone_name, metrics, labels, preds, confs, latency):
    bdir = OUT_DIR / backbone_name
    bdir.mkdir(parents=True, exist_ok=True)
    name  = DISPLAY_NAMES.get(backbone_name, backbone_name)
    color = COLORS.get(backbone_name, "#333333")

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(f"{name} — Inference Report", fontsize=14, fontweight="bold")
    axes = fig.subplots(1, 3)

    # Confidence distribution
    smoke_confs   = confs[labels == 1]
    clean_confs   = confs[labels == 0]
    axes[0].hist(clean_confs, bins=40, alpha=0.6, label="good (no smoke)", color="#4C72B0")
    axes[0].hist(smoke_confs, bins=40, alpha=0.6, label="bad (smoke)",     color="#DD8452")
    axes[0].axvline(0.5, color="red", linestyle="--", linewidth=1, label="threshold=0.5")
    axes[0].set_title("Confidence Distribution")
    axes[0].set_xlabel("P(smoke)"); axes[0].set_ylabel("Count")
    axes[0].legend()

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, confs)
    auc = roc_auc_score(labels, confs)
    axes[1].plot(fpr, tpr, color=color, lw=2, label=f"AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    # Confusion matrix
    cm   = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["good", "bad"])
    disp.plot(ax=axes[2], colorbar=False, cmap="Blues")
    axes[2].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(bdir / "inference_report.png", dpi=150)
    plt.close()
    print(f"   Report saved → {bdir}/inference_report.png")


def save_comparison_report(all_results):
    """
    Multi-backbone comparison: latency bar chart, ROC overlay,
    confidence distributions, summary metrics table.
    """
    fig = plt.figure(figsize=(24, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Inference Benchmark — Backbone Comparison", fontsize=16, fontweight="bold")

    backbones = list(all_results.keys())
    names     = [DISPLAY_NAMES.get(b, b) for b in backbones]
    colors    = [COLORS.get(b, "#333") for b in backbones]

    # latency bar chart 
    ax_lat = fig.add_subplot(gs[0, 0])
    means  = [all_results[b]["latency"]["mean_ms"] for b in backbones]
    stds   = [all_results[b]["latency"]["std_ms"]  for b in backbones]
    bars   = ax_lat.bar(names, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax_lat.set_title("Single-Image Latency (ms)\nlower = faster")
    ax_lat.set_ylabel("Latency (ms)")
    for bar, val in zip(bars, means):
        ax_lat.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax_lat.tick_params(axis="x", labelrotation=15)

    # FPS bar chart 
    ax_fps = fig.add_subplot(gs[0, 1])
    fps_vals = [all_results[b]["latency"]["fps"] for b in backbones]
    bars2    = ax_fps.bar(names, fps_vals, color=colors, alpha=0.85)
    ax_fps.set_title("Throughput (FPS)\nhigher = faster")
    ax_fps.set_ylabel("Frames per Second")
    for bar, val in zip(bars2, fps_vals):
        ax_fps.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax_fps.tick_params(axis="x", labelrotation=15)

    #  ROC overlay 
    ax_roc = fig.add_subplot(gs[0, 2])
    for b, color in zip(backbones, colors):
        r      = all_results[b]
        fpr, tpr, _ = roc_curve(r["labels"], r["confs"])
        auc    = roc_auc_score(r["labels"], r["confs"])
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{DISPLAY_NAMES.get(b,b)} (AUC={auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_title("ROC Curves")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.legend(fontsize=8)

    #  Confidence distributions (stacked) 
    ax_conf = fig.add_subplot(gs[0, 3])
    for b, color in zip(backbones, colors):
        r = all_results[b]
        ax_conf.hist(r["confs"], bins=40, alpha=0.45,
                     label=DISPLAY_NAMES.get(b, b), color=color)
    ax_conf.axvline(0.5, color="red", linestyle="--", linewidth=1)
    ax_conf.set_title("Confidence Distribution\nP(smoke) across val set")
    ax_conf.set_xlabel("P(smoke)"); ax_conf.set_ylabel("Count")
    ax_conf.legend(fontsize=8)

    #  Summary metrics table 
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")

    col_labels = ["Backbone", "Accuracy", "Precision", "Recall",
                  "AUC-ROC", "Mean Latency (ms)", "Std (ms)", "FPS"]
    rows = []
    for b in backbones:
        r   = all_results[b]
        m   = r["metrics"]
        lat = r["latency"]
        auc = roc_auc_score(r["labels"], r["confs"])
        rows.append([
            DISPLAY_NAMES.get(b, b),
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{auc:.4f}",
            f"{lat['mean_ms']:.2f}",
            f"{lat['std_ms']:.2f}",
            f"{lat['fps']:.1f}",
        ])

    table = ax_tbl.table(cellText=rows, colLabels=col_labels,
                         cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Colour header row
    for col in range(len(col_labels)):
        table[0, col].set_facecolor("#2d2d2d")
        table[0, col].set_text_props(color="white", fontweight="bold")

    # Highlight best value per metric column (cols 1-7)
    metric_cols  = [1, 2, 3, 4, 7]   # higher is better
    latency_cols = [5, 6]             # lower is better

    for col_idx in metric_cols:
        vals = [float(rows[r][col_idx]) for r in range(len(rows))]
        best = max(vals)
        for row_idx, val in enumerate(vals):
            if val == best:
                table[row_idx + 1, col_idx].set_facecolor("#d4edda")

    for col_idx in latency_cols:
        vals = [float(rows[r][col_idx]) for r in range(len(rows))]
        best = min(vals)
        for row_idx, val in enumerate(vals):
            if val == best:
                table[row_idx + 1, col_idx].set_facecolor("#d4edda")

    plt.tight_layout()
    out_path = OUT_DIR / "inference_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison report saved → {out_path}")


# summary

def print_single_result(backbone_name, result):
    name = DISPLAY_NAMES.get(backbone_name, backbone_name)
    print(f"\n{'═'*55}")
    print(f"  {name}")
    print(f"{'═'*55}")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Confidence : {result['confidence']:.2f}%")
    print(f"  P(smoke)   : {result['prob_smoke']:.2f}%")
    print(f"  P(clean)   : {result['prob_no_smoke']:.2f}%")
    print(f"  Latency    : {result['latency_ms']:.2f} ms")


def print_benchmark_summary(all_results):
    print(f"\n{'═'*90}")
    print("  INFERENCE BENCHMARK SUMMARY")
    print(f"{'═'*90}")
    header = (f"  {'Backbone':<22} {'Acc':>6} {'Prec':>6} {'Recall':>6} {'AUC':>6} "
              f"{'Mean ms':>9} {'Std ms':>7} {'FPS':>7}")
    print(header)
    print(f"  {'-'*86}")
    for b, r in all_results.items():
        m   = r["metrics"]
        lat = r["latency"]
        auc = roc_auc_score(r["labels"], r["confs"])
        print(f"  {DISPLAY_NAMES.get(b,b):<22} "
              f"{m['accuracy']:>6.4f} {m['precision']:>6.4f} {m['recall']:>6.4f} {auc:>6.4f} "
              f"{lat['mean_ms']:>9.2f} {lat['std_ms']:>7.2f} {lat['fps']:>7.1f}")
    print(f"{'═'*90}\n")


# main

def parse_args():
    parser = argparse.ArgumentParser(description="Smoke detector inference & benchmark")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all",        action="store_true",
                      help="Benchmark all checkpoints in CHECKPOINTS dict")
    mode.add_argument("--backbone",   type=str,
                      choices=list(BACKBONE_CFG_KEYS := ["resnet50", "vit_b16",
                                                          "efficientnet_b0", "efficientnetv2m"]),
                      help="Single backbone to run")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth file (required when using --backbone)")
    parser.add_argument("--image",      type=str, default=None,
                        help="Path to a single image for quick inference (optional)")
    return parser.parse_args()


def run_backbone(backbone_name, ckpt_path, val_loader):
    print(f"\n  Loading {DISPLAY_NAMES.get(backbone_name, backbone_name)} "
          f"from {ckpt_path} …")
    model = load_model(backbone_name, ckpt_path)

    print("  Benchmarking latency …")
    latency = benchmark_latency(model)
    print(f"  Mean latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms  "
          f"({latency['fps']:.1f} FPS)")

    print("  Running full val-set inference …")
    labels, preds, confs = run_inference(model, val_loader)

    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
    }

    return {
        "metrics": metrics,
        "latency": latency,
        "labels":  labels,
        "preds":   preds,
        "confs":   confs,
        "model":   model,
    }


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Device : {DEVICE}")
    print(f"  Output : {OUT_DIR}\n")

    # single image mode
    if args.image:
        assert args.backbone and args.checkpoint, \
            "--image requires --backbone and --checkpoint"
        model = load_model(args.backbone, args.checkpoint)
        result = infer_single(model, args.image)
        print_single_result(args.backbone, result)
        return

    # ── Build val DataLoader (shared across backbones) ──
    print("Collecting validation image paths …")
    val_smoke          = collect_smoke_images(VAL_SMOKE_DIR)
    _, val_clean       = load_csv(VAL_CSV)   # VAL_CSV contains clean frames only
    assert val_smoke,  "No val smoke images — check VAL_SMOKE_DIR."
    assert val_clean,  "No val clean images — check VAL_CSV."
    print(f"  Val: {len(val_smoke)} smoke | {len(val_clean)} clean")

    from torch.utils.data import DataLoader as DL
    val_ds     = SmokeDataset(val_smoke, val_clean, get_val_transform())
    val_loader = DL(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

    # determine which backbones to run
    if args.all:
        targets = CHECKPOINTS
    else:
        assert args.checkpoint, "--backbone requires --checkpoint"
        targets = {args.backbone: args.checkpoint}

    # run each backbone
    all_results = {}
    for backbone_name, ckpt_path in targets.items():
        if not os.path.exists(ckpt_path):
            print(f"   Checkpoint not found, skipping: {ckpt_path}")
            continue
        result = run_backbone(backbone_name, ckpt_path, val_loader)
        all_results[backbone_name] = result
        save_individual_report(
            backbone_name,
            result["metrics"],
            result["labels"],
            result["preds"],
            result["confs"],
            result["latency"],
        )

    if not all_results:
        print("No valid checkpoints found. Exiting.")
        return

    if len(all_results) > 1:
        save_comparison_report(all_results)

    print_benchmark_summary(all_results)
    print("Inference complete.\n")


if __name__ == "__main__":
    main()
