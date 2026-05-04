import os
import csv
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from noise import pnoise2
    HAS_PERLIN = True
except ImportError:
    HAS_PERLIN = False
    print("[WARN] Install 'noise' for better realism: pip install noise")


# base noise

def _perlin_smoke_mask(h, w, scale=80, octaves=6):
    mask = np.zeros((h, w), dtype=np.float32)
    seed = random.randint(0, 10000)

    for i in range(h):
        for j in range(w):
            mask[i, j] = pnoise2(
                (i + seed) / scale,
                (j + seed) / scale,
                octaves=octaves,
                persistence=0.5,
                lacunarity=2.0,
            )

    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def _numpy_smoke_mask(h, w):
    mask = np.random.rand(h, w).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=20, sigmaY=20)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


# improved smoke generation

def generate_smoke_mask(h, w, density=1.0, spread=0.9):
    # Base noise
    if HAS_PERLIN:
        alpha = _perlin_smoke_mask(h, w)
    else:
        alpha = _numpy_smoke_mask(h, w)

    # Spread thresholding
    threshold = 1.0 - spread
    alpha = np.clip((alpha - threshold) / (1 - threshold + 1e-8), 0, 1)

    # multi scale structure
    alpha_large = cv2.GaussianBlur(alpha, (0, 0), sigmaX=10, sigmaY=10)
    alpha_small = cv2.GaussianBlur(alpha, (0, 0), sigmaX=3, sigmaY=3)
    alpha = 0.7 * alpha_large + 0.3 * alpha_small

    # nonlinear contrast
    alpha = alpha ** 0.2

    # directional flow
    for i in range(3):
        alpha = 0.8 * alpha + 0.2 * np.roll(alpha, shift=-5 * (i + 1), axis=0)

    # localized plume
    center_x = random.randint(w // 4, 3 * w // 4)
    center_y = random.randint(h // 4, 3 * h // 4)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    falloff = np.exp(-dist**2 / (2 * (0.35 * w) ** 2))
    alpha *= falloff

    # Final scaling
    alpha = (alpha * density * 255).clip(0, 255).astype(np.uint8)

    # Darker, realistic surgical smoke color
    gray = random.randint(170, 210)
    r = g = b = gray

    smoke = np.zeros((h, w, 4), dtype=np.uint8)
    smoke[:, :, 0] = r
    smoke[:, :, 1] = g
    smoke[:, :, 2] = b
    smoke[:, :, 3] = alpha

    return smoke


def apply_smoke(image_bgr, density=1.0, spread=0.9):
    h, w = image_bgr.shape[:2]
    smoke = generate_smoke_mask(h, w, density, spread)

    img = image_bgr.astype(np.float32)

    alpha = smoke[:, :, 3:4] / 255.0
    alpha = alpha ** 0.2
    core_mask = alpha > 0.45
    alpha[core_mask] = 1.0

    gray_val = random.randint(90,120)
    smoke_color = np.ones_like(img) * gray_val

    out = img * (1 * alpha) + smoke_color * alpha

    out_uint8 = out.clip(0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(out_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1 - 0.6 * alpha.squeeze())  # reduce saturation in smoky areas
    out_final = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out_final


# random parameters

def random_smoke_params():
    return dict(
        density=random.uniform(0.8, 1.4),
        spread=random.uniform(0.7, 1.0),
    )


# dataset builder

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_dataset(input_dir, output_dir, augment_factor=2):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    smoke_dir = output_dir / "smoke"
    clean_dir = output_dir / "no_smoke"

    smoke_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]

    records = []

    for img_path in tqdm(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Clean copy
        clean_out = clean_dir / img_path.name
        shutil.copy2(img_path, clean_out)
        records.append((str(clean_out), 0))

        # Smoke versions
        for i in range(augment_factor):
            params = random_smoke_params()
            smoky = apply_smoke(img, **params)

            out_name = f"{img_path.stem}_smoke_{i}{img_path.suffix}"
            out_path = smoke_dir / out_name

            cv2.imwrite(str(out_path), smoky)
            records.append((str(out_path), 1))

    # CSV labels
    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(records)

    print(f"\n Dataset complete: {csv_path}")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--augment_factor", type=int, default=2)

    args = parser.parse_args()

    build_dataset(args.input_dir, args.output_dir, args.augment_factor)
