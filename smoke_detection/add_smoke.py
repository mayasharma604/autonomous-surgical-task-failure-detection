"""
add_smoke.py
------------
Adds artificial smoke/fog augmentation to endoscopic PNG images.
Builds a mask of non-black pixels so smoke is only applied to
actual tissue content, never to the black border/background.

Usage:
    python add_smoke.py --input_dir /path/to/pngs --output_dir /path/to/output
    python add_smoke.py --input_dir /data/virtuoso_cao_demo/tissue_1/3_resect/20251215-185458-408655/endoscope --output_dir /data/CIS2/smoke_detection/debug_smoke
    python add_smoke.py --input_dir /path/to/pngs --output_dir /path/to/output --copies 3
    python add_smoke.py --input_dir /path/to/pngs --output_dir /path/to/output --debug
    python add_smoke.py --input_dir /path/to/pngs --output_dir /path/to/output --black_threshold 20
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    print("albumentations not found. Install it with: pip install albumentations")
    sys.exit(1)


# augmentation pipeline

# Check albumentations version to handle API differences
import albumentations as _a_check
import importlib.metadata
_albu_version = tuple(int(x) for x in importlib.metadata.version("albumentations").split(".")[:2])

if _albu_version >= (1, 4):
    smoke_aug = A.Compose([
        A.RandomFog(
            fog_coef_range=(0.6, 0.61),
            alpha_coef=0.18,
            p=1.0,
        ),
    ])
else:
    smoke_aug = A.Compose([
        A.RandomFog(
            fog_coef_lower=0.6,
            fog_coef_upper=0.61,
            alpha_coef=0.2,
            p=1.0,
        ),
    ])


# mask detection

def build_content_mask(image: np.ndarray, black_threshold: int = 15,
                        erode_px: int = 20) -> np.ndarray:
    """
    Return a binary mask (255) for every pixel that is NOT near-black,
    then erode inward by erode_px pixels so the augmentation region sits
    comfortably inside the circular boundary with no border bleed.

    black_threshold : raise to 25-40 if border is dark grey not pure black.
    erode_px        : how many pixels to pull the mask inward from the edge.
                      Increase if smoke still spills onto the border.
    """
    not_black = np.any(image > black_threshold, axis=2)
    mask = (not_black * 255).astype(np.uint8)

    # Close small holes inside the tissue area
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    # Erode inward so the augmentation boundary is safely inside the circle
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,
                                                              erode_px * 2 + 1))
    mask = cv2.erode(mask, erode_k, iterations=1)

    return mask


# core functions

def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def apply_smoke_masked(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    1. Erode the mask then crop its tight bounding box — RandomFog only
       ever sees pixels well inside the circle, never the black border.
    2. Augment the crop in isolation.
    3. Feather-blend back using the eroded mask so the smoke fades out
       naturally before it gets anywhere near the actual edge.
    """
    # tight bounding box of the eroded mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # augment only the cropped interior
    crop     = image[y0:y1, x0:x1]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    aug_rgb  = smoke_aug(image=crop_rgb)["image"]
    aug_crop = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

    # feather the eroded mask, smoke fades to zero well before the border
    feathered  = cv2.GaussianBlur(mask, (51, 51), sigmaX=25, sigmaY=25)
    alpha_full = feathered.astype(np.float32) / 255.0
    alpha_crop = alpha_full[y0:y1, x0:x1]
    alpha_3ch  = cv2.merge([alpha_crop, alpha_crop, alpha_crop])

    # blend
    blended = (aug_crop.astype(np.float32) * alpha_3ch
               + crop.astype(np.float32)   * (1.0 - alpha_3ch))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # paste back
    result = image.copy()
    result[y0:y1, x0:x1] = blended
    return result


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def process_folder(
    input_dir: Path,
    output_dir: Path | None,
    copies: int,
    force: bool,
    debug: bool,
    black_threshold: int,
    prefix: str = "",
    erode_px: int = 20,
) -> None:
    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Found {len(png_files)} PNG(s). Generating {copies} augmented copy/copies each.")
    print(f"Black threshold: {black_threshold}\n")

    debug_dir = (output_dir or input_dir) / "debug_masks" if debug else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug masks will be saved to: {debug_dir}\n")

    success, skipped, errors = 0, 0, 0

    for img_path in png_files:
        try:
            image = load_image(img_path)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            errors += 1
            continue

        mask = build_content_mask(image, black_threshold=black_threshold,
                                    erode_px=erode_px)

        if debug_dir:
            # Save the raw mask so you can see exactly which pixels are included
            cv2.imwrite(str(debug_dir / img_path.name), mask)

        for i in range(1, copies + 1):
            suffix = f"_smoke_{i:02d}" if copies > 1 else "_smoke"
            stem = f"{prefix}{img_path.stem}" if prefix else img_path.stem
            out_path = (output_dir or img_path.parent) / f"{stem}{suffix}.png"

            if out_path.exists() and not force:
                print(f"  [SKIP]  {out_path.name} (already exists; use --force to overwrite)")
                skipped += 1
                continue

            augmented = apply_smoke_masked(image, mask)
            save_image(augmented, out_path)
            print(f"  [OK]    {img_path.name}  →  {out_path.name}")
            success += 1

    print(f"\nDone. {success} saved, {skipped} skipped, {errors} errors.")


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add smoke to endoscopic images — skips any pixel that is near-black."
    )
    parser.add_argument("--input_dir", "-i", required=True, type=Path)
    parser.add_argument("--output_dir", "-o", type=Path, default=None)
    parser.add_argument("--copies", "-n", type=int, default=1)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Save the computed mask per image to debug_masks/ for inspection.")
    parser.add_argument("--black_threshold", type=int, default=15,
                        help="Pixels where ALL channels are below this are treated as black "
                             "(default: 15). Raise to 25-40 if your border is dark grey.")
    parser.add_argument("--erode_px", type=int, default=20,
                        help="Pixels to erode mask inward from the circular edge (default: 20). "
                             "Increase if smoke still spills onto the border.")
    parser.add_argument("--prefix", type=str, default="",
                        help="String to prepend to every output filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"Error: input_dir '{args.input_dir}' is not a directory.")
        sys.exit(1)
    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copies=args.copies,
        force=args.force,
        debug=args.debug,
        black_threshold=args.black_threshold,
        prefix=args.prefix,
        erode_px=args.erode_px,
    )


if __name__ == "__main__":
    main()
