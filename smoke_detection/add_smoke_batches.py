"""
run_smoke_batch.py
------------------
Walks tissue_1 through tissue17, finds every
    tissue_N/3_resect/<timestamp>/endoscope/
directory, and runs the smoke augmentation on each.

All augmented images are saved flat into a single output folder.

Usage:
    python run_smoke_batch.py --output_dir /data/smoke_out
    python run_smoke_batch.py --output_dir /data/CIS2/smoke_detection/smoke_val
    python run_smoke_batch.py --output_dir /data/smoke_out --copies 3
    python run_smoke_batch.py --dry_run        # preview folders without processing
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_endoscope_dirs(root: Path, tissue_range: range) -> list[Path]:
    dirs = []
    for n in tissue_range:
        tissue_dir = root / f"tissue_{n}" / "3_resect"
        if not tissue_dir.exists():
            print(f"  [SKIP] {tissue_dir} not found")
            continue
        for ts_dir in sorted(tissue_dir.iterdir()):
            endo = ts_dir / "endoscope"
            if endo.is_dir():
                dirs.append(endo)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch smoke augmentation across tissue_1–tissue_15 / 3_resect."
    )
    parser.add_argument(
        "--root", type=Path,
        default=Path("/data/virtuoso_cao_demo"),
        help="Root directory containing tissue_N folders.",
    )
    parser.add_argument(
        "--output_dir", "-o", type=Path, required=True,
        help="Single flat folder where all augmented images will be saved.",
    )
    parser.add_argument(
        "--tissue_start", type=int, default=31,
        help="First tissue number (default: 31).",
    )
    parser.add_argument(
        "--tissue_end", type=int, default=35,
        help="Last tissue number inclusive (default: 35).",
    )
    parser.add_argument(
        "--copies", "-n", type=int, default=1,
        help="Augmented copies per image (default: 1).",
    )
    parser.add_argument(
        "--black_threshold", type=int, default=15,
        help="Pixel brightness threshold for border detection (default: 15).",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print which folders would be processed without doing anything.",
    )
    parser.add_argument(
        "--smoke_script", type=Path,
        default=Path(__file__).parent / "add_smoke.py",
        help="Path to add_smoke.py (default: same directory as this script).",
    )
    args = parser.parse_args()

    if not args.smoke_script.exists():
        print(f"Error: add_smoke.py not found at {args.smoke_script}")
        sys.exit(1)

    tissue_range = range(args.tissue_start, args.tissue_end + 1)
    endoscope_dirs = find_endoscope_dirs(args.root, tissue_range)

    if not endoscope_dirs:
        print("No endoscope directories found. Check --root and tissue range.")
        sys.exit(1)

    print(f"Found {len(endoscope_dirs)} endoscope folder(s).")
    print(f"Output dir: {args.output_dir}\n")

    for i, endo_dir in enumerate(endoscope_dirs, 1):
        print(f"[{i}/{len(endoscope_dirs)}] {endo_dir}")

        if args.dry_run:
            continue

        # endo_dir is: .../tissue_N/3_resect/<timestamp>/endoscope
        tissue_num = endo_dir.parts[-4]          # e.g. "tissue_3"
        timestamp  = endo_dir.parts[-2]          # e.g. "20251215-185458-408655"
        ts_suffix  = timestamp.replace("-", "")[-4:]  # last 4 digits
        prefix     = f"{tissue_num}_ts{ts_suffix}_"   # e.g. "tissue_3_ts8655_"

        cmd = [sys.executable, str(args.smoke_script),
               "--input_dir",       str(endo_dir),
               "--output_dir",      str(args.output_dir),
               "--copies",          str(args.copies),
               "--black_threshold", str(args.black_threshold),
               "--prefix",          prefix]

        if args.force:
            cmd.append("--force")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [ERROR] exited {result.returncode} for {endo_dir}\n")
        else:
            print()

    if args.dry_run:
        print("\n(Dry run — no files written.)")
    else:
        print("All done.")


if __name__ == "__main__":
    main()
