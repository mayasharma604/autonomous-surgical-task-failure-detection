"""
Smoke Detection — Annotation Tool with Auto-Fill
Press 0 or 1 to label the current frame AND auto-apply that label
to the next N frames (default 15), then jump forward.
Use [ / ] to step back/forward one frame at a time to fix mistakes.

Controls:
    0          → label current frame CLEAN, auto-fill next N frames as CLEAN
    1          → label current frame SMOKE, auto-fill next N frames as SMOKE
    ]  or  →   → step forward one frame  (to review / fix)
    [  or  ←   → step back one frame     (to review / fix)
    0 / 1      → when reviewing a single frame, just relabels that one frame
    s          → skip current frame (no label)
    q          → quit and save

Usage:
    python annotate.py --tissue /data/virtuoso_cao_demo/tissue_16
    python annotate.py --tissue /data/virtuoso_cao_demo/tissue_16 --resume
    python annotate.py --tissue /data/virtuoso_cao_demo/tissue_16 --stride 20
    python annotate.py --tissue /data/virtuoso_cao_demo/tissue_16 --csv ./labels_t16.csv

Requirements:
    pip install rich
"""

import os
import sys
import glob
import csv
import signal
import argparse
from datetime import datetime
from pathlib import Path

import termios
import tty

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DEFAULT_CSV    = Path("./annotations.csv")
DEFAULT_STRIDE = 15          # frames auto-filled per keypress
CSV_FIELDS     = ["path", "label", "annotated_at"]

console = Console()

# ──────────────────────────────────────────────
# KEYBOARD  (handles arrow keys too)
# ──────────────────────────────────────────────

def getch() -> str:
    """
    Read one keypress. Arrow keys send a 3-byte escape sequence:
    ESC [ A/B/C/D  →  we return '<up>' '<down>' '<right>' '<left>'
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":                    # escape — maybe arrow key
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return {
                    "A": "<up>",
                    "B": "<down>",
                    "C": "<right>",
                    "D": "<left>",
                }.get(ch3, "<esc>")
            return "<esc>"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# ──────────────────────────────────────────────
# IMAGE COLLECTION  (3_resect only)
# ──────────────────────────────────────────────

def collect_frames(tissue_dir: str) -> list:
    pattern = os.path.join(tissue_dir, "3_resect", "*", "endoscope", "*.png")
    frames  = sorted(glob.glob(pattern))
    if not frames:
        pattern2 = os.path.join(tissue_dir, "3_resect", "*", "*.png")
        frames   = sorted(glob.glob(pattern2))
    return frames

# ──────────────────────────────────────────────
# CSV HELPERS
# ──────────────────────────────────────────────

def load_existing(csv_path: Path) -> dict:
    existing = {}
    if not csv_path.exists():
        return existing
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[row["path"]] = int(row["label"])
    return existing


def rewrite_csv(csv_path: Path, existing: dict):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for path, label in existing.items():
            writer.writerow({
                "path":         path,
                "label":        label,
                "annotated_at": datetime.now().isoformat(timespec="seconds"),
            })

# ──────────────────────────────────────────────
# VS CODE
# ──────────────────────────────────────────────

def open_in_vscode(path: str):
    os.system(f'code --reuse-window "{path}" 2>/dev/null')

# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

def progress_bar(done: int, total: int, width: int = 40) -> str:
    filled = int(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def experiment_name(frame_path: str) -> str:
    try:
        return Path(frame_path).parent.parent.name
    except Exception:
        return "—"


def label_display(label) -> str:
    if label is None:
        return "[dim]—[/dim]"
    return "[red]SMOKE[/red]" if label == 1 else "[green]CLEAN[/green]"


def render_ui(frames: list, idx: int, existing: dict,
              stride: int, last_action: str, reviewing: bool):

    console.clear()
    total     = len(frames)
    done      = sum(1 for f in frames if f in existing)
    n_smoke   = sum(1 for f in frames if existing.get(f) == 1)
    n_clean   = sum(1 for f in frames if existing.get(f) == 0)
    n_skip    = total - done
    frame     = frames[idx]
    cur_label = existing.get(frame)

    console.print(Rule("[bold white]🔬  Smoke Annotator[/bold white]", style="dim"))
    console.print()

    # Progress
    console.print(
        f"  [{progress_bar(done, total)}]  "
        f"[bold]{done} / {total}[/bold]  "
        f"([cyan]{done / total * 100:.1f}%[/cyan])"
    )
    console.print()

    # Stats
    stats = Table.grid(padding=(0, 5))
    stats.add_row(
        f"[bold red]🔴 Smoke[/bold red]  [white]{n_smoke}[/white]",
        f"[bold green]🟢 Clean[/bold green]  [white]{n_clean}[/white]",
        f"[dim]Unlabeled  {n_skip}[/dim]",
        f"[dim]Stride  {stride}[/dim]",
    )
    console.print(stats)
    console.print()

    # Current frame panel — highlighted differently when reviewing
    border = "yellow" if reviewing else "bright_blue"
    mode   = "[bold yellow]REVIEW MODE[/bold yellow]" if reviewing else f"[bold]Frame {idx + 1} of {total}[/bold]"
    exp    = experiment_name(frame)

    # Show a mini strip of surrounding labels for context
    strip_radius = 7
    strip_start  = max(0, idx - strip_radius)
    strip_end    = min(total, idx + strip_radius + 1)
    strip_parts  = []
    for i in range(strip_start, strip_end):
        lbl = existing.get(frames[i])
        if i == idx:
            char = "[bold white on blue] ● [/bold white on blue]"
        elif lbl == 1:
            char = "[red]█[/red]"
        elif lbl == 0:
            char = "[green]█[/green]"
        else:
            char = "[dim]░[/dim]"
        strip_parts.append(char)
    strip_str = " ".join(strip_parts)

    console.print(Panel(
        f"[bold white]{Path(frame).name}[/bold white]\n"
        f"[dim]Experiment :[/dim]  {exp}\n"
        f"[dim]Current label :[/dim]  {label_display(cur_label)}\n\n"
        f"{strip_str}",
        title=mode,
        border_style=border,
        padding=(0, 1),
    ))
    console.print()

    if last_action:
        console.print(f"  {last_action}\n")

    # Keybindings — two rows depending on context
    console.print(Rule(style="dim"))
    if reviewing:
        console.print("  [bold yellow]Review mode[/bold yellow] — navigate with arrows, relabel with 0/1\n")
        keys = Table.grid(padding=(0, 3))
        keys.add_row(
            "[bold green] 0 [/bold green] Relabel CLEAN",
            "[bold red] 1 [/bold red] Relabel SMOKE",
            "[bold] ← [ [/bold] Prev frame",
            "[bold] → ] [/bold] Next frame",
            "[bold dim] q [/bold dim] Quit & save",
        )
    else:
        keys = Table.grid(padding=(0, 3))
        keys.add_row(
            f"[bold green] 0 [/bold green] CLEAN  ×{stride}",
            f"[bold red] 1 [/bold red] SMOKE  ×{stride}",
            "[bold] ← [ [/bold] Step back",
            "[bold] → ] [/bold] Step fwd",
            "[bold yellow] s [/bold yellow] Skip",
            "[bold dim] q [/bold dim] Quit",
        )
    console.print(keys)
    console.print()
    console.print("  [dim italic]Frame opens in VS Code automatically →[/dim italic]")

# ──────────────────────────────────────────────
# ANNOTATION LOOP
# ──────────────────────────────────────────────

def annotate(tissue_dir: str, csv_path: Path, resume: bool, stride: int):

    frames = collect_frames(tissue_dir)
    if not frames:
        console.print(
            f"\n[red]No frames found[/red] under:\n"
            f"  {tissue_dir}/3_resect/*/endoscope/\n"
        )
        return

    # Always load the existing CSV so annotations from other tissues
    # are never overwritten, regardless of whether --resume is passed.
    existing = load_existing(csv_path)

    # Count how many of THIS tissue's frames are already annotated
    already_done = sum(1 for f in frames if f in existing)

    if already_done and resume:
        # Jump to first unlabeled frame in this tissue
        first_unlabeled = next((i for i, f in enumerate(frames) if f not in existing), 0)
        idx = first_unlabeled
        console.print(
            f"\n[cyan]Resuming:[/cyan] {already_done} frames from this tissue already annotated. "
            f"Starting at frame {idx + 1}.\n"
        )
        import time; time.sleep(1.2)
    else:
        idx = 0

    last_action = ""

    def graceful_exit(sig=None, frame=None):
        rewrite_csv(csv_path, existing)
        console.print("\n\n[yellow]Interrupted — all progress saved.[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)

    while 0 <= idx < len(frames):
        frame     = frames[idx]
        reviewing = frame in existing   # already has a label — we're navigating back

        render_ui(frames, idx, existing, stride, last_action, reviewing)
        open_in_vscode(frame)

        key = getch()

        # ── Label + auto-fill ─────────────────────────────────────
        if key in ("0", "1"):
            label    = int(key)
            label_word = "[red]SMOKE[/red]" if label == 1 else "[green]CLEAN[/green]"

            if reviewing:
                # In review mode: relabel just this one frame
                existing[frame] = label
                rewrite_csv(csv_path, existing)
                last_action = (
                    f"{label_word}  ← relabeled frame {idx + 1}"
                )
                idx += 1

            else:
                # Auto-fill: label current + next (stride-1) unlabeled frames
                filled   = 0
                fill_idx = idx
                while filled < stride and fill_idx < len(frames):
                    existing[frames[fill_idx]] = label
                    filled   += 1
                    fill_idx += 1

                rewrite_csv(csv_path, existing)
                last_action = (
                    f"{label_word}  ← auto-filled [bold]{filled}[/bold] frames "
                    f"({idx + 1} → {fill_idx})"
                )
                idx = fill_idx   # jump past the filled block

        # ── Step back ────────────────────────────────────────────
        elif key in ("[", "<left>"):
            idx         = max(0, idx - 1)
            last_action = f"[dim]← stepped back to frame {idx + 1}[/dim]"

        # ── Step forward ─────────────────────────────────────────
        elif key in ("]", "<right>"):
            idx         = min(len(frames) - 1, idx + 1)
            last_action = f"[dim]→ stepped forward to frame {idx + 1}[/dim]"

        # ── Skip ─────────────────────────────────────────────────
        elif key == "s":
            last_action = f"[yellow]⏭  Skipped[/yellow]  ← frame {idx + 1}"
            idx += 1

        # ── Quit ─────────────────────────────────────────────────
        elif key == "q":
            break

        else:
            last_action = f"[dim]Unknown key — use 0, 1, [ ] ← → s q[/dim]"

    # Save and summary
    rewrite_csv(csv_path, existing)

    n_smoke = sum(1 for v in existing.values() if v == 1)
    n_clean = sum(1 for v in existing.values() if v == 0)

    console.clear()
    console.print(Rule("[bold white]Session Complete[/bold white]"))
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("", style="dim")
    table.add_column("", justify="right", style="bold white")
    table.add_row("🔴 Smoke",           str(n_smoke))
    table.add_row("🟢 Clean",           str(n_clean))
    table.add_row("📋 Total annotated", str(n_smoke + n_clean))
    table.add_row("🔲 Unlabeled",       str(len(frames) - n_smoke - n_clean))
    console.print(table)
    console.print()
    console.print(f"  [green]✅  CSV saved →[/green] [bold]{csv_path.resolve()}[/bold]")
    console.print()

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke annotation tool — auto-fill stride, per-frame nav"
    )
    parser.add_argument(
        "--tissue", type=str, required=True,
        help="Path to tissue folder, e.g. /data/virtuoso_cao_demo/tissue_16"
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"Output CSV path (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Start from the first unlabeled frame instead of the beginning"
    )
    parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE,
        help=f"Number of frames to auto-fill per keypress (default: {DEFAULT_STRIDE})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not sys.stdin.isatty():
        console.print("[red]Error:[/red] Needs an interactive terminal (VS Code terminal).")
        sys.exit(1)

    tissue_dir = args.tissue.rstrip("/")
    if not os.path.isdir(tissue_dir):
        console.print(f"[red]Directory not found:[/red] {tissue_dir}")
        sys.exit(1)

    console.print(f"\n  [bold]Tissue :[/bold] {tissue_dir}")
    console.print(f"  [bold]CSV    :[/bold] {args.csv}")
    console.print(f"  [bold]Stride :[/bold] {args.stride} frames per keypress")
    console.print(f"  [bold]Mode   :[/bold] {'resume' if args.resume else 'fresh start'}\n")

    annotate(tissue_dir, args.csv, resume=args.resume, stride=args.stride)


if __name__ == "__main__":
    main()
