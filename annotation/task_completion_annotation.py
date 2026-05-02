import os
import pandas as pd

# =========================
# CONFIGURATION
# =========================
PHASE_ORDER = [
    "2_resect_start",
    "3_resect",
    "3_resect_recovery",
    "4_resect_home",
    "5_retract_home"
]

DATA_ROOT = "/data/virtuoso_cao_demo/tissue_1/"
MASTER_CSV = "./annotations/resect_continuous_progress.csv"

def generate_continuous_labels():
    # 1. Get sorted trial list per phase
    phase_trials = {}
    for phase in PHASE_ORDER:
        phase_path = os.path.join(DATA_ROOT, phase)
        if os.path.exists(phase_path):
            phase_trials[phase] = sorted([
                d for d in os.listdir(phase_path)
                if os.path.isdir(os.path.join(phase_path, d))
            ])
            print(f"  {phase}: {len(phase_trials[phase])} trials")
        else:
            phase_trials[phase] = []

    n_trials = max(len(t) for t in phase_trials.values())
    print(f"\n🔍 Processing {n_trials} trials matched by sort order.\n")

    all_rows = []

    for trial_idx in range(n_trials):
        # 2. Collect ALL frames across all phases for this trial, in phase order
        trial_frames = []
        for phase in PHASE_ORDER:
            trials = phase_trials.get(phase, [])
            if trial_idx < len(trials):
                tid = trials[trial_idx]
                img_dir = os.path.join(DATA_ROOT, phase, tid, "endoscope")
                if os.path.exists(img_dir):
                    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
                    for f in imgs:
                        trial_frames.append({
                            "relative_path": os.path.join(img_dir, f),
                            "trial_index": trial_idx,
                            "trial_id": tid,
                            "phase": phase,
                            "filename": f,
                        })

        # 3. Assign 0.0 → 0.9 gradually across every frame in this trial
        total_f = len(trial_frames)
        if total_f == 0:
            print(f"⚠️  Trial index {trial_idx}: no frames found, skipping.")
            continue

        print(f"📈 Trial {trial_idx}: {total_f} total frames → 0.0 to 0.9")
        for i, frame in enumerate(trial_frames):
            frame["progress"] = round((i / max(total_f - 1, 1)) * 0.9, 4)
            all_rows.append(frame)

    # 4. Save
    if not all_rows:
        print("❌ No images were found.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(by=["trial_index", "progress", "filename"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(MASTER_CSV), exist_ok=True)
    df.to_csv(MASTER_CSV, index=False)

    print(f"\n✨ Success! Created {MASTER_CSV}")
    print("\n--- Sample: first and last 5 rows per trial ---")
    for idx in df["trial_index"].unique():
        sub = df[df["trial_index"] == idx]
        print(f"\nTrial {idx}:")
        print(sub[["phase", "filename", "progress"]].head(3).to_string())
        print("  ...")
        print(sub[["phase", "filename", "progress"]].tail(3).to_string())

if __name__ == "__main__":
    generate_continuous_labels()
