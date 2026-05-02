import os
import pandas as pd

# =========================
# CONFIGURATION
# =========================
# Keeping your original path structure
DATA_ROOTS = [
    "/data/virtuoso_cao_demo/tissue_1/1_retract/",
    "/data/virtuoso_cao_demo/tissue_1/2_resect_start/",
    "/data/virtuoso_cao_demo/tissue_1/3_resect/",
    "/data/virtuoso_cao_demo/tissue_1/3_resect_recovery/",
    "/data/virtuoso_cao_demo/tissue_1/4_resect_home/",
    "/data/virtuoso_cao_demo/tissue_1/5_retract_home/",
    "/data/virtuoso_cao_demo/tissue_1/6_remove/"
]

# Saving to a new name so you don't overwrite your task completion data
MASTER_CSV = "./annotations/incomplete_cut_labels2.csv"

# =========================
# 1. SCAN AND LABEL (Tissue-Centric Logic)
# =========================
def generate_labels_from_images():
    all_rows = []

    for root in DATA_ROOTS:
        phase = os.path.basename(os.path.normpath(root))
        print(f"\n--- 📂 Processing Phase: {phase} ---")
        
        if not os.path.exists(root):
            print(f"❌ Cannot access: {root}")
            continue

        trial_folders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        
        for trial_id in sorted(trial_folders):
            image_dir = os.path.join(root, trial_id, "endoscope")
            
            if not os.path.exists(image_dir):
                continue

            images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
            if not images:
                continue

            total_f = len(images)
            unique_trial_key = f"{phase}_{trial_id}"
            
            for i, fname in enumerate(images):
                # ---------------------------------------------------------
                # NEW LOGIC: TISSUE SEPARATION INDEX
                # ---------------------------------------------------------
                if phase == "3_resect":
                    # Only the active cutting phase shows progress (0.01 -> 0.99)
                    progress = round(0.01 + (i / total_f) * (0.99 - 0.01), 4)
                
                elif phase == "3_resect_recovery":
                    # PLATEAU: We don't know the exact last frame here easily, 
                    # so we'll set a standard "mid-cut" value or 
                    # you can manually tune this to 0.5 for safety.
                    progress = 0.50 
                
                elif phase in ["4_resect_home", "5_retract_home", "6_remove"]:
                    # These frames mean the cut is 100% physically done
                    progress = 1.0
                
                else:
                    # Phases 1 and 2: The tissue is still 100% intact (0% cut)
                    progress = 0.0
                # ---------------------------------------------------------

                all_rows.append({
                    "phase": phase,
                    "trial_id": trial_id,
                    "unique_key": unique_trial_key,
                    "filename": fname,
                    "relative_path": os.path.join(image_dir, fname),
                    "progress": progress
                })
            
    # Create DataFrame and Save
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(MASTER_CSV), exist_ok=True)
    df.to_csv(MASTER_CSV, index=False)
    return df

if __name__ == "__main__":
    print("🚀 Generating INCOMPLETE CUT Labels (Tissue-Centric)...")
    master_df = generate_labels_from_images()
    print(f"\n✨ Success! Created {MASTER_CSV}")
