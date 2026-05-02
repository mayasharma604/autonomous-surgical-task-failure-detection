import sys
sys.path = [p for p in sys.path if "/opt/ros" not in p]

import os
import cv2
import pandas as pd
import numpy as np
import glob
from mcap_ros2.reader import read_ros2_messages

# =========================
# CONFIGURATION
# =========================
RESECT_ROOT = "/data/2026-02-26-CAO1-JHU/2_resect_start/"
IMAGE_TOPIC = "/ves_camera/image"
MASTER_CSV = "./annotations/master_labels.csv"
PREVIEW_BASE = "./annotations/previews"

os.makedirs(PREVIEW_BASE, exist_ok=True)

# =========================
# 1. AUTO-SYNC ALL TRIALS
# =========================
def sync_all_resect_trials():
    if os.path.exists(MASTER_CSV):
        df = pd.read_csv(MASTER_CSV)
    else:
        df = pd.DataFrame(columns=["trial_id", "filename", "relative_path", "label"])

    # Find every folder you just listed
    mcap_files = glob.glob(os.path.join(RESECT_ROOT, "**/*.mcap"), recursive=True)
    
    for mcap_path in sorted(mcap_files):
        # Use the folder name (e.g., 20260226-153108-352310) as the Trial ID
        trial_id = os.path.basename(os.path.dirname(mcap_path))
        
        if trial_id in df['trial_id'].values:
            continue # Already processed this trial

        print(f"--- Extracting New Trial: {trial_id} ---")
        trial_dir = os.path.join(PREVIEW_BASE, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        
        new_rows = []
        count = 0
        try:
            for msg_view in read_ros2_messages(mcap_path):
                if msg_view.channel.topic == IMAGE_TOPIC:
                    if count % 30 == 0:
                        msg = msg_view.ros_msg
                        fname = f"frame_{msg_view.publish_time}.jpg"
                        rel_path = os.path.join(trial_dir, fname)
                        
                        img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                        if img_np.shape[2] == 3:
                            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(rel_path, img_np, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        
                        new_rows.append({"trial_id": trial_id, "filename": fname, "relative_path": rel_path, "label": ""})
                    count += 1
        except Exception as e:
            print(f"Error extracting {trial_id}: {e}")

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(MASTER_CSV, index=False)
        print(f"--- Added {len(new_rows)} frames from {trial_id} ---")
    
    return df

# =========================
# 2. SPEED LABELER
# =========================
def run_labeler(df):
    import tty, termios

    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    print("\n" + "="*45)
    print("RESECTION LABELER: [1] GOOD | [2] BAD | [Q] QUIT")
    print("="*45)

    for idx, row in df.iterrows():
        if str(row['label']).strip() != "": continue

        full_path = os.path.abspath(row['relative_path'])
        os.system(f"code '{full_path}'")
        
        print(f"\rTrial: {row['trial_id']} | Image {idx+1}/{len(df)} | [1/2/Q]: ", end="", flush=True)
        
        key = get_key()
        if key == '1': df.at[idx, 'label'] = 'good'
        elif key == '2': df.at[idx, 'label'] = 'bad'
        elif key.lower() == 'q': break
        else: continue

        df.to_csv(MASTER_CSV, index=False)

if __name__ == "__main__":
    master_df = sync_all_resect_trials()
    run_labeler(master_df)
