import pandas as pd

# =========================
# CONFIGURATION
# =========================
INFERENCE_CSV = "tissue2_full_sequence_inference.csv"

# Adjusted ranges to ensure total coverage from 0.0 to 0.9
IDEAL_RANGES = {
    "2_resect_start": (0.00, 0.35),
    "3_resect":       (0.35, 0.45),
    "4_resect_home":  (0.45, 0.95)
}

def analyze_total_accuracy(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return

    phase_stats = []
    total_frames_all_phases = 0
    total_correct_all_phases = 0

    print(f"📊 Analyzing Accuracy for: {csv_path}\n")

    for phase, (low, high) in IDEAL_RANGES.items():
        # Match phase name (case-insensitive and partial match to handle folder variations)
        phase_df = df[df['phase'].str.contains(phase, case=False, na=False)]
        
        if phase_df.empty:
            continue
            
        count = len(phase_df)
        # Count frames within range
        in_range = ((phase_df['predicted_progress'] >= low) & 
                    (phase_df['predicted_progress'] < high)).sum()
        
        total_frames_all_phases += count
        total_correct_all_phases += in_range
        
        phase_stats.append({
            "Phase": phase,
            "Frames": count,
            "Accuracy": round((in_range / count) * 100, 2)
        })

    # =========================
    # DISPLAY RESULTS
    # =========================
    print("--- Per-Phase Breakdown ---")
    for stat in phase_stats:
        print(f"{stat['Phase']:<20} | Frames: {stat['Frames']:<5} | Accuracy: {stat['Accuracy']}%")

    if total_frames_all_phases > 0:
        total_accuracy = (total_correct_all_phases / total_frames_all_phases) * 100
        
        print("\n" + "="*30)
        print(f"TOTAL SYSTEM ACCURACY: {total_accuracy:.2f}%")
        print("="*30)
        
        # Verdict
        if total_accuracy >= 85:
            print("🚀 Result: Exceptional. No retraining needed.")
        elif total_accuracy >= 70:
            print("📈 Result: Good. Model is capturing the resection timeline well.")
        elif total_accuracy >= 50:
            print("⚠️ Result: Moderate. Significant overlap between phases.")
        else:
            print("🆘 Result: Low. You should definitely retrain with the 2-4 Phase CSV.")
    else:
        print("❌ No matching phases found in CSV to calculate accuracy.")

if __name__ == "__main__":
    analyze_total_accuracy(INFERENCE_CSV)
