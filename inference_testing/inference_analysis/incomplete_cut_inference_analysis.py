import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load in results
INFERENCE_CSV = "./incomplete_cut_endoscope_only_weights.csv"
df = pd.read_csv(INFERENCE_CSV)

# filter for only the active cutting phase (3_resect)
# We exclude 'recovery' because it's a plateau, not a trend.
resect_df = df[df['phase'].str.contains("3_resect") & ~df['phase'].str.contains("recovery")].copy()

# sort by filename to ensure looking at time-sequence
resect_df = resect_df.sort_values(by=['phase', 'filename'])
resect_df['frame_index'] = range(len(resect_df))

#  find correlation
correlation = resect_df['frame_index'].corr(resect_df['predicted_weight'])

print(f" Trend Analysis for '3_resect' folders:")
print(f"Total Frames: {len(resect_df)}")
print(f"Pearson Correlation (Time vs. Weight): {correlation:.4f}")

if correlation > 0.7:
    print(" Strong Increase: The model successfully captures the cutting progression.")
elif correlation > 0.4:
    print(" Weak Increase: The trend is positive but contains significant noise.")
else:
    print(" No Trend: The model is not correctly identifying the temporal progress.")

# plot the results
plt.figure(figsize=(10, 6))
plt.scatter(resect_df['frame_index'], resect_df['predicted_weight'], s=10, alpha=0.3, label='Predictions')
plt.plot(resect_df['frame_index'], np.linspace(0.01, 0.99, len(resect_df)), color='red', linestyle='--', label='Annotation Target')

plt.title('Weight Progression: 3_resect (Endoscope Only)')
plt.xlabel('Frame Sequence')
plt.ylabel('Predicted Weight')
plt.legend()
plt.savefig('resect_trend_analysis.png') # Saving because plotting on a server can be tricky
print("\nPlot saved as 'resect_trend_analysis.png'")
