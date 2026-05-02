import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import random
from torchvision.models import efficientnet_v2_m

sys.path.append("/data/tensionData/training/shared")
import tension_training as tt

# Set seeds globally to force the tt module to be reproducible
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class SqueezeTime(nn.Module):
    def forward(self, x):
        return x.squeeze(1) 

# =========================
# Config
# =========================
WORKSPACE_ROOT = '/data/tensionData'
WINDOW_SIZE = 1 
PREDICT_FRAMES = 1
BATCH_SIZE = 64 
NUM_WORKERS = 12 
LEARNING_RATE = 1e-4
IMG_SIZE = 224 
EPOCHS = 10
DEVICE = tt.setup_device(seed=42)

# =========================
# Prepare masks & clips
# =========================
mask_lookup = tt.make_mask_lookup(WORKSPACE_ROOT)
seg_mask_lookup = tt.make_seg_mask_lookup(WORKSPACE_ROOT)

# Let the tt module handle indexing and splitting its own way
clips = tt.index_all_clips(WORKSPACE_ROOT, mask_lookup, min_clip_len=WINDOW_SIZE)
train_clips, val_clips = tt.split_clips_by_trial(clips)

# This converts clips into the structured "windows" tt.balance_windows expects
train_windows = tt.build_windows(train_clips, WINDOW_SIZE, PREDICT_FRAMES)
val_windows = tt.build_windows(val_clips, WINDOW_SIZE, PREDICT_FRAMES)

# Now balance_windows should see lists/tuples instead of strings
train_windows = tt.balance_windows(train_windows, PREDICT_FRAMES)
val_windows = tt.balance_windows(val_windows, PREDICT_FRAMES)

# =========================
# Loaders
# =========================
train_loader, val_loader, train_ds, val_ds = tt.make_loaders(
    train_windows, val_windows,
    BATCH_SIZE, NUM_WORKERS,
    IMG_SIZE, PREDICT_FRAMES,
    mask_lookup,
    seg_mask_lookup=seg_mask_lookup,
    dual_stream=True
)

# =========================
# Model: EfficientNetV2-M
# =========================
base_model = efficientnet_v2_m(weights=None) 

original_conv = base_model.features[0][0]
base_model.features[0][0] = nn.Conv2d(
    in_channels=6, 
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=False
)

num_ftrs = base_model.classifier[1].in_features
base_model.classifier[1] = nn.Linear(num_ftrs, 1)

model = nn.Sequential(
    SqueezeTime(),
    base_model
).to(DEVICE)

# =========================
# Optimizer & Scheduler
# =========================
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# =========================
# Training loop
# =========================
tt.train(
    model=model,
    raw_model=base_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=DEVICE,
    checkpoint_dir='./checkpoints_effnet',
    epochs=EPOCHS,
    accum_steps=1,
    early_stop_patience=5
)

# =========================
# MANUAL FINAL EVALUATION (FIXED)
# =========================
print("\n" + "="*40)
print("FORCE-CALCULATING BINARY METRICS")
print("="*40)

best_path = os.path.join('./checkpoints_effnet', 'best_model.pth')
if os.path.exists(best_path):
    # Use weights_only=True to silence that warning
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    
    # Extract state_dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # FIX: If the saved model doesn't have the "1." prefix, add it 
    # OR if it has it and we don't need it, remove it.
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith('0.') and not k.startswith('1.'):
            # If saved as base_model, it needs to be mapped to the second part of our Sequential wrapper
            new_state_dict[f'1.{k}'] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict)
        print("✅ Successfully matched and loaded model weights.")
    except RuntimeError as e:
        print(f"⚠️ Direct load failed, trying base_model load: {e}")
        # Fallback: Load directly into the base_model part of our Sequential
        model[1].load_state_dict(state_dict)
        print("✅ Successfully loaded weights into base_model.")

    model.eval()
    all_preds = []
    all_targets = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            
    y_pred = torch.cat(all_preds).numpy().flatten()
    y_true = torch.cat(all_targets).numpy().flatten()
    
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    print("\n" + "-"*30)
    print(f"Final Validation Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Final Precision:           {precision_score(y_true, y_pred):.4f}")
    print(f"Final Recall:              {recall_score(y_true, y_pred):.4f}")
    print(f"Final F1-Score:            {f1_score(y_true, y_pred):.4f}")
    print("-"*30)
else:
    print("Error: Could not find best_model.pth for final evaluation.")
