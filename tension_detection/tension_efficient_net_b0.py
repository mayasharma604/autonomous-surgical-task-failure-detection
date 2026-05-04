import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
from torchvision.models import efficientnet_b0

# Ensure this matches your directory structure
sys.path.append("/data/tensionData/training/shared")
import tension_training as tt

class SqueezeTime(nn.Module):
    def forward(self, x):
        return x.squeeze(1) 

# config
WORKSPACE_ROOT = '/data/tensionData'
WINDOW_SIZE = 1 
PREDICT_FRAMES = 1
BATCH_SIZE = 64     
NUM_WORKERS = 12    
LEARNING_RATE = 1e-4
IMG_SIZE = 224      
EPOCHS = 10
DEVICE = tt.setup_device(seed=42)

# prepare masks and clips
mask_lookup = tt.make_mask_lookup(WORKSPACE_ROOT)
seg_mask_lookup = tt.make_seg_mask_lookup(WORKSPACE_ROOT)

clips = tt.index_all_clips(WORKSPACE_ROOT, mask_lookup, min_clip_len=WINDOW_SIZE)
train_clips, val_clips = tt.split_clips_by_trial(clips)

train_windows = tt.build_windows(train_clips, WINDOW_SIZE, PREDICT_FRAMES)
val_windows = tt.build_windows(val_clips, WINDOW_SIZE, PREDICT_FRAMES)

train_windows = tt.balance_windows(train_windows, PREDICT_FRAMES)
val_windows = tt.balance_windows(val_windows, PREDICT_FRAMES)

# loaders
train_loader, val_loader, train_ds, val_ds = tt.make_loaders(
    train_windows, val_windows,
    BATCH_SIZE, NUM_WORKERS,
    IMG_SIZE, PREDICT_FRAMES,
    mask_lookup,
    seg_mask_lookup=seg_mask_lookup,
    dual_stream=True
)

# training the model
base_model = efficientnet_b0(weights=None) 

# modify first conv layer for 6 channels
original_conv = base_model.features[0][0]
base_model.features[0][0] = nn.Conv2d(
    in_channels=6, 
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=False
)

# modify classifier for single-output binary classification
num_ftrs = base_model.classifier[1].in_features
base_model.classifier[1] = nn.Linear(num_ftrs, 1)

model = nn.Sequential(
    SqueezeTime(),
    base_model
).to(DEVICE)

# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# training loop
# early_stop_patience=5 prevents overfitting by stopping if Val Loss stalls
tt.train(
    model=model,
    raw_model=base_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=DEVICE,
    checkpoint_dir='./checkpoints_b0',
    epochs=EPOCHS,
    accum_steps=1,
    early_stop_patience=5
)

# final metrics eval
print("\n" + "="*40)
print("REPORTING FINAL METRICS FROM BEST MODEL")
print("="*40)

best_path = os.path.join('./checkpoints_b0', 'best_model.pth')
if os.path.exists(best_path):
    # Load with weights_only=False to support the tt.train checkpoint format
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Prefix handling: ensures weights load correctly into the Sequential wrapper
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith('1.'):
            new_state_dict[f'1.{k}'] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict)
    except:
        model[1].load_state_dict(state_dict)
    
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            
    y_pred = torch.cat(all_preds).numpy().flatten()
    y_true = torch.cat(all_targets).numpy().flatten()
    
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    print(f"\nFinal Validation Results for EfficientNet-B0:")
    print("-" * 30)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print("="*40)
else:
    print("Error: best_model.pth not found in ./checkpoints_b0")
