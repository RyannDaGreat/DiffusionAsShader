#!/usr/bin/env python3
"""
Model Pre-Download Script for Diffusion as Shader (DaS)
-------------------------------------------------------

This script pre-downloads all additional models needed during inference time.
It's typically called by download_checkpoints.sh but can be run standalone.

Purpose:
- During inference, DaS automatically downloads several models if they're not present
- This can cause unexpected delays during the first run
- Pre-downloading them ensures smoother operation from the start

Models downloaded:
1. MoGe (Ruicheng/moge-vitl) - Used for image input processing for 3D awareness
2. VGGT (facebook/VGGT-1B) - Used for camera motion calculations
3. ZoeDepth (Intel/zoedepth-nyu-kitti) - Used for depth estimation with CoTracker
4. CoTracker (facebookresearch/co-tracker) - Used for tracking

The models are downloaded to their default cache locations that will be automatically
found during inference.
"""

import os
import torch
from huggingface_hub import snapshot_download

# Get cache locations
hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
torch_cache = os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch'))

print(f"HuggingFace cache: {hf_cache}")
print(f"PyTorch Hub cache: {torch_cache}")

# Download MoGe model
print("Downloading MoGe model (used for image inputs)...")
moge_path = snapshot_download('Ruicheng/moge-vitl')
print(f"  - Downloaded to: {moge_path}")

# Download VGGT model
print("Downloading VGGT model (used for camera motion)...")
vggt_path = snapshot_download('facebook/VGGT-1B')
print(f"  - Downloaded to: {vggt_path}")

# Download ZoeDepth model
print("Downloading ZoeDepth model (used for CoTracker)...")
zoedepth_path = snapshot_download('Intel/zoedepth-nyu-kitti')
print(f"  - Downloaded to: {zoedepth_path}")

# Download CoTracker model 
print("Downloading CoTracker model...")
try:
    torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline', pretrained=True, skip_validation=True)
    cotracker_path = os.path.join(torch_cache, 'hub', 'facebookresearch_co-tracker_main')
    print(f"  - Downloaded to: {cotracker_path}")
except Exception as e:
    print(f"  - Warning: CoTracker download encountered an issue: {e}")
    print("    (The model may still be usable during inference)")

print("\nAll models downloaded successfully to their respective cache locations!")