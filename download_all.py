#!/usr/bin/env python3
"""
Complete Download Script for Diffusion as Shader (DaS)
-----------------------------------------------------

This script downloads ALL model files needed by DaS:
1. SpatialTracker checkpoint (from Google Drive to ./checkpoints/)
2. Diffusion as Shader model (from HuggingFace to ./diffusion_shader_model/)
3. Additional inference models to their cache locations:
   - MoGe (~/.cache/huggingface/models--Ruicheng--moge-vitl) - For image processing
   - VGGT (~/.cache/huggingface/models--facebook--VGGT-1B) - For camera motion
   - ZoeDepth (~/.cache/huggingface/models--Intel--zoedepth-nyu-kitti) - For depth estimation
   - CoTracker (~/.cache/torch/hub/facebookresearch_co-tracker_main) - For tracking

Purpose:
- Ensures all models are pre-downloaded before the first inference
- Avoids unexpected delays during the first run
- Consolidates all downloading in one script for convenience
"""

import os
import sys
import subprocess
import tempfile

# Create necessary directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("diffusion_shader_model", exist_ok=True)

# Install required packages
print("Installing required packages...")
packages = ["gdown", "huggingface_hub", "torch"]
for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    except subprocess.CalledProcessError:
        print(f"Error installing {package}. Please install it manually.")
        sys.exit(1)

# Now import after installation
import torch
from huggingface_hub import snapshot_download, hf_hub_download

# Step 1: Download SpatialTracker checkpoint from Google Drive
print("\n=== Step 1: Downloading SpatialTracker checkpoint ===")
spatracker_folder_id = "1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ"
try:
    import gdown
    print(f"Downloading SpatialTracker from Google Drive to ./checkpoints/")
    gdown.download_folder(id=spatracker_folder_id, output="checkpoints/", quiet=False)
except Exception as e:
    print(f"Error downloading SpatialTracker: {e}")
    print("You may need to manually download from: https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ")

# Step 2: Download Diffusion as Shader model from HuggingFace
print("\n=== Step 2: Downloading Diffusion as Shader model ===")
try:
    subprocess.check_call(["huggingface-cli", "download", "EXCAI/Diffusion-As-Shader", "--local-dir", "diffusion_shader_model"])
    print(f"Diffusion as Shader model downloaded to ./diffusion_shader_model/")
except subprocess.CalledProcessError as e:
    print(f"Error downloading Diffusion as Shader model: {e}")
    print("You may need to manually download from: https://huggingface.co/EXCAI/Diffusion-As-Shader")

# Step 3: Download additional inference models
print("\n=== Step 3: Downloading additional inference models ===")

# Get cache locations
hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
torch_cache = os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch'))

print(f"HuggingFace cache: {hf_cache}")
print(f"PyTorch Hub cache: {torch_cache}")

# Download MoGe model
print("\nDownloading MoGe model (used for image inputs)...")
try:
    moge_path = snapshot_download('Ruicheng/moge-vitl')
    print(f"  - Downloaded to: {moge_path}")
except Exception as e:
    print(f"  - Error downloading MoGe model: {e}")

# Download VGGT model
print("\nDownloading VGGT model (used for camera motion)...")
try:
    vggt_path = snapshot_download('facebook/VGGT-1B')
    print(f"  - Downloaded to: {vggt_path}")
except Exception as e:
    print(f"  - Error downloading VGGT model: {e}")

# Download ZoeDepth model
print("\nDownloading ZoeDepth model (used for CoTracker)...")
try:
    zoedepth_path = snapshot_download('Intel/zoedepth-nyu-kitti')
    print(f"  - Downloaded to: {zoedepth_path}")
except Exception as e:
    print(f"  - Error downloading ZoeDepth model: {e}")

# Download CoTracker model 
print("\nDownloading CoTracker model...")
try:
    torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline', pretrained=True, skip_validation=True)
    cotracker_path = os.path.join(torch_cache, 'hub', 'facebookresearch_co-tracker_main')
    print(f"  - Downloaded to: {cotracker_path}")
except Exception as e:
    print(f"  - Warning: CoTracker download encountered an issue: {e}")
    print("    (The model may still be usable during inference)")

print("\n=== Download Summary ===")
print("1. SpatialTracker checkpoint: ./checkpoints/")
print("2. Diffusion as Shader model: ./diffusion_shader_model/")
print("3. Additional inference models:")
print(f"   - MoGe: {hf_cache}/models--Ruicheng--moge-vitl")
print(f"   - VGGT: {hf_cache}/models--facebook--VGGT-1B")
print(f"   - ZoeDepth: {hf_cache}/models--Intel--zoedepth-nyu-kitti")
print(f"   - CoTracker: {torch_cache}/hub/facebookresearch_co-tracker_main")
print("\nSetup complete! All models downloaded successfully.")