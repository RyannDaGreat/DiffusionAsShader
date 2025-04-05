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

os.system('git submodule update --init --recursive')

def main():
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
    # Direct file download links for SpatialTracker model files
    spatracker_files = [
        # Main model file - this is the critical one
        {"id": "1yCG6ztR2lB0bxrGIBgLmFeCx9YA9Znct", "output": "checkpoints/spatracker_model.pth"},
        
        # Example files that show successful download (optional)
        {"id": "1BDtvfrvbzEFY84XJPp62Dq1PujIpbOK_", "output": "checkpoints/examples/butterfly_rgb/butterfly.mp4"},
        {"id": "1hlAGFony7LzpLcEAoGLiNaY3zxfiN_bW", "output": "checkpoints/examples/butterfly_rgb/butterfly.png"}
    ]
    try:
        import gdown
        # Create the example directories
        os.makedirs("checkpoints/examples/butterfly_rgb", exist_ok=True)
        
        print(f"Downloading SpatialTracker model files to ./checkpoints/")
        success = True
        for file_info in spatracker_files:
            try:
                gdown.download(id=file_info["id"], output=file_info["output"], quiet=False)
                print(f"Successfully downloaded {file_info['output']}")
            except Exception as file_e:
                print(f"Error downloading {file_info['output']}: {file_e}")
                if "spatracker_model.pth" in file_info["output"]:
                    success = False
        
        if not success:
            print("\nError: Failed to download the critical model file.")
            print("Please manually download from: https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ")
            print("You need the spatracker_model.pth file in your checkpoints/ directory.")
    except Exception as e:
        print(f"Error setting up SpatialTracker download: {e}")
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
    download_inference_models()

def download_inference_models():
    """Downloads additional models needed during inference time"""
    
    import torch
    from huggingface_hub import snapshot_download
    
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
    
    # Check if SpatialTracker model exists in diffusion_shader_model directory
    spatracker_src = "diffusion_shader_model/spatracker/spaT_final.pth"
    spatracker_dst = "checkpoints/spatracker_model.pth"
    if os.path.exists(spatracker_src) and not os.path.exists(spatracker_dst):
        print(f"\nFound SpatialTracker model at {spatracker_src}")
        print(f"Copying to {spatracker_dst}...")
        import shutil
        os.makedirs(os.path.dirname(spatracker_dst), exist_ok=True)
        shutil.copy2(spatracker_src, spatracker_dst)
        print(f"Successfully copied SpatialTracker model to {spatracker_dst}")
    
    # Install rich for better table display
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.table import Table
    
    # Create a list of files to check
    files_to_check = [
        {"name": "SpatialTracker Model", "path": "checkpoints/spatracker_model.pth", "critical": True},
        {"name": "Example Butterfly MP4", "path": "checkpoints/examples/butterfly_rgb/butterfly.mp4", "critical": False},
        {"name": "Example Butterfly PNG", "path": "checkpoints/examples/butterfly_rgb/butterfly.png", "critical": False},
        {"name": "Diffusion Shader Model - Config", "path": "diffusion_shader_model/model_index.json", "critical": True},
        {"name": "Diffusion Shader - Scheduler", "path": "diffusion_shader_model/scheduler/scheduler_config.json", "critical": True},
        {"name": "Diffusion Shader - Text Encoder", "path": "diffusion_shader_model/text_encoder/config.json", "critical": True},
        {"name": "Diffusion Shader - Tokenizer", "path": "diffusion_shader_model/tokenizer/spiece.model", "critical": True},
        {"name": "Diffusion Shader - Transformer", "path": "diffusion_shader_model/transformer/config.json", "critical": True},
        {"name": "Diffusion Shader - VAE", "path": "diffusion_shader_model/vae/config.json", "critical": True}
    ]
    
    # Add inference model checks
    cache_paths = {
        "MoGe": os.path.join(hf_cache, "hub/models--Ruicheng--moge-vitl"),
        "VGGT": os.path.join(hf_cache, "hub/models--facebook--VGGT-1B"),
        "ZoeDepth": os.path.join(hf_cache, "hub/models--Intel--zoedepth-nyu-kitti"),
        "CoTracker": os.path.join(torch_cache, "hub/facebookresearch_co-tracker_main")
    }
    
    for name, path in cache_paths.items():
        files_to_check.append({"name": name, "path": path, "critical": False})
    
    # Check each file and build results
    results = []
    all_critical_present = True
    
    for file_info in files_to_check:
        exists = os.path.exists(file_info["path"])
        if file_info["critical"] and not exists:
            all_critical_present = False
        
        results.append({
            "name": file_info["name"],
            "status": "✅ Present" if exists else "❌ Missing",
            "critical": "Critical" if file_info["critical"] else "Optional",
            "path": file_info["path"]
        })
    
    console = Console()
    table = Table(title="Diffusion as Shader - Model Status")
    
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Type", style="yellow")
    table.add_column("Path", style="green")
    
    for result in results:
        status_style = "green" if "Present" in result["status"] else "red"
        critical_style = "red bold" if result["critical"] == "Critical" else "yellow"
        
        table.add_row(
            result["name"],
            f"[{status_style}]{result['status']}[/]",
            f"[{critical_style}]{result['critical']}[/]",
            result["path"]
        )
    
    console.print("\n")
    console.print(table)
    
    if all_critical_present:
        console.print("\n[green bold]Setup complete! All critical models are present.[/]")
    else:
        console.print("\n[red bold]Setup incomplete! Some critical models are missing.[/]")
        console.print("[yellow]Please check the table above and ensure all critical components are present.[/]")

if __name__ == "__main__":
    main()
