#!/bin/bash

# Create necessary directories
mkdir -p checkpoints
mkdir -p diffusion_shader_model

echo "Downloading SpatialTracker checkpoint..."
# Using gdown to download from Google Drive
# Folder ID: 1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ
gdown https://drive.google.com/uc?id=1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ --folder -O checkpoints/

echo "Downloading Diffusion as Shader checkpoint from Hugging Face..."
# Using Hugging Face CLI to download model
huggingface-cli download EXCAI/Diffusion-As-Shader --local-dir diffusion_shader_model

echo "Checkpoints downloaded successfully!"
echo "SpatialTracker checkpoint: ./checkpoints/"
echo "Diffusion as Shader checkpoint: ./diffusion_shader_model/"
