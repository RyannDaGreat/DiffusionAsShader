#!/bin/bash

# Script to generate ONLY the tracking video (tracking_video.mp4) without creating result.mp4

python demo.py \
    --prompt "Track only" \
    --checkpoint_path diffusion_shader_model \
    --output_dir outputs \
    --input_path example_inputs/firegirl1.mp4 \
    --tracking_method spatracker \
    --gpu 0 \
    --tracking_only