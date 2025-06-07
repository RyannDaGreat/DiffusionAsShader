import time
import sys
import os

print("Diagnosing performance issues...")
print(f"Python: {sys.executable}")
print(f"Working dir: {os.getcwd()}")

# Time imports
start = time.time()
import numpy as np
print(f"Import numpy: {time.time() - start:.3f}s")

start = time.time()
import torch
print(f"Import torch: {time.time() - start:.3f}s")

start = time.time()
import rp
print(f"Import rp: {time.time() - start:.3f}s")

start = time.time()
from render_tracks import random_7_gaussians_video, USE_OPTIMIZED_GAUSSIAN, draw_blobs_videos
print(f"Import render_tracks: {time.time() - start:.3f}s")
print(f"USE_OPTIMIZED_GAUSSIAN: {USE_OPTIMIZED_GAUSSIAN}")

# Check which implementation is loaded
import render_tracks
if hasattr(render_tracks, 'draw_multiple_gaussians_fast'):
    print(f"Using fast implementation from: {render_tracks.draw_multiple_gaussians_fast.__module__}")

# Create test data
print("\n--- Testing gaussian rendering ---")
T, N, VH, VW = 49, 4900, 480, 720

start = time.time()
tracks = torch.randn(T, N, 4)
tracks[:, :, 0] *= VW
tracks[:, :, 1] *= VH
tracks[:, :, 2] = 0
tracks[:, :, 3] = 1
counter_tracks = tracks.clone()
print(f"Create test data: {time.time() - start:.3f}s")

# Test random_7_gaussians_video
start = time.time()
v, c = random_7_gaussians_video(tracks, counter_tracks, VH, VW, sigma=5.0)
elapsed = time.time() - start
print(f"random_7_gaussians_video: {elapsed:.3f}s")

# Test draw_blobs_videos
print("\n--- Testing draw_blobs_videos ---")
video = torch.randn(T, 3, VH, VW)
counter_video = torch.randn(T, 3, VH, VW)

start = time.time()
result = draw_blobs_videos(video, counter_video, tracks, counter_tracks, visualize=False)
elapsed = time.time() - start
print(f"draw_blobs_videos (no viz): {elapsed:.3f}s")

start = time.time()
result = draw_blobs_videos(video, counter_video, tracks, counter_tracks, visualize=True)
elapsed = time.time() - start
print(f"draw_blobs_videos (with viz): {elapsed:.3f}s")

print("\n--- Testing components ---")
# Test the actual gaussian function directly
from ultra_fast_gaussian import draw_multiple_gaussians_fast

# Select 7 tracks
selected_tracks = tracks[:, :7, :].cpu().float().numpy()
selected_counter = counter_tracks[:, :7, :].cpu().float().numpy()
colors = np.ones((7, 4), dtype=np.float32)

start = time.time()
v_np, c_np = draw_multiple_gaussians_fast(selected_tracks, selected_counter, colors, VH, VW, 5.0)
print(f"draw_multiple_gaussians_fast directly: {time.time() - start:.3f}s")

# Time individual operations
start = time.time()
_ = tracks.cpu().float().numpy()
print(f"Tensor to numpy conversion (full): {time.time() - start:.3f}s")

start = time.time()
_ = torch.from_numpy(v_np)
print(f"Numpy to tensor conversion: {time.time() - start:.3f}s")