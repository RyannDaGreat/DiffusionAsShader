import numpy as np
import time
from render_tracks import _draw_multiple_gaussians_numba
from ultra_fast_gaussian import draw_multiple_gaussians_fast

# Test like the actual main() function would
T, N = 49, 4900  # 49 frames, 70x70 grid
VH, VW = 480, 720

# Generate tracks
tracks = np.random.rand(T, N, 4).astype(np.float32)
tracks[:, :, 0] *= VW
tracks[:, :, 1] *= VH
tracks[:, :, 2] = 0
tracks[:, :, 3] = 1  # All visible

# Select 7 random tracks (like random_7_gaussians_video)
np.random.seed(42)
selected_idx = np.random.choice(N, 7, replace=False)
selected_tracks = tracks[:, selected_idx, :]
selected_counter = selected_tracks.copy()

# Colors
colors = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
], dtype=np.float32)

print("Testing performance with typical usage pattern...")
print(f"Frames: {T}, Selected tracks: 7 (from {N}), Resolution: {VH}x{VW}")

# Warm up numba
print("\nWarming up numba...")
_ = _draw_multiple_gaussians_numba(
    selected_tracks[:1, :1, :], selected_counter[:1, :1, :], 
    colors[:1], VH, VW, 5.0
)

# Test numba
runs = []
for i in range(5):
    start = time.time()
    v1, c1 = _draw_multiple_gaussians_numba(selected_tracks, selected_counter, colors, VH, VW, 5.0)
    runs.append(time.time() - start)
numba_time = np.median(runs)
print(f"\nNumba: {numba_time:.3f}s (median of 5 runs)")

# Test ultra fast
runs = []
for i in range(5):
    start = time.time()
    v2, c2 = draw_multiple_gaussians_fast(selected_tracks, selected_counter, colors, VH, VW, 5.0)
    runs.append(time.time() - start)
fast_time = np.median(runs)
print(f"Ultra fast: {fast_time:.3f}s (median of 5 runs)")
print(f"Speedup: {numba_time/fast_time:.1f}x")

# Check accuracy
diff = np.abs(v1 - v2).max()
print(f"\nMax difference: {diff:.8f}")

# Test with different blob_colors settings
print("\n\nTesting 'random_of_7' simulation...")
num_colors = np.random.randint(1, 8)
selected_colors = colors[:num_colors]
selected_tracks_subset = selected_tracks[:, :num_colors, :]
selected_counter_subset = selected_counter[:, :num_colors, :]

start = time.time()
v1, c1 = _draw_multiple_gaussians_numba(
    selected_tracks_subset, selected_counter_subset, selected_colors, VH, VW, 5.0
)
numba_rand = time.time() - start

start = time.time()
v2, c2 = draw_multiple_gaussians_fast(
    selected_tracks_subset, selected_counter_subset, selected_colors, VH, VW, 5.0
)
fast_rand = time.time() - start

print(f"Random {num_colors} colors - Numba: {numba_rand:.3f}s, Ultra fast: {fast_rand:.3f}s")
print(f"Speedup: {numba_rand/fast_rand:.1f}x")