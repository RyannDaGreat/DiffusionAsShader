import numpy as np
import time
from render_tracks import _draw_multiple_gaussians_numba
from optimized_gaussian_render_fast import draw_multiple_gaussians_fast, _draw_multiple_gaussians_fast_numpy

def create_test_data(T, N, VH, VW, visibility_prob=0.8):
    """Create realistic test data."""
    tracks = np.random.rand(T, N, 4).astype(np.float32)
    tracks[:, :, 0] *= VW  # X positions
    tracks[:, :, 1] *= VH  # Y positions
    tracks[:, :, 2] = 0    # Z
    tracks[:, :, 3] = (np.random.rand(T, N) > (1-visibility_prob)).astype(np.float32)
    
    counter_tracks = tracks + np.random.randn(T, N, 4).astype(np.float32) * 10
    counter_tracks[:, :, 3] = tracks[:, :, 3]
    
    colors = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=np.float32)
    
    return tracks, counter_tracks, colors

def benchmark_implementations():
    """Comprehensive benchmark of all implementations."""
    print("=== Gaussian Rendering Benchmark ===\n")
    
    # Test configurations
    configs = [
        # (name, T, N, VH, VW, sigma)
        ("Small", 5, 50, 256, 256, 5.0),
        ("Medium", 10, 100, 480, 640, 5.0),
        ("Large", 20, 200, 720, 1280, 5.0),
        ("XLarge", 49, 490, 1080, 1920, 5.0),
    ]
    
    for config_name, T, N, VH, VW, sigma in configs:
        print(f"\n{config_name} Test: {T} frames, {N} tracks, {VH}x{VW}")
        print("-" * 50)
        
        # Create test data
        tracks, counter_tracks, colors = create_test_data(T, N, VH, VW)
        
        # Select 7 tracks like random_7_gaussians_video does
        np.random.seed(42)
        selected_idx = np.random.choice(N, min(7, N), replace=False)
        selected_tracks = tracks[:, selected_idx, :]
        selected_counter = counter_tracks[:, selected_idx, :]
        selected_colors = colors[:len(selected_idx)]
        
        # Warm-up numba
        if config_name == "Small":
            print("Warming up numba...")
            _ = _draw_multiple_gaussians_numba(
                selected_tracks[:1], selected_counter[:1], 
                selected_colors, VH, VW, sigma
            )
        
        # Benchmark numba
        num_runs = 5 if config_name != "XLarge" else 3
        
        times_numba = []
        for _ in range(num_runs):
            start = time.time()
            v1, c1 = _draw_multiple_gaussians_numba(
                selected_tracks, selected_counter, selected_colors, VH, VW, sigma
            )
            times_numba.append(time.time() - start)
        numba_time = np.median(times_numba)
        
        # Benchmark optimized batch version
        times_opt = []
        for _ in range(num_runs):
            start = time.time()
            v2, c2 = draw_multiple_gaussians_fast(
                selected_tracks, selected_counter, selected_colors, VH, VW, sigma
            )
            times_opt.append(time.time() - start)
        opt_time = np.median(times_opt)
        
        # Calculate difference
        diff = np.abs(v1 - v2).max()
        
        # Print results
        print(f"Numba:     {numba_time:.4f}s (median of {num_runs} runs)")
        print(f"Optimized: {opt_time:.4f}s (median of {num_runs} runs)")
        print(f"Speedup:   {numba_time/opt_time:.2f}x")
        print(f"Max diff:  {diff:.8f}")
        
        # Memory usage estimate
        mem_usage = (2 * T * 4 * VH * VW * 4) / (1024**2)  # 2 videos, 4 channels, float32
        print(f"Output size: ~{mem_usage:.1f} MB")

def test_accuracy():
    """Test numerical accuracy across different scenarios."""
    print("\n=== Accuracy Tests ===\n")
    
    test_cases = [
        ("Non-overlapping", 3, 3, 100, 100, 3.0, [(25, 25), (50, 50), (75, 75)]),
        ("Overlapping", 3, 4, 100, 100, 8.0, [(50, 50), (55, 50), (50, 55), (55, 55)]),
        ("Edge positions", 2, 4, 50, 50, 5.0, [(-5, 25), (25, -5), (55, 25), (25, 55)]),
        ("Single pixel", 1, 1, 10, 10, 0.5, [(5, 5)]),
    ]
    
    for test_name, T, N, VH, VW, sigma, positions in test_cases:
        print(f"{test_name}:")
        
        # Create specific test data
        tracks = np.zeros((T, N, 4), dtype=np.float32)
        for i, (x, y) in enumerate(positions[:N]):
            tracks[:, i, 0] = x
            tracks[:, i, 1] = y
            tracks[:, i, 3] = 1  # visible
        
        counter_tracks = tracks.copy()
        counter_tracks[:, :, :2] += 2  # Small offset
        
        colors = np.eye(4, dtype=np.float32)[:N]  # Different colors
        
        # Run both versions
        v1, c1 = _draw_multiple_gaussians_numba(tracks, counter_tracks, colors, VH, VW, sigma)
        v2, c2 = draw_multiple_gaussians_fast(tracks, counter_tracks, colors, VH, VW, sigma)
        
        # Compare
        video_diff = np.abs(v1 - v2)
        counter_diff = np.abs(c1 - c2)
        
        print(f"  Video   - Max: {video_diff.max():.8f}, Mean: {video_diff.mean():.8f}")
        print(f"  Counter - Max: {counter_diff.max():.8f}, Mean: {counter_diff.mean():.8f}")
        print()

def profile_bottlenecks():
    """Profile where time is spent."""
    print("\n=== Profiling Bottlenecks ===\n")
    
    T, N, VH, VW = 10, 100, 480, 640
    tracks, counter_tracks, colors = create_test_data(T, N, VH, VW)
    
    # Select tracks
    selected_idx = np.random.choice(N, 7, replace=False)
    selected_tracks = tracks[:, selected_idx, :]
    selected_counter = counter_tracks[:, selected_idx, :]
    selected_colors = colors[:7]
    
    # Time different parts
    start = time.time()
    for _ in range(10):
        # Just the gaussian kernel creation
        radius = int(np.ceil(3.0 * 5.0))
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        gaussian = np.exp(-(x*x + y*y) / (2.0 * 5.0 * 5.0))
    kernel_time = (time.time() - start) / 10
    
    print(f"Gaussian kernel creation: {kernel_time*1000:.2f} ms")
    
    # Time array allocation
    start = time.time()
    for _ in range(10):
        video = np.zeros((T, 4, VH, VW), dtype=np.float32)
        counter = np.zeros((T, 4, VH, VW), dtype=np.float32)
    alloc_time = (time.time() - start) / 10
    
    print(f"Array allocation: {alloc_time*1000:.2f} ms")
    print(f"Total expected overhead: {(kernel_time + alloc_time)*1000:.2f} ms")

if __name__ == "__main__":
    benchmark_implementations()
    test_accuracy()
    profile_bottlenecks()