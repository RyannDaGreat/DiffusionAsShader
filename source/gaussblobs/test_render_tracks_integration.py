import rp
import torch
import numpy as np
import time
from render_tracks import random_7_gaussians_video, _draw_multiple_gaussians_numba
from optimized_gaussian_render import draw_multiple_gaussians_fast

def test_integration():
    """Test that the optimized version works as a drop-in replacement."""
    print("Testing integration with render_tracks.py...")
    
    # Create test data
    T, N, VH, VW = 5, 100, 128, 128
    tracks = torch.randn(T, N, 4) * 50 + 64  # Random positions around center
    tracks[:, :, 2] = 0  # Z = 0
    tracks[:, :, 3] = (torch.rand(T, N) > 0.3).float()  # 70% visible
    
    counter_tracks = tracks + torch.randn_like(tracks) * 5  # Slight offset
    counter_tracks[:, :, 3] = tracks[:, :, 3]  # Same visibility
    
    # Test with default 7 colors
    print("\n1. Testing with default 7 colors...")
    start = time.time()
    video_orig, counter_orig = random_7_gaussians_video(tracks, counter_tracks, VH, VW, sigma=5.0)
    orig_time = time.time() - start
    print(f"Original implementation time: {orig_time:.3f}s")
    
    # Test with 'random_of_7'
    print("\n2. Testing with 'random_of_7' colors...")
    video_rand, counter_rand = random_7_gaussians_video(
        tracks, counter_tracks, VH, VW, sigma=5.0, blob_colors='random_of_7'
    )
    
    # Test direct comparison between numba and optimized
    print("\n3. Direct comparison test...")
    colors = np.array([
        [1, 0, 0, 0.8],
        [0, 1, 0, 0.8],
        [0, 0, 1, 0.8],
    ], dtype=np.float32)
    
    # Select same tracks as random_7_gaussians_video would
    torch.manual_seed(42)
    selected_indices = torch.randperm(N)[:3]
    selected_tracks = tracks[:, selected_indices, :].cpu().float().numpy()
    selected_counter = counter_tracks[:, selected_indices, :].cpu().float().numpy()
    
    # Run both implementations
    start = time.time()
    video_numba, counter_numba = _draw_multiple_gaussians_numba(
        selected_tracks, selected_counter, colors, VH, VW, 5.0
    )
    numba_time = time.time() - start
    
    start = time.time()
    video_opt, counter_opt = draw_multiple_gaussians_fast(
        selected_tracks, selected_counter, colors, VH, VW, 5.0
    )
    opt_time = time.time() - start
    
    # Compare results
    video_diff = np.abs(video_numba - video_opt)
    counter_diff = np.abs(counter_numba - counter_opt)
    
    print(f"\nNumba time: {numba_time:.3f}s")
    print(f"Optimized time: {opt_time:.3f}s")
    print(f"Speedup: {numba_time/opt_time:.1f}x")
    print(f"Max video difference: {video_diff.max():.8f}")
    print(f"Mean video difference: {video_diff.mean():.8f}")
    print(f"Max counter difference: {counter_diff.max():.8f}")
    print(f"Mean counter difference: {counter_diff.mean():.8f}")
    
    # Visual test
    print("\n4. Visual comparison...")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    frame = 0
    axes[0, 0].imshow(video_numba[frame].transpose(1, 2, 0))
    axes[0, 0].set_title("Numba Implementation")
    
    axes[0, 1].imshow(video_opt[frame].transpose(1, 2, 0))
    axes[0, 1].set_title("Optimized Implementation")
    
    axes[1, 0].imshow(video_diff[frame].transpose(1, 2, 0) * 100)
    axes[1, 0].set_title("Difference x100")
    
    # Show sum across frames
    axes[1, 1].imshow(video_diff.sum(axis=0).transpose(1, 2, 0))
    axes[1, 1].set_title("Cumulative Difference")
    
    plt.tight_layout()
    plt.savefig("integration_test.png")
    print("Saved visual comparison to integration_test.png")
    
    # Test edge cases
    print("\n5. Testing edge cases...")
    
    # Empty tracks
    empty_tracks = np.zeros((1, 0, 4), dtype=np.float32)
    empty_colors = np.zeros((0, 4), dtype=np.float32)
    video_empty, counter_empty = draw_multiple_gaussians_fast(
        empty_tracks, empty_tracks, empty_colors, 10, 10, 1.0
    )
    assert video_empty.shape == (1, 4, 10, 10) and np.all(video_empty == 0)
    print("✓ Empty tracks test passed")
    
    # All invisible tracks
    invisible_tracks = np.ones((2, 5, 4), dtype=np.float32)
    invisible_tracks[:, :, 3] = 0  # All invisible
    video_inv, counter_inv = draw_multiple_gaussians_fast(
        invisible_tracks, invisible_tracks, colors, 20, 20, 2.0
    )
    assert np.all(video_inv == 0) and np.all(counter_inv == 0)
    print("✓ Invisible tracks test passed")
    
    # Out of bounds tracks
    oob_tracks = np.array([[[150, 150, 0, 1]]], dtype=np.float32)  # Way out of 50x50 canvas
    oob_colors = np.array([[1, 1, 1, 1]], dtype=np.float32)
    video_oob, counter_oob = draw_multiple_gaussians_fast(
        oob_tracks, oob_tracks, oob_colors, 50, 50, 3.0
    )
    # Should handle gracefully without errors
    print("✓ Out of bounds test passed")
    
    print("\n✅ All integration tests passed!")
    
def benchmark_performance():
    """Benchmark performance on realistic workload."""
    print("\n=== Performance Benchmark ===")
    
    sizes = [(10, 50), (20, 100), (49, 490)]  # (frames, tracks)
    
    for T, N in sizes:
        print(f"\nBenchmarking {T} frames, {N} tracks...")
        
        # Generate realistic data
        tracks = np.random.rand(T, N, 4).astype(np.float32)
        tracks[:, :, 0] *= 720  # X positions
        tracks[:, :, 1] *= 480  # Y positions
        tracks[:, :, 2] = 0     # Z
        tracks[:, :, 3] = (np.random.rand(T, N) > 0.2).astype(np.float32)  # 80% visible
        
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
        
        # Select 7 random tracks
        selected_idx = np.random.choice(N, min(7, N), replace=False)
        selected_tracks = tracks[:, selected_idx, :]
        selected_counter = counter_tracks[:, selected_idx, :]
        
        # Benchmark numba
        start = time.time()
        v1, c1 = _draw_multiple_gaussians_numba(
            selected_tracks, selected_counter, colors[:len(selected_idx)], 480, 720, 5.0
        )
        numba_time = time.time() - start
        
        # Benchmark optimized
        start = time.time()
        v2, c2 = draw_multiple_gaussians_fast(
            selected_tracks, selected_counter, colors[:len(selected_idx)], 480, 720, 5.0
        )
        opt_time = time.time() - start
        
        diff = np.abs(v1 - v2).max()
        
        print(f"  Numba: {numba_time:.3f}s")
        print(f"  Optimized: {opt_time:.3f}s")
        print(f"  Speedup: {numba_time/opt_time:.1f}x")
        print(f"  Max difference: {diff:.8f}")

if __name__ == "__main__":
    test_integration()
    benchmark_performance()