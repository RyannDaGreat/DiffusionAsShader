import numpy as np
from render_tracks import _draw_multiple_gaussians_numba
from optimized_gaussian_render_fast import draw_multiple_gaussians_fast

def test_alpha_blending_logic():
    """Test to understand the exact alpha blending difference."""
    
    # Simple test case - single gaussian, single color
    T, N, VH, VW = 1, 1, 20, 20
    sigma = 3.0
    
    # Place gaussian at center
    tracks = np.array([[[10, 10, 0, 1]]], dtype=np.float32)
    counter_tracks = tracks.copy()
    
    # Test different color/alpha combinations
    test_cases = [
        ("RGBA full", np.array([[1, 0, 0, 1]], dtype=np.float32)),
        ("RGBA half", np.array([[1, 0, 0, 0.5]], dtype=np.float32)),
        ("RGB red", np.array([[1, 0, 0]], dtype=np.float32)),
        ("RGB gray", np.array([[0.5, 0.5, 0.5]], dtype=np.float32)),
    ]
    
    for name, colors in test_cases:
        print(f"\n{name} - shape {colors.shape}:")
        
        v1, c1 = _draw_multiple_gaussians_numba(tracks, counter_tracks, colors, VH, VW, sigma)
        v2, c2 = draw_multiple_gaussians_fast(tracks, counter_tracks, colors, VH, VW, sigma)
        
        # Check center pixel
        cy, cx = 10, 10
        C = colors.shape[1]
        
        print("Center pixel values:")
        for ch in range(C):
            numba_val = v1[0, ch, cy, cx]
            opt_val = v2[0, ch, cy, cx]
            diff = abs(numba_val - opt_val)
            print(f"  Ch{ch}: numba={numba_val:.6f}, opt={opt_val:.6f}, diff={diff:.6f}")
        
        # Check a pixel at distance 1
        print("Pixel at (10,11):")
        for ch in range(C):
            numba_val = v1[0, ch, cy, cx+1]
            opt_val = v2[0, ch, cy, cx+1]
            diff = abs(numba_val - opt_val)
            print(f"  Ch{ch}: numba={numba_val:.6f}, opt={opt_val:.6f}, diff={diff:.6f}")
        
        # Overall stats
        diff = np.abs(v1 - v2)
        print(f"Max diff: {diff.max():.8f}")
        print(f"Mean diff: {diff.mean():.8f}")
        print(f"Non-zero pixels in diff: {(diff > 0.0001).sum()}")

if __name__ == "__main__":
    test_alpha_blending_logic()