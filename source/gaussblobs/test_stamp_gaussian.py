import rp
import torch
import numpy as np
import time
from render_tracks import _draw_multiple_gaussians_numba

def create_gaussian_sprite(sigma, dtype=torch.float32):
    """Create a gaussian sprite with given sigma."""
    radius = int(np.ceil(3.0 * sigma))
    size = 2 * radius + 1
    
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    y = y - radius
    x = x - radius
    
    # Compute gaussian
    dist_sq = x * x + y * y
    gaussian = torch.exp(-dist_sq / (2.0 * sigma * sigma))
    
    return gaussian.to(dtype)

def draw_multiple_gaussians_torch(tracks_np, counter_tracks_np, colors_np, VH, VW, sigma):
    """Non-numba implementation using rp.stamp_tensor."""
    T, num_tracks, XYZV = tracks_np.shape
    num_colors, C = colors_np.shape
    Xi, Yi, Zi, Vi = 0, 1, 2, 3
    
    # Convert to torch tensors
    tracks = torch.from_numpy(tracks_np)
    counter_tracks = torch.from_numpy(counter_tracks_np)
    colors = torch.from_numpy(colors_np)
    
    # Initialize output tensors
    video = torch.zeros((T, C, VH, VW), dtype=torch.float32)
    counter_video = torch.zeros((T, C, VH, VW), dtype=torch.float32)
    
    # Pre-compute gaussian sprite
    gaussian_sprite = create_gaussian_sprite(sigma)
    
    # Process each frame
    for t in range(T):
        for track_idx in range(min(num_tracks, num_colors)):
            color_idx = track_idx % num_colors
            color = colors[color_idx]
            
            # Process original track
            center_x = tracks[t, track_idx, Xi].item()
            center_y = tracks[t, track_idx, Yi].item()
            vv = int(tracks[t, track_idx, Vi].item())
            
            if vv:
                # Create colored gaussian sprite for this track
                if C == 4:
                    # RGBA - use alpha channel
                    alpha_sprite = gaussian_sprite * color[3]
                    colored_sprite = gaussian_sprite.unsqueeze(0) * color.view(C, 1, 1)
                else:
                    # RGB - use color intensity as alpha
                    alpha_sprite = gaussian_sprite * color.max()
                    colored_sprite = gaussian_sprite.unsqueeze(0) * color.view(C, 1, 1)
                
                # For each channel, apply alpha compositing
                for c in range(C):
                    # Create sprite for this channel
                    channel_sprite = colored_sprite[c]
                    
                    # Define custom compositing function for this channel
                    def alpha_composite(canvas_val, sprite_val):
                        # Get the overlapping region of alpha_sprite
                        # The sprite_val shape matches the overlapping region
                        sprite_h, sprite_w = sprite_val.shape
                        center_h, center_w = alpha_sprite.shape[0] // 2, alpha_sprite.shape[1] // 2
                        
                        # Calculate the bounds of the overlapping region in the sprite
                        y_off = int(center_y - center_h)
                        x_off = int(center_x - center_w)
                        
                        # Determine sprite region bounds
                        sprite_y_start = max(0, -y_off)
                        sprite_x_start = max(0, -x_off)
                        sprite_y_end = sprite_y_start + sprite_h
                        sprite_x_end = sprite_x_start + sprite_w
                        
                        # Extract the corresponding alpha region
                        alpha_region = alpha_sprite[sprite_y_start:sprite_y_end, sprite_x_start:sprite_x_end]
                        
                        # Apply alpha compositing
                        return canvas_val * (1 - alpha_region) + sprite_val
                    
                    # Stamp the gaussian onto the canvas
                    video[t, c] = rp.stamp_tensor(
                        canvas=video[t, c],
                        sprite=channel_sprite,
                        offset=[center_y, center_x],
                        mode=alpha_composite,
                        sprite_origin='center',
                        mutate=True
                    )
            
            # Process counter track
            counter_center_x = counter_tracks[t, track_idx, Xi].item()
            counter_center_y = counter_tracks[t, track_idx, Yi].item()
            counter_vv = int(counter_tracks[t, track_idx, Vi].item())
            
            if counter_vv:
                # Create colored gaussian sprite for this track
                if C == 4:
                    # RGBA - use alpha channel
                    alpha_sprite = gaussian_sprite * color[3]
                    colored_sprite = gaussian_sprite.unsqueeze(0) * color.view(C, 1, 1)
                else:
                    # RGB - use color intensity as alpha
                    alpha_sprite = gaussian_sprite * color.max()
                    colored_sprite = gaussian_sprite.unsqueeze(0) * color.view(C, 1, 1)
                
                # For each channel, apply alpha compositing
                for c in range(C):
                    # Create sprite for this channel
                    channel_sprite = colored_sprite[c]
                    
                    # Define custom compositing function for this channel
                    def alpha_composite(canvas_val, sprite_val):
                        # Get the overlapping region of alpha_sprite
                        # The sprite_val shape matches the overlapping region
                        sprite_h, sprite_w = sprite_val.shape
                        center_h, center_w = alpha_sprite.shape[0] // 2, alpha_sprite.shape[1] // 2
                        
                        # Calculate the bounds of the overlapping region in the sprite
                        y_off = int(counter_center_y - center_h)
                        x_off = int(counter_center_x - center_w)
                        
                        # Determine sprite region bounds
                        sprite_y_start = max(0, -y_off)
                        sprite_x_start = max(0, -x_off)
                        sprite_y_end = sprite_y_start + sprite_h
                        sprite_x_end = sprite_x_start + sprite_w
                        
                        # Extract the corresponding alpha region
                        alpha_region = alpha_sprite[sprite_y_start:sprite_y_end, sprite_x_start:sprite_x_end]
                        
                        # Apply alpha compositing
                        return canvas_val * (1 - alpha_region) + sprite_val
                    
                    # Stamp the gaussian onto the canvas
                    counter_video[t, c] = rp.stamp_tensor(
                        canvas=counter_video[t, c],
                        sprite=channel_sprite,
                        offset=[counter_center_y, counter_center_x],
                        mode=alpha_composite,
                        sprite_origin='center',
                        mutate=True
                    )
    
    # Convert back to numpy
    return video.numpy(), counter_video.numpy()

def test_non_overlapping_gaussians():
    """Test with non-overlapping gaussians."""
    print("\n=== Testing Non-Overlapping Gaussians ===")
    
    # Create test data
    T, VH, VW = 3, 100, 100
    num_tracks = 3
    sigma = 3.0
    
    # Non-overlapping positions
    tracks = np.zeros((T, num_tracks, 4), dtype=np.float32)
    counter_tracks = np.zeros((T, num_tracks, 4), dtype=np.float32)
    
    # Set positions far apart
    positions = [(25, 25), (50, 50), (75, 75)]
    for i, (x, y) in enumerate(positions):
        for t in range(T):
            tracks[t, i, 0] = x  # X
            tracks[t, i, 1] = y  # Y
            tracks[t, i, 2] = 0  # Z
            tracks[t, i, 3] = 1  # Visibility
            
            counter_tracks[t, i, 0] = x + 5  # Offset slightly
            counter_tracks[t, i, 1] = y + 5
            counter_tracks[t, i, 2] = 0
            counter_tracks[t, i, 3] = 1
    
    # Colors
    colors = np.array([
        [1, 0, 0, 1],  # Red
        [0, 1, 0, 1],  # Green
        [0, 0, 1, 1],  # Blue
    ], dtype=np.float32)
    
    # Run both implementations
    start = time.time()
    video_numba, counter_numba = _draw_multiple_gaussians_numba(tracks, counter_tracks, colors, VH, VW, sigma)
    numba_time = time.time() - start
    
    start = time.time()
    video_torch, counter_torch = draw_multiple_gaussians_torch(tracks, counter_tracks, colors, VH, VW, sigma)
    torch_time = time.time() - start
    
    # Compare results
    video_diff = np.abs(video_numba - video_torch)
    counter_diff = np.abs(counter_numba - counter_torch)
    
    print(f"Numba time: {numba_time:.4f}s")
    print(f"Torch time: {torch_time:.4f}s")
    print(f"Speedup: {numba_time/torch_time:.2f}x")
    print(f"Max video difference: {video_diff.max():.8f}")
    print(f"Mean video difference: {video_diff.mean():.8f}")
    print(f"Max counter difference: {counter_diff.max():.8f}")
    print(f"Mean counter difference: {counter_diff.mean():.8f}")
    
    # Visual verification
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show first frame
    axes[0, 0].imshow(video_numba[0].transpose(1, 2, 0))
    axes[0, 0].set_title("Numba Video")
    
    axes[0, 1].imshow(video_torch[0].transpose(1, 2, 0))
    axes[0, 1].set_title("Torch Video")
    
    axes[0, 2].imshow(video_diff[0].transpose(1, 2, 0) * 100)  # Scale up diff
    axes[0, 2].set_title("Difference x100")
    
    axes[1, 0].imshow(counter_numba[0].transpose(1, 2, 0))
    axes[1, 0].set_title("Numba Counter")
    
    axes[1, 1].imshow(counter_torch[0].transpose(1, 2, 0))
    axes[1, 1].set_title("Torch Counter")
    
    axes[1, 2].imshow(counter_diff[0].transpose(1, 2, 0) * 100)
    axes[1, 2].set_title("Counter Diff x100")
    
    plt.tight_layout()
    plt.savefig("non_overlapping_test.png")
    print("Saved visualization to non_overlapping_test.png")

def test_overlapping_gaussians():
    """Test with overlapping gaussians."""
    print("\n=== Testing Overlapping Gaussians ===")
    
    # Create test data
    T, VH, VW = 3, 100, 100
    num_tracks = 4
    sigma = 8.0  # Larger sigma for more overlap
    
    # Overlapping positions
    tracks = np.zeros((T, num_tracks, 4), dtype=np.float32)
    counter_tracks = np.zeros((T, num_tracks, 4), dtype=np.float32)
    
    # Set positions close together
    center_x, center_y = 50, 50
    radius = 15
    angles = np.linspace(0, 2*np.pi, num_tracks, endpoint=False)
    
    for i, angle in enumerate(angles):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        for t in range(T):
            tracks[t, i, 0] = x
            tracks[t, i, 1] = y
            tracks[t, i, 2] = 0
            tracks[t, i, 3] = 1
            
            # Counter tracks move inward
            counter_tracks[t, i, 0] = center_x + (radius/2) * np.cos(angle)
            counter_tracks[t, i, 1] = center_y + (radius/2) * np.sin(angle)
            counter_tracks[t, i, 2] = 0
            counter_tracks[t, i, 3] = 1
    
    # Semi-transparent colors for better overlap visualization
    colors = np.array([
        [1, 0, 0, 0.7],   # Red
        [0, 1, 0, 0.7],   # Green
        [0, 0, 1, 0.7],   # Blue
        [1, 1, 0, 0.7],   # Yellow
    ], dtype=np.float32)
    
    # Run both implementations
    start = time.time()
    video_numba, counter_numba = _draw_multiple_gaussians_numba(tracks, counter_tracks, colors, VH, VW, sigma)
    numba_time = time.time() - start
    
    start = time.time()
    video_torch, counter_torch = draw_multiple_gaussians_torch(tracks, counter_tracks, colors, VH, VW, sigma)
    torch_time = time.time() - start
    
    # Compare results
    video_diff = np.abs(video_numba - video_torch)
    counter_diff = np.abs(counter_numba - counter_torch)
    
    print(f"Numba time: {numba_time:.4f}s")
    print(f"Torch time: {torch_time:.4f}s")
    print(f"Speedup: {numba_time/torch_time:.2f}x")
    print(f"Max video difference: {video_diff.max():.8f}")
    print(f"Mean video difference: {video_diff.mean():.8f}")
    print(f"Max counter difference: {counter_diff.max():.8f}")
    print(f"Mean counter difference: {counter_diff.mean():.8f}")
    
    # Visual verification
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show first frame
    axes[0, 0].imshow(video_numba[0].transpose(1, 2, 0))
    axes[0, 0].set_title("Numba Video (Overlapping)")
    
    axes[0, 1].imshow(video_torch[0].transpose(1, 2, 0))
    axes[0, 1].set_title("Torch Video (Overlapping)")
    
    axes[0, 2].imshow(video_diff[0].transpose(1, 2, 0) * 100)
    axes[0, 2].set_title("Difference x100")
    
    axes[1, 0].imshow(counter_numba[0].transpose(1, 2, 0))
    axes[1, 0].set_title("Numba Counter (Overlapping)")
    
    axes[1, 1].imshow(counter_torch[0].transpose(1, 2, 0))
    axes[1, 1].set_title("Torch Counter (Overlapping)")
    
    axes[1, 2].imshow(counter_diff[0].transpose(1, 2, 0) * 100)
    axes[1, 2].set_title("Counter Diff x100")
    
    plt.tight_layout()
    plt.savefig("overlapping_test.png")
    print("Saved visualization to overlapping_test.png")

def test_edge_cases():
    """Test edge cases like out-of-bounds positions."""
    print("\n=== Testing Edge Cases ===")
    
    T, VH, VW = 2, 50, 50
    num_tracks = 4
    sigma = 5.0
    
    tracks = np.zeros((T, num_tracks, 4), dtype=np.float32)
    counter_tracks = np.zeros((T, num_tracks, 4), dtype=np.float32)
    
    # Edge positions
    positions = [
        (-5, 25),      # Left edge (partially out)
        (25, -5),      # Top edge (partially out)
        (55, 25),      # Right edge (partially out)
        (25, 55),      # Bottom edge (partially out)
    ]
    
    for i, (x, y) in enumerate(positions):
        tracks[0, i, 0] = x
        tracks[0, i, 1] = y
        tracks[0, i, 3] = 1
        
        counter_tracks[0, i, 0] = VW - x - 1  # Mirror position
        counter_tracks[0, i, 1] = VH - y - 1
        counter_tracks[0, i, 3] = 1
    
    colors = np.array([[1, 1, 1, 1]] * 4, dtype=np.float32)
    
    # Run both
    video_numba, counter_numba = _draw_multiple_gaussians_numba(tracks, counter_tracks, colors, VH, VW, sigma)
    video_torch, counter_torch = draw_multiple_gaussians_torch(tracks, counter_tracks, colors, VH, VW, sigma)
    
    # Compare
    video_diff = np.abs(video_numba - video_torch)
    counter_diff = np.abs(counter_numba - counter_torch)
    
    print(f"Edge cases - Max video difference: {video_diff.max():.8f}")
    print(f"Edge cases - Max counter difference: {counter_diff.max():.8f}")

if __name__ == "__main__":
    print("Testing gaussian rendering implementations...")
    
    # Run tests
    test_non_overlapping_gaussians()
    test_overlapping_gaussians()
    test_edge_cases()
    
    print("\nAll tests completed!")