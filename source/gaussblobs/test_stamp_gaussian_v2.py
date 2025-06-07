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

def draw_multiple_gaussians_torch_v2(tracks_np, counter_tracks_np, colors_np, VH, VW, sigma):
    """Optimized non-numba implementation using rp.stamp_tensor."""
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
    
    # Pre-compute all colored gaussian sprites for each color
    colored_gaussians = []
    for color_idx in range(num_colors):
        color = colors[color_idx]
        # Create RGBA gaussian sprite
        colored_gaussian = gaussian_sprite.unsqueeze(0) * color.view(C, 1, 1)
        colored_gaussians.append(colored_gaussian)
    
    # Process each frame
    for t in range(T):
        # Create temporary canvases for this frame
        frame_video = video[t].clone()
        frame_counter = counter_video[t].clone()
        
        for track_idx in range(min(num_tracks, num_colors)):
            color_idx = track_idx % num_colors
            colored_gaussian = colored_gaussians[color_idx]
            
            # Process original track
            if int(tracks[t, track_idx, Vi].item()):
                center_x = tracks[t, track_idx, Xi].item()
                center_y = tracks[t, track_idx, Yi].item()
                
                # Define alpha compositing function based on the numba implementation
                if C == 4:
                    # RGBA - use alpha channel
                    alpha_weight = colors[color_idx, 3].item()
                    
                    def alpha_composite_rgba(canvas_val, sprite_val):
                        # sprite_val already contains gaussian * color
                        # We need gaussian values to compute alpha
                        gauss_val = sprite_val / (colors[color_idx].view(C, 1, 1) + 1e-10)
                        alpha = alpha_weight * gauss_val[3]  # Use the alpha channel
                        return canvas_val * (1 - alpha) + sprite_val
                    
                    frame_video = rp.stamp_tensor(
                        canvas=frame_video,
                        sprite=colored_gaussian,
                        offset=[center_y, center_x],
                        mode=alpha_composite_rgba,
                        sprite_origin='center',
                        mutate=True
                    )
                else:
                    # RGB - use color intensity as alpha
                    def alpha_composite_rgb(canvas_val, sprite_val):
                        # For RGB, the numba code uses: alpha = color_value * gauss_weight
                        # sprite_val = gaussian * color, so we need to extract gaussian
                        # Since we're stamping all channels at once, we need per-channel alpha
                        gauss_val = sprite_val / (colors[color_idx].view(C, 1, 1) + 1e-10)
                        # Use the gaussian value from any channel (they're all the same)
                        gauss = gauss_val[0]  # Just use first channel's gaussian
                        
                        # Apply per-channel alpha compositing
                        result = canvas_val.clone()
                        for c in range(C):
                            color_val = colors[color_idx, c].item()
                            alpha = color_val * gauss
                            result[c] = canvas_val[c] * (1 - alpha) + color_val * gauss
                        return result
                    
                    frame_video = rp.stamp_tensor(
                        canvas=frame_video,
                        sprite=colored_gaussian,
                        offset=[center_y, center_x],
                        mode=alpha_composite_rgb,
                        sprite_origin='center',
                        mutate=True
                    )
            
            # Process counter track
            if int(counter_tracks[t, track_idx, Vi].item()):
                counter_center_x = counter_tracks[t, track_idx, Xi].item()
                counter_center_y = counter_tracks[t, track_idx, Yi].item()
                
                # Same compositing for counter track
                if C == 4:
                    alpha_weight = colors[color_idx, 3].item()
                    
                    def alpha_composite_rgba_counter(canvas_val, sprite_val):
                        gauss_val = sprite_val / (colors[color_idx].view(C, 1, 1) + 1e-10)
                        alpha = alpha_weight * gauss_val[3]
                        return canvas_val * (1 - alpha) + sprite_val
                    
                    frame_counter = rp.stamp_tensor(
                        canvas=frame_counter,
                        sprite=colored_gaussian,
                        offset=[counter_center_y, counter_center_x],
                        mode=alpha_composite_rgba_counter,
                        sprite_origin='center',
                        mutate=True
                    )
                else:
                    def alpha_composite_rgb_counter(canvas_val, sprite_val):
                        gauss_val = sprite_val / (colors[color_idx].view(C, 1, 1) + 1e-10)
                        gauss = gauss_val[0]
                        
                        result = canvas_val.clone()
                        for c in range(C):
                            color_val = colors[color_idx, c].item()
                            alpha = color_val * gauss
                            result[c] = canvas_val[c] * (1 - alpha) + color_val * gauss
                        return result
                    
                    frame_counter = rp.stamp_tensor(
                        canvas=frame_counter,
                        sprite=colored_gaussian,
                        offset=[counter_center_y, counter_center_x],
                        mode=alpha_composite_rgb_counter,
                        sprite_origin='center',
                        mutate=True
                    )
        
        # Store the composited frames
        video[t] = frame_video
        counter_video[t] = frame_counter
    
    # Convert back to numpy
    return video.numpy(), counter_video.numpy()

def draw_multiple_gaussians_torch_v3(tracks_np, counter_tracks_np, colors_np, VH, VW, sigma):
    """Even simpler implementation - stamp gaussians sequentially with proper alpha blending."""
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
    gaussian_base = create_gaussian_sprite(sigma)
    
    # Process each frame and track
    for t in range(T):
        for track_idx in range(min(num_tracks, num_colors)):
            color_idx = track_idx % num_colors
            color = colors[color_idx]
            
            # Process original track
            if int(tracks[t, track_idx, Vi].item()):
                center_x = tracks[t, track_idx, Xi].item()
                center_y = tracks[t, track_idx, Yi].item()
                
                # For each channel, create the sprite and composite
                for c in range(C):
                    color_value = color[c].item()
                    
                    # Create gaussian sprite for this channel
                    if C == 4:
                        # RGBA mode: use actual alpha channel
                        alpha_value = color[3].item()
                        gaussian_alpha = gaussian_base * alpha_value
                        gaussian_color = gaussian_base * color_value
                    else:
                        # RGB mode: use color value as alpha
                        gaussian_alpha = gaussian_base * color_value
                        gaussian_color = gaussian_base * color_value
                    
                    # Define compositing function for this specific alpha
                    def composite_fn(canvas_val, sprite_val):
                        # Get alpha for this specific sprite region
                        h, w = sprite_val.shape
                        cy, cx = int(center_y), int(center_x)
                        
                        # Calculate sprite bounds in gaussian_alpha
                        gh, gw = gaussian_alpha.shape
                        gy_start = max(0, gh//2 - cy)
                        gx_start = max(0, gw//2 - cx)
                        gy_end = min(gh, gy_start + h)
                        gx_end = min(gw, gx_start + w)
                        
                        # Handle edge cases
                        if cy < gh//2:
                            gy_start = gh//2 - cy
                            gy_end = gy_start + h
                        if cx < gw//2:
                            gx_start = gw//2 - cx
                            gx_end = gx_start + w
                        
                        alpha_region = gaussian_alpha[gy_start:gy_end, gx_start:gx_end]
                        return canvas_val * (1 - alpha_region) + sprite_val
                    
                    # Stamp this channel
                    video[t, c] = rp.stamp_tensor(
                        canvas=video[t, c],
                        sprite=gaussian_color,
                        offset=[center_y, center_x],
                        mode=composite_fn,
                        sprite_origin='center',
                        mutate=True
                    )
            
            # Process counter track
            if int(counter_tracks[t, track_idx, Vi].item()):
                counter_center_x = counter_tracks[t, track_idx, Xi].item()
                counter_center_y = counter_tracks[t, track_idx, Yi].item()
                
                # Same process for counter track
                for c in range(C):
                    color_value = color[c].item()
                    
                    if C == 4:
                        alpha_value = color[3].item()
                        gaussian_alpha = gaussian_base * alpha_value
                        gaussian_color = gaussian_base * color_value
                    else:
                        gaussian_alpha = gaussian_base * color_value
                        gaussian_color = gaussian_base * color_value
                    
                    def composite_fn_counter(canvas_val, sprite_val):
                        h, w = sprite_val.shape
                        cy, cx = int(counter_center_y), int(counter_center_x)
                        
                        gh, gw = gaussian_alpha.shape
                        gy_start = max(0, gh//2 - cy)
                        gx_start = max(0, gw//2 - cx)
                        gy_end = min(gh, gy_start + h)
                        gx_end = min(gw, gx_start + w)
                        
                        if cy < gh//2:
                            gy_start = gh//2 - cy
                            gy_end = gy_start + h
                        if cx < gw//2:
                            gx_start = gw//2 - cx
                            gx_end = gx_start + w
                        
                        alpha_region = gaussian_alpha[gy_start:gy_end, gx_start:gx_end]
                        return canvas_val * (1 - alpha_region) + sprite_val
                    
                    counter_video[t, c] = rp.stamp_tensor(
                        canvas=counter_video[t, c],
                        sprite=gaussian_color,
                        offset=[counter_center_y, counter_center_x],
                        mode=composite_fn_counter,
                        sprite_origin='center',
                        mutate=True
                    )
    
    # Convert back to numpy
    return video.numpy(), counter_video.numpy()

def test_implementations():
    """Test all implementations for correctness and performance."""
    print("Testing gaussian rendering implementations...")
    
    # Test 1: Non-overlapping gaussians
    print("\n=== Test 1: Non-Overlapping Gaussians ===")
    T, VH, VW = 3, 100, 100
    num_tracks = 3
    sigma = 3.0
    
    tracks = np.zeros((T, num_tracks, 4), dtype=np.float32)
    counter_tracks = np.zeros((T, num_tracks, 4), dtype=np.float32)
    
    positions = [(25, 25), (50, 50), (75, 75)]
    for i, (x, y) in enumerate(positions):
        for t in range(T):
            tracks[t, i] = [x, y, 0, 1]
            counter_tracks[t, i] = [x + 5, y + 5, 0, 1]
    
    colors = np.array([
        [1, 0, 0, 1],  # Red
        [0, 1, 0, 1],  # Green  
        [0, 0, 1, 1],  # Blue
    ], dtype=np.float32)
    
    # Test all versions
    start = time.time()
    video_numba, counter_numba = _draw_multiple_gaussians_numba(tracks, counter_tracks, colors, VH, VW, sigma)
    numba_time = time.time() - start
    
    start = time.time()
    video_v3, counter_v3 = draw_multiple_gaussians_torch_v3(tracks, counter_tracks, colors, VH, VW, sigma)
    v3_time = time.time() - start
    
    # Compare results
    video_diff = np.abs(video_numba - video_v3)
    counter_diff = np.abs(counter_numba - counter_v3)
    
    print(f"Numba time: {numba_time:.4f}s")
    print(f"Torch V3 time: {v3_time:.4f}s")
    print(f"Speedup: {numba_time/v3_time:.2f}x")
    print(f"Max video difference: {video_diff.max():.8f}")
    print(f"Mean video difference: {video_diff.mean():.8f}")
    print(f"Max counter difference: {counter_diff.max():.8f}")
    print(f"Mean counter difference: {counter_diff.mean():.8f}")
    
    # Test 2: Overlapping gaussians
    print("\n=== Test 2: Overlapping Gaussians ===")
    num_tracks = 4
    sigma = 8.0
    
    tracks = np.zeros((T, num_tracks, 4), dtype=np.float32)
    counter_tracks = np.zeros((T, num_tracks, 4), dtype=np.float32)
    
    center_x, center_y = 50, 50
    radius = 15
    angles = np.linspace(0, 2*np.pi, num_tracks, endpoint=False)
    
    for i, angle in enumerate(angles):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        for t in range(T):
            tracks[t, i] = [x, y, 0, 1]
            counter_tracks[t, i] = [center_x + (radius/2) * np.cos(angle),
                                   center_y + (radius/2) * np.sin(angle), 0, 1]
    
    colors = np.array([
        [1, 0, 0, 0.7],
        [0, 1, 0, 0.7],
        [0, 0, 1, 0.7],
        [1, 1, 0, 0.7],
    ], dtype=np.float32)
    
    video_numba, counter_numba = _draw_multiple_gaussians_numba(tracks, counter_tracks, colors, VH, VW, sigma)
    video_v3, counter_v3 = draw_multiple_gaussians_torch_v3(tracks, counter_tracks, colors, VH, VW, sigma)
    
    video_diff = np.abs(video_numba - video_v3)
    counter_diff = np.abs(counter_numba - counter_v3)
    
    print(f"Overlapping - Max video difference: {video_diff.max():.8f}")
    print(f"Overlapping - Mean video difference: {video_diff.mean():.8f}")
    print(f"Overlapping - Max counter difference: {counter_diff.max():.8f}")
    print(f"Overlapping - Mean counter difference: {counter_diff.mean():.8f}")
    
    # Visualize results
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(video_numba[0].transpose(1, 2, 0))
    axes[0, 0].set_title("Numba (Overlapping)")
    
    axes[0, 1].imshow(video_v3[0].transpose(1, 2, 0))
    axes[0, 1].set_title("Torch V3 (Overlapping)")
    
    axes[0, 2].imshow(video_diff[0].transpose(1, 2, 0) * 100)
    axes[0, 2].set_title("Difference x100")
    
    axes[1, 0].imshow(counter_numba[0].transpose(1, 2, 0))
    axes[1, 0].set_title("Counter Numba")
    
    axes[1, 1].imshow(counter_v3[0].transpose(1, 2, 0))
    axes[1, 1].set_title("Counter Torch V3")
    
    axes[1, 2].imshow(counter_diff[0].transpose(1, 2, 0) * 100)
    axes[1, 2].set_title("Counter Diff x100")
    
    plt.tight_layout()
    plt.savefig("test_comparison_v3.png")
    print("Saved visualization to test_comparison_v3.png")
    
    # Test 3: RGB mode (no alpha channel)
    print("\n=== Test 3: RGB Mode (No Alpha) ===")
    colors_rgb = colors[:, :3]  # Drop alpha channel
    
    video_numba_rgb, counter_numba_rgb = _draw_multiple_gaussians_numba(tracks, counter_tracks, colors_rgb, VH, VW, sigma)
    video_v3_rgb, counter_v3_rgb = draw_multiple_gaussians_torch_v3(tracks, counter_tracks, colors_rgb, VH, VW, sigma)
    
    video_diff_rgb = np.abs(video_numba_rgb - video_v3_rgb)
    counter_diff_rgb = np.abs(counter_numba_rgb - counter_v3_rgb)
    
    print(f"RGB Mode - Max video difference: {video_diff_rgb.max():.8f}")
    print(f"RGB Mode - Mean video difference: {video_diff_rgb.mean():.8f}")
    print(f"RGB Mode - Max counter difference: {counter_diff_rgb.max():.8f}")
    print(f"RGB Mode - Mean counter difference: {counter_diff_rgb.mean():.8f}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_implementations()