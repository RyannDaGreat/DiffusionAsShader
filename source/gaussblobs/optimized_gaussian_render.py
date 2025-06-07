import rp
import torch
import numpy as np

def _draw_multiple_gaussians_optimized(tracks_np, counter_tracks_np, colors_np, VH, VW, sigma):
    """
    Optimized gaussian rendering using rp.stamp_tensor.
    Designed to be a drop-in replacement for _draw_multiple_gaussians_numba.
    """
    T, num_tracks, XYZV = tracks_np.shape
    num_colors, C = colors_np.shape
    Xi, Yi, Zi, Vi = 0, 1, 2, 3
    
    # Convert to torch tensors
    tracks = torch.from_numpy(tracks_np.astype(np.float32))
    counter_tracks = torch.from_numpy(counter_tracks_np.astype(np.float32))
    colors = torch.from_numpy(colors_np.astype(np.float32))
    
    # Initialize output tensors
    video = torch.zeros((T, C, VH, VW), dtype=torch.float32)
    counter_video = torch.zeros((T, C, VH, VW), dtype=torch.float32)
    
    # Pre-compute gaussian kernel
    radius = int(np.ceil(3.0 * sigma))
    size = 2 * radius + 1
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    y = y - radius
    x = x - radius
    dist_sq = x.float() * x.float() + y.float() * y.float()
    gaussian_kernel = torch.exp(-dist_sq / (2.0 * sigma * sigma))
    
    # Process each frame
    for t in range(T):
        for track_idx in range(min(num_tracks, num_colors)):
            color_idx = track_idx % num_colors
            color = colors[color_idx]
            
            # Process original track
            vv = int(tracks[t, track_idx, Vi].item())
            if vv:
                center_x = tracks[t, track_idx, Xi].item()
                center_y = tracks[t, track_idx, Yi].item()
                
                # For each channel
                for c in range(C):
                    color_value = color[c].item()
                    
                    # Create the gaussian sprite for this channel
                    if C == 4:
                        # RGBA mode: alpha = color[3] * gaussian
                        alpha_multiplier = color[3].item()
                        sprite = gaussian_kernel * color_value
                        
                        # Custom mode function for RGBA
                        def mode_rgba(canvas, sprite_region):
                            # Extract corresponding gaussian values for alpha
                            gauss_region = sprite_region / (color_value + 1e-10)
                            alpha = gauss_region * alpha_multiplier
                            return canvas * (1 - alpha) + sprite_region
                        
                        video[t, c] = rp.stamp_tensor(
                            video[t, c], sprite, [center_y, center_x],
                            mode=mode_rgba, sprite_origin='center', mutate=True
                        )
                    else:
                        # RGB mode: alpha = color_value * gaussian
                        sprite = gaussian_kernel * color_value
                        
                        # Custom mode function for RGB
                        def mode_rgb(canvas, sprite_region):
                            # Alpha is the sprite region itself (color * gaussian)
                            alpha = sprite_region
                            return canvas * (1 - alpha) + sprite_region
                        
                        video[t, c] = rp.stamp_tensor(
                            video[t, c], sprite, [center_y, center_x],
                            mode=mode_rgb, sprite_origin='center', mutate=True
                        )
            
            # Process counter track
            counter_vv = int(counter_tracks[t, track_idx, Vi].item())
            if counter_vv:
                counter_center_x = counter_tracks[t, track_idx, Xi].item()
                counter_center_y = counter_tracks[t, track_idx, Yi].item()
                
                # Same process for counter
                for c in range(C):
                    color_value = color[c].item()
                    
                    if C == 4:
                        alpha_multiplier = color[3].item()
                        sprite = gaussian_kernel * color_value
                        
                        def mode_rgba_counter(canvas, sprite_region):
                            gauss_region = sprite_region / (color_value + 1e-10)
                            alpha = gauss_region * alpha_multiplier
                            return canvas * (1 - alpha) + sprite_region
                        
                        counter_video[t, c] = rp.stamp_tensor(
                            counter_video[t, c], sprite, [counter_center_y, counter_center_x],
                            mode=mode_rgba_counter, sprite_origin='center', mutate=True
                        )
                    else:
                        sprite = gaussian_kernel * color_value
                        
                        def mode_rgb_counter(canvas, sprite_region):
                            alpha = sprite_region
                            return canvas * (1 - alpha) + sprite_region
                        
                        counter_video[t, c] = rp.stamp_tensor(
                            counter_video[t, c], sprite, [counter_center_y, counter_center_x],
                            mode=mode_rgb_counter, sprite_origin='center', mutate=True
                        )
    
    # Convert back to numpy
    return video.numpy(), counter_video.numpy()


# Now let's create the wrapper function to replace the numba version in render_tracks.py
def draw_multiple_gaussians_fast(tracks_np, counter_tracks_np, colors_np, VH, VW, sigma):
    """
    Fast replacement for _draw_multiple_gaussians_numba using rp.stamp_tensor.
    This function has the exact same interface and behavior as the numba version.
    """
    return _draw_multiple_gaussians_optimized(tracks_np, counter_tracks_np, colors_np, VH, VW, sigma)