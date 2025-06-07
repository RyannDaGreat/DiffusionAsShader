import numpy as np
import rp

def _draw_multiple_gaussians_fast_numpy(tracks_np, counter_tracks_np, colors_np, VH, VW, sigma):
    """
    Ultra-fast implementation using numpy arrays throughout with in-place operations.
    Minimizes data transfers and uses rp.stamp_tensor with mutate=True.
    """
    T, num_tracks, XYZV = tracks_np.shape
    num_colors, C = colors_np.shape
    Xi, Yi, Zi, Vi = 0, 1, 2, 3
    
    # Initialize output arrays
    video_np = np.zeros((T, C, VH, VW), dtype=np.float32)
    counter_video_np = np.zeros((T, C, VH, VW), dtype=np.float32)
    
    # Pre-compute gaussian kernel once
    radius = int(np.ceil(3.0 * sigma))
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    gaussian_kernel = np.exp(-(x*x + y*y) / (2.0 * sigma * sigma)).astype(np.float32)
    
    # Pre-compute colored gaussian sprites for all colors
    colored_sprites = []
    for color_idx in range(num_colors):
        color = colors_np[color_idx]
        # For RGB mode, we'll handle alpha differently
        if C == 4:
            # RGBA: separate color and alpha
            color_sprite = gaussian_kernel[:, :, np.newaxis] * color[:3]  # RGB channels
            alpha_sprite = gaussian_kernel * color[3]  # Alpha channel
            colored_sprites.append((color_sprite, alpha_sprite))
        else:
            # RGB: color doubles as alpha
            color_sprite = gaussian_kernel[:, :, np.newaxis] * color
            colored_sprites.append((color_sprite, None))
    
    # Process each frame
    for t in range(T):
        # Process all tracks for this frame
        for track_idx in range(min(num_tracks, num_colors)):
            color_idx = track_idx % num_colors
            color_sprite, alpha_sprite = colored_sprites[color_idx]
            
            # Original track
            if tracks_np[t, track_idx, Vi]:
                cx = tracks_np[t, track_idx, Xi]
                cy = tracks_np[t, track_idx, Yi]
                
                if C == 4:
                    # RGBA mode - stamp each channel with alpha blending
                    for c in range(3):  # RGB channels
                        def rgba_blend(canvas, sprite):
                            # Alpha blend: canvas * (1 - alpha) + sprite
                            return canvas * (1 - alpha_sprite) + sprite
                        
                        video_np[t, c] = rp.stamp_tensor(
                            video_np[t, c], color_sprite[:, :, c],
                            offset=[cy, cx], mutate=True, mode=rgba_blend,
                            sprite_origin='center'
                        )
                    
                    # Alpha channel - use max blending
                    video_np[t, 3] = rp.stamp_tensor(
                        video_np[t, 3], alpha_sprite,
                        offset=[cy, cx], mutate=True, mode='max',
                        sprite_origin='center'
                    )
                else:
                    # RGB mode - color is alpha
                    for c in range(C):
                        sprite_c = color_sprite[:, :, c]
                        
                        def rgb_blend(canvas, sprite):
                            # In RGB mode, sprite value IS the alpha
                            return canvas * (1 - sprite) + sprite
                        
                        video_np[t, c] = rp.stamp_tensor(
                            video_np[t, c], sprite_c,
                            offset=[cy, cx], mutate=True, mode=rgb_blend,
                            sprite_origin='center'
                        )
            
            # Counter track - same logic
            if counter_tracks_np[t, track_idx, Vi]:
                cx = counter_tracks_np[t, track_idx, Xi]
                cy = counter_tracks_np[t, track_idx, Yi]
                
                if C == 4:
                    for c in range(3):
                        def rgba_blend_counter(canvas, sprite):
                            return canvas * (1 - alpha_sprite) + sprite
                        
                        counter_video_np[t, c] = rp.stamp_tensor(
                            counter_video_np[t, c], color_sprite[:, :, c],
                            offset=[cy, cx], mutate=True, mode=rgba_blend_counter,
                            sprite_origin='center'
                        )
                    
                    counter_video_np[t, 3] = rp.stamp_tensor(
                        counter_video_np[t, 3], alpha_sprite,
                        offset=[cy, cx], mutate=True, mode='max',
                        sprite_origin='center'
                    )
                else:
                    for c in range(C):
                        sprite_c = color_sprite[:, :, c]
                        
                        def rgb_blend_counter(canvas, sprite):
                            return canvas * (1 - sprite) + sprite
                        
                        counter_video_np[t, c] = rp.stamp_tensor(
                            counter_video_np[t, c], sprite_c,
                            offset=[cy, cx], mutate=True, mode=rgb_blend_counter,
                            sprite_origin='center'
                        )
    
    return video_np, counter_video_np


def _draw_multiple_gaussians_batch_optimized(tracks_np, counter_tracks_np, colors_np, VH, VW, sigma):
    """
    Even faster version that batches operations and minimizes function calls.
    """
    T, num_tracks, XYZV = tracks_np.shape
    num_colors, C = colors_np.shape
    Xi, Yi, Zi, Vi = 0, 1, 2, 3
    
    # Initialize output arrays
    video_np = np.zeros((T, C, VH, VW), dtype=np.float32)
    counter_video_np = np.zeros((T, C, VH, VW), dtype=np.float32)
    
    # Pre-compute gaussian kernel
    radius = int(np.ceil(3.0 * sigma))
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    gaussian_kernel = np.exp(-(x*x + y*y) / (2.0 * sigma * sigma)).astype(np.float32)
    
    # Pre-allocate work buffers
    temp_sprite = np.empty((size, size), dtype=np.float32)
    
    # Process each frame
    for t in range(T):
        # Get visible tracks for this frame
        vis_mask = tracks_np[t, :, Vi].astype(bool)
        vis_indices = np.where(vis_mask)[0]
        
        # Process visible tracks
        for idx in vis_indices[:num_colors]:  # Limit to number of colors
            color_idx = idx % num_colors
            color = colors_np[color_idx]
            
            cx = tracks_np[t, idx, Xi]
            cy = tracks_np[t, idx, Yi]
            
            # Inline the stamping logic to avoid function call overhead
            # Calculate bounds
            x0 = int(cx - radius)
            y0 = int(cy - radius)
            x1 = x0 + size
            y1 = y0 + size
            
            # Clip to canvas bounds
            sx0 = max(0, -x0)
            sy0 = max(0, -y0)
            sx1 = min(size, size - max(0, x1 - VW))
            sy1 = min(size, size - max(0, y1 - VH))
            
            dx0 = max(0, x0)
            dy0 = max(0, y0)
            dx1 = min(VW, x1)
            dy1 = min(VH, y1)
            
            if dx0 < dx1 and dy0 < dy1:
                # Get the visible region of gaussian
                gauss_region = gaussian_kernel[sy0:sy1, sx0:sx1]
                
                # Apply to each channel
                for c in range(C):
                    canvas_region = video_np[t, c, dy0:dy1, dx0:dx1]
                    
                    if C == 4:
                        # RGBA mode
                        alpha = gauss_region * color[3]
                        sprite_val = gauss_region * color[c]  # This already includes gaussian
                        # Alpha compositing: canvas * (1 - alpha) + color * alpha
                        np.multiply(canvas_region, 1 - alpha, out=canvas_region)
                        canvas_region += color[c] * alpha  # NOT sprite_val, but color * alpha!
                    else:
                        # RGB mode - follows numba logic exactly
                        alpha = gauss_region * color[c]  # alpha = color * gaussian
                        # Alpha compositing: canvas * (1 - alpha) + color * alpha
                        # Note: color * alpha = color * (color * gaussian) = color^2 * gaussian
                        np.multiply(canvas_region, 1 - alpha, out=canvas_region)
                        canvas_region += color[c] * alpha
        
        # Same for counter tracks
        vis_mask_counter = counter_tracks_np[t, :, Vi].astype(bool)
        vis_indices_counter = np.where(vis_mask_counter)[0]
        
        for idx in vis_indices_counter[:num_colors]:
            color_idx = idx % num_colors
            color = colors_np[color_idx]
            
            cx = counter_tracks_np[t, idx, Xi]
            cy = counter_tracks_np[t, idx, Yi]
            
            x0 = int(cx - radius)
            y0 = int(cy - radius)
            x1 = x0 + size
            y1 = y0 + size
            
            sx0 = max(0, -x0)
            sy0 = max(0, -y0)
            sx1 = min(size, size - max(0, x1 - VW))
            sy1 = min(size, size - max(0, y1 - VH))
            
            dx0 = max(0, x0)
            dy0 = max(0, y0)
            dx1 = min(VW, x1)
            dy1 = min(VH, y1)
            
            if dx0 < dx1 and dy0 < dy1:
                gauss_region = gaussian_kernel[sy0:sy1, sx0:sx1]
                
                for c in range(C):
                    canvas_region = counter_video_np[t, c, dy0:dy1, dx0:dx1]
                    
                    if C == 4:
                        alpha = gauss_region * color[3]
                        sprite_val = gauss_region * color[c]
                        np.multiply(canvas_region, 1 - alpha, out=canvas_region)
                        canvas_region += color[c] * alpha
                    else:
                        alpha = gauss_region * color[c]
                        np.multiply(canvas_region, 1 - alpha, out=canvas_region)
                        canvas_region += color[c] * alpha
    
    return video_np, counter_video_np


# Export the fastest version
draw_multiple_gaussians_fast = _draw_multiple_gaussians_batch_optimized