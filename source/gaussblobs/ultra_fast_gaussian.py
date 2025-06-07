import numpy as np

def draw_multiple_gaussians_fast(tracks_np, counter_tracks_np, colors_np, VH, VW, sigma):
    """
    Ultra-fast gaussian rendering matching numba implementation exactly.
    No fancy features, just raw speed.
    """
    T, num_tracks, XYZV = tracks_np.shape
    num_colors, C = colors_np.shape
    Xi, Yi, Zi, Vi = 0, 1, 2, 3
    
    video_np = np.zeros((T, C, VH, VW), dtype=np.float32)
    counter_video_np = np.zeros((T, C, VH, VW), dtype=np.float32)
    
    # Precompute gaussian values
    radius = int(np.ceil(3.0 * sigma))
    inv_2sigma_sq = 1.0 / (2.0 * sigma * sigma)
    
    # Build lookup table for gaussian values
    max_dist_sq = radius * radius * 2
    gauss_lut = np.exp(-np.arange(max_dist_sq + 1) * inv_2sigma_sq).astype(np.float32)
    
    # Process each frame
    for t in range(T):
        for track_idx in range(min(num_tracks, num_colors)):
            color_idx = track_idx % num_colors
            
            # Original track
            if tracks_np[t, track_idx, Vi]:
                center_x = int(tracks_np[t, track_idx, Xi])
                center_y = int(tracks_np[t, track_idx, Yi])
                
                # Calculate bounds once
                y_start = max(0, center_y - radius)
                y_end = min(VH, center_y + radius + 1)
                x_start = max(0, center_x - radius)
                x_end = min(VW, center_x + radius + 1)
                
                # Skip if no pixels to draw
                if y_start >= y_end or x_start >= x_end:
                    continue
                
                # Create coordinate arrays for vectorized operations
                y_coords = np.arange(y_start, y_end, dtype=np.int32)
                x_coords = np.arange(x_start, x_end, dtype=np.int32)
                
                # Vectorized distance calculation
                dy = y_coords[:, None] - center_y
                dx = x_coords[None, :] - center_x
                dist_sq = dy * dy + dx * dx
                
                # Look up gaussian values (clamp to max dist)
                dist_sq = np.minimum(dist_sq, max_dist_sq)
                gauss_region = gauss_lut[dist_sq]
                
                # Get canvas region view
                canvas_region = video_np[t, :, y_start:y_end, x_start:x_end]
                
                if C == 4:
                    # RGBA mode
                    alpha = gauss_region * colors_np[color_idx, 3]
                    for c in range(C):
                        color_val = colors_np[color_idx, c]
                        canvas_region[c] *= (1 - alpha)
                        canvas_region[c] += color_val * alpha
                else:
                    # RGB mode
                    for c in range(C):
                        color_val = colors_np[color_idx, c]
                        alpha = gauss_region * color_val
                        canvas_region[c] *= (1 - alpha)
                        canvas_region[c] += color_val * alpha
            
            # Counter track
            if counter_tracks_np[t, track_idx, Vi]:
                center_x = int(counter_tracks_np[t, track_idx, Xi])
                center_y = int(counter_tracks_np[t, track_idx, Yi])
                
                y_start = max(0, center_y - radius)
                y_end = min(VH, center_y + radius + 1)
                x_start = max(0, center_x - radius)
                x_end = min(VW, center_x + radius + 1)
                
                # Skip if no pixels to draw
                if y_start >= y_end or x_start >= x_end:
                    continue
                    
                y_coords = np.arange(y_start, y_end, dtype=np.int32)
                x_coords = np.arange(x_start, x_end, dtype=np.int32)
                
                dy = y_coords[:, None] - center_y
                dx = x_coords[None, :] - center_x
                dist_sq = dy * dy + dx * dx
                
                # Clamp to max dist
                dist_sq = np.minimum(dist_sq, max_dist_sq)
                gauss_region = gauss_lut[dist_sq]
                canvas_region = counter_video_np[t, :, y_start:y_end, x_start:x_end]
                
                if C == 4:
                    alpha = gauss_region * colors_np[color_idx, 3]
                    for c in range(C):
                        color_val = colors_np[color_idx, c]
                        canvas_region[c] *= (1 - alpha)
                        canvas_region[c] += color_val * alpha
                else:
                    for c in range(C):
                        color_val = colors_np[color_idx, c]
                        alpha = gauss_region * color_val
                        canvas_region[c] *= (1 - alpha)
                        canvas_region[c] += color_val * alpha
    
    return video_np, counter_video_np