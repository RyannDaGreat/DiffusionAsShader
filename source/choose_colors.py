# 2025-06-16 18:50:42.902255
# 2025-06-16 17:55:05.925671
import numpy as np
from typing import List, Tuple
import colorsys
import cv2
from rp import unique


def find_next_color(colors: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """
    Find RGB color where at least one component (R,G,B) has greater than 50% intensity and 
    that has maximum distance from existing colors.
    
    Args:
        colors: List of RGB tuples (0-255 range)
    
    Returns:
        RGB tuple (0-255 range) with maximum distance from all existing colors
    """
    if not colors:
        # If no colors exist, return color with at least one component > 50%
        return (255, 128, 128)
    
    # Convert existing colors to normalized RGB (0-1 range)
    normalized_colors = np.array([(r/255, g/255, b/255) for r, g, b in colors])
    
    # Generate a grid of RGB values directly
    r_values = np.linspace(0.0, 1.0, 64)  # 20 red values
    g_values = np.linspace(0.0, 1.0, 64)  # 20 green values
    b_values = np.linspace(0.0, 1.0, 64)  # 20 blue values
    
    # Create meshgrid for all combinations
    r_grid, g_grid, b_grid = np.meshgrid(r_values, g_values, b_values, indexing='ij')
    
    # Flatten the grids to get all combinations
    r_flat = r_grid.flatten()
    g_flat = g_grid.flatten()
    b_flat = b_grid.flatten()
    
    # Combine into RGB array
    rgb_flat = np.column_stack((r_flat, g_flat, b_flat))
    
    # Filter to keep only colors where min(max(R,G,B)) > 0.5
    max_per_color = np.max(rgb_flat, axis=1)
    valid_indices = max_per_color > 0.5
    rgb_flat = rgb_flat[valid_indices]
    
    # Calculate distances to all existing colors for all candidate colors
    min_distances = np.full(len(rgb_flat), np.inf)
    
    for i, existing in enumerate(normalized_colors):
        # Broadcasting to calculate distances to this existing color for all candidates
        distances = np.sqrt(np.sum((rgb_flat - existing)**2, axis=1))
        # Update minimum distances
        min_distances = np.minimum(min_distances, distances)
    
    # Find the color with maximum minimum distance
    best_idx = np.argmax(min_distances)
    best_color = rgb_flat[best_idx]
    
    # Convert back to 0-255 range
    return tuple(int(c * 255) for c in best_color)


# Helper functions
def as_rgb_float_colors(color_names: List[str]) -> List[Tuple[float, float, float]]:
    """Convert color names to RGB float values (0-1 range)"""
    color_map = {
        'white': (1.0, 1.0, 1.0),
        'red': (1.0, 0.0, 0.0),
        'blue': (0.0, 0.0, 1.0),
        'green': (0.0, 1.0, 0.0),
        'cyan': (0.0, 1.0, 1.0),
        'magenta': (1.0, 0.0, 1.0),
        'yellow': (1.0, 1.0, 0.0),
        'black': (0.0, 0.0, 0.0)
    }
    return [color_map.get(name.lower(), (0.5, 0.5, 0.5)) for name in color_names]

def float_colors_to_byte_colors(float_colors: List[Tuple[float, float, float]]) -> List[Tuple[int, int, int]]:
    """Convert RGB float values (0-1 range) to byte values (0-255 range)"""
    return [tuple(int(c * 255) for c in color) for color in float_colors]

def byte_colors_to_float_colors(byte_colors: List[Tuple[int, int, int]]) -> List[Tuple[float, float, float]]:
    """Convert RGB byte values (0-255 range) to float values (0-1 range)"""
    return [tuple(c / 255 for c in color) for color in byte_colors]

def uniform_float_color_image(width: int, height: int, color: Tuple[float, float, float]) -> np.ndarray:
    """Create a uniform color image with the given dimensions and float RGB color"""
    return np.full((height, width, 3), color, dtype=np.float32)

def color_distance(color1: Tuple[float, float, float], color2: Tuple[float, float, float]) -> float:
    """Calculate Euclidean distance between two RGB colors in 0-1 range"""
    return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

def tiled_images(images: List[np.ndarray], colors: List[Tuple[float, float, float]]) -> np.ndarray:
    """Arrange images in a grid with distance labels"""
    # Determine grid size
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Get image dimensions
    img_h, img_w = images[0].shape[:2]
    
    # Create canvas
    canvas = np.ones((rows * img_h, cols * img_w, 3), dtype=np.float32)
    
    # Place images with distance labels
    for i, img in enumerate(images):
        r, c = i // cols, i % cols
        y_pos = r * img_h
        x_pos = c * img_w
        
        # Place the image
        canvas[y_pos:y_pos+img_h, x_pos:x_pos+img_w] = img
        
        # Calculate distances
        # Convert to OpenCV format (0-255 range and BGR order)
        cv_img = (canvas[y_pos:y_pos+img_h, x_pos:x_pos+img_w] * 255).astype(np.uint8)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        
        # Determine text color (white or black) based on background brightness
        brightness = 0.299 * colors[i][0] + 0.587 * colors[i][1] + 0.114 * colors[i][2]
        text_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)
        
        if i > 0:
            # Calculate minimum distance to all previous colors
            min_dist = min(color_distance(colors[i], colors[j]) for j in range(i))
            
            # Add just the min distance with 2 decimal places
            cv2.putText(
                cv_img, 
                f"{min_dist:.2f}", 
                (5, img_h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                text_color, 
                1, 
                cv2.LINE_AA
            )
        
        # Convert back to RGB float format
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        canvas[y_pos:y_pos+img_h, x_pos:x_pos+img_w] = cv_img.astype(np.float32) / 255
    
    return canvas

# Main execution
colors = float_colors_to_byte_colors(as_rgb_float_colors('gray'.split()))
while len(colors) < 100:
    next_color = find_next_color(colors)
    colors.append(next_color)
    colors = unique(colors)
    print(len(colors))
float_colors = byte_colors_to_float_colors(colors)

# Create and display tiled images
color_images = [uniform_float_color_image(50, 50, color) for color in float_colors]
result = tiled_images(color_images, float_colors)
print(f"Generated {len(colors)} distinct colors with at least one RGB component greater than 50%")

# Save the image if a display_image function is not available
cv2.imwrite('color_tiles.png', cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# If display_image function exists in the environment, use it
try:
    display_image(result)
except NameError:
    print("Image saved as 'color_tiles.png'")
