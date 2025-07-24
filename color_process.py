import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import joblib

def correct_transparent_color(solution_rgb, background_rgb, alpha_estimate=0.7):
    """
    Original correction method using alpha blending.
    """
    solution_rgb = np.array(solution_rgb, dtype=float)
    background_rgb = np.array(background_rgb, dtype=float)
    white_bg = np.array([255, 255, 255], dtype=float)
    
    corrected_color = (solution_rgb - (1 - alpha_estimate) * background_rgb) / alpha_estimate
    final_color = alpha_estimate * corrected_color + (1 - alpha_estimate) * white_bg
    final_color = np.clip(final_color, 0, 255)
    
    return final_color.astype(int)

def correct_transparent_color_beer_lambert(solution_rgb, background_rgb, path_length=1.0):
    """
    Beer-Lambert Law correction method.
    """
    solution_rgb = np.array(solution_rgb, dtype=float) / 255.0
    background_rgb = np.array(background_rgb, dtype=float) / 255.0
    white_bg = np.array([1.0, 1.0, 1.0])
    
    background_rgb = np.maximum(background_rgb, 0.001)
    transmission = solution_rgb / background_rgb
    transmission = np.clip(transmission, 0.001, 1.0)
    
    color_on_white = white_bg * (transmission ** path_length)
    final_color = (color_on_white * 255).astype(int)
    
    return final_color

def correct_color_with_model(model, observed_rgb, background_rgb):
    """
    Linear model correction method.
    """
    x = np.array(observed_rgb + background_rgb).reshape(1, -1)
    prediction = model.predict(x)
    return np.clip(prediction.round().astype(int), 0, 255)[0]

def average_rgb_with_outlier_removal(image_path, center_ratio=0.3, std_threshold=2.0):
    """
    Calculate average RGB from center region of image, excluding outlier colors.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img_rgb.shape[:2]
    center_h, center_w = int(h * center_ratio), int(w * center_ratio)
    start_h, start_w = (h - center_h) // 2, (w - center_w) // 2
    
    center_region = img_rgb[start_h:start_h + center_h, start_w:start_w + center_w]
    pixels = center_region.reshape(-1, 3).astype(float)
    
    mean_color = np.mean(pixels, axis=0)
    distances = np.sqrt(np.sum((pixels - mean_color) ** 2, axis=1))
    threshold = np.mean(distances) + std_threshold * np.std(distances)
    
    valid_pixels = pixels[distances <= threshold]
    
    if len(valid_pixels) == 0:
        valid_pixels = pixels
    
    avg_color = np.mean(valid_pixels, axis=0)
    return avg_color.astype(int)

def create_color_swatch(color, size=(100, 100)):
    """Create a color swatch for visualization."""
    swatch = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return swatch

def process_solution_images(list_dir="./list", background_filename="background.png", 
                          alpha_estimate=0.5, path_length=1.0, model_path="linear_color_correction_model.pkl"):
    """
    Process all solution images and compare three correction methods.
    """
    list_path = Path(list_dir)
    background_path = list_path / background_filename
    
    # Load the linear model
    try:
        model = joblib.load(model_path)
        print(f"Loaded linear model from {model_path}")
    except:
        print(f"Could not load model from {model_path}")
        model = None
    
    if not background_path.exists():
        print(f"Background image not found: {background_path}")
        return
    
    print("Processing background image...")
    background_color = average_rgb_with_outlier_removal(background_path)
    print(f"Background color (RGB): {background_color}")
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in list_path.iterdir() 
                   if f.suffix.lower() in image_extensions and f.name != background_filename]
    
    if not image_files:
        print("No solution images found!")
        return
    
    print(f"\nFound {len(image_files)} solution images to process...")
    
    results = []
    
    for img_path in sorted(image_files):
        try:
            print(f"\nProcessing: {img_path.name}")
            
            original_color = average_rgb_with_outlier_removal(img_path)
            print(f"Original average color: {original_color}")
            
            # Original method
            corrected_original = correct_transparent_color(original_color, background_color, alpha_estimate)
            print(f"Original method: {corrected_original}")
            
            # Beer-Lambert method
            corrected_beer_lambert = correct_transparent_color_beer_lambert(original_color, background_color, path_length)
            print(f"Beer-Lambert method: {corrected_beer_lambert}")
            
            # Linear model method
            corrected_linear = None
            if model is not None:
                corrected_linear = correct_color_with_model(model, list(original_color), list(background_color))
                print(f"Linear model method: {corrected_linear}")
            else:
                print("Linear model method: Not available")
            
            results.append({
                'filename': img_path.name,
                'original': original_color,
                'corrected_original': corrected_original,
                'corrected_beer_lambert': corrected_beer_lambert,
                'corrected_linear': corrected_linear
            })
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    if results:
        create_comparison_plot_three_methods(results, background_color)
        create_concatenated_color_grid(results, background_color)
    
    return results

def create_concatenated_color_grid(results, background_color):
    """Create a grid of color swatches with no spacing, text, or labels."""
    n_images = len(results)
    n_cols = 4 if results[0]['corrected_linear'] is not None else 3  # Removed background column
    
    # Create color swatches for each method
    swatch_size = 100
    grid = np.zeros((n_images * swatch_size, n_cols * swatch_size, 3), dtype=np.uint8)
    
    for i, result in enumerate(results):
        row_start = i * swatch_size
        row_end = (i + 1) * swatch_size
        col = 0
        
        # Original color
        grid[row_start:row_end, col*swatch_size:(col+1)*swatch_size] = result['original']
        col += 1
        
        # Alpha blending method
        grid[row_start:row_end, col*swatch_size:(col+1)*swatch_size] = result['corrected_original']
        col += 1
        
        # Beer-Lambert method
        grid[row_start:row_end, col*swatch_size:(col+1)*swatch_size] = result['corrected_beer_lambert']
        col += 1
        
        # Linear model method (if available)
        if result['corrected_linear'] is not None:
            grid[row_start:row_end, col*swatch_size:(col+1)*swatch_size] = result['corrected_linear']
            col += 1
        
        # Background column removed
    
    # Flip with respect to x-axis (flip up-down)
    grid = np.flipud(grid)
    
    # Rotate right 90 degrees (clockwise)
    grid = np.rot90(grid, k=-1)  # k=-1 for clockwise rotation
    
    # Display the concatenated grid
    fig, ax = plt.subplots(1, 1, figsize=(n_images * 2, n_cols * 2))  # Swapped dimensions due to rotation
    ax.imshow(grid)
    ax.axis('off')
    
    # Remove all margins and make background transparent
    fig.patch.set_alpha(0)  # Transparent figure background
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all margins
    plt.tight_layout(pad=0)
    
    # Save with transparent background
    plt.savefig('color_grid.png', bbox_inches='tight', pad_inches=0, 
                facecolor='none', edgecolor='none', transparent=True, dpi=150)
    print("Saved color grid as 'color_grid.png' with transparent background")
    
    plt.show()

def create_comparison_plot_three_methods(results, background_color):
    """Create a visualization comparing all three correction methods."""
    n_images = len(results)
    n_cols = 5 if results[0]['corrected_linear'] is not None else 4
    fig, axes = plt.subplots(n_images, n_cols, figsize=(4 * n_cols, 3 * n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        col = 0
        
        # Original color
        original_swatch = create_color_swatch(result['original'])
        axes[i, col].imshow(original_swatch)
        axes[i, col].set_title(f"Original\n{result['filename']}")
        axes[i, col].axis('off')
        axes[i, col].text(0.5, -0.1, f"RGB: {result['original']}", 
                         transform=axes[i, col].transAxes, ha='center', fontsize=8)
        col += 1
        
        # Original method
        corrected_swatch = create_color_swatch(result['corrected_original'])
        axes[i, col].imshow(corrected_swatch)
        axes[i, col].set_title("Alpha Blending")
        axes[i, col].axis('off')
        axes[i, col].text(0.5, -0.1, f"RGB: {result['corrected_original']}", 
                         transform=axes[i, col].transAxes, ha='center', fontsize=8)
        col += 1
        
        # Beer-Lambert method
        beer_lambert_swatch = create_color_swatch(result['corrected_beer_lambert'])
        axes[i, col].imshow(beer_lambert_swatch)
        axes[i, col].set_title("Beer-Lambert")
        axes[i, col].axis('off')
        axes[i, col].text(0.5, -0.1, f"RGB: {result['corrected_beer_lambert']}", 
                         transform=axes[i, col].transAxes, ha='center', fontsize=8)
        col += 1
        
        # Linear model method (if available)
        if result['corrected_linear'] is not None:
            linear_swatch = create_color_swatch(result['corrected_linear'])
            axes[i, col].imshow(linear_swatch)
            axes[i, col].set_title("Linear Model")
            axes[i, col].axis('off')
            axes[i, col].text(0.5, -0.1, f"RGB: {result['corrected_linear']}", 
                             transform=axes[i, col].transAxes, ha='center', fontsize=8)
            col += 1
        
        # Background
        bg_swatch = create_color_swatch(background_color)
        axes[i, col].imshow(bg_swatch)
        axes[i, col].set_title("Background")
        axes[i, col].axis('off')
        axes[i, col].text(0.5, -0.1, f"RGB: {background_color}", 
                         transform=axes[i, col].transAxes, ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Run the analysis comparing all three methods
results = process_solution_images(alpha_estimate=0.5, path_length=1.0)