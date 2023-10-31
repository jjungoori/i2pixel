from PIL import Image, ImageFilter, ImageEnhance, ImageChops
from sklearn.cluster import KMeans
import numpy as np
import cv2

def get_dominant_colors(image, n_colors):
    """Extract dominant colors from the image using k-means clustering."""
    pixels = np.array(image)
    pixels = pixels.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_colors, init='random', n_init=10)
    kmeans.fit(pixels)
    
    return kmeans.cluster_centers_

def map_colors_to_palette(image, palette):
    """Map image colors to the nearest color in the palette."""
    quantized = np.array(image)
    shape = quantized.shape
    quantized = quantized.reshape(-1, 3)
    
    for i, pixel in enumerate(quantized):
        distances = np.sum((palette - pixel) ** 2, axis=1)
        quantized[i] = palette[np.argmin(distances)]
    
    return Image.fromarray(quantized.reshape(shape).astype(np.uint8))

def apply_canny_edge(image):
    """Apply the Canny edge detection algorithm for outlining."""
    # Convert the image to grayscale for edge detection
    grayscale = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply the Canny edge detection
    edges = cv2.Canny(grayscale, 50, 150)
    
    # Convert edges back to a PIL Image and invert the colors (for a black outline)
    edge_image = Image.fromarray(255 - edges).convert('RGB')
    
    # Blend the edges with the original image
    outlined_image = ImageChops.multiply(image, edge_image)
    
    return outlined_image

def emphasize_high_contrast(image):
    """Emphasize high contrast areas and suppress softer colors."""
    img = np.array(image)
    
    # Apply bilateral filter
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Subtract filtered image from original and add the result back to the original
    detail = cv2.subtract(img, bilateral)
    emphasized = cv2.add(img, detail)
    
    return Image.fromarray(emphasized)

def remove_background(image, rect=None):
    """Apply the GrabCut algorithm to segment the image and remove the background."""
    img = np.array(image)
    
    # If no rectangle is provided, use the entire image
    if rect is None:
        rect = (5, 5, img.shape[1]-5, img.shape[0]-5)
    
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Extract the foreground using the mask
    foreground = img * mask2[:, :, np.newaxis]
    
    return Image.fromarray(foreground)


def convert_to_pixel_art_with_palette(image_path, output_size=(64, 64), num_colors=4, contrast=True, upscale_factor=10, emphasize_contrast=True):
    """Convert the image to pixel art, upscale it, and optionally remove the background."""
    image = Image.open(image_path).convert('RGB')
    
    # Step 2: Remove the background
    #image = remove_background(image)
    
    # Step 3: Emphasize high contrast areas
    if emphasize_contrast:
        image = emphasize_high_contrast(image)
    
    # Step 4: Resize for pixel art effect
    image = image.resize(output_size, Image.LANCZOS)

    # Step 5: Adjust contrast
    if contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
    
    # Step 6: Map to dominant color palette
    dominant_colors = get_dominant_colors(image, num_colors)
    palette = [tuple(map(int, color)) for color in dominant_colors]
    image = map_colors_to_palette(image, palette)
    
    # Step 7: Upscale to desired size
    upscaled_size = (image.width * upscale_factor, image.height * upscale_factor)
    image = image.resize(upscaled_size, Image.NEAREST)
    
    return image




# For demonstration, you'd run the following in your environment:
image_path = "OIG.jpg"
pixel_art_image = convert_to_pixel_art_with_palette(image_path)
pixel_art_image.show()
