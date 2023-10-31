from PIL import Image, ImageFilter, ImageEnhance, ImageChops
from sklearn.cluster import KMeans
import numpy as np

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

def convert_to_pixel_art_with_palette(image_path, output_size=(64, 64), num_colors=8, dithering=True, outline=True, contrast=True, upscale_factor=10):
    """Convert the image to pixel art using improved palette selection and upscale the result."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(output_size, Image.LANCZOS)

    dominant_colors = get_dominant_colors(image, num_colors)
    palette = [tuple(map(int, color)) for color in dominant_colors]

    if contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Slightly increase contrast
    
    image = map_colors_to_palette(image, palette)
    
    if outline:
        edges = image.filter(ImageFilter.FIND_EDGES)
        image = ImageChops.blend(image, edges, alpha=0.2)

    # Upscaling the image
    upscaled_size = (image.width * upscale_factor, image.height * upscale_factor)
    image = image.resize(upscaled_size, Image.NEAREST)
    
    return image


# For demonstration, you'd run the following in your environment:
image_path = "OIG.jpg"
pixel_art_image = convert_to_pixel_art_with_palette(image_path)
pixel_art_image.show()
