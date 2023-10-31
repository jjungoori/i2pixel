import cv2
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageChops, ImageDraw
from sklearn.cluster import KMeans
import numpy as np

# Load the pre-trained object detection model
model = tf.saved_model.load("faster_rcnn_resnet50_v1_640x640_coco17_tpu-8\saved_model")
infer = model.signatures["serving_default"]

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

def detect_objects(image):
    """Detect objects in the image using the pre-trained model."""
    # Convert image to a numpy array
    image_np = np.array(image)
    
    # Explicitly cast to uint8
    image_np = image_np.astype(np.uint8)
    
    # Print the dtype to make sure it's uint8
    print("image_np dtype:", image_np.dtype)
    
    # Convert numpy array to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor([image_np], dtype=tf.uint8)
    
    # Print the dtype of the tensor to make sure it's uint8
    print("input_tensor dtype:", input_tensor.dtype)
    
    # Use named argument for inference
    detections = infer(input_tensor=input_tensor)
    
    # Extract bounding boxes from the detections
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    boxes = boxes[scores > 0.5]  # Consider only boxes with a detection score > 0.5
    
    return boxes




def apply_object_outlines(image):
    """Apply outlines to detected objects in the image."""
    
    # Convert the image to numpy array
    np_image = np.array(image)

    # Apply Canny Edge Detection on the entire image
    edges = cv2.Canny(np_image, 100, 200)

    # Dilate the edges to make them thicker
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Overlay the edges onto the original image
    np_image[edges == 255] = [0, 0, 255]  # Setting edge pixels to blue color

    # Convert the modified numpy array back to a PIL Image
    outlined_image = Image.fromarray(np_image)

    return outlined_image

def convert_to_pixel_art_with_palette(image_path, output_size=(50, 50), num_colors=16, dithering=True, outline=True, contrast=True, upscale_factor=10):
    """Convert the image to pixel art, upscale it, and apply outlining to detected objects."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(output_size, Image.LANCZOS)

    dominant_colors = get_dominant_colors(image, num_colors)
    palette = [tuple(map(int, color)) for color in dominant_colors]

    if contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
    
    image = map_colors_to_palette(image, palette)
    
    if outline:
        image = apply_object_outlines(image)

    # Upscaling the image
    upscaled_size = (image.width * upscale_factor, image.height * upscale_factor)
    image = image.resize(upscaled_size, Image.NEAREST)
    
    return image

def get_edges(image):
    """Get edges from the image using Canny edge detection."""
    
    # Convert the image to numpy array and grayscale
    np_image = np.array(image)
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(gray_image, 100, 200)

    # Convert the edges numpy array back to a PIL Image
    edge_image = Image.fromarray(edges)

    return edge_image

# For demonstration, you'd run the following in your environment:
image_path = "OIG.jpg"

image = Image.open(image_path)
    
edges_image = get_edges(image)
edges_image.show()
    
pixel_art_image = convert_to_pixel_art_with_palette(image_path)
pixel_art_image.show()
