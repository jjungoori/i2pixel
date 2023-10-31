from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

# ... [Rest of the code, including the get_dominant_colors function] ...

def extract_and_show_colors(image_path, n_colors=16):
    """Extract dominant colors and display them."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((50, 50), Image.LANCZOS)
    
    dominant_colors = get_dominant_colors(image, n_colors)
    
    # Create an image displaying the dominant colors
    color_image = Image.new('RGB', (n_colors * 20, 20))
    draw = ImageDraw.Draw(color_image)
    
    for idx, color in enumerate(dominant_colors):
        draw.rectangle([idx * 20, 0, (idx + 1) * 20, 20], fill=tuple(map(int, color)))
    
    color_image.show()

# Extract and display dominant colors
image_path = "OIG.jpg"
extract_and_show_colors(image_path)
