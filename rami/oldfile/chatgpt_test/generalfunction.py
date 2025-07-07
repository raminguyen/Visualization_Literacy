import matplotlib.pyplot as plt
from PIL import Image

def plot_image(image_path, title=None, figsize=(8, 6)):
    img = Image.open(image_path)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')  # Hide axis
    if title:
        plt.title(title, fontsize=14)
    plt.show()
