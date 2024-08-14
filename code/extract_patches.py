import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_image(image_path):
    """ Load an image from a file path. """
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def color_quantization(image, clusters):
    """ Reduce the number of colors using K-means clustering. """
    pixels = image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    quantized_image = colors[labels].reshape(image.shape)
    return quantized_image, labels, colors

def extract_color_patches(image, labels, colors):
    """ Extract patches based on the quantized colors. """
    patches = []
    mask = labels.reshape(image.shape[:2])  # reshape labels to the image size
    
    for color_index, color in enumerate(colors):
        component_mask = (mask == color_index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        patch = np.zeros_like(image)
        cv2.drawContours(patch, contours, -1, color.tolist(), -1)
        patches.append(patch)
    
    return patches

def plot_results(images, titles):
    """ Plot a series of images. """
    plt.figure(figsize=(15, 5))
    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
    plt.show()

image = load_image('data/train_color/image0009.jpg')

quantized_image, labels, colors = color_quantization(image, clusters=5)

patches = extract_color_patches(image, labels, colors)

plot_results([image, quantized_image] + patches, ['Original Image', 'Quantized Image'] + [f'Patch {i+1}' for i in range(len(patches))])
