import os
import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def segment_with_color(dataset_dir, filename, color_space, n_clusters=5):
    assert color_space == 'rgb' or color_space == 'hsv'
    assert n_clusters > 0
    converter = cv2.COLOR_BGR2HSV if color_space == 'hsv' else cv2.COLOR_BGR2RGB
    converter_inv = cv2.COLOR_HSV2RGB if color_space == 'hsv' else None
    img = cv2.imread(os.path.join(dataset_dir, filename))
    color_sp_img = cv2.cvtColor(img, converter)
    pixels = color_sp_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(img.shape).astype(np.uint8)
    segmented_img_rgb = cv2.cvtColor(segmented_img, converter_inv) if converter_inv is not None else segmented_img

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    name, _ = os.path.splitext(filename)
    res_filename = os.path.join(results_dir, f'{name}-{color_space}-cl-{n_clusters}.png')
    plt.imsave(res_filename, segmented_img_rgb)

def main():
    dataset_dir = 'data/train_color'
    files = os.listdir(dataset_dir)
    for n_clusters in [5, 10, 15]:
        for i in range(1):
            segment_with_color(dataset_dir, files[i], 'rgb', n_clusters)
            segment_with_color(dataset_dir, files[i], 'hsv', n_clusters)

if __name__ == '__main__':
    main()
