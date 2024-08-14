import os
import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

def segment_with_color(dataset_dir, filename, color_space, n_clusters=5):
    assert color_space == 'rgb' or color_space == 'hsv'
    assert n_clusters > 0
    converter = cv2.COLOR_BGR2HSV if color_space == 'hsv' else cv2.COLOR_BGR2RGB
    converter_inv = cv2.COLOR_HSV2RGB if color_space == 'hsv' else None
    img = cv2.imread(os.path.join(dataset_dir, filename))
    color_sp_img = cv2.cvtColor(img, converter)
    pixels = color_sp_img.reshape(-1, 3)
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    print(pixels.shape)
    agglomerative.fit(pixels)
    # segmented_img = np.zeros_like(pixels, dtype=np.uint8)
    # for i in range(n_clusters):
    #     segmented_img[agglomerative.labels_ == i] = np.mean(pixels[agglomerative.labels_ == i], axis=0)
    # segmented_img = segmented_img.reshape(img.shape)
    # segmented_img_rgb = cv2.cvtColor(segmented_img, converter_inv) if converter_inv is not None else segmented_img

    # results_dir = 'results'
    # os.makedirs(results_dir, exist_ok=True)
    # name, _ = os.path.splitext(filename)
    # res_filename = os.path.join(results_dir, f'{name}-{color_space}-cl-{n_clusters}.png')
    # plt.imsave(res_filename, segmented_img_rgb)


def main():
    dataset_dir = 'data/train_color'
    files = os.listdir(dataset_dir)
    # segment_with_color(dataset_dir, files[0], 'rgb')
    # segment_with_color(dataset_dir, files[0], 'hsv')
    # for n_clusters in [5, 10, 15]:
    #     for i in range(3):
    #         segment_with_color(dataset_dir, files[i], 'rgb', n_clusters)
    #         segment_with_color(dataset_dir, files[i], 'hsv', n_clusters)

    segment_with_color(dataset_dir, files[10], 'hsv', 5)
if __name__ == '__main__':
    main()