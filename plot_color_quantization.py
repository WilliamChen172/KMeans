# -*- coding: utf-8 -*-
"""
==================================
Color Quantization using K-Means
==================================

Performs a pixel-wise Vector Quantization (VQ) of an image of the summer palace
(China), reducing the number of colors required to show the image from 96,615
unique colors to 64, while preserving the overall appearance quality.

In this example, pixels are represented in a 3D-space and K-means is used to
find 64 color clusters. In the image processing literature, the codebook
obtained from K-means (the cluster centers) is called the color palette. Using
a single byte, up to 256 colors can be addressed, whereas an RGB encoding
requires 3 bytes per pixel. The GIF file format, for example, uses such a
palette.

For comparison, a quantized image using a random codebook (colors picked up
randomly) is also shown.
"""
# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

print(__doc__)
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

if not len(sys.argv) == 2:
    print()
    print("Usage: python3 plot_color_quantization.py [name of directory that contains all the frames]")
    print()
    print("Example: python3 plot_color_quantization.py '12 YEARS A SLAVE - Official Trailer (HD)'")
    print()
    exit()

n_colors = 15

# Load the Summer Palace photo
movie_path = sys.argv[1]
SSIM = "SSIM"
MSE = "MSE"
Histogram = "Histogram"
Histogram_MSE = "Histogram+MSE"
phase2_list = [MSE, Histogram, Histogram_MSE, SSIM]


def create_frame_array(path, method):
    print("creating frames array...")
    frames = os.listdir(path)
    array_list = []
    for frame_path in frames:
        frame = cv2.imread(movie_path + "/" + method + "/" + frame_path)
        frame_np = np.array(frame, dtype=np.float64) / 255
        w, h, d = tuple(frame_np.shape)
        assert d == 3
        frame_array = np.reshape(frame_np, (w * h, d))
        array_list.append(frame_array)
    frames_array = np.concatenate(array_list)
    return frames_array


def k_means_codebook(frames_array):
    print("creating kmeans codebook...")
    frames_array_sample = shuffle(frames_array, random_state=0)[:1000000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(frames_array_sample)
    return kmeans.cluster_centers_


def create_palette_image(codebook):
    print("creating colour palette...")
    d = codebook.shape[1]
    len = codebook.shape[0]
    image = np.zeros((100, 100 * len, d))
    for i in range(n_colors):
        for j in range(100):
            for k in range(100):
                image[j][i * 100 + k] = codebook[i]
    return image


def plot_and_save(images):
    print("plotting...")
    rows = 4
    cols = 1
    axes = []
    fig = plt.figure(1)
    plt.clf()
    plt.axis('off')
    for a in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, a + 1))
        subplot_title = (phase2_list[a])
        axes[-1].set_title(subplot_title)
        axes[-1].axis('off')
        plt.imshow(images[a])
    fig.tight_layout()
    plt.suptitle('Colour Palette With 4 Different Frame Reduce Methods')
    plt.show()
    plt.savefig(movie_path + "/" + movie_path + ".png", dpi=fig.dpi)


if __name__ == '__main__':
    palette_images = []
    for method in phase2_list:
        print("Processing {}...".format(method))
        path = movie_path + "/" + method
        frame_array = create_frame_array(path, method)
        codebook = k_means_codebook(frame_array)
        image = create_palette_image(codebook)
        palette_images.append(image)
    plot_and_save(palette_images)

# china = load_sample_image("china.jpg")
#
# # Convert to floats instead of the default 8 bits integer coding. Dividing by
# # 255 is important so that plt.imshow behaves works well on float data (need to
# # be in the range [0-1])
# china = np.array(china, dtype=np.float64) / 255

# # Load Image and transform to a 2D numpy array.
# w, h, d = original_shape = tuple(china.shape)
# assert d == 3
# image_array = np.reshape(china, (w * h, d))
#
# print("Fitting model on a small sub-sample of the data")
# t0 = time()
# image_array_sample = shuffle(image_array, random_state=0)[:1000]
# kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
# print(kmeans.cluster_centers_)
# print("done in %0.3fs." % (time() - t0))
#
# # Get labels for all points
# print("Predicting color indices on the full image (k-means)")
# t0 = time()
# labels = kmeans.predict(image_array)
# print("done in %0.3fs." % (time() - t0))
#
# codebook_random = shuffle(image_array, random_state=0)[:n_colors]
# print("Predicting color indices on the full image (random)")
# t0 = time()
# labels_random = pairwise_distances_argmin(codebook_random,
#                                           image_array,
#                                           axis=0)
# print("done in %0.3fs." % (time() - t0))
#
#
# def recreate_image(codebook, labels, w, h):
#     """Recreate the (compressed) image from the code book & labels"""
#     d = codebook.shape[1]
#     image = np.zeros((w, h, d))
#     label_idx = 0
#     for i in range(w):
#         for j in range(h):
#             image[i][j] = codebook[labels[label_idx]]
#             label_idx += 1
#     return image


# Display all results, alongside original image
# plt.figure(1)
# plt.clf()
# plt.axis('off')
# plt.title('Original image (96,615 colors)')
# plt.imshow(china)
#
# plt.figure(2)
# plt.clf()
# plt.axis('off')
# plt.title('Quantized image (64 colors, K-Means)')
# plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
#
# plt.figure(3)
# plt.clf()
# plt.axis('off')
# plt.title('Quantized image (64 colors, Random)')
# plt.imshow(recreate_image(codebook_random, labels_random, w, h))
# plt.show()
