import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time

n_colors = 15

root_dir = "filtered_frames/"
SSIM = "SSIM"
MSE = "MSE"
Histogram = "Histogram"
Histogram_MSE = "Histogram+MSE"
phase2_list = [MSE, Histogram, Histogram_MSE, SSIM]


def create_frame_array(path, method, movie_path):
    print(" creating frames array...")
    frames = os.listdir(path)
    array_list = []
    t0 = time()
    for frame_path in frames:
        full_path = movie_path + "/" + method + "/" + frame_path
        if not os.path.exists(full_path):
            print(full_path)
            exit()
        frame = cv2.imread(movie_path + "/" + method + "/" + frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_np = np.array(frame, dtype=np.float64) / 255
        w, h, d = tuple(frame_np.shape)
        assert d == 3
        frame_array = np.reshape(frame_np, (w * h, d))
        array_list.append(frame_array)
    frames_array = np.concatenate(array_list)
    print("     done in %0.3fs." % (time() - t0))
    return frames_array


def k_means_codebook(frames_array):
    print(" creating kmeans codebook...")
    t0 = time()
    frames_array_sample = shuffle(frames_array, random_state=0)[:1000000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(frames_array_sample)
    print("     done in %0.3fs." % (time() - t0))
    return kmeans.cluster_centers_


def create_palette_image(codebook):
    print(" creating colour palette...")
    t0 = time()
    d = codebook.shape[1]
    len = codebook.shape[0]
    image = np.zeros((100, 100 * len, d))
    for i in range(n_colors):
        for j in range(100):
            for k in range(100):
                image[j][i * 100 + k] = codebook[i]
    print("     done in %0.3fs." % (time() - t0))
    return image


def plot_and_save(images, movie_path, movie_name):
    print("Plotting...")
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
    fig.subplots_adjust(top=0.9)
    plt.suptitle(movie_name)
    save_path = movie_path + "/" + movie_name + ".png"
    plt.savefig(save_path)
    # plt.show()
    print("Image saved to " + save_path + ".\n\n")


def create_palette_for_movie(trailer_name):
    global movie_path
    movie_path = root_dir + trailer_name
    palette_images = []
    if os.path.exists(movie_path + "/" + trailer_name + ".png"):
        return
    for method in phase2_list:
        print("Processing {}...".format(method))
        t0 = time()
        path = movie_path + "/" + method
        frame_array = create_frame_array(path, method, movie_path)
        codebook = k_means_codebook(frame_array)
        image = create_palette_image(codebook)
        palette_images.append(image)
        print("Finished " + method + " in %0.3fs.\n" % (time() - t0))
    plot_and_save(palette_images, movie_path, trailer_name)
