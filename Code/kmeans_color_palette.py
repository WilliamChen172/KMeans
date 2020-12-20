import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time

n_colors = 25

root_dir = "../filtered_frames/"


# palette_names = ["filtered_dark_colours"]


def create_frame_array(movie_path):
    print(" creating frames array...")
    frames = os.listdir(movie_path)
    if len(frames) == 0:
        print("No frames exist in {}!".format(movie_path))
        return []
    t0 = time()
    array_list = []
    for frame_path in frames:
        if frame_path[:-4] == "palette":
            continue
        if int(frame_path[:-4]) % 5 != 0:
            continue
        print(frame_path)
        full_path = movie_path + "/" + frame_path
        if not os.path.exists(full_path):
            exit()
        frame = cv2.imread(full_path)
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


def filter_dark_colours(codebook):
    delete_index = []
    for i in range(codebook.shape[0]):
        color = codebook[i]
        # print(color)
        if color[0] < 0.2 and color[1] < 0.2 and color[2] < 0.2:
            delete_index.append(i)
    codebook = np.delete(codebook, delete_index, axis=0)
    while len(codebook) > 20:
        codebook = np.delete(codebook, -1, axis=0)
    return codebook


def sRGBtoLin(colorChannel):
    if colorChannel <= 0.04045:
        return colorChannel / 12.92
    else:
        return ((colorChannel + 0.055) / 1.055) ** 2.4


def calculate_average_luminance(frames_array):
    print(" calculating luminance...")
    rY = 0.212655
    gY = 0.715158
    bY = 0.072187
    if len(frames_array) == 0:
        print("No frames exists!")
        return []

    sum_luminance = 0
    t0 = time()
    i = 0
    for pixel in frames_array:
        t1 = time()
        linear_p = (sRGBtoLin(pixel[0]), sRGBtoLin(pixel[1]), sRGBtoLin(pixel[2]))

        t2 = time()
        # print("t2:", t2-t1)
        luminance = rY * linear_p[0] + gY * linear_p[1] + bY * linear_p[2]

        t3 = time()
        # print("t3:", t3-t2)

        sum_luminance += luminance
        print(i, time() - t1)
        i += 1

    luminance_array = [(0.072187 * sRGBtoLin(pixel[2]) + 0.715158 * sRGBtoLin(pixel[1]) + 0.212655 * sRGBtoLin(pixel[0])) for pixel in frames_array]
    # print("     done in %0.3fs." % (time() - t0))
    mean_luminance = np.mean(luminance_array)
    # mean_luminance = sum_luminance / len(frames_array)
    print("     done in %0.3fs." % (time() - t0))
    return mean_luminance


def create_palette_image(codebook):
    print(" creating colour palette...")
    t0 = time()
    # print(codebook.shape)
    d = codebook.shape[1]
    len = codebook.shape[0]
    image = np.zeros((100, 100 * len, d))
    for i in range(len):
        for j in range(100):
            for k in range(100):
                image[j][i * 100 + k] = codebook[i]
    print("     done in %0.3fs." % (time() - t0))
    return image


def create_palette_for_movie(trailer_name):
    movie_path = root_dir + trailer_name
    palette_images = []
    if os.path.exists(movie_path + "/" + trailer_name + ".png"):
        return
    frames_array = create_frame_array(movie_path)
    codebook = k_means_codebook(frames_array)
    filtered_codebook = filter_dark_colours(codebook)
    filtered_image = create_palette_image(filtered_codebook)
    palette_images.append(filtered_image)
    avg_luminance = float(calculate_average_luminance(frames_array))
    plot_and_save(palette_images, avg_luminance, movie_path, trailer_name)
    return filtered_codebook, avg_luminance


def plot_and_save(images, luminance, movie_path, movie_name):
    print("Plotting...")
    rows = 1
    cols = 1
    axes = []
    fig = plt.figure(1)
    plt.clf()
    plt.axis('off')
    for a in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, a + 1))
        lumin = "Average Luminance: {:.3f}".format(luminance)
        subplot_title = (lumin)
        axes[-1].set_title(subplot_title)
        axes[-1].axis('off')
        plt.imshow(images[a])
    fig.tight_layout()
    plt.suptitle(movie_name)
    save_path = movie_path + "/palette.png"
    plt.savefig(save_path)
    # plt.show()
    print("Image saved to " + save_path + ".\n\n")
