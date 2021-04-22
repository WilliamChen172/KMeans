import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
from multiprocessing import Pool

n_colors = 8

root_dir = "../filtered_frames/"
trailer = ""

# Weights of saliency and saturation respectively. This array is passed when calculating palettes for trailers.
# The number of weight pairs is equal to the number of palettes saved for each trailer for comparison.
w = np.array([[0, 0], [1, 0], [1, 4], [1, 8], [1, 16]])


def create_frame_array(movie_path, sal_w, sat_w):
    print(" creating frames array...")
    frames = os.listdir(movie_path)
    if len(frames) == 0:
        print("  No frames exist in {}!".format(movie_path))
        return []
    t0 = time()
    color_dict = {}
    for frame_path in frames:
        if frame_path[:-4] == "palette":
            continue
        if int(frame_path[:-4]) % 5 != 0:
            continue
        full_path = movie_path + "/" + frame_path
        if not os.path.exists(full_path):
            exit()
        frame = cv2.imread(full_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_np = np.array(frame, dtype=np.float64) / 255
        w, h, d = tuple(frame_np.shape)
        assert d == 3
        for i in range(w):
            for j in range(h):
                color = tuple(frame_np[i, j])
                if color in color_dict.keys():
                    color_dict[color] += 1
                else:
                    color_dict[color] = 1
        # Add salience weights
        color_dict = add_saliency_weights(frame, color_dict, sal_w)
        # Add saturation weights
        color_dict = add_saturation_weights(frame, color_dict, sat_w)
    frames_array = list(color_dict.keys())
    weights = list(color_dict.values())
    print("   done in %0.3fs." % (time() - t0))
    return frames_array, weights


def add_saliency_weights(frame, color_dict, weight):
    block_size = 16
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_np = np.array(frame, dtype=np.float64) / 255
    w, h, d = tuple(frame_np.shape)
    sift_map = np.zeros((w, h))
    sift = cv2.SIFT_create()
    m = 0
    n = 0
    while n < h:
        while m < w:
            p = min(m + block_size, w)
            q = min(n + block_size, h)
            # plt.imshow(frame_gray, cmap='gray', vmin=0, vmax=255)
            # plt.show()
            kp, des = sift.detectAndCompute(frame_gray[m:p, n:q], None)
            sift_map[m:p, n:q] = len(kp)
            m = p
        n = min(n + block_size, h)
        m = 0
    sift_map = cv2.GaussianBlur(sift_map, (17, 17), 0)

    for i in range(w):
        for j in range(h):
            color = tuple(frame_np[i, j])
            color_dict[color] += sift_map[i, j] * weight

    return color_dict


def add_saturation_weights(frame, color_dict, weight):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_np = np.array(frame, dtype=np.float64) / 255
    w, h, d = tuple(frame_np.shape)
    hsv_map = frame_hsv[:, :, 1]
    hsv_map = np.array(hsv_map, dtype=np.float64) / 255

    for i in range(w):
        for j in range(h):
            color = tuple(frame_np[i, j])
            color_dict[color] += hsv_map[i, j] * weight

    return color_dict


def k_means_codebook(frames_array, weights):
    print(" creating kmeans codebook...")
    t0 = time()
    frames_array_sample = shuffle(frames_array, random_state=0)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(frames_array_sample, sample_weight=weights)
    print("   done in %0.3fs." % (time() - t0))
    return kmeans.cluster_centers_


def filter_dark_colours(codebook):
    delete_index = []
    if codebook.shape[0] > 20:
        rm = codebook.shape[0] - 20
        for i in reversed(range(codebook.shape[0])):
            color = codebook[i]
            # print(color)
            if color[0] < 0.2 and color[1] < 0.2 and color[2] < 0.2 and rm > 0:
                delete_index.append(i)
                rm -= 1
        codebook = np.delete(codebook, delete_index, axis=0)
    while len(codebook) > 20:
        codebook = np.delete(codebook, -1, axis=0)
    return codebook


def sRGBtoLin(colorChannel):
    if colorChannel <= 0.04045:
        return colorChannel / 12.92
    else:
        return ((colorChannel + 0.055) / 1.055) ** 2.4


def calculate_frame_luminance(frame_path):
    rY = 0.212655
    gY = 0.715158
    bY = 0.072187
    if not os.path.exists(frame_path):
        print("Full path doesn't exist!")
        return
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_np = np.array(frame, dtype=np.float64) / 255
    w, h, d = tuple(frame_np.shape)
    assert d == 3
    frame_array = np.reshape(frame_np, (w * h, d))
    lumin_array = [
        (rY * sRGBtoLin(frame_array[i][0]) + gY * sRGBtoLin(frame_array[i][1]) + bY * sRGBtoLin(frame_array[i][2]))
        for
        i in range(0, len(frame_array), 5)]
    mean_lumin = np.mean(lumin_array)
    return mean_lumin


def calculate_average_luminance(movie_path):
    print(" calculating luminance...")

    frames = os.listdir(movie_path)
    if len(frames) == 0:
        print("No frames exists!")
        return []
    t0 = time()

    cpu = os.cpu_count() - 1
    if cpu == 0:
        cpu = 1

    frame_arr = []
    for frame in frames:
        full_path = movie_path + "/" + frame
        frame_arr.append(full_path)

    pool = Pool(processes=cpu)
    lumin_list = pool.map(calculate_frame_luminance, frame_arr)
    pool.close()
    pool.join()
    mean_luminance = np.mean(lumin_list)

    print("   done in %0.3fs." % (time() - t0))
    return mean_luminance


def create_palette_image(codebook):
    print(" creating colour palette...")
    t0 = time()
    d = codebook.shape[1]
    len = codebook.shape[0]
    image = np.zeros((100, 100 * len, d))
    for i in range(len):
        for j in range(100):
            for k in range(100):
                image[j][i * 100 + k] = codebook[i]
    print("   done in %0.3fs." % (time() - t0))
    return image


def create_palette_for_movie(trailer_name):
    global trailer, w
    trailer = trailer_name
    movie_path = root_dir + trailer_name
    palette_images = []
    filtered_codebook = []

    # A colour palette is created for each pair of weights. All palettes from the same movie are saved in
    # the same image.
    for weight in w:
        print(weight)
        frames_array, weights = create_frame_array(movie_path, weight[0], weight[1])
        codebook = k_means_codebook(frames_array, weights)
        filtered_codebook = filter_dark_colours(codebook)
        print(filtered_codebook)
        filtered_image = create_palette_image(filtered_codebook)
        palette_images.append(filtered_image)

    avg_luminance = float(calculate_average_luminance(movie_path))

    plot_and_save(palette_images, avg_luminance, trailer_name, w)
    return filtered_codebook, avg_luminance


def plot_and_save(images, luminance, movie_name, weights):
    print(" Plotting ...")
    rows = len(images)
    cols = 1
    axes = []
    fig = plt.figure(1)
    plt.clf()
    plt.axis('off')
    for a in range(rows * cols):
        weight = weights[a]
        axes.append(fig.add_subplot(rows, cols, a + 1))
        # modify the title of the plot to show luminance
        lumin = "Average Luminance: {:.3f}\n".format(luminance)
        w = "Saliency weight: {} Saturation weight: {}".format(weight[0], weight[1])
        subplot_title = w
        axes[-1].set_title(subplot_title)
        axes[-1].axis('off')
        plt.imshow(images[a])
    fig.tight_layout()
    plt.suptitle(movie_name)
    save_path = "../color_palettes/" + movie_name + ".png"
    plt.savefig(save_path)
    # plt.show()
    print(" Image saved to " + save_path + ".\n\n")
