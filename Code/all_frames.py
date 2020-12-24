import os
import kmeans_color_palette
import slicer
from multiprocessing import Pool
from time import time
import sys
import os
import numpy as np
import csv

frames_per_output = 1
step = 5


def clamp(x):
    return max(0, min(round(x * 255), 255))


def RGB2HEX(palette):
    hex_list = []
    for rgb in palette:
        hex = "#{0:02x}{1:02x}{2:02x}".format(clamp(rgb[0]), clamp(rgb[1]), clamp(rgb[2]))
        hex_list.append(hex)
    return np.array(hex_list)


def create_folders():
    trailers = os.listdir("../trailer_videos")

    if not os.path.exists("../filtered_frames"):
        os.mkdir("../filtered_frames")

    if not os.path.exists("../color_palettes"):
        os.mkdir("../color_palettes")

    for trailer in trailers:
        if trailer[len(trailer) - 2:] == "py" or trailer == "__pycache__":
            continue
        if not os.path.exists("../filtered_frames/" + trailer[:len(trailer) - 4]):
            os.mkdir("../filtered_frames/" + trailer[:len(trailer) - 4])


def process_videos(trailer):
    # for trailer in trailers:
    start = time()
    trailer_name = trailer[:-4]
    print("---------------------{}-------------------".format(trailer))
    slicer.remove_MPAA_frames(trailer, frames_per_output)
    palette, luminance = kmeans_color_palette.create_palette_for_movie(trailer_name)

    trailer_palette = np.array([trailer_name, luminance])
    hex_palette = RGB2HEX(palette)
    trailer_palette = np.concatenate([trailer_palette, hex_palette])

    # Delete frames to save space after finishing processing the trailer
    files = os.listdir("../filtered_frames/" + trailer_name)
    for file in files:
        os.remove("../filtered_frames/" + trailer_name + "/" + file)

    with open("../palettes.csv", "a+", newline='\n') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(trailer_palette)
        my_csv.close()

    print("{}: Total time: {}\n".format(trailer, time() - start))
    # return trailer_palette


if __name__ == "__main__":

    create_folders()

    files = os.listdir("../trailer_videos")
    trailers = []

    for f in files:
        if f[len(f) - 2:] == "py" or f == "__pycache__":
            continue
        trailers.append(f)

    cpu = 2  # int(os.cpu_count() / 2)
    if cpu == 0:
        cpu = 1

    if os.path.exists("../palettes.csv"):
        with open('../palettes.csv') as csvfile:
            palette_list = csv.reader(csvfile)
            for row in palette_list:
                trailer = row[0] + ".mp4"
                if trailer in trailers and len(row) == 22:
                    trailers.remove(trailer)

    pool = Pool(processes=cpu)
    pool.map(process_videos, trailers)
    pool.close()
    pool.join()

    if os.path.exists("../palettes.csv"):
        file = []
        with open('../palettes.csv') as csvfile:
            palette_list = csv.reader(csvfile)
            for row in palette_list:
                file.append(np.array(row))
        file = np.array(file)
        file = file[file[:, 0].argsort()]
        with open("../palettes.csv", "w+", newline='\n') as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(file)
            my_csv.close()
    # result = np.array(result)
    # process_videos(trailers)
    # print(result.shape)
    # print(result)
    # np.transpose(result)
    # with open("../palettes.csv", "a+", newline='\n') as my_csv:
    #     csvWriter = csv.writer(my_csv, delimiter=',')
    #     csvWriter.writerows(result)
    #     my_csv.close()
