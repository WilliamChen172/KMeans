import os
import kmeans_color_palette
import slicer
from multiprocessing import Pool
import time
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
        print(hex)
        hex_list.append(hex)
    return np.array(hex_list)


def process_videos(trailer):
    # result = []
    # for trailer in trailers:
    start = time.time()
    trailer_name = trailer[:-4]
    print("---------------------{}-------------------".format(trailer))
    slicer.remove_MPAA_frames(trailer, frames_per_output)
    palette, luminance = kmeans_color_palette.create_palette_for_movie(trailer_name)

    trailer_palette = np.array([trailer_name, luminance])
    print(palette)
    hex_palette = RGB2HEX(palette)
    trailer_palette = np.concatenate([trailer_palette, hex_palette])
    # result.append(trailer_palette)

    # Delete frames to save space after finishing processing the trailer
    files = os.listdir("../filtered_frames/" + trailer_name)
    for file in files:
        os.remove("../filtered_frames/" + trailer_name + "/" + file)

    with open("../palettes.csv", "a+", newline='\n') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(trailer_palette)
        my_csv.close()

    print("{}: Total time: {}\n".format(trailer, time.time() - start))
    # return np.array(result)


if __name__ == "__main__":

    files = os.listdir("../trailer_videos")
    trailers = []

    for f in files:
        if f[len(f) - 2:] == "py" or f == "__pycache__":
            continue
        trailers.append(f)

    cpu = int(os.cpu_count() / 2)
    if cpu == 0:
        cpu = 1

    if os.path.exists("../palettes.csv"):
        with open('../palettes.csv') as csvfile:
            palette_list = csv.reader(csvfile)
            for row in palette_list:
                trailer = row[0] + ".mp4"
                if trailer in trailers:
                    trailers.remove(trailer)
    # sys.exit(0)
    pool = Pool(processes=cpu)
    pool.map(process_videos, trailers)
    pool.close()
    pool.join()
    # result = np.array(result)
    # result = process_videos(trailers)
    # print(result.shape)
    # print(result)
    # np.transpose(result)
    # with open("../palettes.csv", "a+", newline='\n') as my_csv:
    #     csvWriter = csv.writer(my_csv, delimiter=',')
    #     csvWriter.writerows(result)
    #     my_csv.close()


