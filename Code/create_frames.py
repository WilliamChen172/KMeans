import os
import csv
from multiprocessing import Pool
from time import time
import slicer

frames_per_output = 1


def create_folders(trailers):
    if not os.path.exists("../filtered_frames"):
        os.mkdir("../filtered_frames")

    for trailer in trailers:
        if trailer[len(trailer) - 2:] == "py" or trailer == "__pycache__":
            continue
        if not os.path.exists("../filtered_frames/" + trailer[:len(trailer) - 4]):
            os.mkdir("../filtered_frames/" + trailer[:len(trailer) - 4])


def create_frames(trailer):
    print("---------------------{}-------------------".format(trailer))
    start = time()
    slicer.remove_MPAA_frames(trailer, frames_per_output)
    print("{}: Total time: {:.3f}s\n".format(trailer, time() - start))


if __name__ == "__main__":

    files = os.listdir("../trailer_videos")
    trailers = []

    for f in files:
        if f[len(f) - 2:] == "py" or f == "__pycache__":
            continue
        trailers.append(f)

    # Code to prevent duplication, comment out when debugging
    # if os.path.exists("../palettes.csv"):
    #     with open('../palettes.csv') as csvfile:
    #         palette_list = csv.reader(csvfile)
    #         for row in palette_list:
    #             trailer = row[0] + ".mp4"
    #             if trailer in trailers:
    #                 trailers.remove(trailer)

    create_folders(trailers)
    cpu = os.cpu_count() - 1
    if cpu == 0:
        cpu = 1

    pool = Pool(processes=cpu)
    pool.map(create_frames, trailers)
    pool.close()
    pool.join()

