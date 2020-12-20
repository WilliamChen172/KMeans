import os
import kmeans_color_palette
import slicer
from multiprocessing import Pool
import time
import sys
import os
import numpy as np
import glob

frames_per_output = 1
step = 5


def process_videos(trailer):
    start = time.time()
    trailer_palettes = []
    print("---------------------{}-------------------".format(trailer))
    slicer.remove_MPAA_frames(trailer, frames_per_output)
    palette, luminance = kmeans_color_palette.create_palette_for_movie(trailer[:-4])
    trailer_palettes.append((palette, luminance))
    print("{}: Total time: {}\n".format(trailer, time.time() - start))
    return trailer_palettes


if __name__ == "__main__":

    files = os.listdir("../trailer_videos")
    trailers = []

    for f in files:
        if f[len(f)-2:] == "py" or f == "__pycache__":
            continue
        trailers.append(f)

    cpu = os.cpu_count() - 1
    if cpu == 0:
        cpu = 1

    pool = Pool(processes=cpu)
    result = np.array(pool.map(process_videos, trailers))
    pool.close()
    pool.join()
    result = np.array(result).flatten()
    print(result.shape)
    print(result)
    np.savetxt("palettes.csv", result)

