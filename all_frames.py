import sys
import os
import glob
import kmeans_color_palette
import time
import threading


def process_videos(trailers):
    start = time.time()
    for trailer in trailers:
        print("---------------------{}-------------------".format(trailer))
        kmeans_color_palette.create_palette_for_movie(trailer)
        print("{}: Total time: {}\n".format(trailer, time.time() - start))


if __name__ == "__main__":

    files = os.listdir("./filtered_frames")
    trailers = []

    for f in files:
        if f[len(f)-2:] == "py" or f == "__pycache__":
            continue
        trailers.append(f)

    process_videos(trailers)
