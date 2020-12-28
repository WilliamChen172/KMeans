import os
import kmeans_color_palette
from time import time
import os
import numpy as np
import csv


def clamp(x):
    return max(0, min(round(x * 255), 255))


def RGB2HEX(palette):
    hex_list = []
    for rgb in palette:
        hex = "#{0:02x}{1:02x}{2:02x}".format(clamp(rgb[0]), clamp(rgb[1]), clamp(rgb[2]))
        hex_list.append(hex)
    return np.array(hex_list)


def create_folders():
    if not os.path.exists("../color_palettes"):
        os.mkdir("../color_palettes")


def process_videos(trailers):
    for trailer in trailers:
        trailer_name = trailer[:-4]
        print("---------------------{}-------------------".format(trailer))
        start = time()
        palette, luminance = kmeans_color_palette.create_palette_for_movie(trailer_name)

        trailer_palette = np.array([trailer_name, luminance])
        hex_palette = RGB2HEX(palette)
        trailer_palette = np.concatenate([trailer_palette, hex_palette])

        # Delete frames to save space after finishing processing the trailer
        files = os.listdir("../filtered_frames/" + trailer_name)
        for file in files:
            os.remove("../filtered_frames/" + trailer_name + "/" + file)
        os.rmdir("../filtered_frames/" + trailer_name)

        with open("../palettes.csv", "a+", newline='\n') as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerow(trailer_palette)
            my_csv.close()

        print("{}: Total time: {:.3f}s\n".format(trailer, time() - start))


if __name__ == "__main__":

    files = os.listdir("../trailer_videos")
    trailers = []

    for f in files:
        if f[len(f) - 2:] == "py" or f == "__pycache__":
            continue
        trailers.append(f)

    if os.path.exists("../palettes.csv"):
        with open('../palettes.csv') as csvfile:
            palette_list = csv.reader(csvfile)
            for row in palette_list:
                trailer = row[0] + ".mp4"
                if trailer in trailers:
                    print(row)
                    trailers.remove(trailer)

    create_folders()
    process_videos(trailers)

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
