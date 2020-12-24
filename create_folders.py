import os

trailers = os.listdir("trailer_videos")

if not os.path.exists("./filtered_frames"):
    os.mkdir("./filtered_frames")

if not os.path.exists("./color_palettes"):
    os.mkdir("./color_palettes")

for trailer in trailers:
    if trailer[len(trailer)-2:] == "py" or trailer == "__pycache__":
        continue
    if not os.path.exists("./filtered_frames/" + trailer[:len(trailer) - 4]):
        os.mkdir("./filtered_frames/" + trailer[:len(trailer) - 4])
