import os

trailers = os.listdir("filtered_frames")
print(trailers)
root_dir = "filtered_frames/"

for trailer in trailers:
    if trailer[len(trailer) - 2:] == "py" or trailer == "__pycache__":
        continue
    histogram_path = "./" + root_dir + trailer + "/Histogram"
    if not os.path.exists(histogram_path):
        os.mkdir(histogram_path)
    histogram_MSE_path = "./" + root_dir + trailer + "/Histogram+MSE"
    if not os.path.exists(histogram_MSE_path):
        os.mkdir(histogram_MSE_path)
    MSE_path = "./" + root_dir + trailer + "/MSE"
    if not os.path.exists(MSE_path):
        os.mkdir(MSE_path)
    SSIM_path = "./" + root_dir + trailer + "/SSIM"
    if not os.path.exists(SSIM_path):
        os.mkdir(SSIM_path)
