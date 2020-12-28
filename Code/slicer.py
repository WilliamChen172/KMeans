import cv2
import os
import numpy as np
from time import time

sensitivity = 15
# height, width, depth = img.shape
# Green threshhold values
green_lower_h = 60 - sensitivity
green_lower_s = 100
green_lower_v = 50
green_upper_h = 60 + sensitivity
green_upper_s = 255
green_upper_v = 255
# Red threshhold values
red_lower_h_1 = 0
red_lower_s = 100
red_lower_v = 100
red_upper_h_1 = 10
red_upper_s = 255
red_upper_v = 255
red_lower_h_2 = 160
red_upper_h_2 = 179
# Blue threshhold values
blue_lower_h = 120 - sensitivity
blue_lower_s = 50
blue_lower_v = 50
blue_upper_h = 120 + sensitivity
blue_upper_s = 255
blue_upper_v = 255
# Black threshhold values
black_lower_h = 0
black_lower_s = 0
black_lower_v = 0
black_upper_h = 180
black_upper_s = 255
black_upper_v = 40
# White threshhold values
white_lower_h = 0
white_lower_s = 0
white_lower_v = 168
white_upper_h = 172
white_upper_s = 111
white_upper_v = 255


def check_green(pixels_0, pixels_1, pixels_2):
    if pixels_0 >= green_lower_h and pixels_1 >= green_lower_s and pixels_2 >= green_lower_v and pixels_0 <= green_upper_h and pixels_1 <= green_upper_s and pixels_2 <= green_upper_v:
        return True
    return False


def check_red(pixels_0, pixels_1, pixels_2):
    if (
            pixels_0 >= red_lower_h_1 and pixels_1 >= red_lower_s and pixels_2 >= red_lower_v and pixels_0 <= red_upper_h_1 and pixels_1 <= red_upper_s and pixels_2 <= red_upper_v) or (
            pixels_0 >= red_lower_h_2 and pixels_1 >= red_lower_s and pixels_2 >= red_lower_v and pixels_0 <= red_upper_h_2 and pixels_1 <= red_upper_s and pixels_2 <= red_upper_v):
        return True
    return False


def check_blue(pixels_0, pixels_1, pixels_2):
    if pixels_0 >= blue_lower_h and pixels_1 >= blue_lower_s and pixels_2 >= blue_lower_v and pixels_0 <= blue_upper_h and pixels_1 <= blue_upper_s and pixels_2 <= blue_upper_v:
        return True
    return False


def check_black(pixels_0, pixels_1, pixels_2):
    if pixels_0 >= black_lower_h and pixels_1 >= black_lower_s and pixels_2 >= black_lower_v and pixels_0 <= black_upper_h and pixels_1 <= black_upper_s and pixels_2 <= black_upper_v:
        return True
    return False


def check_white(pixels_0, pixels_1, pixels_2):
    if pixels_0 >= white_lower_h and pixels_1 >= white_lower_s and pixels_2 >= white_lower_v and pixels_0 <= white_upper_h and pixels_1 <= white_upper_s and pixels_2 <= white_upper_v:
        return True
    return False


def filter_frames(img, height, width, step=5):
    color_variation = {
        "green": 0,
        "red": 0,
        "blue": 0,
        "black": 0,
        "white": 0,
        "other": 0
    }
    green_variation = {}
    blue_variation = {}
    red_variation = {}
    total = 0
    for i in range(0, height, step):
        for j in range(0, width, step):
            pixels_0 = img.item(i, j, 0)
            pixels_1 = img.item(i, j, 1)
            pixels_2 = img.item(i, j, 2)
            hsv = "{},{},{}".format(pixels_0, pixels_1, pixels_2)
            other = True
            if check_green(pixels_0, pixels_1, pixels_2):
                if hsv in green_variation:
                    green_variation[hsv] += 1
                else:
                    green_variation[hsv] = 1
                color_variation["green"] += 1
                other = False
            if check_red(pixels_0, pixels_1, pixels_2):
                if hsv in red_variation:
                    red_variation[hsv] += 1
                else:
                    red_variation[hsv] = 1
                color_variation["red"] += 1
                other = False
            if check_blue(pixels_0, pixels_1, pixels_2):
                if hsv in blue_variation:
                    blue_variation[hsv] += 1
                else:
                    blue_variation[hsv] = 1
                color_variation["blue"] += 1
                other = False
            if check_black(pixels_0, pixels_1, pixels_2):
                color_variation["black"] += 1
                other = False
            if check_white(pixels_0, pixels_1, pixels_2):
                color_variation["white"] += 1
                other = False
            if other:
                color_variation["other"] += 1
            total += 1

    variation_tolerance_ratio = 0.105
    # If more than 55% of frame is within some tolerance of green, blue or red, and not much variation between the
    # colors, then remove the frame
    if (color_variation["green"] / total >= 0.55 and len(green_variation) / color_variation[
        "green"] <= variation_tolerance_ratio) or (
            color_variation["blue"] / total >= 0.55 and len(blue_variation) / color_variation[
        "blue"] <= variation_tolerance_ratio) or (
            color_variation["red"] / total >= 0.55 and len(red_variation) / color_variation[
        "red"] <= variation_tolerance_ratio):
        # print("Too much green/blue/red")
        return True
    # or, if more than 99% is black, remove this frame
    elif color_variation["black"] / total >= 0.90 or (
            color_variation["black"] / total >= 0.95 and color_variation["white"] >= 0.01):
        # print("Too much black")
        return True
    return False


def get_crop_dimensions(frame):
    frame_np = np.array(frame)
    w, h, d = original_shape = tuple(frame_np.shape)
    assert d == 3
    top = bottom = left = right = 0
    for i in range(w):
        for j in range(h):
            if frame_np[i][j][0] >= 30 or frame_np[i][j][1] >= 30 or frame_np[i][j][2] >= 30:
                # print(frame_np[i][j])
                bottom = i
                break
    for i in reversed(range(w)):
        for j in range(h):
            if frame_np[i][j][0] >= 30 or frame_np[i][j][1] >= 30 or frame_np[i][j][2] >= 30:
                # print(frame_np[i][j])
                top = i
                break
    for j in range(h):
        for i in range(w):
            if frame_np[i][j][0] >= 30 or frame_np[i][j][1] >= 30 or frame_np[i][j][2] >= 30:
                right = j
                break
    for j in reversed(range(h)):
        for i in range(w):
            if frame_np[i][j][0] >= 30 or frame_np[i][j][1] >= 30 or frame_np[i][j][2] >= 30:
                left = j
                break
    return top, bottom, left, right


def remove_black_borders(frame, crop_dimensions):
    frame_np = np.array(frame)
    w, h, d = tuple(frame_np.shape)
    assert d == 3
    top, bottom, left, right = crop_dimensions
    return frame_np[top:bottom, left:right]


def remove_MPAA_frames(trailer, frame_step):
    print(" creating frames...")
    t0 = time()
    trailer_name = trailer[:-4]
    frames = os.listdir("../filtered_frames/" + trailer_name)
    if len(frames) > 0:
        return
    cap = cv2.VideoCapture("../trailer_videos/" + trailer)
    index = 0
    amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    crop_dimension = 0

    cap2 = cv2.VideoCapture("../trailer_videos/" + trailer)
    mid_index = amount_of_frames // 2
    cap2.set(cv2.CAP_PROP_POS_FRAMES, mid_index)
    _, mid_img = cap2.retrieve(mid_index)
    height, width, depth = mid_img.shape
    height = min(height, 480)
    width = min(width, 854)
    mid_img = cv2.resize(mid_img, (width, height))

    cap3 = cv2.VideoCapture("../trailer_videos/" + trailer)
    third_index = amount_of_frames // 3
    cap3.set(cv2.CAP_PROP_POS_FRAMES, third_index)
    _, third_img = cap3.retrieve(third_index)
    height, width, depth = third_img.shape
    height = min(height, 480)
    width = min(width, 854)
    third_img = cv2.resize(third_img, (width, height))

    while cap.isOpened():
        if index > amount_of_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, original_img = cap.retrieve(index)
        # Resize frame to be max of 854x480
        height, width, depth = original_img.shape
        height = int(min(height * 854 / width, 480))
        width = min(width, 854)
        resized_img = cv2.resize(original_img, (width, height))

        # Convert to hsv
        img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

        # Process the image and remove MPAA and all-black/mostly-black frames
        remove_frame = filter_frames(img, height, width)
        if not remove_frame:
            if not crop_dimension:
                # print("getting new dimension")
                dimension_1 = get_crop_dimensions(resized_img)
                dimension_2 = get_crop_dimensions(mid_img)
                dimension_3 = get_crop_dimensions(third_img)
                # print(dimension_1, dimension_2, dimension_3)
                dimension_1_size = (dimension_1[1] - dimension_1[0]) * (dimension_1[3] - dimension_1[2])
                dimension_2_size = (dimension_2[1] - dimension_2[0]) * (dimension_2[3] - dimension_2[2])
                dimension_3_size = (dimension_3[1] - dimension_3[0]) * (dimension_3[3] - dimension_3[2])
                if dimension_1_size >= dimension_2_size and dimension_1_size >= dimension_3_size:
                    crop_dimension = dimension_1
                elif dimension_2_size >= dimension_1_size and dimension_2_size >= dimension_3_size:
                    crop_dimension = dimension_2
                elif dimension_3_size >= dimension_1_size and dimension_3_size >= dimension_2_size:
                    crop_dimension = dimension_3
            rmvd_img = remove_black_borders(resized_img, crop_dimension)
            img_path = "../filtered_frames/" + trailer_name + "/" + str(index) + ".png"
            # print(img_path)
            cv2.imwrite(img_path, rmvd_img)

        # End processing
        index = index + frame_step

    cap.release()
    print("   done in %0.3fs." % (time() - t0))
