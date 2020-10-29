from PIL import Image
import numpy as np
import math

import sys
import os
from multiprocessing import Pool
import time
from pathlib import Path
import json
import cv2
import functools, collections, operator

from mediancut_palette_methods.median_cut import median_cut
from mediancut_palette_methods.utility.colour_list import ColourList
from mediancut_palette_methods.utility.colour import Colour


def load_image_file_median_cut(file):
    """
    Loads the image from the file
    :param file: The file name (or image name) as a string
                 :type file: str
    :return: The image acquired
    :raises: TypeError if file isn't a str
    """
    # Check type
    if not isinstance(file, str):
        raise TypeError("file must be a str\n" +
                        "\t\t\tValue:\t" + str(file) + "\n" +
                        "\t\t\tType:\t" + str(type(file)))

    # Load the file
    file_path = "frames/" + sys.argv[1] + "/" + file
    try:
        image = Image.open(file_path)
    except FileNotFoundError:
        print("File: " + file_path + " not found")
        return

    # Turn it into an image
    image.load()
    image = image.getcolors(maxcolors=image.size[0] * image.size[1])
    return image


def calculate_median_cut(filename):
    """
    :param filename: path to image being processed
    :return: A dictionary of a list, where each element in the list is a tuple containing:
                 The percentage of elements that were in this chunk
                 A tuple containing:
                     The average R value for this chunk
                     The average G value for this chunk
                     The average B value for this chunk
    """

    median_cuts_result = {
        "medcut_1": [],
        "medcut_2": [],
        "medcut_3": []
    }

    _image = load_image_file_median_cut(filename)

    _colour_list_1 = ColourList()
    _colour_list_2 = ColourList()
    _colour_list_3 = ColourList()
    for _colour in _image:
        _colour_list_1.append(
            Colour(_colour[1][0], _colour[1][1], _colour[1][2], _colour[0]))
        _colour_list_3.append(
            Colour(_colour[1][0], _colour[1][1], _colour[1][2], 1))
        for _times in range(0, _colour[0]):
            _colour_list_2.append(
                Colour(_colour[1][0], _colour[1][1], _colour[1][2], 1))
    _total_num_colours_1 = _colour_list_1.get_num_colours()
    _total_num_colours_2 = _colour_list_2.get_num_colours()
    _total_num_colours_3 = _colour_list_3.get_num_colours()
    _colour_lists_1 = [_colour_list_1]
    _colour_lists_2 = [_colour_list_2]
    _colour_lists_3 = [_colour_list_3]

    _output_1 = median_cut(_colour_lists_1)
    _output_2 = median_cut(_colour_lists_2)
    _output_3 = median_cut(_colour_lists_3)
    list.sort(_output_1[0], reverse=True)
    list.sort(_output_2[0], reverse=True)
    list.sort(_output_3[0], reverse=True)

    # we need to change first element of each row into percentage to avoid
    #   passing back all 3 of the _total_num_colours_#
    # convert to list so we can edit first element
    #   NOTE: RGB values are still a tuple after this
    medcut_1 = list(map(list, _output_1[0]))
    medcut_2 = list(map(list, _output_2[0]))
    medcut_3 = list(map(list, _output_3[0]))
    # change first element of each row into percentage
    for idx, item in enumerate(medcut_1):
        medcut_1[idx][0] = (item[0] / _total_num_colours_1) * 100
    for idx, item in enumerate(medcut_2):
        medcut_2[idx][0] = (item[0] / _total_num_colours_2) * 100
    for idx, item in enumerate(medcut_3):
        medcut_3[idx][0] = (item[0] / _total_num_colours_3) * 100

    # print
    print("Median Cut Approach 1:")
    for _item in medcut_1:
        rgb_percent = (_item[0], (_item[1][0], _item[1][1], _item[1][2]))
        print(rgb_percent)
        median_cuts_result["medcut_1"].append(rgb_percent)
    print("Median Cut Approach 2:")
    for _item in medcut_2:
        rgb_percent = (_item[0], (_item[1][0], _item[1][1], _item[1][2]))
        print(rgb_percent)
        median_cuts_result["medcut_2"].append(rgb_percent)
    print("Median Cut Approach 3:")
    for _item in medcut_3:
        rgb_percent = (_item[0], (_item[1][0], _item[1][1], _item[1][2]))
        print(rgb_percent)
        median_cuts_result["medcut_3"].append(rgb_percent)

    return median_cuts_result

# Author: Alex Zdanowicz
#         azdanowi@edu.uwaterloo.ca

# Citing my sources:
#   https://spin.atomicobject.com/2016/12/07/pixels-and-palettes-extracting-color-palettes-from-images/
#   https://github.com/mattnedrich/palette-maker

from PIL import Image
import numpy as np
import math

import sys
import os
from multiprocessing import Pool
import time
from pathlib import Path
import json
import cv2
import functools, collections, operator

# from utility.colour_list import ColourList
# from utility.colour import Colour

from palette_methods.median_cut import median_cut
from utility.colour_list import ColourList
from utility.colour import Colour

# Constants:
MAX_COLOUR = 256

# Dictionary that will store YouTube ID, & RGB color and their percentage values for each color palette extraction method,
#  and later be written into a json file
output_dict = {
    "YouTube_ID": -1,
    "Histogram": [],
    "Median Cut 1": [],
    "Median Cut 2": [],
    "Median Cut 3": [],
    "Quantized": []
}


def load_image_file(file):
    """
    Loads the image from the file
    :param file: The file as a string
    :return: The image acquired
    """
    image = Image.open(file)
    image.load()
    return image


def create_colour_list(image):
    """
    Creates a list of all the colours in the image sorted in reverse order,
    where each element is a tuple containing:
        the amount of times this colour appeared
        a tuple containing:
            the red value
            the green value
            the blue value
    :param image: The input image
    :return: The list of colours sorted in reverse order
    """
    colour_list = image.getcolors(maxcolors=image.size[0] * image.size[1])
    list.sort(colour_list, reverse=True)
    return colour_list


def create_colour_matrix(image, colour_list=None):
    """
    Creates a 3D matrix, such that [Red][Green][Blue],
    which contains the amount of times that specific colour occurred in each cell
    :param image: The image to grab the colours from. Can be None if there is a value for colour_list
    :param colour_list: Optional element. If this is passed in, image can be set to None
                        A list of all of the colours in the image, where each element is a tuple containing:
                            the amount of times this colour appeared
                            a tuple containing:
                                the red value
                                the green value
                                the blue value
                                ...
                            ...
    :return: The 3D matrix (of size 256x256x256) which contains the number of times that colour of the cell appeared
    """
    global MAX_COLOUR

    if colour_list is None:
        colour_list = create_colour_list(image)
    colour_matrix = np.zeros((MAX_COLOUR, MAX_COLOUR, MAX_COLOUR))

    for colour in colour_list:
        colour_matrix[colour[1][0]][colour[1][1]][colour[1][2]] = colour[0]

    return colour_matrix


def get_num_distinct_colours(image, colour_list=None):
    """
    Returns the number of distinct colours in an image or in a list of colours
    :param image: The image to grab the colours from. Can be None if colour_list is not None
    :param colour_list: Optional param. If this is passed in, image can be set to None
                        A list of all of the colours in the image, where each element is or contains a distinct colour
    :return: The number of distinct colours in the image
    """
    if colour_list is None:
        colour_list = create_colour_list(image)
    return len(colour_list)


def get_num_colours(image, colour_list=None):
    """
    Returns the number of colours in an image (non-distinct; so essentially, number of pixels)
    :param image: The image to grab the colours from. Can be None if colour_list is not None
    :param colour_list: Optional param. If this is passed in, image can be set to None
                        A list of all of the colours in the image, where each colour is a tuple containing:
                            The number times that colour appeared in the image
                            ...
    :return: The number of colours (or pixels) in the image
    """
    if colour_list is None:
        colour_list = create_colour_list(image)

    total_num_colours = 0
    for colour in colour_list:
        total_num_colours += colour[0]

    return total_num_colours


def iterate_chunk(colour_matrix, R_start, R_size, G_start, G_size, B_start, B_size):
    """
    A helper function for the histogram function
    Goes through the chunk specified by R/G/B_start/_size and returns teh average colour
    :param colour_matrix: The colour matrix where each cell contains the number of times that colour appears in the
                          picture
    :param R_start: The starting element for the red value of this chunk
    :param R_size: The size of the red value for this chunk
    :param G_start: The starting element for the green value of this chunk
    :param G_size: The size of the green value for this chunk
    :param B_start: The starting element for teh blue value of this chunk
    :param B_size: The size of the blue value for this chunk
    :return: None if there were no colours in this chunk
             A tuple containing:
                 The total number of colours in this chunk (NOT distinct)
                 A tuple containing:
                     The average red value of this chunk
                     The average green value of this chunk
                     The average blue value of this chunk
    """
    num_colours = 0
    R_avg = 0
    G_avg = 0
    B_avg = 0

    for R in range(R_start, R_start + R_size):
        for G in range(G_start, G_start + G_size):
            for B in range(B_start, B_start + B_size):
                if colour_matrix[R][G][B] == 0:
                    continue

                num_colour = colour_matrix[R][G][B]
                num_colours += num_colour
                R_avg += R * num_colour
                G_avg += G * num_colour
                B_avg += B * num_colour

    if num_colours == 0:
        return None
    return num_colours, (R_avg / num_colours, G_avg / num_colours, B_avg / num_colours)


def histogram(colour_matrix, total_num_colours, size=3):
    """
    Makes a histogram of based off of the colour matrix
    :param colour_matrix: The colour matrix where each cell contains the number of times that colour appears in the
                          picture
    :param total_num_colours: The total number of (NON-distinct) colours in the matrix
    :param size: The number of chunks = size^3. Has to be >= 2
    :return: A list, where each element is a tuple containing:
                 The percentage of elements that were in this chunk
                 A tuple containing:
                     The average R value for this chunk
                     The average G value for this chunk
                     The average B value for this chunk
    """
    global MAX_COLOUR

    # The output list
    output = []

    # Get the size of the sub-matrices. Round down
    sub_size = MAX_COLOUR / size
    sub_size = math.floor(sub_size)
    # Because of rounding down, the last size might need to be larger
    last_size = MAX_COLOUR - sub_size * (size - 1)

    for chunk_R in range(0, size):
        # chunk_R_size = 0
        if chunk_R == size - 1:
            chunk_R_size = last_size
        else:
            chunk_R_size = sub_size
        for chunk_G in range(0, size):
            # chunk_G_size = 0
            if chunk_G == size - 1:
                chunk_G_size = last_size
            else:
                chunk_G_size = sub_size
            for chunk_B in range(0, size):
                # chunk_B_size = 0
                if chunk_B == size - 1:
                    chunk_B_size = last_size
                else:
                    chunk_B_size = sub_size

                start_R = sub_size * chunk_R
                start_G = sub_size * chunk_G
                start_B = sub_size * chunk_B

                chunk_output = iterate_chunk(colour_matrix, start_R, chunk_R_size, start_G, chunk_G_size, start_B,
                                             chunk_B_size)
                if chunk_output is None:
                    continue

                output.append((round((chunk_output[0] / total_num_colours) * 100, 2),
                               (round(chunk_output[1][0]), round(chunk_output[1][1]), round(chunk_output[1][2]))))
    # print(output)
    return output


def calculate_histogram_on_all_frames(_frames):
    """
    Runs the histogram() method for creating colour palettes on a sequence of frames/images
    :param _frames: List of image/frame names
                    type: List of str
    :return: A dictionary, where:
                 key: R, G, B value as a string
                 value: Sum of all % for this (R,G,B) color from all the frames
    """
    _path_to_movie = "frames/" + sys.argv[1]
    total_percentage = {}

    print("Calculating palette using histogram approach on all frames of a given movie/trailer...")

    loop_number = 1
    for _file in _frames:
        print("{} Loading image...".format(loop_number))
        loop_number += 1

        _image = load_image_file(_path_to_movie + "/" + _file)

        _colour_list = create_colour_list(_image)
        _total_colours = get_num_colours(None, _colour_list)
        _colour_matrix = create_colour_matrix(None, _colour_list)

        _hist = histogram(_colour_matrix, _total_colours)
        list.sort(_hist, reverse=True)
        # Loop through _hist and populate total_percentage
        for _elem in _hist:
            rgb_value = str(_elem[1])
            if not rgb_value in total_percentage:
                total_percentage[rgb_value] = _elem[0]
            else:
                total_percentage[rgb_value] += _elem[0]

    # Loop ends
    print("Palette for all frames calculated!\n")
    return total_percentage


def load_image_file_median_cut(file):
    """
    Loads the image from the file
    :param file: The file name (or image name) as a string
                 :type file: str
    :return: The image acquired
    :raises: TypeError if file isn't a str
    """
    # Check type
    if not isinstance(file, str):
        raise TypeError("file must be a str\n" +
                        "\t\t\tValue:\t" + str(file) + "\n" +
                        "\t\t\tType:\t" + str(type(file)))

    # Load the file
    file_path = "frames/" + sys.argv[1] + "/" + file
    try:
        image = Image.open(file_path)
    except FileNotFoundError:
        print("File: " + file_path + " not found")
        return

    # Turn it into an image
    image.load()
    image = image.getcolors(maxcolors=image.size[0] * image.size[1])
    return image


def calculate_median_cut(filename):
    """
    :param filename: path to image being processed
    :return: A dictionary of a list, where each element in the list is a tuple containing:
                 The percentage of elements that were in this chunk
                 A tuple containing:
                     The average R value for this chunk
                     The average G value for this chunk
                     The average B value for this chunk
    """

    median_cuts_result = {
        "medcut_1": [],
        "medcut_2": [],
        "medcut_3": []
    }

    _image = load_image_file_median_cut(filename)

    _colour_list_1 = ColourList()
    _colour_list_2 = ColourList()
    _colour_list_3 = ColourList()
    for _colour in _image:
        _colour_list_1.append(
            Colour(_colour[1][0], _colour[1][1], _colour[1][2], _colour[0]))
        _colour_list_3.append(
            Colour(_colour[1][0], _colour[1][1], _colour[1][2], 1))
        for _times in range(0, _colour[0]):
            _colour_list_2.append(
                Colour(_colour[1][0], _colour[1][1], _colour[1][2], 1))
    _total_num_colours_1 = _colour_list_1.get_num_colours()
    _total_num_colours_2 = _colour_list_2.get_num_colours()
    _total_num_colours_3 = _colour_list_3.get_num_colours()
    _colour_lists_1 = [_colour_list_1]
    _colour_lists_2 = [_colour_list_2]
    _colour_lists_3 = [_colour_list_3]

    _output_1 = median_cut(_colour_lists_1)
    _output_2 = median_cut(_colour_lists_2)
    _output_3 = median_cut(_colour_lists_3)
    list.sort(_output_1[0], reverse=True)
    list.sort(_output_2[0], reverse=True)
    list.sort(_output_3[0], reverse=True)

    # we need to change first element of each row into percentage to avoid
    #   passing back all 3 of the _total_num_colours_#
    # convert to list so we can edit first element
    #   NOTE: RGB values are still a tuple after this
    medcut_1 = list(map(list, _output_1[0]))
    medcut_2 = list(map(list, _output_2[0]))
    medcut_3 = list(map(list, _output_3[0]))
    # change first element of each row into percentage
    for idx, item in enumerate(medcut_1):
        medcut_1[idx][0] = (item[0] / _total_num_colours_1) * 100
    for idx, item in enumerate(medcut_2):
        medcut_2[idx][0] = (item[0] / _total_num_colours_2) * 100
    for idx, item in enumerate(medcut_3):
        medcut_3[idx][0] = (item[0] / _total_num_colours_3) * 100

    # print
    print("Median Cut Approach 1:")
    for _item in medcut_1:
        rgb_percent = (_item[0], (_item[1][0], _item[1][1], _item[1][2]))
        print(rgb_percent)
        median_cuts_result["medcut_1"].append(rgb_percent)
    print("Median Cut Approach 2:")
    for _item in medcut_2:
        rgb_percent = (_item[0], (_item[1][0], _item[1][1], _item[1][2]))
        print(rgb_percent)
        median_cuts_result["medcut_2"].append(rgb_percent)
    print("Median Cut Approach 3:")
    for _item in medcut_3:
        rgb_percent = (_item[0], (_item[1][0], _item[1][1], _item[1][2]))
        print(rgb_percent)
        median_cuts_result["medcut_3"].append(rgb_percent)

    return median_cuts_result


def calculate_median_cut_on_all_frames(_frames):
    """
    Runs the calculate_median_cut() method for creating colour palettes on a sequence of frames/images
    :param _frames: List of image/frame names
                    type: List of str
    :return: A list of 3 dictionaries (one for each Median Cut methods), where:
                 key: R, G, B value as a string
                 value: Sum of all % for this (R,G,B) color from all the frames
    """
    medcut_1_total_percentage = {}
    medcut_2_total_percentage = {}
    medcut_3_total_percentage = {}

    print("Calculating palette using median cut approach on all frames of a given movie/trailer...")

    loop_number = 1

    for _file in _frames:

        print("{} Loading image...".format(loop_number))
        loop_number += 1
        _median_cut = calculate_median_cut(_file)
        _medcut_1 = _median_cut["medcut_1"]
        _medcut_2 = _median_cut["medcut_2"]
        _medcut_3 = _median_cut["medcut_3"]
        # Loop through _medcut_1 and populate medcut_1_total_percentage
        for _elem in _medcut_1:
            rgb_value = str(_elem[1])
            if not rgb_value in medcut_1_total_percentage:
                medcut_1_total_percentage[rgb_value] = _elem[0]
            else:
                medcut_1_total_percentage[rgb_value] += _elem[0]
        # Loop through _medcut_2 and populate medcut_2_total_percentage
        for _elem in _medcut_2:
            rgb_value = str(_elem[1])
            if not rgb_value in medcut_2_total_percentage:
                medcut_2_total_percentage[rgb_value] = _elem[0]
            else:
                medcut_2_total_percentage[rgb_value] += _elem[0]
        # Loop through _medcut_3 and populate medcut_3_total_percentage
        for _elem in _medcut_3:
            rgb_value = str(_elem[1])
            if not rgb_value in medcut_3_total_percentage:
                medcut_3_total_percentage[rgb_value] = _elem[0]
            else:
                medcut_3_total_percentage[rgb_value] += _elem[0]
    # Loop ends
    print("Palette for all frames calculated!\n")
    print(medcut_1_total_percentage)
    print(medcut_2_total_percentage)
    print(medcut_3_total_percentage)
    return [medcut_1_total_percentage, medcut_2_total_percentage, medcut_3_total_percentage]


# results, no_of_frames, sys.argv[1], "Histogram"/"Median Cut 1" etc...


def combine_results(result, no_of_frames, palette_method_name):
    """
    Combines results from different processors that run histrogram() or calculate_median_cut() methods,
    averages out % for each R,G,B color and stores the average % value in output_dict
    :param result: Return value of histrogram() or calculate_median_cut()
    :param no_of_frames: Total number of frames for the trailer/movie
    :param palette_method_name: Name of the color palette extraction method being used (i.e. "Histogram", "Median Cut 1", "Median Cut 2", "Median Cut 3")
    """
    # Unionize results from all processors
    total_percentage = {}
    # print("result", result)
    for r in result:
        # print("r", r)
        for rgb_value in r:
            if not rgb_value in total_percentage:
                # print(rgb_value)
                total_percentage[rgb_value] = r[rgb_value]
            else:
                print("{} repeated".format(rgb_value))
                total_percentage[rgb_value] += r[rgb_value]

    # Loop through total_percentage and populate average
    total = 0
    average = {}
    for rgb_value in total_percentage:
        average[rgb_value] = total_percentage[rgb_value] / no_of_frames
        total += average[rgb_value]

    # Sort average and store value in output_dict
    sorted_average = {}
    palette_method_dict = {}

    for w in sorted(average, key=average.get, reverse=True):
        sorted_average[w] = average[w]
        if sorted_average[w] == 0.0:
            break
        palette_method_dict[w] = str(sorted_average[w]/total * 100)

    output_dict[palette_method_name].append(palette_method_dict)
    print("{}: Total average of rgb values sums to: {}".format(
        palette_method_name, total))


def process_median_cuts(_frames, no_of_frames):
    """
    Uses muli-processing to run calculate_median_cut_on_all_frames(),
    and then combines the results from different processors by calling combine_results()
    :param _frames: List of image/frame names
                    type: List of str
    :param no_of_frames: Total number of frames for the trailer/movie
    """
    cpu = os.cpu_count() - 1
    if cpu == 0:
        cpu = 1
    chunks = [_frames[i::cpu] for i in range(cpu)]
    pool = Pool(processes=cpu)
    # # result: a list of dictionaries
    result = pool.map(calculate_median_cut_on_all_frames, chunks)
    pool.close()
    pool.join()
    medcut_1_total_percentage = [r[0] for r in result]
    medcut_2_total_percentage = [r[1] for r in result]
    medcut_3_total_percentage = [r[2] for r in result]
    combine_results(medcut_1_total_percentage, no_of_frames, "Median Cut 1")
    combine_results(medcut_2_total_percentage, no_of_frames, "Median Cut 2")
    combine_results(medcut_3_total_percentage, no_of_frames, "Median Cut 3")


def process_histogram(_frames, no_of_frames):
    """
    Uses muli-processing to run calculate_histogram_on_all_frames(),
    and then combines the results from different processors by calling combine_results()
    :param _frames: List of image/frame names
                    type: List of str
    :param no_of_frames: Total number of frames for the trailer/movie
    """
    cpu = os.cpu_count() - 1
    if cpu == 0:
        cpu = 1
    chunks = [_frames[i::cpu] for i in range(cpu)]
    print(len(chunks))
    pool = Pool(processes=cpu)
    # result: a list of dictionaries
    # print(_frames)
    result = pool.map(calculate_histogram_on_all_frames, chunks)
    pool.close()
    pool.join()
    combine_results(result, no_of_frames, "Histogram")


def create_channels(num):
    """
    Creates a dictionary with num**3 keys where each key is a tuple of three channels
    with value 0
    create_channels: Nat -> (Dictof tuples)
    Example: create_channels(1) -> {(0, 0, 0): 0, (0, 0, 1): 0, (0, 1, 0): 0, (0, 1, 1):0,
                                    (1, 0, 0): 0, (1, 0, 1): 0, (1, 1, 0): 0, (1, 1, 1): 0}
    """
    L = list(range(num))
    color = [0, 0, 0]
    d = {}
    return first_num(L, color, d)


def first_num(L, color, d):
    i = 0
    while i <= L[-1]:
        color[0] = L[i]
        second_num(L, color, d)
        i += 1
        color = [0, 0, 0]
    return d


def second_num(L, color, d):
    j = 0
    while j <= L[-1]:
        color[-2] = L[j]
        last_num(L, color, d)
        j += 1
    return d


def last_num(L, color, d):
    k = 0
    while k <= L[-1]:
        color[-1] = L[k]
        n = tuple(color)
        d[n] = 0
        k += 1
    return d


def quantize_colour(pixels):
    """
    Creates a new tuple with the values quantized to 1 of 512 colours

    quantize_colour: tuple(of 3 channels) -> tuple(of 3 channels)

    Example: quantize_colour((0, 2, 4)) -> (0, 1, 1)
    """
    M = 255 / 7
    quantized = tuple(round(c / M) for c in pixels)
    return quantized


def color_identification(img):
    """
    Produces a dictionary with 512 colors with values for each color incremented depending
    on how many pixels in img resemble that color.

    Note: color_dic is generated by calling create_channels(8)

    color_identification: Str (Dictof tuples) -> (Dictof tuples)

    Example: color_identification("img_processing.jpg", create_channels(8))
                                                    -> returns an incremented dic
    """
    colour_dic = create_channels(8)
    print(img)
    image = cv2.imread(img, 1)
    (h, w, d) = image.shape
    i = 0
    r_dic = {}
    while i < h:
        j = 0
        while j < w and i < h:
            if j == (w - 1):
                # print("i: ", i, "h: ", h)
                (B, G, R) = image[i, j]
                if (B, G, R) in r_dic:
                    colour_dic[r_dic[(B, G, R)]] += 1
                else:
                    qnt_colour = quantize_colour((B, G, R))
                    colour_dic[qnt_colour] += 1
                    r_dic[(B, G, R)] = qnt_colour
                j = 0
                i += 1
            else:
                # print("j: ", j, "w: ", w)
                (B, G, R) = image[i, j]
                if (B, G, R) in r_dic:
                    colour_dic[r_dic[(B, G, R)]] += 1
                else:
                    qnt_colour = quantize_colour((B, G, R))
                    colour_dic[qnt_colour] += 1
                    r_dic[(B, G, R)] = qnt_colour
                j += 1
    return colour_dic


def frames(directory):
    """
    Returns a list of all the paths to each frame in the directory
    """
    ListofImgs = []
    for entry in os.scandir(directory):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
            ListofImgs.append(entry.path)
    return ListofImgs


def palette_txt(directory):
    dic = color_identification(directory, create_channels(8))
    df = DataFrame(list(dic.items()), columns=['RGB', 'Count'])
    new_title = directory[0:-3] + "txt"
    np.savetxt(new_title, df, fmt="%s")


def process_quantized_histogram(directory):
    dir_frames = frames(directory)
    cpu = os.cpu_count() - 1
    if cpu == 0:
        cpu = 1
    # x is the folder of trailers run through the function frames(). It is a list of
    # the paths to each individual frame of the trailer.
    # chunks = [individual_frame[i::cpu] for i in range(cpu)]#[0]
    pool = Pool(processes=cpu)
    result = pool.map(color_identification, dir_frames)
    pool.close()
    pool.join()
    quantized_dic = dict(functools.reduce(operator.add, map(collections.Counter, result)))
    # print(quantized_dic)
    output_dict["Quantized"].append(create_quantized_palette(dir_frames, quantized_dic))
    # df = DataFrame(list(final_dic.items()), columns=['RGB', 'Count'])
    # np.savetxt('Quantized_Histogram.txt', df, fmt="%s")


def create_quantized_palette(frames, color_dic):
    M = 255 / 7
    quantized_palette = {}
    image = cv2.imread(frames[0], 1)
    (h, w, d) = image.shape
    total_pixels = h * w * len(frames)
    total = 0
    for key, value in color_dic.items():
        percentage = value / total_pixels * 100
        (B, G, R) = tuple([math.ceil(M*e) for e in key])
        quantized_palette[str((B, G, R))] = percentage
    sorted_palette = {}
    palette_method_dict = {}
    for w in sorted(quantized_palette, key=quantized_palette.get, reverse=True):
        sorted_palette[w] = quantized_palette[w]
        if sorted_palette[w] == 0.0:
            break
        palette_method_dict[w] = str(sorted_palette[w])
        total += sorted_palette[w]
    print(total)
    # print(quantized_palette)
    return palette_method_dict


if __name__ == '__main__':
    # print(len(sys.argv))
    if not len(sys.argv) == 3:
        print()
        print(
            "Usage: python3 colorPaletter_on_all_frames.py [name of directory that contains all the frames] [ID of Youtube Video from where these frames were gathered]")
        print("ID of a YouTube video is usually found at the end of its url")
        print()
        print(
            "Example: python3 colorPaletter_on_all_frames.py '12 YEARS A SLAVE - Official Trailer (HD)' 'z02Ie8wKKRg'")
        print()
        exit()

    start = time.time()

    _path_to_movie = "frames/" + sys.argv[1]
    youtube_id = sys.argv[2]
    output_dict["YouTube_ID"] = youtube_id
    _frames = os.listdir(_path_to_movie)
    no_of_frames = len(_frames)

    # Process Median cuts
    process_median_cuts(_frames, no_of_frames)
    # Process Histogram
    # process_histogram(_frames, no_of_frames)
    # Process Quantized Histogram
    # process_quantized_histogram(_path_to_movie)

    # All json files are stored under the "json_output" directory
    base_path = Path(__file__).parent
    rel_path = "json_output/" + sys.argv[1] + ".json"
    file_path = (base_path / rel_path).resolve()

    open(file_path, "w").close()
    f = open(file_path, "a")
    json.dump(output_dict, f)
    f.close()

    print("Total Time - {}".format(time.time() - start))
