# Author: Alex Zdanowicz
#         azdanowi@edu.uwaterloo.ca

# Citing my sources:
#   https://spin.atomicobject.com/2016/12/07/pixels-and-palettes-extracting-color-palettes-from-images/
#   https://github.com/mattnedrich/palette-maker

from PIL import Image
import numpy as np
import math

import sys

# Constants:
MAX_COLOUR = 256


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
    colour_list = image.getcolors(maxcolors=image.size[0]*image.size[1])
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
    return num_colours, (R_avg/num_colours, G_avg/num_colours, B_avg/num_colours)


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
    last_size = MAX_COLOUR - sub_size * (size-1)

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

                output.append((round((chunk_output[0]/total_num_colours) * 100, 2),
                               (round(chunk_output[1][0]), round(chunk_output[1][1]), round(chunk_output[1][2]))))

    return output


def run_histogram(file):
    """
    Runs the histogram method for creating colour palettes
    """

    print("Loading image...")
    image = load_image_file(file)
    print("Image loaded!\n")

    print("Creating intermediary data...")
    colour_list = create_colour_list(image)
    total_colours = get_num_colours(None, colour_list)
    colour_matrix = create_colour_matrix(None, colour_list)
    print("Data created!\n")

    print("Calculating palette using histogram approach...")
    hist = histogram(colour_matrix, total_colours)
    list.sort(hist, reverse=True)
    print("Palette calculated!\n")

    print('\n[')
    print('\t(% of total image, (R, G, B))')
    print(']')

    print('\nHistogram:\n[')
    for elem in hist:
        print('\t(', end='')
        print(' ' * (5 - len(str(elem[0]))), end='')
        print(str(elem[0]), end=', ')
        print('(', end='')
        print(' ' * (5 - len(str(elem[1][0]))), end='')
        print(str(elem[1][0]), end=', ')
        print(' ' * (5 - len(str(elem[1][1]))), end='')
        print(str(elem[1][1]), end=', ')
        print(' ' * (5 - len(str(elem[1][2]))), end='')
        print(str(elem[1][2]), end='))')
        if elem is not hist[-1]:
            print(',')
        else:
            print('')
    print(']')

    return hist


####################
# Start of Program #
####################

if __name__ == '__main__':

    print("Loading image...")
    if len(sys.argv) != 2:
        _file = "../images/screen_shot.png"
    else:
        _file = "../" + sys.argv[1]

    _image = load_image_file(_file)
    print("Image loaded!\n")

    print("Creating intermediary data...")
    _colour_list = create_colour_list(_image)
    _total_colours = get_num_colours(None, _colour_list)
    _colour_matrix = create_colour_matrix(None, _colour_list)
    print("Data created!\n")

    print("Calculating palette using histogram approach...")
    _hist = histogram(_colour_matrix, _total_colours)
    list.sort(_hist, reverse=True)
    print("Palette calculated!\n")

    print('\n[')
    print('\t(% of total image, (R, G, B))')
    print(']')

    print('\nHistogram:\n[')
    for _elem in _hist:
        print('\t(', end='')
        print(' ' * (5 - len(str(_elem[0]))), end='')
        print(str(_elem[0]), end=', ')
        print('(', end='')
        print(' ' * (5 - len(str(_elem[1][0]))), end='')
        print(str(_elem[1][0]), end=', ')
        print(' ' * (5 - len(str(_elem[1][1]))), end='')
        print(str(_elem[1][1]), end=', ')
        print(' ' * (5 - len(str(_elem[1][2]))), end='')
        print(str(_elem[1][2]), end='))')
        if _elem is not _hist[-1]:
            print(',')
        else:
            print('')
    print(']')
