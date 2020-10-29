# Author: Alex Zdanowicz
#         azdanowi@edu.uwaterloo.ca

# from ColourPaletter.utility.colour_list import ColourList
# from ColourPaletter.utility.colours import Colours

from mediancut_palette_methods.utility.colour_list import ColourList
from mediancut_palette_methods.utility.colours import Colours


def get_max_colour_range(colour_list):
    """
    Finds the colour with the largest range in a given list and returns that colour and the range
    If the list is empty, returns False
    If all three colours have the same range, returns that red has the largest range
    :param colour_list: The colour list to find the type
                        :type colour_list: ColourList
    :return: False if the colour_list is empty, otherwise
             the colour with the largest range and the range of that colour
    :rtype: False or (Colours, float or int)
    :raises: TypeError if colour_list is not ColourList
    """
    # Check type
    if not isinstance(colour_list, ColourList):
        raise TypeError("colour_list has to be of type ColourList\n" +
                        "\t\t\tValue:\t" + str(colour_list) + "\n" +
                        "\t\t\tType:\t" + str(type(colour_list)))

    # Get the ranges
    red_range = colour_list.get_red_range()
    green_range = colour_list.get_green_range()
    blue_range = colour_list.get_blue_range()

    # If any of them are false, return false (empty colour_list)
    # TODO: Maybe return an error in the future? idk
    if red_range is False or green_range is False or blue_range is False:
        return False

    # This will also take care of the case if all of them are equal
    if red_range >= green_range and red_range >= blue_range:
        return Colours.RED, red_range

    if green_range >= red_range and green_range >= blue_range:
        return Colours.GREEN, green_range

    return Colours.BLUE, blue_range


def find_broadest_list(colour_lists):
    """
    Finds the colour list with the largest range and returns the colour, the range, and the colour list
    If all of the lists are empty, returns False
    If two colour lists have the same range, returns the first one
    :param colour_lists: The list of colour lists to go through
                         :type colour_lists: list, tuple, or set of ColourLists
    :return: False if all of the colour lists were empty, otherwise
             the colour with the largest range, the range, and the colour list that had that range
    :rtype: False or (Colours, float or int, ColourList)
    :raises: TypeError if colour_lists is not a list, tuple, or set;
             TypeError if the elements inside colour_lists are not of type ColourList
    """
    # Check types
    if not isinstance(colour_lists, (list, tuple, set)):
        raise TypeError("colour_lists has to be of type list, tuple, or set\n" +
                        "\t\t\tValue:\t" + str(colour_lists) + "\n" +
                        "\t\t\tType:\t" + str(type(colour_lists)))
    for colour_list in colour_lists:
        if not isinstance(colour_list, ColourList):
            raise TypeError("The elements inside colour_lists have to be of type ColourList\n" +
                            "\t\t\tValue:\t" + str(colour_list) + "\n" +
                            "\t\t\tType:\t" + str(type(colour_list)))

    largest_colour_list = None
    largest_colour = None
    largest_range = -1

    for colour_list in colour_lists:
        # TODO: Optimize this with the other for loop
        output = get_max_colour_range(colour_list)

        # If the list is empty, skip it
        if output is False:
            continue

        # If the range is larger, update
        if output[1] > largest_range:
            largest_range = output[1]
            largest_colour = output[0]
            largest_colour_list = colour_list

    # If all of the lists were empty, return false
    if largest_range == -1:
        return False
    return largest_colour, largest_range, largest_colour_list


def format_output(colour_lists):
    """
    Gives a prettified output for a list of ColourLists

    :param colour_lists: The collection of ColourLists
                         :type colour_lists: list of ColourLists
                         :type colour_lists: tuple of ColourLists
                         :type colour_lists: set of ColourLists

    :return: a list of tuples that contain the number of occurrences for each ColourList,
             along with the average RGB value for that list in a tuple
    :rtype:  list of tuples that contain:
                    int,
                    tuple that contains:
                          float or int,
                          float or int,
                          float or int

    :raises: TypeError if colour_lists is not of type list, tuple, or set
    :raises: TypeError if the elements of colour_lists are not of type ColourLists
    """
    # Check types
    if not isinstance(colour_lists, (list, tuple, set)):
        raise TypeError("colour_lists has to be of type list, tuple, or set\n" +
                        "\t\t\tValue:\t" + str(colour_lists) + "\n" +
                        "\t\t\tType:\t" + str(type(colour_lists)))
    for colour_list in colour_lists:
        if not isinstance(colour_list, ColourList):
            raise TypeError("The elements inside colour_lists have to be of type ColourList\n" +
                            "\t\t\tValue:\t" + str(colour_list) + "\n" +
                            "\t\t\tType:\t" + str(type(colour_list)))

    output = []
    for colour_list in colour_lists:
        # TODO: Optimize this with the other for list
        output.append((colour_list.get_num_colours(),
                       (colour_list.get_avg_red(), colour_list.get_avg_green(), colour_list.get_avg_blue())))

    return output


def median_cut(colour_lists, num_palettes=10):
    """
    Runs the median cut algorithm on the ColourLists passed in until num_palettes number of palettes are created

    :param colour_lists: The collection of ColourLists on which to do the median_cut program
                         :type colour_lists: list or tuple or set of ColourLists
    :param num_palettes: The end number of palettes that we want to make from the colours
                         (10 by default)
                         :type num_palettes: int

    :return: a tuple that contains the formatted output of the median cuts and
             the ColourLists after having done the median cuts
    :rtype:  tuple of:
                list of tuples of:
                    int,
                    tuple of:
                        float or int,
                        float or int,
                        float or int
                list of:
                    ColourLists

    :raises: TypeError  if colour_lists is not of type list or tuple or set
    :raises: TypeError  if the items in colour_lists are not of type ColourList
    :raises: TypeError  if num_palettes is not of type int
    :raises: ValueError if num_palettes < 1
    :raises: ValueError if colour_lists has no elements inside of it
    :raises: ValueError if there seems to not be enough colours to create a palette with num_palettes number of palettes
    """
    # Check types
    if not isinstance(colour_lists, (list, tuple, set)):
        raise TypeError("colour_lists has to be of type list, tuple, or set\n" +
                        "\t\t\tValue:\t" + str(colour_lists) + "\n" +
                        "\t\t\tType:\t" + str(type(colour_lists)))
    for colour_list in colour_lists:
        if not isinstance(colour_list, ColourList):
            raise TypeError("The elements inside colour_lists have to be of type ColourList\n" +
                            "\t\t\tValue:\t" + str(colour_list) + "\n" +
                            "\t\t\tType:\t" + str(type(colour_list)))
    if not isinstance(num_palettes, int):
        raise TypeError("num_palettes has to be an int\n" +
                        "\t\t\tValue:\t" + str(num_palettes) + "\n" +
                        "\t\t\tType:\t" + str(type(num_palettes)))

    # Check values
    if num_palettes < 1:
        raise ValueError("num_palettes has to be at least 1\n" +
                         "\t\t\tValue:\t" + str(num_palettes))
    if len(colour_lists) == 0:
        raise ValueError("colour_lists has to at least have 1 list inside of it\n" +
                         "\t\t\tLength:\t" + str(len(colour_lists)))

    # Create the colour palettes
    while len(colour_lists) < num_palettes:

        # Find the colour_list with the largest range
        broadest = find_broadest_list(colour_lists)
        if broadest is False:
            raise ValueError("Something went wrong while creating the palettes. Maybe too many palettes "
                             "or not enough colours?")

        # Get the colour (R/G/B) of the largest range and the list that had that range
        colour = broadest[0]
        colour_list = broadest[2]

        # Remove that list from the master list (we don't want duplicates)
        colour_lists.remove(colour_list)

        # Split the list
        result = colour_list.split_colour_median(colour)

        # Add the two splits back into the master list\
        if result is not False:
            colour_lists.append(result[0])
            colour_lists.append(result[1])


    # Return a prettified version of the output
    return format_output(colour_lists), colour_lists
