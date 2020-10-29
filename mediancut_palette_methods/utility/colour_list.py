# Author: Alex Zdanowicz
#         azdanowi@edu.uwaterloo.ca

from copy import deepcopy
# from ColourPaletter.utility.colour import Colour
# from ColourPaletter.utility.colours import Colours

from mediancut_palette_methods.utility.colour import Colour
from mediancut_palette_methods.utility.colours import Colours


class ColourList:
    """
    A class to hold a list of colours and do a lot of the computations of a list of colours
    Attributes:
        list_of_colours     The internal list to hold the colours
                            (list of Colour)

        num_colours         The total number of colours (non-unique)
                            (summation of all of the amounts of all the colours)
                            (integer between [0, inf))

        avg_red             The average red colour across all of the colours in the list
        avg_green           The average green colour across all of the colours in the list
        avg_blue            The average blue colour across all of the colours in the list
                            (number between [0, 255])

        max_red             The maximum red colour across all of the colours in the list
        max_green           The maximum green colour across all of the colours in the list
        max_blue            The maximum blue colour across all of the colours in the list
                            (number between [0, 255])

        min_red             The minimum red colour across all of the colours in the list
        min_green           The minimum green colour across all of the colours in the list
        min_blue            The minimum blue colour across all of the colours in the list
                            (number between [0, 255])

        TODO: Get rid of these; since I'm setting the max values to out-of-bound values when they aren't updated,
              these variables aren't needed
        max_red_updated     If the max_red value is updated
        max_green_updated   If the max_green value is updated
        max_blue_updated    If the max_blue value is updated
                            (boolean)

        TODO: Get rid of these; since I'm setting the min values to out-of-bound values when they aren't updated,
              these variables aren't needed
        min_red_updated     If the min_red value is updated
        min_green_updated   If the min_green value is updated
        min_blue_updated    If the min_blue value is updated
                            (boolean)
    """

    def __init__(self, colour_list=None, list_of_colours=None):
        """
        The constructor method for the ColourList.
        There are 3 possible ways of constructing a ColourList object:
            Pass in another ColourList object and copy all of the values over (deepcopying the internal list)
            Pass in a list of Colours and iterate over the list creating the values (deepcopying the list)
            Pass in nothing and create a ColourList from scratch with the values being initialized to nothing

        :param colour_list:         The other ColourList object; default to None
                                    :type colour_list: ColourList or None
        :param list_of_colours:     The list of Colour objects; default to None
                                    :type list_of_colours: set or tuple or list or None

        :raises: TypeError  if colour_list is not of type ColourList or None
        :raises: TypeError  if list_of_colours is not of type list or tuple or set or None
        :raises: TypeError  if items in list_of_colours are not of type Colour
        """
        # Another ColourList was passed in;
        #    copy all of its values over (deepcopying the internal list)
        if colour_list is not None:
            # Check type
            if not isinstance(colour_list, ColourList):
                raise TypeError("colour_list has to be of type ColourList\n" +
                                "\t\t\t(if you want to pass in a list of Colours, " +
                                "do 'ColourList(list_of_colours=my_list_of_colours)'\n" +
                                "\t\t\tValue:\t" + str(colour_list) + "\n" +
                                "\t\t\tType:\t" + str(type(colour_list)))

            # Misc
            self.list_of_colours = deepcopy(colour_list.list_of_colours)
            self.num_colours = colour_list.num_colours

            # Red
            self.total_red = colour_list.total_red
            self.max_red = colour_list.max_red
            self.min_red = colour_list.min_red
            self.max_red_updated = colour_list.max_red_updated
            self.min_red_updated = colour_list.min_red_updated

            # Green
            self.total_green = colour_list.total_green
            self.max_green = colour_list.max_green
            self.min_green = colour_list.min_green
            self.max_green_updated = colour_list.max_green_updated
            self.min_green_updated = colour_list.min_green_updated

            # Blue
            self.total_blue = colour_list.total_blue
            self.max_blue = colour_list.max_blue
            self.min_blue = colour_list.min_blue
            self.max_blue_updated = colour_list.max_blue_updated
            self.min_blue_updated = colour_list.min_blue_updated

        # A list of Colours was passed in;
        #    deepcopy the list in and iterate through all the elements to generate the attributes of the ColourList
        elif list_of_colours is not None:
            # Check types
            if not isinstance(list_of_colours, (list, tuple, set)):
                raise TypeError("list_of_colours has to be a list\n" +
                                "\t\t\t(if you want to pass in a ColourList, " +
                                "do 'ColourList(colour_list=my_colour_list'\n" +
                                "\t\t\tValue:\t" + str(list_of_colours) + "\n" +
                                "\t\t\tType:\t" + str(type(list_of_colours)))
            for colour in list_of_colours:
                # TODO: This could be optimized by combining with the other for loop
                if not isinstance(colour, Colour):
                    raise TypeError("The elements inside list_of_colours have to be of type Colour\n" +
                                    "\t\t\tValue:\t" + str(colour) + "\n" +
                                    "\t\t\tType:\t" + str(type(colour)))

            # Misc:
            self.list_of_colours = deepcopy(list_of_colours)
            self.num_colours = 0

            # Red
            self.total_red = Colour.MIN_COLOUR_VALUE
            self.max_red = Colour.MIN_COLOUR_VALUE - 1
            self.min_red = Colour.MAX_COLOUR_VALUE + 1
            self.max_red_updated = False
            self.min_red_updated = False

            # Green
            self.total_green = Colour.MIN_COLOUR_VALUE
            self.max_green = Colour.MIN_COLOUR_VALUE - 1
            self.min_green = Colour.MAX_COLOUR_VALUE + 1
            self.max_green_updated = False
            self.min_green_updated = False

            # Blue
            self.total_blue = Colour.MIN_COLOUR_VALUE
            self.max_blue = Colour.MIN_COLOUR_VALUE - 1
            self.min_blue = Colour.MAX_COLOUR_VALUE + 1
            self.max_blue_updated = False
            self.min_blue_updated = False

            # Iterate over the colours in the list
            for colour in self.list_of_colours:
                self.num_colours += colour.get_amount()

                self.total_red += colour.get_red() * colour.get_amount()
                self.total_green += colour.get_green() * colour.get_amount()
                self.total_blue += colour.get_blue() * colour.get_amount()

                if colour.get_red() > self.max_red:
                    self.max_red = colour.get_red()
                    self.max_red_updated = True
                if colour.get_red() < self.min_red:
                    self.min_red = colour.get_red()
                    self.min_red_updated = True

                if colour.get_green() > self.max_green:
                    self.max_green = colour.get_green()
                    self.max_green_updated = True
                if colour.get_green() < self.min_green:
                    self.min_green = colour.get_green()
                    self.min_green_updated = True

                if colour.get_blue() > self.max_blue:
                    self.max_blue = colour.get_blue()
                    self.max_blue_updated = True
                if colour.get_blue() < self.min_blue:
                    self.min_blue = colour.get_blue()
                    self.min_blue_updated = True

            if len(self.list_of_colours) == 0:
                self.total_red = Colour.MIN_COLOUR_VALUE - 1
                self.total_green = Colour.MIN_COLOUR_VALUE - 1
                self.total_blue = Colour.MIN_COLOUR_VALUE - 1

        # Neither a list of Colours nor a ColourList was passed in;
        #    create a ColourList from scratch, with nothing initialized
        else:
            self.list_of_colours = []
            self.num_colours = 0

            # Red
            self.total_red = Colour.MIN_COLOUR_VALUE - 1
            self.max_red = Colour.MIN_COLOUR_VALUE - 1
            self.min_red = Colour.MAX_COLOUR_VALUE + 1
            self.max_red_updated = False
            self.min_red_updated = False

            # Green
            self.total_green = Colour.MIN_COLOUR_VALUE - 1
            self.max_green = Colour.MIN_COLOUR_VALUE - 1
            self.min_green = Colour.MAX_COLOUR_VALUE + 1
            self.max_green_updated = False
            self.min_green_updated = False

            # Blue
            self.total_blue = Colour.MIN_COLOUR_VALUE - 1
            self.max_blue = Colour.MIN_COLOUR_VALUE - 1
            self.min_blue = Colour.MAX_COLOUR_VALUE + 1
            self.max_blue_updated = False
            self.min_blue_updated = False

    def get(self, index):
        """
        Gets the item in the internal list
        :param index: The index of the item to get
                      :type index: int
        :return: The item in the internal list
        :rtype: Colour
        :raises: TypeError if index is not of type int
        """
        # Check type
        if not isinstance(index, int):
            raise TypeError("index must be an int\n" +
                            "\t\t\tValue:\t" + str(index) + "\n" +
                            "\t\t\tType:\t" + str(type(index)))

        return self.list_of_colours[index]

    def get_avg_red(self):
        """
        If there are colours in the list, returns the average red colour, otherwise False
        :return: average red colour (number between [0, 255]); if the list is empty, returns False
        :rtype: bool
        :rtype: float
        """
        if len(self.list_of_colours) == 0:
            return False
        return self.total_red / self.num_colours

    def get_avg_green(self):
        """
        If there are colours in the list, returns the average green colour, otherwise False
        :return: average green colour (number between [0, 255]); if the list is empty, returns False
        :rtype: bool
        :rtype: float
        """
        if len(self.list_of_colours) == 0:
            return False
        return self.total_green / self.num_colours

    def get_avg_blue(self):
        """
        If there are colours in the list, returns the average blue colour, otherwise False
        :return: average blue colour (number between [0, 255]); if the list is empty, returns False
        :rtype: bool
        :rtype: float
        """
        if len(self.list_of_colours) == 0:
            return False
        return self.total_blue / self.num_colours

    def get_avg_colour(self, colour):
        """
        If there are colours in the list, returns the average value for the given colour, otherwise False
        :param colour: The colour to get the average
        :return: average colour value (number between [0, 255]); if the list is empty, returns False
        :rtype: bool
        :rtype: float
        :raises: TypeError  if colour is not of type Colours
                 ValueError if colour is not one of:
                            Colours.RED
                            Colours.GREEN
                            Colours.BLUE
        """
        # Check type
        if not isinstance(colour, Colours):
            raise TypeError("colour has to be of type Colours\n" +
                            "\t\t\tValue:\t" + str(colour) + "\n" +
                            "\t\t\tType:\t" + str(type(colour)))

        if colour == Colours.RED:
            return self.get_avg_red()
        elif colour == Colours.GREEN:
            return self.get_avg_green()
        elif colour == Colours.BLUE:
            return self.get_avg_blue()
        else:
            raise ValueError("Unrecognized value for colour\n" +
                             "\t\t\tValue:\t" + str(colour) + "\n" +
                             "\t\t\tType:\t" + str(type(colour)))

    def get_num_colours(self):
        """
        :return: the number of colours (non-unique) (integer between [0, inf))
        """
        return self.num_colours

    def get_num_unique_colours(self):
        """
        Returns the length of the internal list
        NOTE: Doesn't necessarily need to be unique, but that's the way I designed it
              If you're using it so that there are duplicate colours in the list,
              it won't affect any of the other functions
        :return: the number of unique colours (length of list) (integer between [0, inf))
        """
        return len(self.list_of_colours)

    def get_max_red(self):
        """
        If the list is empty, returns False; otherwise returns the highest red value in the list
        NOTE: If the maximum red value is known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :return: The highest red value (number between [0, 255]) or False
        :rtype: bool
        :rtype: float
        :rtype: int
        """
        if self.max_red_updated:
            return self.max_red

        if len(self.list_of_colours) == 0:
            return False

        self.max_red = self.list_of_colours[0].get_red()
        for colour in self.list_of_colours:
            if colour.get_red() > self.max_red:
                self.max_red = colour.get_red()
        self.max_red_updated = True

        return self.max_red

    def get_min_red(self):
        """
        If the list is empty, returns False; otherwise returns the lowest red value in the list
        NOTE: If the minimum red value is known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :return: The lowest red value (number between [0, 255]) or False
        :rtype: bool
        :rtype: float
        :rtype: int
        """
        if self.min_red_updated:
            return self.min_red

        if len(self.list_of_colours) == 0:
            return False

        self.min_red = self.list_of_colours[0].get_red()
        for colour in self.list_of_colours:
            if colour.get_red() < self.min_red:
                self.min_red = colour.get_red()
        self.min_red_updated = True

        return self.min_red

    def get_max_green(self):
        """
        If the list is empty, returns False; otherwise returns the highest green value in the list
        NOTE: If the maximum green value is known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :return: The highest green value (number between [0, 255]) or False
        :rtype: bool
        :rtype: float
        :rtype: int
        """
        if self.max_green_updated:
            return self.max_green

        if len(self.list_of_colours) == 0:
            return False

        self.max_green = self.list_of_colours[0].get_green()
        for colour in self.list_of_colours:
            if colour.get_green() > self.max_green:
                self.max_green = colour.get_green()
        self.max_green_updated = True

        return self.max_green

    def get_min_green(self):
        """
        If the list is empty, returns False; otherwise returns the lowest green value in the list
        NOTE: If the minimum green value is known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :return: The lowest green value (number between [0, 255]) or False
        :rtype: bool
        :rtype: float
        :rtype: int
        """
        if self.min_green_updated:
            return self.min_green

        if len(self.list_of_colours) == 0:
            return False

        self.min_green = self.list_of_colours[0].get_green()
        for colour in self.list_of_colours:
            if colour.get_green() < self.min_green:
                self.min_green = colour.get_green()
        self.min_green_updated = True

        return self.min_green

    def get_max_blue(self):
        """
        If the list is empty, returns False; otherwise returns the highest blue value in the list
        NOTE: If the maximum blue value is known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :return: The highest blue value (number between [0, 255]) or False
        :rtype: bool
        :rtype: float
        :rtype: int
        """
        if self.max_blue_updated:
            return self.max_blue

        if len(self.list_of_colours) == 0:
            return False

        self.max_blue = self.list_of_colours[0].get_blue()
        for colour in self.list_of_colours:
            if colour.get_blue() > self.max_blue:
                self.max_blue = colour.get_blue()
        self.max_blue_updated = True

        return self.max_blue

    def get_min_blue(self):
        """
        If the list is empty, returns False; otherwise returns the lowest blue value in the list
        NOTE: If the minimum blue value is known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :return: The lowest blue value (number between [0, 255]) or False
        :rtype: bool
        :rtype: float
        :rtype: int
        """
        if self.min_blue_updated:
            return self.min_blue

        if len(self.list_of_colours) == 0:
            return False

        self.min_blue = self.list_of_colours[0].get_blue()
        for colour in self.list_of_colours:
            if colour.get_blue() < self.min_blue:
                self.min_blue = colour.get_blue()
        self.min_blue_updated = True

        return self.min_blue

    def get_max_colour(self, colour):
        """
        If the list is empty, returns False; otherwise returns the highest colour value in the list
        NOTE: If the maximum colour value is known, it rusn in O(1) time,
              otherwise it runs in O(n) time
        :param colour: The colour for which to grab the max value
                       :type colour: Colours
        :return: The highest colour value (number between [0, 255]) or False
        :rtype: bool
        :rtype: float
        :rtype: int
        :raises: TypeError  if colour is not of type Colours
                 ValueError if colour is not one of:
                            Colours.RED
                            Colours.GREEN
                            Colours.BLUE
        """
        # Check type
        if not isinstance(colour, Colours):
            raise TypeError("colour has to be of type Colours\n" +
                            "\t\t\tValue:\t" + str(colour) + "\n" +
                            "\t\t\tType:\t" + str(type(colour)))

        if colour == Colours.RED:
            return self.get_max_red()
        elif colour == Colours.GREEN:
            return self.get_max_green()
        elif colour == Colours.BLUE:
            return self.get_max_blue()
        else:
            raise ValueError("Unrecognized value for colour\n" +
                             "\t\t\tValue:\t" + str(colour) + "\n" +
                             "\t\t\tType:\t" + str(type(colour)))

    def get_min_colour(self, colour):
        """
        If the list is empty, returns False; otherwise returns the lowest colour value in the list
        NOTE: If the minimum colour value is known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :param colour: The colour for which to grab the min value
                       :type colour: Colours
        :return: The highest colour value (number between [0, 255]) or False
        :rtype: bool
        :rtype: float
        :rtype: int
        :raises: TypeError  if colour is not of type Colours
                 ValueError if colour is not one of:
                            Colours.RED
                            Colours.GREEN
                            Colours.BLUE
        """
        # Check type
        if not isinstance(colour, Colours):
            raise TypeError("colour has to be of type Colours\n" +
                            "\t\t\tValue:\t" + str(colour) + "\n" +
                            "\t\t\tType:\t" + str(type(colour)))

        if colour == Colours.RED:
            return self.get_min_red()
        elif colour == Colours.GREEN:
            return self.get_min_green()
        elif colour == Colours.BLUE:
            return self.get_min_blue()
        else:
            raise ValueError("Unrecognized value for colour\n" +
                             "\t\t\tValue:\t" + str(colour) + "\n" +
                             "\t\t\tType:\t" + str(type(colour)))

    def get_red_range(self):
        """
        If the list is empty, returns False;
        otherwise returns the range of red values in the list (max_red - min_red)
        NOTE: If the maximum and minimum red values are known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :return: The range of red values (number between [0, 255]) or False if the list is empty
        :rtype: bool
        :rtype: float
        """
        max_red = self.get_max_red()
        min_red = self.get_min_red()
        if max_red is False or min_red is False:
            return False
        return max_red - min_red

    def get_green_range(self):
        """
        If the list is empty, returns False;
        otherwise returns the range of green values in the list (max_green - min_green)
        NOTE: If the maximum and minimum green values are known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :return: The range of green values (number between [0, 255]) or False if the list is empty
        :rtype: bool
        :rtype: float
        """
        max_green = self.get_max_green()
        min_green = self.get_min_green()
        if max_green is False or min_green is False:
            return False
        return max_green - min_green

    def get_blue_range(self):
        """
        If the list is empty, returns False;
        otherwise returns the range of blue values in the list (max_blue - min_blue)
        NOTE: If the maximum and minimum blue values are known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :return: The range of blue values (number between [0, 255]) or False if the list is empty
        :rtype: bool
        :rtype: float
        """
        max_blue = self.get_max_blue()
        min_blue = self.get_min_blue()
        if max_blue is False or min_blue is False:
            return False
        return max_blue - min_blue

    def get_colour_range(self, colour):
        """
        If the list is empty, returns False;
        otherwise returns the range of colour values in the list(max_colour - min_colour) for a given colour
        NOTE: If the maximum and minimum colour values are known, it runs in O(1) time,
              otherwise it runs in O(n) time
        :param colour: The colour to get the range for
                       :type colour: Colours
        :return: The range of values for the given colour (number between [0, 255]) or False if the list is empty
        :rtype: bool
        :rtype: float
        :raises: TypeError  if colour is not of type Colours
        :raises: ValueError if colour is not one of:
                            Colours.RED
                            Colours.GREEN
                            Colours.BLUE
        """
        # Check type
        if not isinstance(colour, Colours):
            raise TypeError("colour has to be of type Colours\n" +
                            "\t\t\tValue:\t" + str(colour) + "\n" +
                            "\t\t\tType:\t" + str(type(colour)))

        if colour == Colours.RED:
            return self.get_red_range()
        elif colour == Colours.GREEN:
            return self.get_green_range()
        elif colour == Colours.BLUE:
            return self.get_blue_range()
        else:
            raise ValueError("Unrecognized value for colour\n" +
                             "\t\t\tValue:\t" + str(colour) + "\n" +
                             "\t\t\tType:\t" + str(type(colour)))

    def append(self, colour):
        """
        Appends a given colour to the internal list, similar to list.append(),
        however first deepcopies the colour
        Also:
              updates num_colours
              updates the average red colour
              updates the average green colour
              updates the average blue colour
              updates max red colour if this is the new max red colour
              updates min red colour if this is the new min red colour
              updates max green colour if this is the new max green colour
              updates min green colour if this is the new min green colour
              updates max blue colour if this is the new max blue colour
              updates min blue colour if this is the new min blue colour
        :param colour: The colour to be passed in
                       :type colour: Colour
        :return: nothing
        :raises: TypeError if colour is not of type Colour
        """
        if not isinstance(colour, Colour):
            raise TypeError("colour must be of type Colour\n" +
                            "\t\t\tValue:\t" + str(colour) + "\n" +
                            "\t\t\tType:\t" + str(type(colour)))

        if colour.get_red() >= self.max_red:
            self.max_red = colour.get_red()
            self.max_red_updated = True
        if colour.get_red() <= self.min_red:
            self.min_red = colour.get_red()
            self.min_red_updated = True
        if colour.get_green() >= self.max_green:
            self.max_green = colour.get_green()
            self.max_green_updated = True
        if colour.get_green() <= self.min_green:
            self.min_green = colour.get_green()
            self.min_green_updated = True
        if colour.get_blue() >= self.max_blue:
            self.max_blue = colour.get_blue()
            self.max_blue_updated = True
        if colour.get_blue() <= self.min_blue:
            self.min_blue = colour.get_blue()
            self.min_blue_updated = True

        if len(self.list_of_colours) == 0:
            self.total_red = Colour.MIN_COLOUR_VALUE
            self.total_green = Colour.MIN_COLOUR_VALUE
            self.total_blue = Colour.MIN_COLOUR_VALUE

        self.total_red += colour.get_red() * colour.get_amount()
        self.total_green += colour.get_green() * colour.get_amount()
        self.total_blue += colour.get_blue() * colour.get_amount()

        self.num_colours += colour.get_amount()

        new_colour = deepcopy(colour)
        self.list_of_colours.append(new_colour)

    def pop(self, index):
        """
        Pops a given colour out of the list based on a given index, similar to list.pop()
        Also:
              updates the average red colour
              updates the average green colour
              updates the average blue colour
              updates max red if this colour was the max red
              updates min red if this colour was the min red
              updates max green if this colour was the max green
              updates min green if this colour was the min green
              updates max blue if this colour was the max blue
              updates min blue if this colour was the min blue
        :param index: index entry of the colour
                      :type index: int
        :return: The popped colour
        :rtype: Colour
        :raises: TypeError if index is not of type int
        """
        # Check type
        if not isinstance(index, int):
            raise TypeError("index must be an int\n" +
                            "\t\t\tValue:\t" + str(index) + "\n"
                                                            "\t\t\tType:\t" + str(type(index)))

        colour = self.list_of_colours.pop(index)

        if self.max_red == colour.get_red():
            self.max_red = Colour.MIN_COLOUR_VALUE - 1
            self.max_red_updated = False
        if self.min_red == colour.get_red():
            self.min_red = Colour.MAX_COLOUR_VALUE + 1
            self.min_red_updated = False
        if self.max_green == colour.get_green():
            self.max_green = Colour.MIN_COLOUR_VALUE - 1
            self.max_green_updated = False
        if self.min_green == colour.get_green():
            self.min_green = Colour.MAX_COLOUR_VALUE + 1
            self.min_green_updated = False
        if self.max_blue == colour.get_blue():
            self.max_blue = Colour.MIN_COLOUR_VALUE - 1
            self.max_blue_updated = False
        if self.min_blue == colour.get_blue():
            self.min_blue = Colour.MAX_COLOUR_VALUE + 1
            self.min_blue_updated = False

        self.total_red -= colour.get_red() * colour.get_amount()
        self.total_green -= colour.get_green() * colour.get_amount()
        self.total_blue -= colour.get_blue() * colour.get_amount()

        self.num_colours -= colour.get_amount()

        if self.num_colours == 0:
            self.total_red = Colour.MIN_AMOUNT - 1
            self.total_green = Colour.MIN_AMOUNT - 1
            self.total_blue = Colour.MIN_AMOUNT - 1

        return colour

    def remove(self, colour):
        """
        Removes a given colour from the list, similar to list.remove()
        Also:
              updates the average red colour
              updates the average green colour
              updates the average blue colour
              updates max red if this colour was the max red
              updates min red if this colour was the min red
              updates max green if this colour was the max green
              updates min green if this colour was the min green
              updates max blue if this colour was the max blue
              updates min blue if this colour was the min blue
        :param colour: The colour to remove
                       :type colour: Colour
        :return: nothing
        """
        index = self.list_of_colours.index(colour)
        self.pop(index)

    def sort_red(self, reverse=False):
        """
        Sorts the internal list on the red colour
        NOTE: Sorts increasing by default
        :param reverse: Determines which way the list gets sorted (increasing or decreasing)
                        (False by default)
                        :type reverse: bool
        :return: nothing
        :raises: TypeError if reverse is not a bool
        """
        # Check type
        if not isinstance(reverse, bool):
            raise TypeError("reverse has to be of type bool\n" +
                            "\t\t\tValue:\t" + str(reverse) + "\n" +
                            "\t\t\tType:\t" + str(type(reverse)))

        self.list_of_colours.sort(
            key=lambda colour: colour.get_red(), reverse=reverse)

    def sort_green(self, reverse=False):
        """
        Sorts the internal list on the green colour
        NOTE: Sorts increasing by default
        :param reverse: Determines which way the list gets sorted (increasing or decreasing)
                        (False by default)
                        :type reverse: bool
        :return: nothing
        :raises: TypeError if reverse is not a bool
        """
        # Check type
        if not isinstance(reverse, bool):
            raise TypeError("reverse has to be of type bool\n" +
                            "\t\t\tValue:\t" + str(reverse) + "\n" +
                            "\t\t\tType:\t" + str(type(reverse)))

        self.list_of_colours.sort(
            key=lambda colour: colour.get_green(), reverse=reverse)

    def sort_blue(self, reverse=False):
        """
        Sorts the internal list on the blue colour
        NOTE: Sorts increasing by default
        :param reverse: Determines which way the list gets sorted (increasing or decreasing)
                        (False by default)
                        :type reverse: bool
        :return: nothing
        :raises: TypeError if reverse is not a bool
        """
        # Check type
        if not isinstance(reverse, bool):
            raise TypeError("reverse has to be of type bool\n" +
                            "\t\t\tValue:\t" + str(reverse) + "\n" +
                            "\t\t\tType:\t" + str(type(reverse)))

        self.list_of_colours.sort(
            key=lambda colour: colour.get_blue(), reverse=reverse)

    def sort_colour(self, colour, reverse=False):
        """
        Sorts the internal list on the given colour
        :param colour: The colour to sort on
                       :type colour: Colours
        :param reverse: Determines which was the list gets sorted (increasing or decreasing)
                        (False by default)
                        :type reverse: bool
        :return: nothing
        :raises: TypeError  if colour is not of type Colours
        :raises: TypeError  if reverse is not of type bool
        :raises: ValueError if colour is not one of:
                            Colours.RED
                            Colours.GREEN
                            Colours.BLUE

        """
        # Check types
        if not isinstance(colour, Colours):
            raise TypeError("colour has to be of type Colours\n" +
                            "\t\t\tValue:\t" + str(colour) + "\n" +
                            "\t\t\tType:\t" + str(type(colour)))
        if not isinstance(reverse, bool):
            raise TypeError("reverse has to be of type bool\n" +
                            "\t\t\tValue:\t" + str(reverse) + "\n" +
                            "\t\t\tType:\t" + str(type(reverse)))

        if colour == Colours.RED:
            return self.sort_red(reverse)
        elif colour == Colours.GREEN:
            return self.sort_green(reverse)
        elif colour == Colours.BLUE:
            return self.sort_blue(reverse)
        else:
            raise ValueError("Unrecognized value for colour\n" +
                             "\t\t\tValue:\t" + str(colour) + "\n" +
                             "\t\t\tType:\t" + str(type(colour)))

    def sort_amount(self, reverse=True):
        """
        Sorts the internal list on the amount
        NOTE: Sorts decreasing by default
        :param reverse: Determines which way the list gets sorted (increasing or decreasing)
                        (True by default)
                        :type reverse: bool
        :return: nothing
        :raises: TypeError if reverse is not of type bool
        """
        # Check type
        if not isinstance(reverse, bool):
            raise TypeError("reverse has to be of type bool\n" +
                            "\t\t\tValue:\t" + str(reverse) + "\n" +
                            "\t\t\tType:\t" + str(type(reverse)))

        self.list_of_colours.sort(
            key=lambda colour: colour.get_amount(), reverse=reverse)

    def split_red_median(self):
        """
        Splits the list into two two sub-lists
            (one with values below the red median value,
            the other with values above the red median)
        These two sub-lists are ordered with this colour increasing

        NOTE:        The list must have at least 2 elements
        NOTE:        Does not shrink the size of the current list (copies out the values)
        NOTE:        The values are deepcopied
        SIDE EFFECT: Sorts the list by red increasing

        :return: Two sub-lists, one with values below the red median,
                 the other with values above the red median
        :rtype: tuple of ColourLists

        :raises: ValueError if the list has < 2 items
        """
        # Start off by sorting the list by red values
        self.sort_red()

        # Get the index of the median
        med_index = int(len(self.list_of_colours) / 2)

        # Check that this index doesn't equal to zero
        # If it does, that means there are either 0 or 1 items in the list,
        # and there is no point in splitting.
        # Throw an error (TODO: In the future decide if we want to throw an error or return False)
        if med_index == 0:
            return False
            raise ValueError("Attempting to split a list with 0 or 1 elements\n" +
                             "\t\t\tLen of list:\t" + str(len(self.list_of_colours)) + "\n" +
                             "\t\t\tMedian Index:\t" + str(med_index))

        # Create the two lists
        low_list = self.list_of_colours[0:med_index]
        high_list = self.list_of_colours[med_index:len(self.list_of_colours)]

        # Create the ColourList objects (the internal values will be deepcopied over during the constructor)
        low = ColourList(list_of_colours=low_list)
        high = ColourList(list_of_colours=high_list)

        return low, high

    def split_green_median(self):
        """
        Splits the list into two two sub-lists
            (one with values below the green median value,
            the other with values above the green median)
        These two sub-lists are ordered with this colour increasing

        NOTE:        The list must have at least 2 elements
        NOTE:        Does not shrink the size of the current list (copies out the values)
        NOTE:        The values are deepcopied
        SIDE EFFECT: Sorts the list by green increasing

        :return: Two sub-lists, one with values below the green median,
                 the other with values above the green median
        :rtype: tuple of ColourLists
        :raises: ValueError if the list has < 2 items
        """
        # Start off by sorting the list by green values
        self.sort_green()

        # Get the index of the median
        med_index = int(len(self.list_of_colours) / 2)

        # Check that this index doesn't equal to zero
        # If it does, that means there are either 0 or 1 items in the list,
        # and there is no point in splitting
        # Throw an error (TODO: In the future decide if we want to throw an error or return False)
        if med_index == 0:
            return False
            raise ValueError("Attempting to split a list with 0 or 1 elements\n" +
                             "\t\t\tLen of list:\t" + str(len(self.list_of_colours)) + "\n" +
                             "\t\t\tMedian Index:\t" + str(med_index))

        # Create the two lists
        low_list = self.list_of_colours[0:med_index]
        high_list = self.list_of_colours[med_index:len(self.list_of_colours)]

        # Create the ColourLIst objects (the internal values will be deepcopied over during the constructor)
        low = ColourList(list_of_colours=low_list)
        high = ColourList(list_of_colours=high_list)

        return low, high

    def split_blue_median(self):
        """
        Splits the list into two two sub-lists
            (one with values below the blue median value,
            the other with values above the blue median)
        These two sub-lists are ordered with this colour increasing

        NOTE:        The list must have at least 2 elements
        NOTE:        Does not shrink the size of the current list (copies out the values)
        NOTE:        The values are deepcopied
        SIDE EFFECT: Sorts the list by blue increasing

        :return: Two sub-lists, one with values below the blue median,
                 the other with values above the blue median
        :rtype: tuple of ColourLists
        :raises: ValueError if the list has < 2 items
        """
        # Start off by sorting the list by blue values
        self.sort_blue()

        # Get the index of the median
        med_index = int(len(self.list_of_colours) / 2)

        # Check that this index doesn't equal to zero
        # If it does, that means there are either 0 or 1 items in the list,
        # and there is no point in splitting
        # Throw an error (TODO: In the future decide if we want to throw an error or return False)
        if med_index == 0:
            return False
            raise ValueError("Attempting to split a list with 0 or 1 elements\n" +
                             "\t\t\tLen of list:\t" + str(len(self.list_of_colours)) + "\n" +
                             "\t\t\tMedian Index:\t" + str(med_index))

        # Create the two lists
        low_list = self.list_of_colours[0:med_index]
        high_list = self.list_of_colours[med_index:len(self.list_of_colours)]

        # Create the ColourLIst objects (the internal values will be deepcopied over during the constructor)
        low = ColourList(list_of_colours=low_list)
        high = ColourList(list_of_colours=high_list)

        return low, high

    def split_colour_median(self, colour):
        """
        Splits the list into two two sub-lists
            (one with values below the colour median value,
            the other with values above the colour median)
        These two sub-lists are ordered with this colour increasing

        NOTE:        The list must have at least 2 elements
        NOTE:        Does not shrink the size of the current list (copies out the values)
        NOTE:        The values are deepcopied
        SIDE EFFECT: Sorts the list by that colour increasing

        :param colour: The colour to split by
                       :type colour: Colours
        :return: Two sub-lists, one with values below the colour median,
                 the other with values above the colour median
        :rtype: tuple of ColourLists
        :raises: TypeError  if colour was not of type Colours
                 ValueError if colour is not one of:
                            Colours.RED
                            Colours.GREEN
                            Colours.BLUE
        """
        # Check type
        if not isinstance(colour, Colours):
            raise TypeError("colour has to be of type Colours\n" +
                            "\t\t\tValue:\t" + str(colour) + "\n" +
                            "\t\t\tType:\t" + str(type(colour)))

        if colour == Colours.RED:
            return self.split_red_median()
        elif colour == Colours.GREEN:
            return self.split_green_median()
        elif colour == Colours.BLUE:
            return self.split_blue_median()
        else:
            raise ValueError("Unrecognized value for colour\n" +
                             "\t\t\tValue:\t" + str(colour) + "\n" +
                             "\t\t\tType:\t" + str(type(colour)))
