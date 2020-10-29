# Author: Alex Zdanowicz
#         azdanowi@edu.uwaterloo.ca

# from ColourPaletter.utility.colours import Colours
from mediancut_palette_methods.utility.colours import Colours


class Colour:
    """
    A class to hold the properties of a colour
    Constants:
            MIN_COLOUR_VALUE    The minimum value a colour can have
                                (0)
                                :type MIN_COLOUR_VALUE: int

            MAX_COLOUR_VALUE    The maximum value a colour can have
                                (255)
                                :type MAX_COLOUR_VALUE: int

            MIN_AMOUNT          The minimum value amount can be
                                (1)
                                :type MIN_AMOUNT: int

    Attributes:
            red     The red value
                    (number between [0, 255])
                    :type red: float
                    :type red: int

            green   The green value
                    (number between [0, 255])
                    :type green: float
                    :type green: int

            blue    The blue value
                    (number between [0, 255])
                    :type blue: float
                    :type blue: int

            amount  The amount of times this colour appears
                    (integer between [1, inf))
                    :type amount: int
    """

    # Colour
    MIN_COLOUR_VALUE = 0
    MAX_COLOUR_VALUE = 255

    # Amount
    MIN_AMOUNT = 1

    def __init__(self, red, green, blue, amount=1):
        """
        Constructor for the Colour class

        :param red:    The red value (number between [0, 255])
                       :type red: float
                       :type red: int
        :param green:  The green value (number between [0, 255])
                       :type green: float
                       :type green: int
        :param blue:   The blue value (number between [0, 255])
                       :type blue: float
                       :type blue: int
        :param amount: The amount this colour shows up (number [1, inf))
                       :type amount: int

        :raises: TypeError  if red is not of type float or int
        :raises: ValueError if red is not MIN_COLOUR_VALUE <= red <= MAX_COLOUR VALUE
        :raises: TypeError  if green is not of type float or int
        :raises: ValueError if green is not MIN_COLOUR_VALUE <= green <= MAX_COLOUR_VALUE
        :raises: TypeError  if blue is not of type float or int
        :raises: ValueError if blue is not MIN_COLOUR_VALUE <= blue <= MAX_COLOUR_VALUE
        :raises: TypeError  if amount is not of type int
        :raises: ValueError if amount is not MIN_AMOUNT_VALUE <= amount
        """
        # Check types
        if not isinstance(red, (float, int)):
            raise TypeError("red has to be a float or an int\n" +
                            "\t\t\tValue:\t" + str(red) + "\n" +
                            "\t\t\tType:\t" + str(type(red)))
        if not isinstance(green, (float, int)):
            raise TypeError("green has to be a float or an int\n" +
                            "\t\t\tValue:\t" + str(green) + "\n" +
                            "\t\t\tType:\t" + str(type(green)))
        if not isinstance(blue, (float, int)):
            raise TypeError("blue has to be a float or an int\n" +
                            "\t\t\tValue:\t" + str(blue) + "\n" +
                            "\t\t\tType:\t" + str(type(blue)))
        if not isinstance(amount, int):
            raise TypeError("amount has to be an int\n" +
                            "\t\t\tValue:\t" + str(amount) + "\n" +
                            "\t\t\tType:\t" + str(type(amount)))

        # Red
        if Colour.MIN_COLOUR_VALUE <= red <= Colour.MAX_COLOUR_VALUE:
            self.red = red
        else:
            if red < Colour.MIN_COLOUR_VALUE:
                raise ValueError("Too low of a value for red\n" +
                                 "\t\t\tValue:\t" + str(red) + "\n" +
                                 "\t\t\tMinimum: " + str(Colour.MIN_COLOUR_VALUE))
            else:
                raise ValueError("Too high of a value for red\n" +
                                 "\t\t\tValue:\t " + str(red) + "\n" +
                                 "\t\t\tMaximum: " + str(Colour.MAX_COLOUR_VALUE))

        # Green
        if Colour.MIN_COLOUR_VALUE <= green <= Colour.MAX_COLOUR_VALUE:
            self.green = green
        else:
            if green < Colour.MIN_COLOUR_VALUE:
                raise ValueError("Too low of a value for green\n" +
                                 "\t\t\tValue:\t" + str(green) + "\n" +
                                 "\t\t\tMinimum: " + str(Colour.MIN_COLOUR_VALUE))
            else:
                raise ValueError("Too high of a value for green\n" +
                                 "\t\t\tValue:\t " + str(green) + "\n" +
                                 "\t\t\tMaximum: " + str(Colour.MAX_COLOUR_VALUE))

        # Blue
        if Colour.MIN_COLOUR_VALUE <= blue <= Colour.MAX_COLOUR_VALUE:
            self.blue = blue
        else:
            if blue < Colour.MIN_COLOUR_VALUE:
                raise ValueError("Too low of a value for blue\n" +
                                 "\t\t\tValue:\t" + str(blue) + "\n" +
                                 "\t\t\tMinimum: " + str(Colour.MIN_COLOUR_VALUE))
            else:
                raise ValueError("Too high of a value for blue\n" +
                                 "\t\t\tValue:\t " + str(blue) + "\n" +
                                 "\t\t\tMaximum: " + str(Colour.MAX_COLOUR_VALUE))

        # Amount
        if amount >= Colour.MIN_AMOUNT:
            self.amount = amount
        else:
            raise ValueError("Too low of a value for amount\n" +
                             "\t\t\tValue:\t " + str(amount) + "\n" +
                             "\t\t\tMinimum: " + str(Colour.MIN_AMOUNT))

    def get_red(self):
        """
        :return: the red value (number between [0, 255])
        :rtype: float
        :rtype: int
        """
        return self.red

    def get_green(self):
        """
        :return: the green value (number between [0, 255])
        :rtype: float
        :rtype: int
        """
        return self.green

    def get_blue(self):
        """
        :return: the blue value (number between [0, 255])
        :rtype: float
        :rtype: int
        """
        return self.blue

    def get_colour(self, colour):
        """
        :param colour: the specified colour
                       :type colour: Colours 
        :return: the specified colour value (number between [0, 255])
        :rtype: float or int
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
            return self.get_red()
        elif colour == Colours.GREEN:
            return self.get_green()
        elif colour == Colours.BLUE:
            return self.get_blue()
        else:
            raise ValueError("Unrecognized value for colour\n" +
                             "\t\t\tValue:\t" + str(colour) + "\n" +
                             "\t\t\tType:\t" + str(type(colour)))

    def get_amount(self):
        """
        :return: the amount of times this colour appears (int between [0, inf))
        :rtype: int
        """
        return self.amount

    def set_red(self, red):
        """
        Sets the red value if the passed-in value is between [0, 255]
        :param red: The red value (number between [0, 255])
                    :type red: float
                    :type red: int
        :return: nothing
        :rtype: None
        :raises: TypeError  if red is not float or int
                 ValueError if red is not Colour.MIN_COLOUR_VALUE <= red <= Colour.MAX_COLOUR_VALUE
        """
        # TODO: Throw RuntimeError if COLOUR_SET_ERROR is True
        # Check type
        if not isinstance(red, (float, int)):
            raise TypeError("red has to be a float or an int\n" +
                            "\t\t\tValue:\t" + str(red) + "\n" +
                            "\t\t\tType:\t" + str(type(red)))

        # Check range and set
        if Colour.MIN_COLOUR_VALUE <= red <= Colour.MAX_COLOUR_VALUE:
            self.red = red
            return

        # Otherwise throw error
        if red < Colour.MIN_COLOUR_VALUE:
            raise ValueError("Too low of a value for red\n" +
                             "\t\t\tValue:\t" + str(red) + "\n" +
                             "\t\t\tMinimum: " + str(Colour.MIN_COLOUR_VALUE))
        else:
            raise ValueError("Too high of a value for red\n" +
                             "\t\t\tValue:\t " + str(red) + "\n" +
                             "\t\t\tMaximum: " + str(Colour.MAX_COLOUR_VALUE))

    def set_green(self, green):
        """
        Sets the green value if the passed-in value is between [0, 255]
        :param green: The green value (number between [0, 255])
                      :type green: float
                      :type green: int
        :return: nothing
        :rtype: None
        :raises: TypeError  if green is not float or int
                 ValueError if green is not Colour.MIN_COLOUR_VALUE <= green <= Colour.MAX_COLOUR_VALUE
        """
        # TODO: Throw RuntimeError if COLOUR_SET_ERROR is True
        # Check type
        if not isinstance(green, (float, int)):
            raise TypeError("green has to be a float or an int\n" +
                            "\t\t\tValue:\t" + str(green) + "\n" +
                            "\t\t\tType:\t" + str(type(green)))

        # Check range and set
        if Colour.MIN_COLOUR_VALUE <= green <= Colour.MAX_COLOUR_VALUE:
            self.green = green
            return

        # Otherwise throw error
        if green < Colour.MIN_COLOUR_VALUE:
            raise ValueError("Too low of a value for green\n" +
                             "\t\t\tValue:\t" + str(green) + "\n" +
                             "\t\t\tMinimum: " + str(Colour.MIN_COLOUR_VALUE))
        else:
            raise ValueError("Too high of a value for green\n" +
                             "\t\t\tValue:\t " + str(green) + "\n" +
                             "\t\t\tMaximum: " + str(Colour.MAX_COLOUR_VALUE))

    def set_blue(self, blue):
        """
        Sets the blue value if the passed-in value is between [0, 255]
        :param blue: The blue value (number between [0, 255])
                     :type blue: float
                     :type blue: int
        :return: nothing
        :rtype: None
        :raises: TypeError  if blue is not float or int
                 ValueError if blue is not Colour.MIN_COLOUR_VALUE <= blue <= Colour.MAX_COLOUR_VALUE
        """
        # TODO: Throw RuntimeError if COLOUR_SET_ERROR is True
        # Check type
        if not isinstance(blue, (float, int)):
            raise TypeError("blue has to be a float or an int\n" +
                            "\t\t\tValue:\t" + str(blue) + "\n" +
                            "\t\t\tType:\t" + str(type(blue)))

        # Check range and set
        if Colour.MIN_COLOUR_VALUE <= blue <= Colour.MAX_COLOUR_VALUE:
            self.blue = blue
            return

        # Otherwise throw error
        if blue < Colour.MIN_COLOUR_VALUE:
            raise ValueError("Too low of a value for blue\n" +
                             "\t\t\tValue:\t" + str(blue) + "\n" +
                             "\t\t\tMinimum: " + str(Colour.MIN_COLOUR_VALUE))
        else:
            raise ValueError("Too high of a value for blue\n" +
                             "\t\t\tValue:\t " + str(blue) + "\n" +
                             "\t\t\tMaximum: " + str(Colour.MAX_COLOUR_VALUE))

    def set_colour(self, colour, value):
        """
        Sets the specified colour if the passed-in value is between [0, 255]
        :param colour: the specified colour
                       :type colour: Colours
        :param value:  the value (number between [0, 255])
                       :type value: float
                       :type value: int
        :return: nothing
        :raises: TypeError  if colour is not Colours
                 TypeError  if value is not float or int
                 ValueError if value is not Colour.MIN_COLOUR_VALUE <= value <= Colour.MAX_COLOUR_VALUE
                 ValueError if colour is not one of:
                            Colours.RED
                            Colours.GREEN
                            Colours.Blue
        """
        # TODO: Throw RuntimeError if COLOUR_SET_ERROR is True
        # Check types
        if not isinstance(colour, Colours):
            raise TypeError("colour has to be of type Colours\n" +
                            "\t\t\tValue:\t" + str(colour) + "\n" +
                            "\t\t\tType:\t" + str(type(colour)))
        if not isinstance(value, (float, int)):
            raise TypeError("value has to be a float or an int\n" +
                            "\t\t\tValue:\t" + str(value) + "\n" +
                            "\t\t\tType:\t" + str(type(value)))

        if colour == Colours.RED:
            return self.set_red(value)
        elif colour == Colours.GREEN:
            return self.set_green(value)
        elif colour == Colours.BLUE:
            return self.set_blue(value)
        else:
            raise ValueError("Unrecognized value for colour\n" +
                             "\t\t\tValue:\t" + str(colour) + "\n" +
                             "\t\t\tType:\t" + str(type(colour)))

    def set_amount(self, amount):
        """
        Sets the amount if the passed-in value is between [0, inf)
        :param amount: The amount (number between [0, inf))
                       :type amount: int
        :return: nothing
        :raises: TypeError  if amount is not of type int
        :raises: ValueError if amount is < Colour.MIN_AMOUNT
        """
        # TODO: Throw RuntimeError if COLOUR_SET_ERROR is True
        # Check type
        if not isinstance(amount, int):
            raise TypeError("amount has to be an int\n" +
                            "\t\t\tValue:\t" + str(amount) + "\n" +
                            "\t\t\tType:\t" + str(type(amount)))

        # Check range and set
        if amount >= Colour.MIN_AMOUNT:
            self.amount = amount
            return

        # Otherwise throw error
        raise ValueError("Too low of a value for amount\n" +
                         "\t\t\tValue:\t " + str(amount) + "\n" +
                         "\t\t\tMinimum: " + str(Colour.MIN_AMOUNT))

    def equals(self, other_colour):
        """
        Compares the data of two Colour objects.
            NOTE: This is NOT the same as `colour == other_colour`
            `==` compares if the objects are the same
            `.equals` compares if the data inside the objects are the same (but could be different objects)
        :param other_colour: The other colour to compare against
                             :type other_colour: Colour
        :return: True if the values of this colour and other_colour are the same
        :rtype: bool
        :raises: TypeError if other_colour is not of type Colour
        """
        # Check type
        if not isinstance(other_colour, Colour):
            raise TypeError("other_colour has to be Colour\n" +
                            "\t\t\tValue:\t" + str(other_colour) + "\n" +
                            "\t\t\tType:\t" + str(type(other_colour)))

        return (self.get_amount() == other_colour.get_amount() and
                self.get_red() == other_colour.get_red() and
                self.get_green() == other_colour.get_green() and
                self.get_blue() == other_colour.get_blue())
