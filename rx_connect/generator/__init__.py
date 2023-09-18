from enum import Enum, auto

"""This module contains the currently supported colors and shapes in the dataset generation
pipeline.
"""


class Colors(Enum):
    BLACK = auto()
    BLUE = auto()
    BROWN = auto()
    GRAY = auto()
    GREEN = auto()
    ORANGE = auto()
    PINK = auto()
    PURPLE = auto()
    RED = auto()
    TURQUOISE = auto()
    WHITE = auto()
    YELLOW = auto()


class Shapes(Enum):
    BULLET = auto()
    CAPSULE = auto()
    DIAMOND = auto()
    DOUBLE_CIRCLE = auto()
    FREEFORM = auto()
    HEXAGON = auto()
    OCTAGON = auto()
    OVAL = auto()
    PENTAGON = auto()
    RECTANGLE = auto()
    ROUND = auto()
    SEMI_CIRCLE = auto()
    SQUARE = auto()
    TEAR = auto()
    TRAPEZOID = auto()
    TRIANGLE = auto()


COLORS_LIST = [color.name.upper() for color in Colors]
SHAPES_LIST = [shape.name.upper() for shape in Shapes]
