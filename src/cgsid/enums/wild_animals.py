from __future__ import annotations

from .common import StrEnum


class WildPredator(StrEnum):
    FOX = "fox"
    DOG = "dog"
    CAT = "cat"


class WildPrey(StrEnum):
    HARE = "hare"
    MOUSE = "mouse"


class WildPredatorColor(StrEnum):
    RED = "ginger"
    WHITE = "white"
    BLACK = "black"
    BROWN = "brown"


class WildPreyColor(StrEnum):
    RED = "ginger"
    WHITE = "white"
    BLACK = "black"
    BROWN = "brown"


class WildSeason(StrEnum):
    SUMMER = "summer"
    WINTER = "winter"


class CameraAngle(StrEnum):
    LOW_ANGLE = "low angle"
    EYE_LEVEL = "eye level"
    SIDE_VIEW = "side view"
    HIGH_ANGLE = "high angle"
    AERIAL_VIEW = "aerial view"


class CameraDistance(StrEnum):
    WIDE_SHOT = "wide shot"
    MEDIUM_DISTANCE = "medium distance"
    CLOSE_UP = "close-up"
    LONG_DISTANCE_WILDLIFE_SHOT = "long distance wildlife shot"
