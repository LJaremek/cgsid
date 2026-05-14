from __future__ import annotations

from .common import StrEnum


class PetAnimal(StrEnum):
    DOG = "dog"
    CAT = "cat"


class PetColor(StrEnum):
    BLACK = "dark"
    BROWN = "brown"
    WHITE = "white"
    RED = "ginger"
    GRAY = "gray"


class PetPosition(StrEnum):
    LYING = "lying"
    STANDING = "standing"


class PetRoom(StrEnum):
    LIVING_ROOM = "living room"
    KITCHEN = "kitchen"


class BallObject(StrEnum):
    BALL = "ball"
    NONE = "none"


class BallColor(StrEnum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    ORANGE = "orange"
    PURPLE = "purple"
