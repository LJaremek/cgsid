from __future__ import annotations

from .common import StrEnum


class BirdColor(StrEnum):
    BLUE = "blue"
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLACK = "black"
    WHITE = "white"


class BirdPosition(StrEnum):
    FLYING = "flying in the sky"
    NEST = "sitting in a nest"
    BRANCH = "standing on a branch"


class BeakState(StrEnum):
    EMPTY = "with an empty beak"
    WORM = "with a worm in its beak"


class TimeOfDayBirds(StrEnum):
    DAY = "day"
    EVENING = "evening"
