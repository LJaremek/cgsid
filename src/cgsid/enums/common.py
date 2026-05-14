from __future__ import annotations

from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass


class PromptMode(StrEnum):
    SEQUENTIAL = "sequential"
    RANDOM_NO_REPEAT = "random_no_repeat"


class TimeOfDay(StrEnum):
    DAY = "day"
    EVENING = "evening"
