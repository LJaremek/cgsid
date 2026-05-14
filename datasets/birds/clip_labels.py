from __future__ import annotations

from cgsid.core.labels import LabelSpec, label_specs
from cgsid.enums import BirdColor, TimeOfDayBirds


def build_birds_label_specs() -> list[LabelSpec]:
    labels: list[LabelSpec] = []
    labels.extend(
        label_specs(
            values=(
                (
                    "flying",
                    (
                        "a photo of a bird flying in the open sky",
                        "a bird in flight with wings spread",
                        "a flying bird, not perched and not in a nest",
                    ),
                ),
                (
                    "sitting_in_nest",
                    (
                        "a photo of a bird sitting inside a nest",
                        "a bird resting in a nest",
                        "a perched bird in a nest, not flying",
                    ),
                ),
                (
                    "standing_on_branch",
                    (
                        "a photo of a bird standing on a tree branch",
                        "a bird perched on a branch",
                        "a bird on a branch, not flying and not in a nest",
                    ),
                ),
            ),
            group="pose",
            key_prefix="pose",
        )
    )
    labels.extend(
        label_specs(
            values=(
                (
                    "holding_worm_in_beak",
                    (
                        "a close-up photo of a bird holding a visible worm between the tips of its beak",
                        "a bird carrying a small worm or larva in its beak, with the worm clearly protruding",
                        "a bird with food in its beak, specifically a worm held by the beak",
                        "the bird's beak is gripping a visible worm",
                    ),
                ),
                (
                    "empty_beak",
                    (
                        "a close-up photo of a bird with an empty beak and no food",
                        "a bird with no worm, insect, or object in its beak",
                        "the bird's beak is clearly empty, with nothing between the beak tips",
                        "a bird whose beak is visible and not holding any food",
                    ),
                ),
            ),
            group="interaction",
            key_prefix="interaction",
        )
    )
    labels.extend(
        label_specs(
            values=(
                (
                    TimeOfDayBirds.DAY.value,
                    (
                        "a bright daytime photo of a bird",
                        "a bird photographed during the day",
                        "a bird scene in daylight",
                    ),
                ),
                (
                    TimeOfDayBirds.EVENING.value,
                    (
                        "an evening photo of a bird",
                        "a bird photographed at dusk",
                        "a bird scene with evening light",
                    ),
                ),
            ),
            group="time_of_day",
            key_prefix="time",
        )
    )
    labels.extend(
        label_specs(
            values=tuple(
                (
                    color.value,
                    (
                        f"a photo of a bird with {color.value} feathers",
                        f"the bird is {color.value}",
                        f"a {color.value} bird",
                    ),
                )
                for color in BirdColor
            ),
            group="color_presence",
            key_prefix="contains_color",
        )
    )
    return labels
