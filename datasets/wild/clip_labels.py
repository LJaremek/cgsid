from __future__ import annotations

from cgsid.core.labels import LabelSpec, enum_label_specs, label_specs
from cgsid.enums import TimeOfDay, WildPredator, WildPredatorColor, WildPrey, WildPreyColor, WildSeason


WILD_OBJECT_PRESENCE_LABELS = (
    ("animal", "a photo containing an animal"),
    ("dog", "a photo containing a dog"),
    ("cat", "a photo containing a cat"),
    ("fox", "a photo containing a fox"),
    ("hare", "a photo containing a hare"),
    ("mouse", "a photo containing a mouse"),
)

WILD_SCENE_LABELS = (
    ("two_animals", "a photo containing two animals"),
    ("wildlife_scene", "a photo containing a wildlife scene"),
    ("chasing_interaction", "a photo of one animal chasing another animal"),
)


def build_wild_label_specs() -> list[LabelSpec]:
    labels: list[LabelSpec] = []
    labels.extend(label_specs(values=WILD_OBJECT_PRESENCE_LABELS, group="object_presence", key_prefix="contains"))
    labels.extend(label_specs(values=WILD_SCENE_LABELS, group="scene", key_prefix="scene"))
    labels.extend(
        enum_label_specs(
            values=WildPredator,
            group="predator",
            key_prefix="wild_predator",
            text_builder=lambda predator: f"a photo of a {predator.value} chasing another animal",
        )
    )
    labels.extend(
        enum_label_specs(
            values=WildPrey,
            group="prey",
            key_prefix="wild_prey",
            text_builder=lambda prey: f"a photo of a wild animal chasing a {prey.value}",
        )
    )
    labels.extend(
        enum_label_specs(
            values=WildPredatorColor,
            group="predator_color",
            key_prefix="wild_predator_color",
            text_builder=lambda color: f"a photo of a {color.value} predator animal",
        )
    )
    labels.extend(
        enum_label_specs(
            values=WildPreyColor,
            group="prey_color",
            key_prefix="wild_prey_color",
            text_builder=lambda color: f"a photo of a {color.value} prey animal",
        )
    )
    labels.extend(
        enum_label_specs(
            values=TimeOfDay,
            group="time_of_day",
            key_prefix="wild_time",
            text_builder=lambda value: f"a photo of a wild animal chase during {value.value}",
        )
    )
    labels.extend(
        enum_label_specs(
            values=WildSeason,
            group="season",
            key_prefix="season",
            text_builder=lambda season: f"a photo of a wild animal chase in {season.value}",
        )
    )
    labels.extend(build_wild_pair_label_specs())
    return labels


def build_wild_pair_label_specs() -> list[LabelSpec]:
    labels: list[LabelSpec] = []
    for predator in WildPredator:
        for prey in WildPrey:
            labels.append(
                LabelSpec(
                    key=f"wild_pair_{predator.value.replace(' ', '_')}_{prey.value.replace(' ', '_')}",
                    texts=f"a photo of a {predator.value} chasing a {prey.value}",
                    group="pair",
                )
            )
    return labels
