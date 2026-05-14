from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

LabelTexts = str | tuple[str, ...]


@dataclass(frozen=True)
class LabelSpec:
    key: str
    texts: LabelTexts
    group: str
    subset: str = ""

    @property
    def text_tuple(self) -> tuple[str, ...]:
        if isinstance(self.texts, tuple):
            return self.texts
        return (self.texts,)


def slug(value: str) -> str:
    return value.replace(" ", "_").replace("-", "_")


def label_specs(
    *,
    values: Iterable[tuple[str, LabelTexts]],
    group: str,
    key_prefix: str,
    subset: str = "",
) -> list[LabelSpec]:
    return [
        LabelSpec(
            key=f"{key_prefix}_{slug(key)}",
            texts=texts,
            group=group,
            subset=subset,
        )
        for key, texts in values
    ]


def enum_label_specs(
    *,
    values: Iterable,
    group: str,
    key_prefix: str,
    text_builder,
    subset: str = "",
) -> list[LabelSpec]:
    return [
        LabelSpec(
            key=f"{key_prefix}_{slug(value.value)}",
            texts=text_builder(value),
            group=group,
            subset=subset,
        )
        for value in values
    ]
