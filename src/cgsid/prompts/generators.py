from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from itertools import product
import random
from typing import Any

from cgsid.enums import (
    BallColor,
    BallObject,
    BeakState,
    BirdColor,
    BirdPosition,
    CameraAngle,
    CameraDistance,
    PetAnimal,
    PetColor,
    PetPosition,
    PetRoom,
    PromptMode,
    TimeOfDay,
    WildPredator,
    WildPredatorColor,
    WildPrey,
    WildPreyColor,
    WildSeason,
)

from .builders import (
    build_flux_bird_prompt,
    build_flux_pet_prompt,
    build_wild_animal_prompt,
)


class BasePromptGenerator(ABC):
    def __init__(self, mode: PromptMode | str = PromptMode.SEQUENTIAL, seed: int | None = None) -> None:
        self._mode = PromptMode(mode)
        self._rng = random.Random(seed)
        self._items = self._build_items()
        self._remaining_indexes: list[int] = []
        self._reset_iteration()

    def next_image(self) -> tuple[str, dict[str, Any]]:
        if not self._remaining_indexes:
            raise StopIteration("No prompts left in the current generator.")

        index = self._remaining_indexes.pop(0)
        params = dict(self._items[index])
        prompt = self._build_prompt(params)
        return prompt, self._serialize_params(params)

    def gen_image(self, **params: Any) -> tuple[str, dict[str, Any]]:
        normalized = self._normalize_params(params)
        prompt = self._build_prompt(normalized)
        return prompt, self._serialize_params(normalized)

    def reset(self) -> None:
        self._reset_iteration()

    def _reset_iteration(self) -> None:
        self._remaining_indexes = list(range(len(self._items)))
        if self._mode == PromptMode.RANDOM_NO_REPEAT:
            self._rng.shuffle(self._remaining_indexes)

    def _serialize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, Enum):
                serialized[key] = value.value
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        raise ValueError(f"Cannot convert {value!r} to bool.")

    @abstractmethod
    def _build_items(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def _normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def _build_prompt(self, params: dict[str, Any]) -> str:
        pass


class WildAnimalPromptGenerator(BasePromptGenerator):
    def _build_items(self) -> list[dict[str, Any]]:
        return [
            {
                "chasing_animal": chasing_animal,
                "chased_animal": chased_animal,
                "time_of_day": time_of_day,
                "season": season,
                "chasing_animal_color": chasing_color,
                "chased_animal_color": chased_color,
                "camera_angle": camera_angle,
                "camera_distance": camera_distance,
            }
            for chasing_animal, chased_animal, time_of_day, season, chasing_color, chased_color, camera_angle, camera_distance in product(
                list(WildPredator),
                list(WildPrey),
                list(TimeOfDay),
                list(WildSeason),
                list(WildPredatorColor),
                list(WildPreyColor),
                list(CameraAngle),
                list(CameraDistance),
            )
        ]

    def _normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "chasing_animal": WildPredator(params["chasing_animal"]),
            "chased_animal": WildPrey(params["chased_animal"]),
            "time_of_day": TimeOfDay(params["time_of_day"]),
            "season": WildSeason(params["season"]),
            "chasing_animal_color": WildPredatorColor(params["chasing_animal_color"]),
            "chased_animal_color": WildPreyColor(params["chased_animal_color"]),
            "camera_angle": CameraAngle(params["camera_angle"]),
            "camera_distance": CameraDistance(params["camera_distance"]),
        }

    def _build_prompt(self, params: dict[str, Any]) -> str:
        return build_wild_animal_prompt(**params)


class DomesticAnimalPromptGenerator(BasePromptGenerator):
    def _build_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for animal, animal_color, position, room, time_of_day, is_playing_with_ball in product(
            list(PetAnimal),
            list(PetColor),
            list(PetPosition),
            list(PetRoom),
            list(TimeOfDay),
            [False, True],
        ):
            if is_playing_with_ball:
                for ball_color in BallColor:
                    items.append(
                        {
                            "animal": animal,
                            "animal_color": animal_color,
                            "position": position,
                            "is_playing_with_ball": is_playing_with_ball,
                            "room": room,
                            "time_of_day": time_of_day,
                            "ball_object": BallObject.BALL,
                            "ball_color": ball_color,
                        }
                    )
            else:
                items.append(
                    {
                        "animal": animal,
                        "animal_color": animal_color,
                        "position": position,
                        "is_playing_with_ball": is_playing_with_ball,
                        "room": room,
                        "time_of_day": time_of_day,
                        "ball_object": None,
                        "ball_color": None,
                    }
                )
        return items

    def _normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        is_playing_with_ball = self._coerce_bool(params["is_playing_with_ball"])
        ball_object = params.get("ball_object")
        ball_color = params.get("ball_color")
        return {
            "animal": PetAnimal(params["animal"]),
            "animal_color": PetColor(params["animal_color"]),
            "position": PetPosition(params["position"]),
            "is_playing_with_ball": is_playing_with_ball,
            "room": PetRoom(params["room"]),
            "time_of_day": TimeOfDay(params["time_of_day"]),
            "ball_object": None if ball_object is None else BallObject(ball_object),
            "ball_color": None if ball_color is None else BallColor(ball_color),
        }

    def _build_prompt(self, params: dict[str, Any]) -> str:
        return build_flux_pet_prompt(**params)


class FluxBirdPromptGenerator(BasePromptGenerator):
    def _build_items(self) -> list[dict[str, Any]]:
        return [
            {
                "bird_color": bird_color,
                "position": position,
                "time_of_day": time_of_day,
                "beak_state": beak_state,
            }
            for bird_color, position, time_of_day, beak_state in product(
                list(BirdColor),
                list(BirdPosition),
                list(TimeOfDay),
                list(BeakState),
            )
        ]

    def _normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "bird_color": BirdColor(params["bird_color"]),
            "position": BirdPosition(params["position"]),
            "time_of_day": TimeOfDay(params["time_of_day"]),
            "beak_state": BeakState(params["beak_state"]),
        }

    def _build_prompt(self, params: dict[str, Any]) -> str:
        return build_flux_bird_prompt(**params)
