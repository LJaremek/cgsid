from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from cgsid.enums import CameraAngle, CameraDistance

from .clip import query_dataset_with_clip
from .config import ClipQueryConfig, GenerationConfig, Settings
from .distribution import Relation
from .generation import generate_dataset_from_csv, write_metadata_rows
from .labels import LabelSpec


class BaseSyntheticDataset(ABC):
    name: str
    clip_description: str

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    @property
    def default_output_dir(self) -> Path:
        return self.settings.dataset_dir(self.name)

    @property
    def default_distribution_csv(self) -> Path:
        return self.default_output_dir / "distribution.csv"

    @abstractmethod
    def build_correlations(self) -> list[Relation]:
        raise NotImplementedError

    @abstractmethod
    def build_distribution_rows(self, *, size: int) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def build_label_specs(self) -> list[LabelSpec]:
        raise NotImplementedError

    def generate_distribution(self, *, output_path: Path, size: int) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_metadata_rows(output_path, self.build_distribution_rows(size=size), append=False)
        return output_path

    def generate_images(
        self,
        *,
        distribution_csv: Path,
        output_dir: Path,
        config: GenerationConfig,
        camera_angle: CameraAngle | None = None,
        camera_distance: CameraDistance | None = None,
    ) -> Path:
        return generate_dataset_from_csv(
            distribution_csv=distribution_csv,
            output_dir=output_dir,
            config=config,
            wild_camera_angle=camera_angle or CameraAngle.EYE_LEVEL,
            wild_camera_distance=camera_distance or CameraDistance.MEDIUM_DISTANCE,
        )

    def query_clip(self, config: ClipQueryConfig) -> None:
        query_dataset_with_clip(
            config=config,
            labels=self.build_label_specs(),
            description=self.clip_description,
        )
