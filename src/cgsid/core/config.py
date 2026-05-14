from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv()


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(env_str(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env_str(name, str(default)))


def env_path(name: str, default: str | Path) -> Path:
    return Path(env_str(name, str(default)))


@dataclass(frozen=True)
class Settings:
    data_root: Path = env_path("CGSID_DATA_ROOT", "./generated")
    weights_dir: Path = env_path("CGSID_WEIGHTS_DIR", "./weights")
    model_name: str = env_str("CGSID_MODEL_NAME", "diffusers/FLUX.2-dev-bnb-4bit")

    def dataset_dir(self, dataset_name: str) -> Path:
        return self.data_root / f"{dataset_name}_correlated"


@dataclass(slots=True)
class GenerationConfig:
    width: int = env_int("CGSID_IMAGE_WIDTH", 512)
    height: int = env_int("CGSID_IMAGE_HEIGHT", 512)
    num_inference_steps: int = env_int("CGSID_NUM_INFERENCE_STEPS", 20)
    true_cfg_scale: float = env_float("CGSID_TRUE_CFG_SCALE", 4.0)
    model_name: str = env_str("CGSID_MODEL_NAME", "diffusers/FLUX.2-dev-bnb-4bit")
    cache_dir: str = env_str("CGSID_WEIGHTS_DIR", "./weights")


@dataclass(frozen=True)
class ClipQueryConfig:
    dataset_dir: Path
    output_dirname: str = env_str("CGSID_CLIP_OUTPUT_DIRNAME", "")
    output_filename: str = env_str("CGSID_CLIP_OUTPUT_FILENAME", "clip_scores.csv")
    label_catalog_filename: str = env_str("CGSID_CLIP_LABEL_CATALOG_FILENAME", "clip_label_catalog.csv")
    device: str = env_str("CGSID_CLIP_DEVICE", "cuda")
    batch_size: int = env_int("CGSID_CLIP_BATCH_SIZE", 8)

    @property
    def output_dir(self) -> Path:
        return self.dataset_dir / self.output_dirname

    @property
    def output_csv(self) -> Path:
        return self.output_dir / self.output_filename

    @property
    def label_catalog_csv(self) -> Path:
        return self.output_dir / self.label_catalog_filename
