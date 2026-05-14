# CGSID

CGSID contains tooling for Generating Correlated Synthetic Images, querying the generated images with CLIP, and analysing whether the designed correlations are visible in metadata and CLIP predictions.

The project supports three example datasets:

| Dataset | Scene | Main designed correlations |
|---|---|---|
| `birds` | bird photos | pose -> bird color, pose -> time of day |
| `domestic` | indoor pet photos | animal -> room, room -> ball presence, ball presence -> animal |
| `wild` | wildlife chase photos | predator -> season, time of day -> predator color |

## Repository Layout

- `src/cgsid/core` - shared configuration, distribution generation, image generation, CLIP querying, and dataset base classes.
- `src/cgsid/prompts` - prompt builders and prompt generators for wild, domestic, and bird scenes.
- `src/cgsid/models` - FLUX image generation wrapper.
- `src/cgsid/analysis` - Jensen-Shannon analysis, distribution tables, CLIP accuracy helpers, and plotting utilities.
- `src/cgsid/enums` - controlled label vocabularies used by datasets and prompts.
- `datasets/birds`, `datasets/domestic`, `datasets/wild` - dataset-specific correlations, CLIP labels, and dataset classes.
- `1_generate_dataset_*.py` - generate distributions and images.
- `2_query_dataset_with_clip_*.py` - query generated images with CLIP.
- `3_analyse_*.ipynb` - analyse generated metadata and optional CLIP outputs.
- `cgsid.yml` - conda environment with GPU/Jupyter dependencies.
- `.env_example` - runtime configuration template.

## Setup

Create the conda environment:

```bash
conda env create -f cgsid.yml
conda activate cgsid
```

For notebooks, register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name cgsid --display-name "cgsid"
```

Then select the `cgsid` kernel in Jupyter or VS Code.

## Configuration

Runtime defaults are loaded from `.env`. Start from the example:

```bash
cp .env_example .env
```

Important variables:

```env
CGSID_DATA_ROOT=./generated
CGSID_WEIGHTS_DIR=./weights
CGSID_MODEL_NAME=diffusers/FLUX.2-dev-bnb-4bit

CGSID_IMAGE_WIDTH=512
CGSID_IMAGE_HEIGHT=512
CGSID_NUM_INFERENCE_STEPS=20
CGSID_TRUE_CFG_SCALE=4.0

CGSID_CLIP_DEVICE=cuda
CGSID_CLIP_BATCH_SIZE=8
CGSID_CLIP_OUTPUT_DIRNAME=
CGSID_CLIP_OUTPUT_FILENAME=clip_scores.csv
CGSID_CLIP_LABEL_CATALOG_FILENAME=clip_label_catalog.csv
```

`CGSID_DATA_ROOT` controls where datasets are saved. Each dataset is written to:

```text
{CGSID_DATA_ROOT}/birds_correlated
{CGSID_DATA_ROOT}/domestic_correlated
{CGSID_DATA_ROOT}/wild_correlated
```

Each dataset directory contains:

```text
distribution.csv
metadata.csv
sample_000000.png
sample_000001.png
...
clip_scores.csv
clip_label_catalog.csv
```

CSV files in the dataset directory:

- `distribution.csv` - planned dataset rows before image generation. It contains the target labels and correlations that should be rendered into images.
- `metadata.csv` - generated dataset metadata. It keeps the same label columns as `distribution.csv` and adds the generated `image_path` plus the exact prompt used for each image.
- `clip_scores.csv` - CLIP query results. Each row corresponds to one generated image and stores CLIP probabilities for the configured label prompts.
- `clip_label_catalog.csv` - label catalog used for the CLIP query. It records each score column, the text prompt or prompts behind it, the label group, and prompt count.

`CGSID_WEIGHTS_DIR` controls the Hugging Face/diffusers cache used by the FLUX model. Existing environment variables override values from `.env`.

## Generate Datasets

Generate birds:

```bash
python 1_generate_dataset_birds.py
```

Generate domestic pets:

```bash
python 1_generate_dataset_domestic.py
```

Generate wild chase scenes:

```bash
python 1_generate_dataset_wild.py
```

By default, each generation script creates `3000` samples. Useful options:

```bash
python 1_generate_dataset_birds.py --size 300
python 1_generate_dataset_birds.py --skip-images
python 1_generate_dataset_birds.py --skip-distribution
python 1_generate_dataset_birds.py --width 256 --height 256
```

Generation is resumable: if `metadata.csv` already exists, previously written rows are skipped and new images continue from the next `sample_XXXXXX.png` index.

## Query With CLIP

After images are generated, run the matching CLIP query script:

```bash
python 2_query_dataset_with_clip_birds.py
python 2_query_dataset_with_clip_domestic.py
python 2_query_dataset_with_clip_wild.py
```

The CLIP step reads image paths from `metadata.csv` when available. It writes:

```text
clip_scores.csv
clip_label_catalog.csv
```

Use `CGSID_CLIP_OUTPUT_DIRNAME` or `--output-dirname` to save CLIP results in a subdirectory of the dataset directory.

Example:

```bash
python 2_query_dataset_with_clip_birds.py --batch-size 16
python 2_query_dataset_with_clip_birds.py --device cpu
```

## Analyse Results

Open the matching notebook:

```text
3_analyse_birds.ipynb
3_analyse_domestic.ipynb
3_analyse_wild.ipynb
```

The notebooks read paths through `Settings().dataset_dir(DATASET)`, so they use the same `.env` configuration as the generation and CLIP scripts.

Each notebook computes:

- label distributions from `metadata.csv`;
- expected directed Jensen-Shannon scores for designed relations;
- full directed Jensen-Shannon tables for appendix-style inspection;
- CLIP-predicted metadata from `clip_scores.csv`, if present;
- CLIP label distributions and accuracy tables;
- metadata-vs-CLIP Jensen-Shannon comparison.

## Data Schema

Generated CSV files use the shared schema:

```text
image_path
subset
object_1
object_1_color
object_2
object_2_color
interaction
environment
pose
day_time
season
```

`distribution.csv` contains planned rows before image generation. `metadata.csv` contains generated rows plus the final image path and prompt.

## Notes

- `.env`, generated datasets, weights, Python caches, and VS Code local settings are ignored by git.
- The image generator uses `diffusers.Flux2Pipeline`.
- The CLIP query currently uses `openai/clip-vit-base-patch32`.
- `wild` currently fixes the prey object to `hare` in the dataset class.

## Add a Custom Dataset

To add a new dataset, create a new directory under `datasets/`:

```text
datasets/my_dataset/
  __init__.py
  correlations.py
  clip_labels.py
  dataset.py
```

`correlations.py` should define the designed correlations as `Relation` objects:

```python
from cgsid.core.distribution import Relation


def build_my_dataset_correlations() -> list[Relation]:
    return [
        # Relation(parent_value, {child_value: probability, ...})
    ]
```

`clip_labels.py` should define the CLIP labels used to query generated images:

```python
from cgsid.core.labels import LabelSpec


def build_my_dataset_label_specs() -> list[LabelSpec]:
    return [
        LabelSpec(
            key="example_label",
            texts="a photo containing the example label",
            group="example_group",
        ),
    ]
```

Use `texts="single prompt"` for one prompt, or `texts=("prompt one", "prompt two")` when one label should aggregate multiple prompts.

`dataset.py` should implement `BaseSyntheticDataset`:

```python
from cgsid.core.dataset import BaseSyntheticDataset
from cgsid.core.distribution import Relation
from cgsid.core.labels import LabelSpec


class MyDataset(BaseSyntheticDataset):
    name = "my_dataset"
    clip_description = "Querying my dataset images with CLIP"

    def build_correlations(self) -> list[Relation]:
        ...

    def build_distribution_rows(self, *, size: int) -> list[dict[str, str]]:
        ...

    def build_label_specs(self) -> list[LabelSpec]:
        ...
```

`build_distribution_rows` must return rows using the shared CSV schema:

```text
image_path, subset, object_1, object_1_color, object_2, object_2_color,
interaction, environment, pose, day_time, season
```

Then add thin root scripts following the existing naming pattern:

```text
1_generate_dataset_my_dataset.py
2_query_dataset_with_clip_my_dataset.py
3_analyse_my_dataset.ipynb
```

The generation script should instantiate `MyDataset`, call `generate_distribution`, then call `generate_images`. The CLIP script should instantiate `MyDataset` and call `query_clip`. The analysis notebook can reuse helpers from `cgsid.analysis.tools`; only `COLUMNS`, `EXPECTED_RELATIONS`, and CLIP score mappings should usually be dataset-specific.
