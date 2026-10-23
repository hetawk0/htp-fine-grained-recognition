# HTP Fine-Grained Recognition

Research code for fine-grained House–Tree–Person (HTP) drawing analysis. The
project explores object detection, small-object-aware feature learning, global
sketch representations, and interpretable classification.

This repository currently contains the reusable HTP classification foundation
(SOANet / SOE-FBANet / FBANet) and synthetic-data smoke tests. The project
requirements also propose detector training, Mamba Sketch, CBAM-EfficientNet,
FASA-SOANet, Grad-CAM, and cross-dataset evaluation; those components must be
implemented and validated before they are described as available features.
This is research software, not a diagnostic system.

Datasets, annotations, split manifests, checkpoints, per-image predictions,
and generated experiment outputs are not distributed here. Use only data you
are authorized to access, keep it local, and do not commit it.

## Project outline

The planned workflow is:

1. Prepare authorized HTP datasets and define leakage-resistant splits.
2. Train and evaluate an object detector on the Roboflow HTP annotations.
3. Apply the selected detector to authorized HTP data and derive fine-grained
   object features, including object-area measures.
4. Compare image-classification baselines and small-object-aware models.
5. Evaluate interpretation maps and cross-dataset generalization.

The project brief is in [`docs/exp2/requirement.md`](docs/exp2/requirement.md).
Planned tables, figures, and reporting safeguards are in
[`docs/exp2/results_and_figure_plan.md`](docs/exp2/results_and_figure_plan.md).
These documents describe the intended work, not completed results.

## Download the Roboflow HTP dataset

The Roboflow project is [`htp/htp-lewxu`](https://universe.roboflow.com/htp/htp-lewxu).
The screenshots provided for this project show **version 3**, dated
2023-06-08, with 589 images resized to 640×640, a CC BY 4.0 label, and
displayed classes for house parts (door, window, wall, roof, chimney, and
sun). The page shows two versions; version 2 has 1,026 images. Verify these
details against the export and current terms before use. Version 3's displayed
split is 437 train, 152 validation, and **0 test images**. Do not report its
validation partition as a final held-out test. Check whether the dataset
version/export lets you define a separate test split before final detector
evaluation.

The visible taxonomy does not show tree or person classes, although those
appear in the project brief. Check the selected version's annotations before
deciding whether it supports the intended full HTP detection task; see the
[results and figure plan](docs/exp2/results_and_figure_plan.md#2-dataset-inventory-and-partition-tables).

The download helper requires a Roboflow API key and an exact version number.
The selected version is 3. The helper prompts for the key with hidden input
if `ROBOFLOW_API_KEY` is not set; do not paste the key into this README, a
committed config, or a command that will be saved in shell history.

Install the optional Roboflow SDK and download a version as COCO (the default,
convenient for sharing annotations between detector implementations):

```bash
python -m pip install roboflow
python scripts/download_roboflow_htp.py --version 3
```

The export is saved under `datasets/roboflow_htp/` and remains local/ignored
by Git. Choose `--format yolov8` if you specifically need a YOLOv8 export.
Each version must use an empty output directory; this prevents accidentally
mixing exports or overwriting a previous download. To keep versions separate,
pass an explicit path such as:

```bash
python scripts/download_roboflow_htp.py \
  --version 3 \
  --format coco \
  --output-dir datasets/roboflow_htp/version-3
```

The code snippet embedded in the Roboflow page uses `inference_sdk` to call a
hosted inference workflow (`general-segmentation-api`) on an image. That is
not the dataset download API and does not train a detector or export the
annotated dataset. For a local training export, use the downloader above.

The project is public, but the download API requires an API key. Create one
in Roboflow's account settings and enter it only in the hidden prompt or
provide it through a local `ROBOFLOW_API_KEY` environment variable. Never
commit or send the key. This repository does not include the downloaded data.

## Repository layout

```text
src/
  model/       SOANet, SOE-FBANet, FBANet and reusable model components
  loader/      synthetic, QuickDraw, and local HTP dataset loaders
  utils/       metrics, checkpoints, EMA, and reproducibility helpers
  configs/     active configs and historical config references
scripts/       training, evaluation, smoke tests, and experiment utilities
tests/         model and CLI regression tests
docs/exp2/     project requirements and results/figure plan
datasets/      local-only Roboflow and authorized HTP data (not committed)
```

## Setup and smoke test

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Synthetic data; no dataset download required.
python scripts/smoke_test.py --config src/configs/smoke.yaml --device cpu

# Optional one-epoch synthetic training run.
python scripts/train.py --config src/configs/smoke.yaml --device cpu
```

The smoke config creates synthetic sketches. Training writes checkpoints and
`history.json` under its configured `train.out_dir` (by default,
`outputs/smoke/`). Output directories are ignored by Git.

## Training

YAML remains the reproducible base configuration. Common settings can be
overridden on the command line:

```bash
python scripts/train.py \
  --config src/configs/smoke.yaml \
  --data synthetic \
  --arch soe_fbanet \
  --backbone cnn_lite \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-3 \
  --device cpu \
  --out-dir outputs/cli-smoke
```

Supported direct options include `--data`/`--dataset`, `--arch`/`--model`,
`--backbone`, `--image-size`, `--epochs`, `--batch-size`, `--lr`,
`--weight-decay`, `--pretrained`/`--no-pretrained`, `--device`, `--out-dir`,
`--seed`, and `--resume`. The available model architectures are
`soe_fbanet`, `soanet`, `fbanet`, and `backbone_classifier`; available
backbones are `cnn_lite`, `resnet18`, `resnet50`, and `vgg16`.

Use repeatable `--set section.key=value` options for settings without a
dedicated flag. Values are parsed as YAML, so lists, numbers, booleans, and
`null` keep their types. Direct flags override a matching `--set` value:

```bash
python scripts/train.py \
  --config src/configs/default.yaml \
  --data zhang \
  --set data.root=datasets/raw/my_dataset \
  --set 'data.split_ratios=[0.7, 0.15, 0.15]' \
  --set model.use_sofe=false \
  --epochs 50 \
  --device auto \
  --out-dir outputs/zhang-run
```

The dataset root above is a local example path; replace it with an authorized
dataset layout. The loaders support synthetic data, QuickDraw, and supported
local HTP formats (`zhang`, `pan`, `bilgi`, and `dd`). See
[`src/loader/dataset_loader.py`](src/loader/dataset_loader.py) and
[`src/configs/default.yaml`](src/configs/default.yaml) for current options.

To warm-start from a checkpoint, use `--resume path/to/checkpoint.pt` and
select a new output directory. `--device auto` chooses CUDA, then Apple MPS,
then CPU. For CUDA, install the PyTorch build appropriate for your system if
the standard requirements do not provide CUDA support. Use
`python scripts/train.py --help` for the full CLI.

## Evaluation

Evaluate a checkpoint using the same data/model configuration used for its
training:

```bash
python scripts/eval.py \
  --config src/configs/smoke.yaml \
  --checkpoint outputs/smoke/best.pt \
  --split test \
  --device cpu
```

Evaluation prints JSON metrics to standard output. To save them locally:

```bash
python scripts/eval.py \
  --config src/configs/smoke.yaml \
  --checkpoint outputs/smoke/best.pt \
  --split test > outputs/smoke/eval_test.json
```

For binary classification, the evaluation code reports argmax metrics and,
when a validation split is available, metrics at a threshold selected on
validation. Do not use test data to choose the checkpoint, threshold, or
training settings. See `python scripts/eval.py --help` for available options.

## Current implementation status

| Component | Status |
|---|---|
| SOANet / SOE-FBANet / FBANet classification foundation | Implemented |
| Synthetic-data smoke training and evaluation | Implemented |
| QuickDraw and local HTP classification loaders | Available; require local data |
| Roboflow detector training and inference pipeline | Planned |
| Detector-derived fine-grained features | Planned |
| Mamba Sketch | Planned |
| CBAM-EfficientNet benchmark | Planned |
| FASA-SOANet fusion | Planned |
| Grad-CAM / attention-map evaluation | Planned |
| Cross-dataset evaluation protocol | Planned |

The current classifier is not an object detector and does not predict boxes.
It cannot report detection metrics such as AP or IoU. Detector, Mamba, fusion,
and interpretability work should be added as separate, testable components.

## Data, privacy, and reporting

Keep datasets, source annotations, subject mappings, split manifests,
checkpoints, per-image predictions, and generated figures/logs out of Git.
The repository's `.gitignore` excludes the local `datasets/`, `outputs/`, and
`logs/` directories and model artifacts. Review `git status --short` before
staging changes; ignore rules do not untrack files already committed.

Use de-identified, authorized data only. Confirm permission before publishing
example drawings, annotations, or aggregate results. Prefer subject-grouped
and source-aware splits when metadata permit, freeze split manifests before
model selection, and report uncertainty and limitations. Do not describe a
screening research model as a diagnostic tool.

Historical YAML trial configs remain under `src/configs/archive/` as
implementation references; their outputs and results are not included and
they are not evidence for this project.
