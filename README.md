# Viola HTP — modular training codebase

Config-driven **SOANet / SOE-FBANet** pipeline for House–Tree–Person (HTP) depression recognition. Reusable project code and YAML presets live under `src/`; runnable commands remain in `scripts/`.

This repository contains code and configuration only. Clinical HTP data are not included; provide an approved local dataset to train on clinical examples. Smoke tests use synthetic sketches, and QuickDraw can be downloaded separately for sketch-domain bootstrap.

> FBANet / SOANet / SOE-FBANet have **no official public code**. The modules here are independent, ablation-ready implementations, not copies of unpublished implementations.

---

## Layout

```text
latex/viola/
  src/
    model/         # SOANet, SOE-FBANet, FBANet + SOFE/SOSM/WFF/bi-level attn
    loader/        # synthetic | quickdraw | zhang/pan/bilgi folder protocols
    utils/         # seed, metrics (acc/F1/AUC), checkpoints
    configs/       # active defaults and ablation presets
      archive/     # historical trial configs; see EXPERIMENT_RESULTS.md
    analysis/viz/  # optional dataset visualization code
  scripts/         # train / eval / smoke_test / download_quickdraw
  analysis/        # local reports and generated figures (not included)
  datasets/        # local-only data (not included in this repository)
```

---

## Quick start

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Smoke test (synthetic data, no download)
python scripts/smoke_test.py --config src/configs/smoke.yaml

# One-epoch synthetic training run
python scripts/train.py --config src/configs/smoke.yaml

# Optional QuickDraw bootstrap (~330 MB)
python scripts/download_quickdraw.py
# then set data.name: quickdraw in a config (see src/configs/default.yaml)
```

The smoke config uses generated synthetic sketches and requires no download.
Training writes `best.pt`, `last.pt`, and `history.json` under the configured
`train.out_dir` (for the smoke config: `outputs/smoke/`). These outputs are
local and ignored by Git. To train with the default configuration instead:

```bash
python scripts/train.py --config src/configs/default.yaml
```

Training options include `--epochs N`, `--device auto|cpu|mps|cuda`, and
`--resume path/to/checkpoint.pt`. `auto` selects CUDA, then Apple MPS, then CPU.
For CUDA, install the PyTorch build appropriate for your system if the standard
`requirements.txt` install does not provide CUDA support.

### Direct command-line runs

You can keep a YAML config as the base and override common settings directly
from the command line. For example, this runs a one-epoch synthetic check
without editing a config:

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

The direct options include `--data`/`--dataset`, `--arch`/`--model`,
`--backbone`, `--image-size`, `--epochs`, `--batch-size`, `--lr`,
`--weight-decay`, `--pretrained`/`--no-pretrained`, `--device`, `--out-dir`,
`--seed`, and `--resume`. `--arch` selects a Viola architecture
(`soe_fbanet`, `soanet`, `fbanet`, or `backbone_classifier`); `--backbone`
selects its feature extractor (`cnn_lite`, `resnet18`, `resnet50`, or `vgg16`).
These are Viola model and training options; MedDef-specific attack options
such as `--attack_type` and `--attack_eps` are not implemented in this
classification pipeline.

For less common settings, repeat `--set SECTION.KEY=VALUE`. Values are parsed
as YAML, so booleans, numbers, lists, and `null` retain their types. For
example:

```bash
python scripts/train.py \
  --config src/configs/default.yaml \
  --data zhang \
  --set data.root=datasets/raw/my_dataset \
  --set 'data.split_ratios=[0.7, 0.15, 0.15]' \
  --set model.use_sofe=false \
  --epochs 50 \
  --device auto \
  --out-dir outputs/zhang_run
```

When both `--set` and a direct option specify the same setting, the direct
option takes precedence. Keep dataset paths and other machine-specific values
local. `--help` lists all supported direct options.

### Run examples

Run a quick synthetic smoke training job (no dataset download required):

```bash
python scripts/train.py \
  --config src/configs/smoke.yaml \
  --device cpu
```

Try a different architecture and backbone on synthetic data. Use a distinct
output directory for each run so checkpoints are not overwritten:

```bash
python scripts/train.py \
  --config src/configs/smoke.yaml \
  --arch fbanet \
  --backbone resnet18 \
  --epochs 2 \
  --batch-size 8 \
  --device cpu \
  --out-dir outputs/fbanet-resnet18-test
```

Train on QuickDraw after downloading the house/tree/face sketches as described
in Quick start. The downloader uses this default data directory; override
`data.root` if you saved the files elsewhere:

```bash
python scripts/train.py \
  --config src/configs/default.yaml \
  --data quickdraw \
  --arch soe_fbanet \
  --backbone resnet18 \
  --epochs 20 \
  --device auto \
  --out-dir outputs/quickdraw-resnet18
```

Train on an authorized local dataset organized in class folders or with a
`labels.csv` manifest. Replace the root with your local dataset directory and
choose the matching dataset layout (`zhang`, `pan`, or `bilgi`):

```bash
python scripts/train.py \
  --config src/configs/default.yaml \
  --data zhang \
  --set data.root=datasets/raw/my_dataset \
  --set 'data.split_ratios=[0.7, 0.15, 0.15]' \
  --arch soe_fbanet \
  --backbone resnet50 \
  --pretrained \
  --epochs 50 \
  --batch-size 8 \
  --device auto \
  --out-dir outputs/my-zhang-run
```

To continue from a saved checkpoint, pass `--resume`. This warm-starts the
model weights and starts a fresh training run in the selected output folder:

```bash
python scripts/train.py \
  --config src/configs/local.yaml \
  --arch soe_fbanet \
  --backbone resnet50 \
  --resume outputs/my-zhang-run/best.pt \
  --epochs 20 \
  --out-dir outputs/my-zhang-finetune
```

Evaluate a run using its matching dataset/model config and checkpoint path:

```bash
python scripts/eval.py \
  --config src/configs/smoke.yaml \
  --checkpoint outputs/smoke/best.pt \
  --split test \
  --device cpu
```

For clinical runs, substitute the matching local config and checkpoint, for
example `src/configs/local.yaml` and `outputs/my-zhang-run/best.pt`. The config
must describe the same model and data protocol used to train that checkpoint.
For repeatable comparisons across many settings, prefer checked-in YAML
presets or the experiment-matrix runner instead of manually launching many
CLI variations.

Evaluation prints JSON to stdout. Save it locally if desired:

```bash
python scripts/eval.py \
  --config src/configs/smoke.yaml \
  --checkpoint outputs/smoke/best.pt \
  --split test > outputs/smoke/eval_test.json
```

For available options, run `python scripts/train.py --help`,
`python scripts/eval.py --help`, or `python scripts/smoke_test.py --help`.

For the best recorded clinical single-model recipe, use
`src/configs/soefbanet_resnet50_ddnorm.yaml`. It reproduces the full SOE-FBANet
cross-entropy setup; the
recorded aggregate results and earlier trial comparisons are in
[EXPERIMENT_RESULTS.md](./EXPERIMENT_RESULTS.md). That result has not been
independently reproduced from a checkpoint in this repository.

For binary classification, evaluation reports argmax metrics and, when a
validation split is available, test metrics at a threshold selected on
validation only. `history.json` from training stores the resolved config,
epoch history, best epoch, validation/test argmax metrics, selected threshold,
and complete validation/test metrics at that threshold. `src/utils/metrics.py`
provides accuracy, macro/weighted and per-class precision/recall/F1, class
support, confusion counts, sensitivity/specificity, ROC-AUC, and PR-AUC
(average precision). ROC-AUC and PR-AUC use continuous probabilities and do
not change with the hard-classification threshold. Binary AUC metrics are
undefined when a split contains only one class; multi-class AUC is not
implemented here.

To save standalone evaluation JSON for an auditable checkpoint:

```bash
python scripts/eval.py \
  --config src/configs/soefbanet_resnet50_ddnorm.yaml \
  --checkpoint outputs/soefbanet_resnet50_ddnorm/best.pt \
  --split test > outputs/soefbanet_resnet50_ddnorm/eval_test.json
```

The current model/eval path is classification-only. Small-object-aware feature
branches do not emit predicted boxes, and a classifier checkpoint cannot
recover them. Small-object detection metrics (such as box precision/recall or
IoU/mAP) require image-level ground-truth box annotations, a model that predicts
boxes, and a separate detection evaluator; none is currently implemented.
The optional `has_small_object,x,y,w,h` CSV fields are annotation inputs only,
not detection predictions or detection evaluation.

---

## Ablation flags

Set under `model:` in YAML (or use `src/configs/ablations/`):

| Flag | Default (SOE-FBANet) | Role |
|------|----------------------|------|
| `name` | `soe_fbanet` | `soanet` \| `soe_fbanet` \| `fbanet` |
| `backbone` | `cnn_lite` | `cnn_lite` \| `resnet18` \| `vgg16` |
| `use_sofe` | `true` | Small-object feature enhancement |
| `multi_scale_sofe` | `true` | 1×1 / 2×2 / 3×3 SOFE (SOE-FBANet) |
| `use_sosm` | `true` | Semantic mapping (size/position/detail) |
| `use_wff` | `true` | Weighted feature fusion |
| `learnable_fusion` | `true` | Learnable WFF gates |
| `so_weight` | `1.5` | Small-object emphasis |
| `use_bilevel_attention` | `true` | FBANet-style bi-level attention |
| `use_small_object_branch` | `true` | Dedicated small-object path |
| `use_feature_enhancement` | `true` | FBANet baseline early enhance |

**ideas2-style ablations** (copy presets):

```bash
python scripts/train.py --config src/configs/ablations/no_sofe.yaml
python scripts/train.py --config src/configs/ablations/no_sosm.yaml
python scripts/train.py --config src/configs/ablations/no_wff.yaml
python scripts/train.py --config src/configs/ablations/baseline_fbanet.yaml
```

Training loss: `CE + so_penalty * SmallObjectLoss` when `so_mask` is present (`train.so_penalty`, default `2.0`).

---

## Mapping to meddef

| meddef | Viola |
|--------|--------|
| `model/model_loader.py` + ablation variants | `src/model/model_loader.py` + YAML flags |
| Config YAML + dataset factory | `src/configs/*.yaml` + `src/loader/dataset_loader.py` |
| `train.py` / metrics / checkpoints | `scripts/train.py`, `src/utils/metrics.py`, `src/utils/checkpoint.py` |

---

## Plugging in clinical data

Only use datasets for which you have authorization. Keep source files, labels,
annotations, and split manifests in the local, Git-ignored `datasets/` folder.
Do not upload clinical data or annotation work to a public repository or
sharing service without explicit permission.

The loader supports either class subfolders or a CSV manifest. For class
subfolders, use the class names expected by the selected dataset (`zhang` and
`bilgi`: `normal/` and `depressed/`; `pan`: `negative/` and `positive/`):

```text
datasets/raw/my_dataset/
  normal/
    sample_001.png
  depressed/
    sample_002.png
```

Alternatively, put `labels.csv` at the dataset root and images under `images/`.
The CSV must have `filename` and `label` columns. Image paths can be relative
to the dataset root; a bare image filename is also looked up under `images/`.
Labels are `0` (normal/negative) and `1` (depressed/positive). Optional
small-object annotations use normalized `x,y,w,h` values:

```csv
filename,label,has_small_object,x,y,w,h
images/sample_001.png,0,0,,,
images/sample_002.png,1,1,0.1,0.2,0.15,0.18
```

Create an ignored local config at `src/configs/local.yaml` (this path is in
`.gitignore`) so your machine-specific dataset path is not accidentally
committed:

```yaml
base: default.yaml
data:
  name: zhang             # zhang | pan | bilgi | htp_depression_dd
  root: datasets/raw/my_dataset
  split_ratios: [0.7, 0.15, 0.15]
train:
  epochs: 50
  out_dir: outputs/my_clinical_run
```

Then run the smoke test, train, and evaluate using that config:

```bash
python scripts/smoke_test.py --config src/configs/local.yaml --device cpu
python scripts/train.py --config src/configs/local.yaml --device auto
python scripts/eval.py \
  --config src/configs/local.yaml \
  --checkpoint outputs/my_clinical_run/best.pt \
  --split test \
  --device cpu
```

The loader creates deterministic, stratified train/validation/test splits from
the configured ratios and seed. Keep the test split for final evaluation; do
not use it to select training settings. For the `htp_depression_dd` layout,
labels are parsed from `Nor`/`De` filename tags rather than folder names, and
augmented images are excluded by default.

For the project's best recorded recipe on HTP-Depression-DD, run:

```bash
python scripts/train.py --config src/configs/soefbanet_resnet50_ddnorm.yaml --device auto
python scripts/eval.py \
  --config src/configs/soefbanet_resnet50_ddnorm.yaml \
  --checkpoint outputs/soefbanet_resnet50_ddnorm/best.pt \
  --split test \
  --device auto
```

This config requires the authorized local HTP-Depression-DD dataset and its
frozen split manifest. Historical trial configs are retained under
`src/configs/archive/`; they are not part of the recommended run path.

QuickDraw is an optional public sketch-domain bootstrap, not clinical HTP data.
Download its house/tree/face categories with `python scripts/download_quickdraw.py`;
the files are stored under the ignored `datasets/` directory.

## Before pushing

The repository intentionally excludes datasets, research notes, generated
results, third-party reference clones, machine-specific experiment launchers,
logs, virtual environments, and local environment/credential files. The
collaborator-provided clinical dataset in local development has no verified
redistribution permission, so it is not included. Review `git status --short`
before staging or pushing. `.gitignore` does not remove files already committed
or staged.

---

## What’s stubbed until clinical data arrives

| Piece | Status |
|-------|--------|
| SOFE / SOSM / WFF / bi-level attention | Implemented (draft-aligned stubs) |
| Train / eval / checkpoints / Acc·F1·AUC | Working |
| Synthetic + QuickDraw loaders | Working |
| DD filename-label loader | Ready — requires an authorized local dataset |
| Zhang / Pan / Bilgi folder + CSV loaders | Ready — requires authorized local data |
| Official FBANet weights / paper code | N/A (none public) |
| Small-object detection head / COCO boxes | Protocol only (`bbox` in CSV); no detector yet |
| Grad-CAM / clinical Kappa reporting | Not wired (eval hooks ready for later) |
| VGG16+SVM Hatipoglu baseline | Reference implementation not bundled |

---

## Cite references

- FBANet paper: Wang et al., *Entropy* 2023 (DOI: 10.3390/e25091350) — no official code
- QuickDraw: [googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)
