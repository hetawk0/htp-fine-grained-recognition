# Viola HTP — modular training codebase

Config-driven **SOANet / SOE-FBANet** pipeline for House–Tree–Person (HTP) depression recognition, organized like [`latex/meddef`](../meddef/) (`model/`, `loader/`, `configs/`, `scripts/`).

This repository contains code and configuration only. Clinical HTP data are not included; provide an approved local dataset to train on clinical examples. Smoke tests use synthetic sketches, and QuickDraw can be downloaded separately for sketch-domain bootstrap.

> FBANet / SOANet / SOE-FBANet have **no official public code**. The modules here are independent, ablation-ready implementations, not copies of unpublished implementations.

---

## Layout

```text
latex/viola/
  model/           # SOANet, SOE-FBANet, FBANet + SOFE/SOSM/WFF/bi-level attn
  loader/          # synthetic | quickdraw | zhang/pan/bilgi folder protocols
  configs/         # YAML + ablation presets (meddef-style toggles)
  scripts/         # train / eval / smoke_test / download_quickdraw
  utils/           # seed, metrics (acc/F1/AUC), checkpoints
  analysis/        # visualization code; generated reports stay local
  datasets/        # local-only data (not included in this repository)
```

---

## Quick start

```bash
cd latex/viola
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test (synthetic data, no download)
python scripts/smoke_test.py --config configs/smoke.yaml

# Short train
python scripts/train.py --config configs/smoke.yaml

# Optional QuickDraw bootstrap (~330 MB)
python scripts/download_quickdraw.py
# then set data.name: quickdraw in a config (see configs/default.yaml)

# For an approved local clinical dataset, set data.name and data.root in a config.
```

Evaluate a checkpoint:

```bash
python scripts/eval.py \
  --config configs/smoke.yaml \
  --checkpoint outputs/smoke/best.pt \
  --split test
```

For binary classification, evaluation reports argmax metrics and, when a
validation split is available, test metrics at a threshold selected on
validation only. `history.json` from training stores the resolved config,
epoch history, best epoch, validation/test argmax metrics, selected threshold,
and complete validation/test metrics at that threshold. `utils/metrics.py`
provides accuracy, macro/weighted and per-class precision/recall/F1, class
support, confusion counts, sensitivity/specificity, ROC-AUC, and PR-AUC
(average precision). ROC-AUC and PR-AUC use continuous probabilities and do
not change with the hard-classification threshold. Binary AUC metrics are
undefined when a split contains only one class; multi-class AUC is not
implemented here.

To save standalone evaluation JSON for an auditable checkpoint:

```bash
python scripts/eval.py \
  --config configs/experiments/vabl_focal.yaml \
  --checkpoint outputs/Vabl_focal/best.pt \
  --split test > outputs/Vabl_focal/eval_test.json
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

Set under `model:` in YAML (or use `configs/ablations/`):

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
python scripts/train.py --config configs/ablations/no_sofe.yaml
python scripts/train.py --config configs/ablations/no_sosm.yaml
python scripts/train.py --config configs/ablations/no_wff.yaml
python scripts/train.py --config configs/ablations/baseline_fbanet.yaml
```

Training loss: `CE + so_penalty * SmallObjectLoss` when `so_mask` is present (`train.so_penalty`, default `2.0`).

---

## Mapping to meddef

| meddef | Viola |
|--------|--------|
| `model/model_loader.py` + ablation variants | `model/model_loader.py` + YAML flags |
| `loader/config.yaml` + dataset factory | `configs/*.yaml` + `loader/dataset_loader.py` |
| `train.py` / metrics / checkpoints | `scripts/train.py`, `utils/metrics.py`, `utils/checkpoint.py` |

---

## Plugging in clinical data

Only use datasets for which you have authorization. Keep source files, labels,
annotations, and split manifests in the local, Git-ignored `datasets/` folder.
Do not upload clinical data or annotation work to a public repository or
sharing service without explicit permission.

For example, configure a local dataset root:

```text
datasets/raw/my_dataset/images/
datasets/raw/my_dataset/labels.csv
```

Point the config at that local root:

```yaml
data:
  name: htp_depression_dd
  root: datasets/raw/my_dataset
  include_augmented: false
  invert: false
model:
  name: soe_fbanet
  backbone: resnet18   # optional once data volume allows
  pretrained: true
train:
  epochs: 50
  batch_size: 16
  lr: 1.0e-5
```

4. Optional Label Studio boxes → CSV columns `has_small_object,x,y,w,h` (see `loader/clinical.py`).

CSV schema example:

```csv
filename,label,has_small_object,x,y,w,h
img_001.png,1,1,0.1,0.2,0.15,0.18
```

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
