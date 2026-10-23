# Results and figure plan

This is a reporting blueprint for the HTP fine-grained recognition project.
It is a plan, not a record of completed experiments. Populate result cells
only from saved runs using the final, documented split protocol. Do not copy
numbers or images from the earlier HTP depression-classification project.

## Planned result tables

### Table 1: Dataset composition and split protocol

| Dataset/source | Task | Total samples | Included samples | Train | Validation | Test | Grouping unit |
|---|---|---:|---:|---:|---:|---:|---|
| Roboflow HTP | Object detection | TBD | TBD | TBD | TBD | TBD | TBD |
| Bilgi | Classification / transfer | TBD | TBD | TBD | TBD | TBD | Subject, if available |
| Zhang et al. | Classification / transfer | TBD | TBD | TBD | TBD | TBD | Subject, if available |
| Pan et al. | Cross-dataset evaluation | TBD | TBD | TBD | TBD | TBD | Subject, if available |
| Small-object subset | Classification analysis | TBD | TBD | TBD | TBD | TBD | Subject, if available |

Verify all counts against the authorized source data and inclusion criteria
before filling the table. Report source and class distributions, exclusions,
duplicate handling, and whether labels or subject identifiers are available.

### Table 2: Object detector performance

| Detector | Training data | Class | Precision | Recall | AP50 | AP50:95 | Support |
|---|---|---|---:|---:|---:|---:|---:|
| YOLOv8 | Roboflow HTP | House | TBD | TBD | TBD | TBD | TBD |
| YOLOv8 | Roboflow HTP | Tree | TBD | TBD | TBD | TBD | TBD |
| YOLOv8 | Roboflow HTP | Person | TBD | TBD | TBD | TBD | TBD |
| YOLOv8 | Roboflow HTP | Small details | TBD | TBD | TBD | TBD | TBD |
| Faster R-CNN, if run | Roboflow HTP | Per class and aggregate | TBD | TBD | TBD | TBD | TBD |

State the detector checkpoint-selection rule, input resolution, confidence
threshold, IoU threshold, and whether reported numbers are validation or
held-out test results. Report per-class AP and macro/micro aggregation clearly.

### Table 3: HTP classification model comparison

| Model | Input / detector features | Dataset and split | Accuracy | Macro-F1 | Balanced accuracy | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| ResNet-50 baseline | Image | TBD | TBD | TBD | TBD | TBD | TBD |
| ViT baseline | Image | TBD | TBD | TBD | TBD | TBD | TBD |
| CBAM-EfficientNet | Image | TBD | TBD | TBD | TBD | TBD | TBD |
| SOANet | Image + fine-grained features | TBD | TBD | TBD | TBD | TBD | TBD |
| Mamba Sketch | Image / scan features | TBD | TBD | TBD | TBD | TBD | TBD |
| FASA-SOANet | Final proposed features | TBD | TBD | TBD | TBD | TBD | TBD |

### Table 4: Component ablations

| Condition | CLAHE | Detector features | Small-object loss | CBAM | Mamba | Fusion | Macro-F1 | PR-AUC |
|---|---|---|---|---|---|---|---:|---:|
| Baseline | No | No | No | No | No | No | TBD | TBD |
| Preprocessing | Yes | No | No | No | No | No | TBD | TBD |
| Small-object supervision | As specified | Yes | Yes | No | No | No | TBD | TBD |
| CBAM variant | As specified | Yes | As specified | Yes | No | No | TBD | TBD |
| Mamba variant | As specified | Yes | As specified | No | Yes | No | TBD | TBD |
| Full FASA-SOANet | As specified | Yes | As specified | As specified | As specified | Yes | TBD | TBD |

Define each component and hold all non-ablated settings fixed. The final
ablation table should reflect the actual implemented design, not this initial
proposal.

### Table 5: Cross-dataset generalization

| Train dataset | Evaluation dataset | Model | Accuracy | Macro-F1 | Balanced accuracy | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| TBD | Bilgi | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | Zhang | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | Pan | TBD | TBD | TBD | TBD | TBD | TBD |

Do not describe repeated random image splits as cross-dataset generalization.
Keep all examples from an evaluation source out of training and model
selection; use subject-grouped splits whenever subject IDs are available.

## Planned figures

| Figure | Purpose and content | Evidence / notes |
|---|---|---|
| 1. Project pipeline | Roboflow detector training; frozen detector on authorized DD data; box/area features; classification models; interpretability and external evaluation | Draw anew for this project. Distinguish implemented components from planned ones. |
| 2. Dataset and split flow | Source datasets, inclusion/exclusion, small-object subset, deduplication, and subject/source-aware splits | Recompute from project data; do not reuse previous study counts without verification. |
| 3. HTP sample montage | Representative house/tree/person drawings and annotated small details | Use only approved, de-identified samples with permission to publish. |
| 4. Class/source balance | Counts by class, source, and split | Recompute from the final manifest; show source confounding if present. |
| 5. Detector results | Training/validation curves, per-class PR or AP, qualitative detections and failure cases | Use the held-out detector evaluation and state the confidence/IoU thresholds. |
| 6. Fine-grained feature analysis | Box-area distributions and performance by object-size group | Define size categories before evaluation; avoid drawing clinical conclusions from object size alone. |
| 7. Classifier comparison | Macro-F1 and PR-AUC across ResNet-50, ViT, CBAM-EfficientNet, SOANet, Mamba Sketch, and FASA-SOANet | Use matched data, splits, and model-selection rules. |
| 8. Ablation visualization | Component contribution across CLAHE, detector features, small-object loss, CBAM, Mamba, and fusion | Match Table 4 and report uncertainty where multiple seeds are run. |
| 9. Classification error analysis | Confusion matrices, class-wise recall, ROC/PR curves, and calibration | Select thresholds on validation data only; report test support. |
| 10. Interpretability examples | Grad-CAM/attention maps overlaid on test images with detector boxes | Show successes and failures; heatmaps are explanations of model responses, not clinical evidence. |
| 11. Cross-dataset results | Per-source performance and domain-shift comparison | Evaluation data must remain independent of training and model selection. |

## Prior-study figure references

The following names refer to figures generated for the earlier project. Their
files are removed during cleanup; they are references for report organization
only, not assets or results to reuse:

| Earlier figure | Possible design reference for this project |
|---|---|
| `htp_dataset_class_balance.svg` | Figure 4, after recomputing project-2 counts |
| `htp_dataset_sample_montage.svg` | Figure 3, only with approved project-2 images |
| `training_validation_curves.svg` | Figure 5 for detector curves and Figure 7 for classifier curves |
| `confusion_matrices.svg` | Figure 9 with project-2 test predictions |
| `class_recall_comparison.svg` | Figure 7 or 9 with project-2 test metrics |
| `roc_pr_curves_matrix.svg` | Figure 9 with project-2 test probabilities |
| `calibration_matrix.svg` | Figure 9, after calibration protocol is decided |
| `factorial_macro_f1_heatmap.svg` | Figure 8, only if the project-2 ablation design is factorial |
| `threshold_sensitivity_validation.svg` | Figure 9, using validation-only threshold analysis |

These old figures were under `docs/exp1/results/img/`; no files from that
directory are retained or implied as project-2 evidence.

## Reporting and data safeguards

- Store data, annotations, subject mappings, manifests, checkpoints,
  per-image predictions, and generated outputs outside Git.
- Use authorized, de-identified data only; confirm redistribution rights
  before publishing example images or annotations.
- Freeze split manifests before training and keep subject/source leakage out
  of evaluation. Never tune settings or thresholds on the held-out test set.
- Record seeds, config, code revision, dataset version, checkpoint-selection
  metric, and evaluation command for each run.
- Report aggregate results and uncertainty; do not present a screening model as
  a diagnostic tool or infer psychological state from a single visual cue.
