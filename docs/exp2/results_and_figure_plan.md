# Results, ablations, metrics, and figure plan

This document is the reporting blueprint for the HTP fine-grained recognition
project described in [`requirement.md`](requirement.md). It is a plan, not a
record of completed experiments. Every `TBD` is intentionally empty. Do not
copy metrics, sample counts, images, predictions, or conclusions from the
earlier HTP classification project into this study.

The planned work has two distinct prediction tasks:

1. **Object detection:** find annotated objects in the Roboflow HTP images.
2. **Drawing-level classification:** predict the label defined by the
   authorized DD/clinical dataset protocol, using images and, in selected
   conditions, detector-derived fine-grained features.

Detection scores do not establish classification performance, and
classification scores do not establish clinical validity. Report each task
and its data partitions separately.

## 1. Study design and evidence rules

### 1.1 Freeze the protocol before model comparisons

Record and freeze the following before running the final comparison:

| Protocol item | Required record |
|---|---|
| Dataset identity | Source, version/date, license or data-use permission, download/export identifier, export format, and class map |
| Unit of analysis | Drawing/image, participant, session, or other; explain repeated images per participant |
| Labels | Exact class/score definitions, label source, adjudication, and mapping between datasets |
| Inclusion/exclusion | Predefined criteria, small-object definition, unreadable/corrupt image handling |
| Duplicate control | Exact and near-duplicate checks across train/validation/test and across sources |
| Split unit | Participant/group where available; otherwise state the image-level limitation |
| Split assignment | Frozen manifest, seed, stratification variables, and source-specific counts |
| Preprocessing | Resolution, grayscale/color handling, normalization, inversion, CLAHE parameters |
| Detector protocol | Dataset version, category mapping, train/validation/test partitions, checkpoint rule |
| Classifier protocol | Inputs/features, model initialization, optimizer/schedule, loss, augmentations |
| Selection rules | Validation metric for checkpoint and threshold; no test-set selection |
| Repetitions | Seeds, number of runs, aggregation rule, and uncertainty interval method |
| Software/hardware | Code revision, dependency versions, device, runtime measurement procedure |

The requirement describes a final small-object subset of 210 drawings and
counts from several sources. Treat these as **unverified planning figures**:
recompute them from authorized source files and confirm that source,
participant, label, age, and inclusion definitions reconcile before using
them in any result.

### 1.2 Leakage-resistant evaluation

- Keep all drawings from one participant in a single partition whenever
  participant identifiers are available. If not available, disclose that
  limitation and do not imply subject-independent generalization.
- Keep exact and near duplicates together; document the check and resolution.
- For cross-dataset evaluation, hold the target dataset/source out of all
  training, hyperparameter selection, checkpoint selection, and threshold
  selection. A source used for validation is not an untouched external test.
- Fit normalization statistics, feature selection, and any data-dependent
  small-object cutoffs on the training partition only.
- Select checkpoints, classification thresholds, detector confidence
  thresholds, and calibration parameters on validation data only. Evaluate
  the frozen choices once on test data.
- Keep a final test set sealed during model development. Use nested or
  grouped cross-validation for tuning when sample size and available metadata
  make a single validation split unreliable.
- Train each planned condition with the same data split, augmentations,
  training budget, initialization policy, and selection rule unless the
  condition specifically studies that factor.

### 1.3 Results status and notation

Use `Not run`, `Not available`, or `Not applicable` rather than leaving an
ambiguous blank. Record `N` and the denominator for every metric. Use
percentages to two decimal places for rates and metrics; retain probability
thresholds on a 0–1 scale. Report confidence intervals with the point
estimate. Distinguish validation, internal test, and external test in every
table and figure caption.

No numerical result in this document is a measured result. Proposed models,
ablations, and metrics below are pending implementation, protocol approval,
and evaluation.

## 2. Dataset inventory and partition tables

### Table 1. Source dataset inventory

| Source / version | Task role | Permission / license verified | Images before QC | Included images | Participants | Label types / classes | Annotation types | Exclusions and reason |
|---|---|---|---:|---:|---:|---|---|---|
| Roboflow HTP (`htp-lewxu`, v3) | Detector development | Screenshot indicates CC BY 4.0; verify current terms | 589 in v3; v2 screenshot count 1,026 | TBD | TBD / unavailable | Screenshot shows 7 classes: door, window, house, wall, roof, chimney, sun | Bounding boxes; verify taxonomy per version | TBD |
| Istanbul Bilgi | Classification / external evaluation | TBD | TBD | TBD | TBD | Verify source labels and score scales | Verify available boxes/metadata | TBD |
| Zhang et al. (2023) | Classification / external evaluation | TBD | TBD | TBD | TBD | Verify diagnostic label mapping | Verify annotations/metadata | TBD |
| Pan et al. (2022) | Cross-dataset evaluation | TBD | TBD | TBD | TBD | Verify positive/negative construct | Verify annotations/metadata | TBD |
| Supervisor-provided DD collection | Classification development/evaluation | TBD | TBD | TBD | TBD | Verify diagnosis/label provenance | Verify boxes and metadata | TBD |

Do not assume that similarly named labels across sources are equivalent.
Document label harmonization and any excluded/ambiguous classes. The Roboflow
project screenshots available during planning show version 3 (dated
2023-06-08), 589 images at 640×640, two dataset versions, a CC BY 4.0 license
label, and seven classes: door, window, house, wall, roof, chimney, and sun.
Version 2 is shown with 1,026 images. Verify these details against the
downloaded export and current project terms.

Version 3's displayed split is 437 train, 152 validation, and zero test
images. Its provided validation partition cannot also serve as a final
held-out test. Determine whether a separate detector test split can be
created from the exported images with permission, or whether an independently
annotated test source is required. Do not evaluate on validation and then
report that as final test performance.

**Important scope mismatch:** the shown Roboflow taxonomy is house/house-part
detection. It does not show tree or person classes, despite the project brief
describing house, tree, person, and small details. Before treating this as the
detector for the DD classification stage, inspect the version's class list,
image examples, annotation quality, and overlap with the target HTP drawings.
Decide whether this dataset is only a house-part detector pretraining source,
whether compatible tree/person annotations must be sourced separately, or
whether the detector/classifier scope should be narrowed. Do not relabel
house parts as tree/person or claim full-scene HTP detection unless the
annotations support it.

### Table 2. Final partition and class distribution

| Source / task | Partition | Images | Participants/groups | Class 0 | Class 1 | Other labels/classes | Small-object positive | Small-object negative |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Roboflow detection | Train | TBD | TBD | N/A | N/A | Per detector class: TBD | TBD | TBD |
| Roboflow detection | Validation | TBD | TBD | N/A | N/A | Per detector class: TBD | TBD | TBD |
| Roboflow detection | Test | TBD | TBD | N/A | N/A | Per detector class: TBD | TBD | TBD |
| Clinical/DD classification | Train | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Clinical/DD classification | Validation | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Clinical/DD classification | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Bilgi/Zhang/Pan | External test, each source separately | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Add separate rows for each dataset/source rather than pooling counts. Define
the positive class, class 0/1 mapping, participant count, and image count in
the caption. Report missing labels and unknown small-object status separately.

### Table 3. Annotation and image-quality audit

| Source / partition | Images reviewed | Missing/corrupt | Duplicate / near-duplicate | Images with boxes | Box count by class | Inter-annotator agreement, if measured | Resolution / quality summary |
|---|---:|---:|---:|---:|---|---:|---|
| Roboflow train | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Roboflow validation | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Roboflow test | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DD/classification cohort | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Report how boxes and labels were generated, whether annotations were reviewed
or adjudicated, and whether the detector training and evaluation images
overlap with images used to derive classification features.

## 3. Object-detection evaluation

### 3.1 Detector model comparison

Run only architectures that can be trained and evaluated under the same
Roboflow export, class map, resolution policy, and held-out split. YOLOv8 and
Faster R-CNN are the planned detector families; do not label one as the final
detector until the validation selection rule is applied.

### Table 4. Detector aggregate performance

| Detector / size | Pretraining | Test images | Parameters | FLOPs or MACs | Latency (ms/image) | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 | AP-small | AP-medium | AP-large |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8 variant (specify) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Faster R-CNN (specify backbone) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Table 5. Detector per-class performance

| Detector | Class (exact export name) | GT instances | Predicted instances | Precision | Recall | AP@0.50 | AP@0.75 | AP@0.50:0.95 | Median IoU of matched boxes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8 | House | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8 | Door | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8 | Window | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8 | Wall | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8 | Roof | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8 | Chimney | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8 | Sun | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Faster R-CNN | Repeat rows for each verified class in the selected version | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Detection reporting requirements:

- State the IoU matching rule, confidence threshold, class-wise versus
  class-agnostic matching, AP interpolation/implementation, and whether AP
  uses the COCO 0.50:0.95 convention.
- Report aggregate and per-class scores. Include support (ground-truth
  instances) and identify classes with zero or very low test support.
- Define AP-small/medium/large using the stated area convention and report
  input/image scale. Do not silently conflate detector-size bins with the
  project's separate small-object flag.
- Report the confidence threshold and operating-point precision/recall/F1
  selected on validation. Report the full PR curve/AP separately from that
  operating point.
- Measure latency with fixed hardware, batch size, warm-up, preprocessing,
  and synchronization. Report model-only and end-to-end latency separately;
  state whether detector postprocessing is included.

### Table 6. Detector-to-classifier feature quality

| Detector / operating point | Source / partition | Images | Detection coverage | Missed object rate | False-positive boxes/image | Area-ratio MAE | Small-object sensitivity | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Selected detector | Validation | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Selected detector | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Selected detector | External source, each separately | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Define these task-specific measures before use. For example, detection
coverage can be the proportion of images with at least one correctly matched
box for a target category; area-ratio MAE requires an explicitly defined
ground-truth ratio and reference annotation. If no valid ground truth exists,
mark a measure unavailable rather than inferring it from the detector's own
predictions.

## 4. Drawing-level classification baselines and proposed models

### 4.1 Baseline and model roster

The table below is the minimum comparison roster. Record the exact backbone
variant, pretraining source, input channels/resolution, feature inputs,
parameter count, and training recipe. The proposed transformer and
EfficientNet entries must name the specific model variant before running.

### Table 7. Primary classifier comparison

| ID | Model / condition | Image backbone | Detector-derived features | CLAHE | Small-object loss | CBAM | Mamba | Fusion | Status |
|---|---|---|---|---|---|---|---|---|---|
| B0 | Majority-class predictor (sanity baseline) | None | None | No | No | No | No | No | Planned |
| B1 | ResNet-50 image-only baseline | ResNet-50 | None | No | No | No | No | No | Planned |
| B2 | Vision Transformer image-only baseline | Specify ViT variant | None | No | No | No | No | No | Planned |
| B3 | EfficientNet image-only baseline | Specify EfficientNet variant | None | No | No | No | No | No | Planned |
| B4 | CBAM-EfficientNet | Same EfficientNet as B3 | None | No | No | Yes | No | No | Planned |
| B5 | SOANet image-only | Specify backbone | None | No | As implemented | As specified | No | No | Planned |
| B6 | Detector-feature-only classifier | Specify classifier | Yes | No | No | No | No | Feature branch | Planned |
| B7 | Mamba Sketch image-only | Specify stem and Mamba variant | None | No | No | No | Yes | No | Planned |
| B8 | SOANet + Mamba | Specify backbone and Mamba block | Optional; state explicitly | No | As specified | As specified | Yes | Late/feature | Planned |
| B9 | Full FASA-SOANet | Specify all modules | Yes, if final design includes it | As specified | As specified | As specified | As specified | Specify | Planned |

The majority predictor is a label-frequency sanity baseline, not a deployable
model. Include at least one strong standard image classifier (ResNet-50), the
requested ViT baseline, the CBAM-EfficientNet comparison, SOANet, Mamba
Sketch, and the full proposed FASA-SOANet. A plain EfficientNet row is
required to isolate the effect of adding CBAM. Add a plain SOANet row and
detector-feature-only row so improvements are not attributed to fusion without
controls.

Where feasible, add an image-only SOE-FBANet/FBANet baseline because these
models already exist in the codebase; report them as implementation
baselines, not as substitutes for SOANet unless the architectures are
identical under the frozen configuration.

### Table 8. Classification metrics for the primary comparison

| ID | Model | Split/source | N images / groups | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC / AP | MCC | Brier score | ECE | Sensitivity | Specificity | Threshold |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | Majority predictor | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | N/A | N/A | TBD | N/A | N/A | TBD | TBD | N/A |
| B1 | ResNet-50 | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B2 | ViT (specify) | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B3 | EfficientNet (specify) | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B4 | CBAM-EfficientNet | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B5 | SOANet | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B6 | Detector-feature-only | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B7 | Mamba Sketch | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B8 | SOANet + Mamba | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B9 | FASA-SOANet | Internal test | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Add a separate row for each external dataset and each model; do not aggregate
different sources into one external score unless the aggregation method and
source-level scores are also shown. Provide confidence intervals and class
support in the final paper table or supplement.

### 4.2 Required per-class classification results

### Table 9. Confusion and class-wise outcomes

| Model | Source/split | Positive class | TN | FP | FN | TP | Class 0 precision | Class 0 recall | Class 0 F1 | Positive precision | Sensitivity / positive recall | Positive F1 | N class 0 / class 1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Each primary model | Internal test | Define explicitly | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Define the positive class once and keep it consistent. If the label is a
psychological score rather than diagnosis, name the score and its threshold;
do not relabel it as depression diagnosis. Include the confusion-matrix
decision threshold and its validation-selection rule.

## 5. Ablation design

### 5.1 Ablation factors and controls

Define binary or categorical factor levels in the final resolved config.
“Small-object loss” must specify the target, equation, weight, and source of
supervision. “Fusion” must specify whether it combines image embeddings,
detector-region features, semantic metadata, or another representation.
CLAHE parameters and object-area thresholds must be fixed using training data
or a pre-registered rule.

### Table 10. Planned component ablations

| ID | Condition | CLAHE | Detector-derived feature branch | Small-object auxiliary loss | CBAM | Mamba | Fusion | Purpose |
|---|---|---|---|---|---|---|---|---|
| A0 | Matched image-only backbone control | Off | Off | Off | Off | Off | Off | Reference condition |
| A1 | A0 + CLAHE | On | Off | Off | Off | Off | Off | Preprocessing effect |
| A2 | A0 + detector features | Off | On | Off | Off | Off | Off | Feature contribution |
| A3 | A2 + small-object loss | Off | On | On | Off | Off | Off | Auxiliary supervision effect |
| A4 | A0 + CBAM | Off | Off | Off | On | Off | Off | Attention effect without detector/fusion |
| A5 | A0 + Mamba | Off | Off | Off | Off | On | Off | Global sequence modeling effect |
| A6 | A2 + CBAM | Off | On | Off | On | Off | Feature + CBAM interaction |
| A7 | A2 + Mamba | Off | On | Off | Off | On | Feature + Mamba interaction |
| A8 | SOANet full internal modules, no Mamba/fusion | As specified | As specified | As specified | As specified | Off | Off | SOANet contribution |
| A9 | SOANet + Mamba, no detector fusion | As specified | Off or specified | As specified | As specified | On | Image/feature-level | Mamba integration |
| A10 | Full FASA-SOANet | As specified | As specified | As specified | As specified | As specified | On | Proposed complete method |

Run A0–A5 at minimum to estimate the direct contribution of each core
component. A6–A7 identify important two-way interactions. A8–A10 test
integration and are not substitutes for the matched single-component
controls. Keep the image encoder, split, training budget, and model-selection
rule matched. If compute permits, pre-register a factorial experiment for
the central components; do not present a hand-picked subset as a complete
factorial design.

### Table 11. Ablation metrics

| ID | Condition | Seed(s) | Internal test macro-F1 | Δ macro-F1 vs A0 | Balanced accuracy | PR-AUC | ROC-AUC | Small-object subgroup macro-F1 | Large/no-object subgroup macro-F1 | Parameters | Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0–A10 | One row per condition | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Report subgroup counts and confidence intervals. “Small object” must be
defined independently of test outcomes (prefer annotation-based labels; if
detector-predicted, report detector errors and avoid treating predictions as
ground truth). Do not claim an ablation helps a subgroup based on a handful
of cases without showing the denominator and uncertainty.

### 5.2 Optional small-object size-stratified analysis

| Size stratum | Definition fixed before test | N drawings/groups | Model | Sensitivity | Specificity | Macro-F1 | PR-AUC |
|---|---|---:|---|---:|---:|---:|---:|
| No target object / not annotated | Define | TBD | TBD | TBD | TBD | TBD | TBD |
| Small | Define area ratio and rule | TBD | TBD | TBD | TBD | TBD | TBD |
| Medium | Define area ratio and rule | TBD | TBD | TBD | TBD | TBD | TBD |
| Large | Define area ratio and rule | TBD | TBD | TBD | TBD | TBD | TBD |

Only report this analysis if object annotations/ratios are sufficiently
reliable and available across the evaluated subset. Predefine the area
calculation (box area divided by image area, object-specific or union area)
and cutoff(s); do not choose cutoffs to maximize test performance.

## 6. Cross-dataset generalization

### Table 12. Train-on-source / test-on-source results

| Training source(s) | Validation source | External test source | Model | N participants/images | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC | PR-AUC | Sensitivity | Specificity | Threshold source |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TBD | TBD | Bilgi | Each selected model | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Validation only |
| TBD | TBD | Zhang | Each selected model | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Validation only |
| TBD | TBD | Pan | Each selected model | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Validation only |

Report each source separately, including the label mapping, demographic and
acquisition differences that are permitted to report, missing metadata, and
class support. If labels are not clinically comparable, report the task as
domain transfer/benchmarking rather than pooling metrics.

### Table 13. Cross-dataset transfer protocol

| Train source | Validation source | Test source | Fine-tuning allowed? | Any target labels used? | Preprocessing fit source | Test status |
|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | Train only / specified | External untouched |

Distinguish zero-shot transfer, unsupervised adaptation, and supervised
fine-tuning. If target labels influence any choice, the evaluation is not
zero-shot. Keep a target-domain test partition untouched after adaptation.

## 7. Required classification metrics and definitions

Report metrics at two operating modes where relevant:

1. **Threshold-independent ranking:** ROC-AUC and PR-AUC / average precision
   from continuous probabilities.
2. **Frozen operating point:** hard-label metrics at a threshold selected on
   validation, with the threshold-selection objective stated.

Also report argmax/0.5-threshold results when useful for a model's default
decision behavior, clearly separated from validation-thresholded results.
Never optimize a threshold on the test set.

| Metric | Required interpretation / convention |
|---|---|
| Accuracy | Correct predictions / all evaluated examples; include class counts |
| Balanced accuracy | Mean class recall; state class labels and averaging |
| Precision, recall, F1 per class | Report for every class; positive-class naming must be explicit |
| Macro precision/recall/F1 | Unweighted mean across classes; primary summary for class imbalance unless protocol states otherwise |
| Weighted-F1 | Class F1 weighted by support; not a replacement for macro-F1 |
| ROC-AUC | Continuous score ranking; undefined for a test set containing only one class |
| PR-AUC / average precision | State whether trapezoidal PR area or average precision is used; do not use these names interchangeably without definition |
| Sensitivity / specificity | Positive-class recall / negative-class recall at the frozen operating threshold |
| Confusion matrix | Provide raw TN/FP/FN/TP counts and the decision threshold |
| Matthews correlation coefficient (MCC) | Include as a balanced scalar summary; note undefined edge cases and implementation behavior |
| Brier score | Mean squared probability error; lower is better, for calibrated probabilities |
| Expected calibration error (ECE) | State bin count, binning strategy, and whether bins are equal-width or equal-mass |
| Confidence interval | Prefer participant/group bootstrap when repeated drawings per subject exist; state resampling unit and number of replicates |
| Threshold | Probability cutoff selected on validation; report selection objective and validation value |
| Runtime / efficiency | State hardware, batch size, warm-up, measured stages, and whether preprocessing/postprocessing is included |

For class imbalance, publish support (N per class and, where possible,
participants per class) beside every metric. For multi-class or ordinal tasks,
predefine macro/weighted averaging, one-vs-rest AUC, score cutoffs, and
confusion-matrix ordering. Do not report binary metrics if the target task is
not actually binary.

### Statistical comparison

- Report point estimates and confidence intervals; avoid significance claims
  from a single seed or a single split.
- For paired model comparison, resample at the participant/group level and
  use the same test examples. State the paired statistical method and
  multiplicity handling if formal hypothesis tests are used.
- Prefer multiple pre-registered seeds or grouped folds for internal
  stability; do not treat seeds over the same fixed test subjects as
  independent external validation.
- Include training variability and failures. Do not silently drop failed runs
  or select only favorable seeds.

## 8. Interpretability and localization validation

### Table 14. Explanation/localization audit

| Method | Model | Test examples reviewed | Expert/annotation target | Localization metric or protocol | Agreement / uncertainty | Failure cases retained |
|---|---|---:|---|---|---|---|
| Grad-CAM | TBD | TBD | Annotated region / none | IoU or pointing metric only if target exists | TBD | Yes |
| Attention map | TBD | TBD | Annotated region / none | Specify method | TBD | Yes |
| Occlusion/deletion test | TBD | TBD | Model prediction | Prediction change under fixed perturbation | TBD | Yes |
| Blinded expert review, if approved | TBD | TBD | Clinician-rated relevance | Predefined rubric and inter-rater agreement | TBD | Yes |

Use explanations as qualitative model-behavior analyses, not proof that the
model has learned clinically valid reasoning. Heatmaps do not prove the model
uses a diagnostic element. Quantitative localization claims require
independent region annotations and a pre-specified metric. Show correct and
incorrect predictions, and avoid selecting examples only because their maps
look persuasive.

## 9. Figures and visual assets

| Figure | Required content | Source and safeguards |
|---|---|---|
| 1. End-to-end study flow | Dataset preparation; detector train/validation/test; frozen detector; feature extraction; classifier comparisons; internal/external evaluation; interpretation | Draw for this project. Mark planned versus implemented stages. |
| 2. Dataset flow and split diagram | Source counts, exclusions, duplicate checks, participant grouping, final partitions | Recompute from frozen manifests. Include per-source and per-class counts. |
| 3. Example HTP and detector annotations | House/tree/person and actual annotated detail classes, with bounding boxes | Use approved, de-identified examples. Do not publish unlicensed Roboflow or clinical images. |
| 4. Dataset balance | Counts by source, partition, label, and participant where known | Calculate from the final dataset manifest; reveal source/class confounding. |
| 5. Detector training and validation curves | Losses and per-class validation metrics across epochs | Test data must not be used for epoch selection. |
| 6. Detector PR and size performance | Per-class PR curves/AP; AP by object-size bin; selected operating point | Label IoU and confidence rules; include support. |
| 7. Detector qualitative results | True positives, missed objects, false positives, small/faint cases | Predefine sampling or show representative stratified cases; include failures. |
| 8. Object-area / small-object analysis | Defined area ratios and class/source-stratified distributions | Use ground-truth annotations where possible; show denominators and cutoffs. |
| 9. Classifier learning curves | Train/validation loss and selected metrics for each model | Show multiple seeds/folds or uncertainty when available. Do not plot test metrics epoch by epoch for selection. |
| 10. Main model comparison | Macro-F1, PR-AUC, balanced accuracy and uncertainty across baseline/proposed models | Matched protocol; show test support; distinguish internal and external sources. |
| 11. Ablation effects | Component effects and pre-specified interactions | Match Table 10; do not label incomplete subset factorial. |
| 12. Confusion and error analysis | Confusion matrices, class-wise sensitivity/specificity, examples by error type | Threshold selected on validation; report raw counts and class definitions. |
| 13. ROC/PR and calibration | ROC and PR curves, calibration plot, Brier/ECE | Use held-out test probabilities once; report calibration procedure and bins. |
| 14. Interpretability | Grad-CAM/attention overlays on correct and incorrect cases, with detector boxes | Approved examples only; maps are not clinical validation. |
| 15. Cross-dataset generalization | Per-source metric comparison and uncertainty | Each target source must remain outside development; show domain/source labels. |

Every plot must include a descriptive caption, data partition, sample/group
support, metric definition, uncertainty method, and any threshold or IoU
setting. Use consistent colors and positive-class conventions across all
figures.

### Prior-study figure references (layout ideas only)

Earlier figure filenames may guide organization, but the files were deleted
and their numbers, images, and visual conclusions must not be reused:

| Earlier figure name | Possible new-project analogue |
|---|---|
| `htp_dataset_class_balance.svg` | Figure 4, recomputed from the new data |
| `htp_dataset_sample_montage.svg` | Figure 3, using approved new examples |
| `training_validation_curves.svg` | Figure 5 and Figure 9 |
| `confusion_matrices.svg` | Figure 12 |
| `class_recall_comparison.svg` | Figure 10 or 12 |
| `roc_pr_curves_matrix.svg` | Figure 13, with the new protocol |
| `calibration_matrix.svg` | Figure 13, after calibration protocol is fixed |
| `factorial_macro_f1_heatmap.svg` | Figure 11 only if a complete factorial is actually run |
| `threshold_sensitivity_validation.svg` | Validation-only threshold analysis; do not use a test-tuned threshold |

## 10. Efficiency, reproducibility, and model complexity

### Table 15. Compute and reproducibility summary

| Model / detector | Parameters | Trainable parameters | FLOPs/MACs | Peak memory | Training time | Inference latency | Hardware | Seed(s) | Code/config ID |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| Each primary model | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Use identical measurement conditions where comparisons are made. State input
resolution, precision (FP32/AMP), batch size, warm-up iterations, number of
timed iterations, and whether data loading, detector postprocessing, and
feature extraction are included.

For every completed run, privately retain:

- Immutable run ID, code commit, resolved config, seed, and environment
  version.
- Dataset version/fingerprint and split-manifest fingerprint; do not publish
  fingerprints that expose sensitive file naming.
- Checkpoint-selection metric, best epoch, validation threshold, and exact
  evaluation command.
- Aggregate test metrics with class and participant support, uncertainty,
  and the distinction between internal and external evaluation.
- Detector version/checkpoint and the generated feature-extraction settings
  for any detector-assisted classifier.

Do not commit datasets, source annotations, participant mappings,
split-manifest contents, checkpoints, individual predictions, or private logs.

## 11. Planned reporting summary

The final report should include, at minimum:

1. Dataset provenance, permissions, quality audit, label definitions, source
   counts, participant/group counts, and frozen split protocol.
2. Per-class detector performance, mAP@0.50 and mAP@0.50:0.95, size-specific
   AP, operating-point precision/recall, and detector failure cases.
3. A complete classifier baseline roster: majority predictor, ResNet-50, ViT,
   plain EfficientNet, CBAM-EfficientNet, SOANet, detector-feature-only,
   Mamba Sketch, SOANet+Mamba, and full FASA-SOANet; include existing
   SOE-FBANet/FBANet controls when practical.
4. Predefined component ablations for CLAHE, detector features,
   small-object supervision, CBAM, Mamba, and fusion, with matched controls
   and any interaction design identified honestly.
5. Classification accuracy, balanced accuracy, per-class and macro metrics,
   ROC-AUC, PR-AUC/AP, confusion counts, sensitivity/specificity, threshold,
   calibration, uncertainty, support, and efficiency.
6. Source-separated cross-dataset results with target-set isolation and
   label-harmonization details.
7. Interpretability examples and failures, with quantitative localization
   only when independent region annotations support it.
8. Limitations covering data size, source/participant confounding, label
   comparability, uncertainty, generalization, and the non-diagnostic role of
   the system.

## 12. Prior-study file map

Earlier artifacts were under `docs/exp1/results/` and `analysis/`. Those
directories and generated files have been removed from the working tree.
Their former names are mentioned above only as layout references; there is no
retained blueprint or result asset in the new project. All project-2
measurements, figures, and conclusions must be regenerated from its own
authorized data and runs.
