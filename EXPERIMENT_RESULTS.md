# Experiment results

## Current best recorded run

**SOE-FBANet full CE** (`soefbanet_full_ce`) is the best recorded single-model
result in the local experiment histories: test macro-F1 **0.9186**, with
threshold **0.66** selected on validation. The saved history records validation
macro-F1 0.9287, test depressed-class F1 0.8713, test ROC-AUC 0.9503, and best
checkpoint epoch 14. It used seed 42, ImageNet-pretrained ResNet-50, DD image
normalization, the full SOFE/SOSM/WFF model, and horizontal-flip TTA, with
cross-entropy and no class weighting or sampler.

This supersedes D27 (test macro-F1 0.884) and Vabl_focal (0.9047). The earlier
D27+D28+D29 ensemble scored 0.9019, below this single-model run. These are
recorded local results, not independently reproduced claims: the checkpoint
and clinical dataset are not included here. Although each run's checkpoint and
threshold were selected using validation metrics, the same test split was
compared across many trials; choosing the highest-scoring trial introduces
experiment-level selection bias. Treat 0.9186 as the best observed exploratory
score, not an unbiased estimate of generalization. Re-run the config and
evaluate on a fresh, untouched cohort before relying on or publishing it.
Before publishing this summary, confirm permission to disclose aggregate
clinical-study results.

Recommended reproducibility commands, from the repository root:

```bash
python scripts/train.py --config src/configs/soefbanet_resnet50_ddnorm.yaml
python scripts/eval.py \
  --config src/configs/soefbanet_resnet50_ddnorm.yaml \
  --checkpoint outputs/soefbanet_resnet50_ddnorm/best.pt \
  --split test
```

The clinical dataset and frozen split file must already be present locally in
the paths specified by `src/configs/dd_train.yaml`. Neither is part of this
repository. The recommended config mirrors the recorded `soefbanet_full_ce`
settings; its output directory differs from the original run.

## Phase 1/2 architecture trials

These runs use the same seed-42 frozen DD split and ResNet-50/DD-normalized
protocol. Thresholds were selected on validation. The full cross-entropy run
is the highest recorded model result in this group.

| Run | Best epoch | Val macro-F1 | Test macro-F1 | Test De F1 | Test ROC-AUC |
|-----|------------|--------------|---------------|------------|-------------|
| ResNet-50 backbone baseline | 8 | 0.8921 | 0.8638 | 0.7931 | 0.9147 |
| Attention-only control | 7 | 0.9327 | 0.8874 | 0.8257 | 0.9176 |
| SOFE only | 9 | 0.9228 | 0.8859 | 0.8224 | 0.9405 |
| SOSM only | 11 | 0.9240 | 0.8822 | 0.8182 | 0.9516 |
| WFF only | 62 | 0.9028 | 0.8786 | 0.8142 | 0.9392 |
| SOFE + SOSM | 1 | 0.9144 | 0.8959 | 0.8333 | 0.9234 |
| SOFE + WFF | 35 | 0.9434 | 0.8950 | 0.8350 | 0.9531 |
| SOSM + WFF | 33 | 0.9204 | 0.9075 | 0.8511 | 0.9111 |
| **SOFE + SOSM + WFF, CE** | **14** | **0.9287** | **0.9186** | **0.8713** | **0.9503** |
| SOFE + SOSM + WFF, focal gamma 0.5 | 13 | 0.9042 | 0.9117 | 0.8600 | 0.9510 |

## Later DD-normalized follow-up trials

These additional trials compare individual training changes against the D27
ResNet-50/DD-normalized recipe. Metrics are from local histories; thresholds
were selected on validation. The higher architecture-matrix results above
supersede D27 and Vabl_focal.

| Run | Test macro-F1 | Test De F1 | Test ROC-AUC |
|-----|---------------|------------|-------------|
| Vabl_focal (gamma 0.5) | 0.9047 | 0.8485 | 0.9062 |
| Vabl_ls (label smoothing 0.10) | 0.8549 | 0.7736 | 0.9134 |
| Vabl_amp (AMP + grad clip) | 0.8391 | 0.7544 | 0.9129 |
| Vabl_wd (weight decay 0.003) | 0.8391 | 0.7544 | 0.8981 |
| Vabl_cw (rebalanced class weights) | 0.8336 | 0.7500 | 0.9176 |
| Vabl_cnn_seed7 (CNN-lite, seed 7) | 0.7577 | 0.6462 | 0.8542 |

## Historical clinical trials

Test macro-F1 is shown only when recorded. `—` means no comparable clinical
test result was recorded. Older trials use different initializations or data
normalization and should not be compared as if they were one controlled sweep.

| Run | Test macro-F1 | Test De F1 | Test ROC-AUC | Outcome / note |
|-----|---------------|------------|-------------|----------------|
| FULL, 12-epoch scratch | — | — | 0.663 | Training collapsed late; not a stable best-checkpoint comparison |
| LONG, scratch long run | 0.742 | 0.610 | 0.807 | Early foundation |
| FT, warm-start LONG | 0.744 | — | 0.811 | AUC-selected; not the target metric |
| F1PUSH_v1 | — | — | — | Stopped early; ranking collapsed |
| F1PUSH_v2 | 0.760 | 0.620 | 0.790 | Improved over LONG; later surpassed |
| D1 | 0.773 | 0.655 | 0.829 | Previous early-wave single-model best |
| D2 | 0.756 | 0.614 | 0.796 | De-focused loss/augmentation did not generalize |
| D3 | 0.763 | 0.637 | 0.823 | Focal on older protocol; not comparable to Vabl_focal |
| D4 QuickDraw pretraining | — | — | — | QuickDraw score is not a clinical metric |
| D4b QuickDraw-to-DD | 0.706 | 0.545 | 0.746 | Worse than D1 |
| D5 temperature scaling | 0.773 | 0.655 | 0.829 | No improvement over D1 |
| D6 D1+LONG ensemble | 0.743 | 0.594 | 0.820 | Diluted D1 |
| D6v2 three-model ensemble | 0.760 | 0.639 | 0.813 | Below D1 |
| D7 mild De boost | 0.762 | 0.638 | 0.826 | Below D1 |
| D8 label smoothing 0.10 | 0.763 | 0.637 | 0.828 | Below D1 |
| D9 class-weight retune | 0.770 | 0.649 | 0.828 | No material improvement |
| D10 stronger weight decay | 0.770 | 0.649 | 0.826 | No material improvement |
| D11 lower learning rate | 0.773 | 0.655 | 0.827 | Tied D1 |
| D12 soft De sampler | — | — | — | Superseded by D17 |
| D13 progressive training from D1 | 0.759 | 0.632 | 0.824 | Worse generalization |
| D14 progressive training from LONG | 0.770 | 0.649 | 0.825 | No material improvement |
| D15 EMA head | 0.773 | 0.655 | 0.829 | Tied D1 |
| D16 soft LDAM | 0.763 | 0.637 | 0.827 | Below D1 |
| D17 soft De sampler | 0.742 | 0.610 | 0.825 | Worse than D1 |
| D18 PR-AUC checkpoint selection | 0.770 | 0.649 | 0.826 | No material improvement |
| D19 close/soft augmentation | 0.770 | 0.649 | 0.826 | No material improvement |
| D20 AMP + gradient clipping | 0.770 | 0.649 | 0.826 | Older freeze-fine-tune protocol |
| D21 DD-normalized CNN-lite | 0.778 | 0.667 | 0.842 | Foundation for capacity sweep |
| D22 freeze fine-tune from D21 | 0.763 | — | — | Worse than D21 |
| D23 mild unfreeze from D21 | 0.778 | 0.667 | 0.862 | Tied D21 on F1 |
| D24 long run with EMA | 0.775 | 0.655 | 0.839 | Below D21 |
| D25 early-close augmentation | 0.754 | — | — | Worse than D21 |
| D26 ImageNet ResNet-18 + DD norm | 0.874 | 0.804 | 0.925 | Prior single-model best |
| D27 ImageNet ResNet-50 + DD norm | 0.884 | 0.819 | 0.937 | Superseded by Vabl_focal |
| D28 ResNet-50 with hvflip TTA | 0.868 | 0.792 | 0.907 | Worse than D27 |
| D29 ResNet-50, seed 7 | 0.862 | 0.785 | 0.951 | Worse than D27 |
| D30 ResNet-50, gentle LR, seed 13 | 0.867 | 0.796 | 0.912 | Worse than D27 |
| Ensemble D27+D29 | 0.8965 | 0.838 | 0.951 | Missed 0.90 target |
| Ensemble D26+D27 | 0.886 | 0.822 | 0.933 | No material improvement |
| Ensemble D27+D28+D29 | 0.9019 | 0.846 | 0.943 | Earlier disclosed ensemble pass; lower than Vabl_focal |

## Notes

- All listed F1 scores are macro-F1 at a threshold selected on validation where
  the source history records that protocol. AUC uses continuous probabilities.
- `D4` is a public-sketch pretraining experiment, not a clinical evaluation.
- The Vabl and architecture-matrix results are separate comparisons; their
  settings differ. The same test cohort was used repeatedly, so the top score
  needs confirmation on independent data.
- Archived YAMLs are in `src/configs/archive/`; the trial configurations are
  retained for reference, not as the recommended main flow.
- The source experiment tracker and logs were local. This summary intentionally
  contains aggregate metrics only; no images, labels, manifests, checkpoints,
  or per-person records are included.
