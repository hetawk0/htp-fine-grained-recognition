"""Classification metrics: accuracy, F1, AUC, PR-AUC (binary); threshold sweep."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def positive_class_probs(
    logits: torch.Tensor,
    num_classes: int = 2,
) -> Optional[np.ndarray]:
    """Return P(class=1) for binary classifiers; None if multi-class."""
    if num_classes != 2:
        return None
    # Softmax/sigmoid in fp16 under AMP can overflow — cast to float32 first.
    logits = logits.float()
    if logits.ndim == 2:
        return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return torch.sigmoid(logits).view(-1).detach().cpu().numpy()


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Hard Acc/F1 use argmax (or 0.5) unless ``threshold`` is set on binary P(pos)."""
    y_true = labels.detach().cpu().numpy().astype(int)
    probs = positive_class_probs(logits, num_classes=num_classes)

    if threshold is not None and probs is not None:
        y_pred = (probs >= float(threshold)).astype(int)
    elif logits.ndim == 2:
        y_pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
    else:
        y_pred = (torch.sigmoid(logits) > 0.5).long(
        ).view(-1).detach().cpu().numpy().astype(int)

    out = _classification_metrics(y_true, y_pred, num_classes)
    out["auc"] = float("nan")
    out["pr_auc"] = float("nan")
    if probs is not None and num_classes == 2 and len(np.unique(y_true)) > 1:
        out["auc"] = _roc_auc(y_true, probs)
        out["pr_auc"] = _pr_auc(y_true, probs)
    if threshold is not None:
        out["threshold"] = float(threshold)
    return out


def metrics_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    num_classes: int = 2,
) -> Dict[str, float]:
    y_true = np.asarray(labels).astype(int)
    y_pred = (np.asarray(probs) >= float(threshold)).astype(int)
    out = _classification_metrics(y_true, y_pred, num_classes)
    out["threshold"] = float(threshold)
    return out


def sweep_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2,
    criterion: str = "f1",
    t_min: float = 0.01,
    t_max: float = 0.99,
    n_steps: int = 99,
    de_recall_floor: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Pick τ on val maximizing macro-F1, De-F1, or Youden. Returns (τ, metrics@τ).

    ``f1_de_floor`` / ``f1_constrained``: maximize macro-F1 among thresholds with
    De recall ≥ ``de_recall_floor`` (avoids pure f1_de overfit; D2 lesson).
    """
    y_true = np.asarray(labels).astype(int)
    scores = np.asarray(probs, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        tau = 0.5
        return tau, metrics_at_threshold(scores, y_true, tau, num_classes=num_classes)

    thresholds = np.linspace(t_min, t_max, n_steps)
    best_tau = 0.5
    best_score = -1.0
    best_m: Dict[str, float] = {}
    criterion = (criterion or "f1").lower()
    floor = float(de_recall_floor or 0.0)
    if criterion in ("f1_de_floor", "f1_constrained", "f1_de_recall"):
        floor = max(floor, 0.70)

    for tau in thresholds:
        y_pred = (scores >= tau).astype(int)
        classified = _classification_metrics(y_true, y_pred, num_classes)
        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fn = float(np.sum((y_pred == 0) & (y_true == 1)))
        de_recall = tp / (tp + fn) if (tp + fn) else 0.0
        if criterion in ("youden", "youden_j", "j"):
            tn = float(np.sum((y_pred == 0) & (y_true == 0)))
            fp = float(np.sum((y_pred == 1) & (y_true == 0)))
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            score = float(tpr - fpr)
        elif criterion in ("f1_de", "f1_pos", "pos_f1", "de_f1"):
            score = float(classified.get("f1_c1", 0.0))
        elif criterion in ("f1_de_floor", "f1_constrained", "f1_de_recall"):
            if de_recall + 1e-12 < floor:
                continue
            score = float(classified["f1"])
        else:
            score = float(classified["f1"])
        if score > best_score or (score == best_score and abs(tau - 0.5) < abs(best_tau - 0.5)):
            best_score = score
            best_tau = float(tau)
            best_m = {
                **classified,
                "threshold": best_tau,
                "sweep_score": best_score,
                "sweep_criterion": criterion,
                "de_recall": float(de_recall),
            }
            for c, v in enumerate(_per_class_f1(y_true, y_pred, num_classes)):
                best_m[f"f1_c{c}"] = v
    # Fallback if floor filtered everything.
    if not best_m:
        return sweep_threshold(
            probs,
            labels,
            num_classes=num_classes,
            criterion="f1",
            t_min=t_min,
            t_max=t_max,
            n_steps=n_steps,
            de_recall_floor=0.0,
        )
    return best_tau, best_m


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> list:
    scores = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(float(2 * precision * recall / (precision + recall))
                      if precision + recall else 0.0)
    return scores


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        if 0 <= truth < num_classes and 0 <= prediction < num_classes:
            confusion[int(truth), int(prediction)] += 1

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    out: Dict[str, float] = {"accuracy": accuracy}
    for c in range(num_classes):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum() - tp)
        fn = int(confusion[c, :].sum() - tp)
        support = int(confusion[c, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / \
            (precision + recall) if precision + recall else 0.0
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
        out[f"precision_c{c}"] = float(precision)
        out[f"recall_c{c}"] = float(recall)
        out[f"f1_c{c}"] = float(f1)
        out[f"support_c{c}"] = float(support)

    out["f1"] = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    out["macro_precision"] = float(
        np.mean(per_class_precision)) if per_class_precision else 0.0
    out["macro_recall"] = float(
        np.mean(per_class_recall)) if per_class_recall else 0.0
    out["macro_f1"] = out["f1"]
    out["balanced_accuracy"] = out["macro_recall"]
    support_total = sum(int(confusion[c, :].sum()) for c in range(num_classes))
    out["weighted_f1"] = (
        float(sum(per_class_f1[c] * int(confusion[c, :].sum())
              for c in range(num_classes)) / support_total)
        if support_total else 0.0
    )
    if num_classes == 2:
        tn, fp = int(confusion[0, 0]), int(confusion[0, 1])
        fn, tp = int(confusion[1, 0]), int(confusion[1, 1])
        out.update({
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
            "sensitivity": float(per_class_recall[1]),
            "specificity": float(per_class_recall[0]),
            "precision_de": float(per_class_precision[1]),
            "recall_de": float(per_class_recall[1]),
            "true_negative_rate": float(per_class_recall[0]),
            "false_positive_rate": float(1.0 - per_class_recall[0]),
        })
    for true_class in range(num_classes):
        for predicted_class in range(num_classes):
            out[f"confusion_{true_class}_{predicted_class}"] = float(
                confusion[true_class, predicted_class]
            )
    return out


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    scores = _per_class_f1(y_true, y_pred, num_classes)
    return float(np.mean(scores)) if scores else 0.0


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    correct = 0.0
    for p in pos:
        correct += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(correct / (len(pos) * len(neg)))


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under precision–recall curve)."""
    pos = int(np.sum(y_true == 1))
    if pos == 0 or pos == len(y_true):
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for i, y in enumerate(y_sorted, 1):
        if y == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return float(ap)
