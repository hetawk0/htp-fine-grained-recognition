#!/usr/bin/env python3
"""Train Viola HTP models from a YAML config."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import torch.nn as nn
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from loader.config import load_config
from loader.dataset_loader import build_dataloaders, resolve_data_name
from loader.transforms import cfg_transforms
from model.losses import DepressionLoss
from model.model_loader import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint, torch_load
from utils.ema import ModelEMA
from utils.metrics import compute_metrics, positive_class_probs, sweep_threshold
from utils.seed import set_seed


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _is_better(score: float, best: float) -> bool:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return False
    return score > best or (best < 0 and score >= 0)


def _tta_logits(model, images: torch.Tensor, tta_mode: str) -> torch.Tensor:
    """Average softmax over geometric views; return log-probs as logits.

    Softmax runs in float32 so AMP Half logits do not overflow.
    """
    out = model(images)
    logits = out["logits"] if isinstance(out, dict) else out
    probs = torch.softmax(logits.float(), dim=1)
    n_views = 1
    mode = (tta_mode or "none").lower()
    if mode in ("hflip", "hvflip", "true", "1"):
        out_f = model(torch.flip(images, dims=[-1]))
        logits_f = out_f["logits"] if isinstance(out_f, dict) else out_f
        probs = probs + torch.softmax(logits_f.float(), dim=1)
        n_views += 1
    if mode == "hvflip":
        # Vertical + both flips — sketch symmetry; avoid multi-crop (can hurt medical TTA).
        out_v = model(torch.flip(images, dims=[-2]))
        logits_v = out_v["logits"] if isinstance(out_v, dict) else out_v
        probs = probs + torch.softmax(logits_v.float(), dim=1)
        n_views += 1
        out_hv = model(torch.flip(images, dims=[-1, -2]))
        logits_hv = out_hv["logits"] if isinstance(out_hv, dict) else out_hv
        probs = probs + torch.softmax(logits_hv.float(), dim=1)
        n_views += 1
    if n_views > 1:
        probs = probs / float(n_views)
        return torch.log(probs.clamp_min(1e-8))
    return logits


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    return_logits: bool = False,
    tta_hflip: bool = False,
    tta_mode: Optional[str] = None,
):
    model.eval()
    totals = {"loss": 0.0, "n": 0}
    all_logits, all_labels = [], []
    if tta_mode is None:
        tta_mode = "hflip" if tta_hflip else "none"
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        so_mask = batch["so_mask"].to(device)
        out = model(images)
        logits = out["logits"] if isinstance(out, dict) else out
        losses = criterion(logits, labels, so_mask)
        if tta_mode and tta_mode.lower() not in ("none", "false", "0"):
            logits = _tta_logits(model, images, tta_mode)
        bs = images.size(0)
        totals["loss"] += float(losses["loss"].detach()) * bs
        totals["n"] += bs
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits_cat, labels_cat, num_classes=num_classes)
    metrics["loss"] = totals["loss"] / max(totals["n"], 1)
    if return_logits:
        return metrics, logits_cat, labels_cat
    return metrics


def _apply_lr_warmup(optimizer: torch.optim.Optimizer, step_idx: int, warmup_steps: int) -> None:
    """Linear LR rise for the first ``warmup_steps`` optimizer steps (Ultralytics-style).

    Targets each group's ``scheduled_lr`` (the post-scheduler LR), not a hardcoded constant.
    No-op when ``warmup_steps`` is 0. D1 fine-tunes leave this off: the champion is epoch 1.
    """
    if warmup_steps <= 0 or step_idx > warmup_steps:
        return
    scale = step_idx / float(warmup_steps)
    for group in optimizer.param_groups:
        target = float(group.get("scheduled_lr", group["lr"]))
        group["lr"] = target * scale


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    log_every: int,
    num_classes: int = 2,
    ema: Optional[ModelEMA] = None,
    global_step: int = 0,
    warmup_steps: int = 0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip_norm: float = 0.0,
):
    model.train()
    running = 0.0
    n = 0
    all_logits, all_labels = [], []
    use_amp = scaler is not None
    pbar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(pbar, 1):
        global_step += 1
        _apply_lr_warmup(optimizer, global_step, warmup_steps)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        so_mask = batch["so_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(images)
            logits = out["logits"] if isinstance(out, dict) else out
            losses = criterion(logits, labels, so_mask)
            loss = losses["loss"]
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        if ema is not None:
            ema.update(model)
        bs = images.size(0)
        running += float(loss.detach()) * bs
        n += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        if step % log_every == 0:
            pbar.set_postfix(loss=running / max(n, 1))
    train_m = compute_metrics(
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        num_classes=num_classes,
    )
    train_m["loss"] = running / max(n, 1)
    train_m["global_step"] = global_step
    return train_m


def build_scheduler(optimizer, cfg_train: dict, epochs: int):
    name = str(cfg_train.get("lr_schedule") or "").strip().lower()
    if not name or name in ("none", "null", "off"):
        return None
    if name in ("cosine", "cosineannealing", "cos"):
        t_max = int(cfg_train.get("lr_t_max", epochs))
        eta_min = float(cfg_train.get("lr_min", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(t_max, 1), eta_min=eta_min
        )
    if name in ("step", "steplr"):
        step_size = int(cfg_train.get("lr_step_size", max(epochs // 3, 1)))
        gamma = float(cfg_train.get("lr_gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name in ("multistep", "multi_step", "multisteplr"):
        milestones = cfg_train.get("lr_milestones")
        if milestones is None:
            milestones = [max(epochs // 2, 1), max(3 * epochs // 4, 1)]
        milestones = [int(m) for m in milestones]
        gamma = float(cfg_train.get("lr_gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    raise ValueError(
        f"Unknown train.lr_schedule={name!r} (use cosine|step|multistep|none)"
    )


def _resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p)


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> int:
    """Freeze/unfreeze ``model.backbone`` if present. Returns #params touched."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return 0
    n = 0
    for p in backbone.parameters():
        p.requires_grad = trainable
        n += p.numel()
    return n


def _norm_types() -> tuple:
    return tuple(v for k, v in nn.__dict__.items() if isinstance(v, type) and "Norm" in k)


def _is_no_decay(module: nn.Module, fullname: str, norm_types: tuple) -> bool:
    """Ultralytics rule: no weight decay on bias, Norm affine, or logit scales."""
    return "bias" in fullname or "logit_scale" in fullname or isinstance(module, norm_types)


def build_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    """Adam with backbone LR multiplier and no weight decay on Norm/bias.

    Decay is applied only to convolutional and linear weights. BatchNorm scale
    and bias are excluded, matching ``BaseTrainer.build_optimizer``. Head and
    backbone stay separate groups when ``backbone_lr_mult != 1``.
    """
    base_lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-3))
    bb_mult = float(train_cfg.get("backbone_lr_mult", 1.0))
    backbone = getattr(model, "backbone", None)
    split_backbone = backbone is not None and abs(bb_mult - 1.0) >= 1e-12
    bb_ids = {id(p) for p in backbone.parameters()
              } if backbone is not None else set()
    norm_types = _norm_types()
    buckets: dict[tuple[str, bool], list] = {
        ("head", False): [],
        ("head", True): [],
        ("backbone", False): [],
        ("backbone", True): [],
    }
    seen: set[int] = set()
    for mod_name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            fullname = f"{mod_name}.{pname}" if mod_name else pname
            where = "backbone" if id(param) in bb_ids else "head"
            buckets[(where, _is_no_decay(module, fullname, norm_types))].append(
                param)

    def _add(params: list, lr: float, decay: float, name: str, groups: list) -> None:
        if not params:
            return
        groups.append(
            {
                "params": params,
                "lr": lr,
                "weight_decay": decay,
                "name": name,
                "scheduled_lr": lr,
            }
        )

    groups: list[dict] = []
    if split_backbone:
        _add(buckets[("head", False)], base_lr,
             weight_decay, "head_decay", groups)
        _add(buckets[("head", True)], base_lr, 0.0, "head_nodecay", groups)
        _add(buckets[("backbone", False)], base_lr *
             bb_mult, weight_decay, "bb_decay", groups)
        _add(buckets[("backbone", True)], base_lr *
             bb_mult, 0.0, "bb_nodecay", groups)
    else:
        decay_params = buckets[("head", False)] + buckets[("backbone", False)]
        nodecay_params = buckets[("head", True)] + buckets[("backbone", True)]
        _add(decay_params, base_lr, weight_decay, "decay", groups)
        _add(nodecay_params, base_lr, 0.0, "nodecay", groups)
    if not groups:
        raise RuntimeError("No trainable parameters for optimizer")
    return torch.optim.Adam(groups)


def _sync_scheduled_lrs(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group["scheduled_lr"] = float(group["lr"])


def _group_lr(optimizer: torch.optim.Optimizer, name_prefix: str) -> Optional[float]:
    for group in optimizer.param_groups:
        if str(group.get("name", "")).startswith(name_prefix):
            return float(group["lr"])
    return None


def rebuild_optimizer_and_scheduler(model, train_cfg, epochs, device_note: str = ""):
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, epochs)
    _sync_scheduled_lrs(optimizer)
    summary = ", ".join(
        f"{g.get('name')} lr={g['lr']:.2e} wd={g['weight_decay']:.1e} n={sum(p.numel() for p in g['params'])}"
        for g in optimizer.param_groups
    )
    print(
        f"optimizer rebuilt{device_note}: {summary}  "
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Train Viola HTP model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "src/configs/default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to warm-start (overrides train.resume)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.device is not None:
        cfg.setdefault("train", {})["device"] = args.device
    if args.resume is not None:
        cfg.setdefault("train", {})["resume"] = args.resume

    reproducibility_cfg = cfg.get("reproducibility", {})
    deterministic = bool(reproducibility_cfg.get("deterministic", True))
    deterministic_warn_only = bool(
        reproducibility_cfg.get("warn_only", True)
    )
    set_seed(
        int(cfg.get("seed", 42)),
        deterministic=deterministic,
        warn_only=deterministic_warn_only,
    )
    device = resolve_device(str(cfg.get("train", {}).get("device", "auto")))
    out_dir = Path(cfg.get("train", {}).get("out_dir", "outputs/run"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    train_cfg = cfg.get("train", {})
    resume_raw = train_cfg.get("resume")
    if resume_raw:
        resume_path = _resolve_path(str(resume_raw), ROOT)
        # Weights-only warm start; fresh optimizer at fine-tune LR.
        # resume_strict=false: keep backbone when head dims differ (QD→DD).
        resume_strict = bool(train_cfg.get("resume_strict", True))
        ckpt = load_checkpoint(
            resume_path, model, map_location=device, strict=resume_strict
        )
        print(
            f"resume={resume_path}  epoch={ckpt.get('epoch')}  "
            f"strict={resume_strict}  metrics={ckpt.get('metrics')}"
        )
        if ckpt.get("_partial_load"):
            pl = ckpt["_partial_load"]
            print(
                f"  partial load: {pl['loaded']} tensors; "
                f"reset heads={pl['skipped_missing']}"
            )

    freeze_epochs = int(train_cfg.get("freeze_epochs", 0))
    if freeze_epochs > 0:
        n_frozen = set_backbone_trainable(model, False)
        print(
            f"freeze backbone for first {freeze_epochs} epochs ({n_frozen} params)")

    cw = train_cfg.get("class_weights")
    focal_gamma = float(train_cfg.get("focal_gamma", 0.0))
    ldam_max_m = float(train_cfg.get("ldam_max_m", 0.0))
    ldam_s = float(train_cfg.get("ldam_s", 1.0))
    cls_num_list = train_cfg.get("cls_num_list")
    criterion = DepressionLoss(
        so_penalty=float(train_cfg.get("so_penalty", 2.0)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        class_weight=cw,
        focal_gamma=focal_gamma,
        dice_weight=float(train_cfg.get("dice_weight", 0.0)),
        ldam_max_m=ldam_max_m,
        ldam_s=ldam_s,
        cls_num_list=cls_num_list,
    ).to(device)
    optimizer, scheduler = rebuild_optimizer_and_scheduler(
        model, train_cfg, int(train_cfg.get("epochs", 5))
    )

    ema_decay = float(train_cfg.get("ema_decay", 0.0) or 0.0)
    # Ultralytics ramps decay with tau (default 2000). Omit / 0 → fixed decay (D15).
    ema_tau_raw = train_cfg.get("ema_tau", None)
    ema_tau = float(ema_tau_raw) if ema_tau_raw not in (
        None, "", 0, 0.0) else None
    ema = (
        ModelEMA(model, decay=ema_decay, tau=ema_tau)
        if ema_decay > 0.0
        else None
    )
    ema_eval = bool(train_cfg.get("ema_eval", True)) and ema is not None

    num_classes = int(cfg.get("model", {}).get("num_classes", 2))
    epochs = int(train_cfg.get("epochs", 5))
    log_every = int(train_cfg.get("log_every", 10))
    # f1_tau: val macro-F1 after τ sweep; f1_de_tau: De-class F1@τ (minority bottleneck)
    monitor = str(train_cfg.get("monitor", "auc")).lower()
    _monitor_aliases = {
        "f1_de": "f1_de_tau",
        "f1_c1_tau": "f1_de_tau",
        "de_f1_tau": "f1_de_tau",
    }
    monitor = _monitor_aliases.get(monitor, monitor)
    if monitor not in ("auc", "f1", "pr_auc", "f1_tau", "f1_de_tau"):
        raise ValueError(
            f"train.monitor must be auc|f1|pr_auc|f1_tau|f1_de_tau, got {monitor!r}"
        )
    patience = int(train_cfg.get("patience", 0))
    threshold_criterion = str(train_cfg.get(
        "threshold_criterion", "f1")).lower()
    de_recall_floor = float(train_cfg.get("de_recall_floor", 0.0) or 0.0)
    tta_hflip = bool(train_cfg.get("tta_hflip", False))
    tta_mode = str(train_cfg.get("tta_mode") or (
        "hflip" if tta_hflip else "none"))
    # Persist best τ alongside best.pt when monitoring f1_tau
    best_tau_saved = None  # type: Optional[float]

    best_score = -1.0
    best_epoch = 0
    epochs_since_improve = 0
    history = []
    backbone_unfrozen = freeze_epochs <= 0
    global_step = 0
    warmup_epochs = float(train_cfg.get("warmup_epochs", 0.0) or 0.0)
    # Ultralytics floors warmup at 100 steps. Keep that only when warmup is requested.
    # D1/D17/D18 leave warmup_epochs at 0 so the epoch-1 champion is not delayed.
    n_batches = max(len(train_loader), 1)
    warmup_steps = (
        max(round(warmup_epochs * n_batches), 100) if warmup_epochs > 0 else 0
    )
    strong_aug = bool(cfg.get("data", {}).get("strong_aug", False))
    close_aug_raw = train_cfg.get("close_strong_aug_epochs", None)
    if close_aug_raw is None:
        # Same default as close_mosaic: last 10 epochs drop heavy aug. No-op if strong_aug is off.
        close_strong_aug_epochs = 10 if strong_aug else 0
    else:
        close_strong_aug_epochs = int(close_aug_raw)
    strong_aug_closed = not strong_aug or close_strong_aug_epochs <= 0
    # Ultralytics close_mosaic analog for *mild* train aug (crop/flip/rot):
    # keep train aug for the first N epochs, then switch to eval transforms.
    # Unlike close_strong_aug, this works when strong_aug is false (D1 recipe).
    soft_aug_epochs = int(train_cfg.get("soft_aug_epochs", 0) or 0)
    close_train_aug_epochs = int(
        train_cfg.get("close_train_aug_epochs", 0) or 0)
    if soft_aug_epochs <= 0 and close_train_aug_epochs > 0:
        # close last K epochs → soft_aug_epochs = epochs - K
        soft_aug_epochs = max(epochs - close_train_aug_epochs, 0)
    train_aug_closed = soft_aug_epochs <= 0

    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    data_name = resolve_data_name(str(cfg.get("data", {}).get("name", "?")))
    display_name = cfg.get("data", {}).get("display_name", data_name)
    img_size = cfg.get("data", {}).get("image_size", 224)
    print(
        f"device={device}  model={cfg['model']['name']}  "
        f"data={display_name} ({data_name})  image_size={img_size}"
    )
    print(
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)}  "
        f"class_weights={cw}  sampler={train_cfg.get('class_balanced_sampler', False)}  "
        f"de_boost={train_cfg.get('sampler_de_boost', 1.0)}  "
        f"focal_gamma={focal_gamma}  "
        f"dice_weight={train_cfg.get('dice_weight', 0.0)}  "
        f"ldam_max_m={ldam_max_m}  ema_decay={ema_decay}  ema_tau={ema_tau}"
    )
    print(
        f"epochs={epochs}  monitor=val_{monitor}  patience={patience}  "
        f"lr_schedule={train_cfg.get('lr_schedule') or 'none'}  "
        f"threshold_criterion={threshold_criterion}  "
        f"de_recall_floor={de_recall_floor}  "
        f"freeze_epochs={freeze_epochs}  "
        f"backbone_lr_mult={train_cfg.get('backbone_lr_mult', 1.0)}  "
        f"tta_mode={tta_mode}  strong_aug={strong_aug}  "
        f"close_strong_aug_epochs={close_strong_aug_epochs}  "
        f"soft_aug_epochs={soft_aug_epochs}  "
        f"amp={use_amp}  grad_clip_norm={grad_clip_norm}  "
        f"warmup_steps={warmup_steps}"
    )
    print(
        f"reproducibility: deterministic={deterministic}  "
        f"warn_only={deterministic_warn_only}  seed={int(cfg.get('seed', 42))}"
    )

    stopped_early = False
    for epoch in range(1, epochs + 1):
        if (not backbone_unfrozen) and epoch > freeze_epochs:
            set_backbone_trainable(model, True)
            remaining = max(epochs - epoch + 1, 1)
            optimizer, scheduler = rebuild_optimizer_and_scheduler(
                model, train_cfg, remaining, device_note=" after unfreeze"
            )
            backbone_unfrozen = True
            print(
                f"unfroze backbone at epoch {epoch} (cosine T_max={remaining})")

        if (
            strong_aug
            and not strong_aug_closed
            and close_strong_aug_epochs > 0
            and epoch > epochs - close_strong_aug_epochs
        ):
            train_loader.dataset.transform = cfg_transforms(
                cfg, "train", strong_aug=False
            )
            strong_aug_closed = True
            print(
                f"closed strong_aug at epoch {epoch} "
                f"(last {close_strong_aug_epochs} epochs use mild crop/flip/rotation only)"
            )

        # Mild train-aug OFF → eval transforms (true close_mosaic for sketches).
        if (
            not train_aug_closed
            and soft_aug_epochs > 0
            and epoch > soft_aug_epochs
        ):
            train_loader.dataset.transform = cfg_transforms(
                cfg, "train", strong_aug=False, augment=False
            )
            train_aug_closed = True
            print(
                f"closed train_aug at epoch {epoch} "
                f"(epochs 1–{soft_aug_epochs} had train aug; now eval-like Resize only)"
            )

        train_m = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            log_every,
            num_classes=num_classes,
            ema=ema,
            global_step=global_step,
            warmup_steps=warmup_steps,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
        )
        global_step = int(train_m.get("global_step", global_step))
        tr_loss = float(train_m["loss"])
        eval_model = ema.ema if ema_eval else model
        val_m, val_logits, val_labels = evaluate(
            eval_model,
            val_loader,
            criterion,
            device,
            num_classes,
            return_logits=True,
            tta_hflip=tta_hflip,
            tta_mode=tta_mode,
        )

        # Val τ sweep every epoch (for monitor=f1_tau and logging)
        tau_ep = None
        val_at_tau_ep = None
        if num_classes == 2:
            val_probs = positive_class_probs(
                val_logits, num_classes=num_classes)
            if val_probs is not None:
                tau_ep, val_at_tau_ep = sweep_threshold(
                    val_probs,
                    val_labels.numpy(),
                    num_classes=num_classes,
                    criterion=threshold_criterion,
                    de_recall_floor=de_recall_floor,
                )
                val_m["f1_tau"] = float(val_at_tau_ep["f1"])
                val_m["tau"] = float(tau_ep)
                for k in ("f1_c0", "f1_c1"):
                    if k in val_at_tau_ep:
                        val_m[f"{k}_tau"] = float(val_at_tau_ep[k])
                if "f1_c1_tau" in val_m:
                    val_m["f1_de_tau"] = float(val_m["f1_c1_tau"])
                if "de_recall" in val_at_tau_ep:
                    val_m["de_recall_tau"] = float(val_at_tau_ep["de_recall"])

        lr_now = _group_lr(optimizer, "head")
        if lr_now is None:
            lr_now = _group_lr(optimizer, "decay")
        if lr_now is None:
            lr_now = float(optimizer.param_groups[0]["lr"])
        bb_lr = _group_lr(optimizer, "bb")
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_f1": float(train_m.get("f1", float("nan"))),
            "train_acc": float(train_m.get("accuracy", float("nan"))),
            "lr": lr_now,
            "backbone_lr": bb_lr,
            "backbone_unfrozen": bool(backbone_unfrozen),
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)
        f1_tau_str = (
            f"  f1@τ={val_m.get('f1_tau', float('nan')):.3f}"
            f"@τ={val_m.get('tau', float('nan')):.2f}"
            f"  f1_De@τ={val_m.get('f1_de_tau', float('nan')):.3f}"
            if "f1_tau" in val_m
            else ""
        )
        bb_lr_str = f"  bb_lr={bb_lr:.2e}" if bb_lr is not None else ""
        print(
            f"epoch {epoch}/{epochs}  train_loss={tr_loss:.4f}  "
            f"train_f1={train_m.get('f1', float('nan')):.3f}  "
            f"val_loss={val_m['loss']:.4f}  acc={val_m['accuracy']:.3f}  "
            f"f1={val_m['f1']:.3f}  auc={val_m['auc']}  pr_auc={val_m['pr_auc']}"
            f"{f1_tau_str}  lr={lr_now:.2e}{bb_lr_str}"
        )
        save_checkpoint(
            out_dir / "last.pt", model, optimizer, epoch, val_m, cfg
        )

        score = val_m.get(monitor, float("nan"))
        if _is_better(float(score) if score is not None else float("nan"), best_score):
            best_score = float(score)
            best_epoch = epoch
            epochs_since_improve = 0
            # Prefer EMA weights for best.pt when ema_eval (generalization target).
            save_model = eval_model if ema_eval else model
            save_checkpoint(
                out_dir / "best.pt", save_model, optimizer, epoch, val_m, cfg
            )
            if tau_ep is not None:
                best_tau_saved = float(tau_ep)
            tag = "ema" if ema_eval else "raw"
            print(f"  ^ saved best.pt ({tag} val_{monitor}={best_score:.4f})")
        else:
            epochs_since_improve += 1

        if scheduler is not None:
            scheduler.step()
            _sync_scheduled_lrs(optimizer)

        if patience > 0 and epochs_since_improve >= patience:
            print(
                f"early stop at epoch {epoch}: no val_{monitor} improvement "
                f"for {patience} epochs (best epoch {best_epoch}, "
                f"val_{monitor}={best_score:.4f})"
            )
            stopped_early = True
            break

    best_path = out_dir / "best.pt"
    if best_path.is_file():
        ckpt = load_checkpoint(best_path, model, map_location=device)
        print(
            f"loaded best.pt for final eval "
            f"(epoch={ckpt.get('epoch')}, metrics={ckpt.get('metrics')})"
        )
    else:
        print("WARNING: best.pt missing; evaluating last in-memory weights")

    val_m, val_logits, val_labels = evaluate(
        model,
        val_loader,
        criterion,
        device,
        num_classes,
        return_logits=True,
        tta_hflip=tta_hflip,
        tta_mode=tta_mode,
    )
    test_m, test_logits, test_labels = evaluate(
        model,
        test_loader,
        criterion,
        device,
        num_classes,
        return_logits=True,
        tta_hflip=tta_hflip,
        tta_mode=tta_mode,
    )

    # Argmax / 0.5 hard metrics (ranking AUC/PR-AUC unchanged)
    print(
        f"val@best (argmax)  acc={val_m['accuracy']:.3f}  f1={val_m['f1']:.3f}  "
        f"auc={val_m['auc']}  pr_auc={val_m['pr_auc']}  "
        f"f1_Nor={val_m.get('f1_c0')}  f1_De={val_m.get('f1_c1')}"
    )
    print(
        f"test@best (argmax)  acc={test_m['accuracy']:.3f}  f1={test_m['f1']:.3f}  "
        f"auc={test_m['auc']}  pr_auc={test_m['pr_auc']}  "
        f"f1_Nor={test_m.get('f1_c0')}  f1_De={test_m.get('f1_c1')}"
    )

    threshold_info = None
    val_at_tau = None
    test_at_tau = None
    if num_classes == 2:
        val_probs = positive_class_probs(val_logits, num_classes=num_classes)
        test_probs = positive_class_probs(test_logits, num_classes=num_classes)
        if val_probs is not None and test_probs is not None:
            tau, val_at_tau = sweep_threshold(
                val_probs,
                val_labels.numpy(),
                num_classes=num_classes,
                criterion=threshold_criterion,
                de_recall_floor=de_recall_floor,
            )
            test_at_tau = compute_metrics(
                test_logits, test_labels, num_classes=num_classes, threshold=tau
            )
            # Keep ranking metrics from continuous scores; overwrite hard Acc/F1 at τ
            test_at_tau["auc"] = test_m["auc"]
            test_at_tau["pr_auc"] = test_m["pr_auc"]
            test_at_tau["loss"] = test_m.get("loss")
            val_at_tau = {
                **val_at_tau,
                "auc": val_m["auc"],
                "pr_auc": val_m["pr_auc"],
                "loss": val_m.get("loss"),
            }
            threshold_info = {
                "tau": tau,
                "criterion": threshold_criterion,
                "val": val_at_tau,
                "test": test_at_tau,
            }
            print(
                f"val@tau={tau:.3f} ({threshold_criterion})  "
                f"acc={val_at_tau['accuracy']:.3f}  f1={val_at_tau['f1']:.3f}  "
                f"f1_Nor={val_at_tau.get('f1_c0')}  f1_De={val_at_tau.get('f1_c1')}"
            )
            print(
                f"test@tau={tau:.3f}  "
                f"acc={test_at_tau['accuracy']:.3f}  f1={test_at_tau['f1']:.3f}  "
                f"f1_Nor={test_at_tau.get('f1_c0')}  f1_De={test_at_tau.get('f1_c1')}  "
                f"auc={test_at_tau['auc']}  pr_auc={test_at_tau['pr_auc']}"
            )
            # Persist τ on best checkpoint
            if best_path.is_file():
                ckpt = torch_load(
                    best_path, map_location="cpu", weights_only=False)
                metrics = dict(ckpt.get("metrics") or {})
                metrics["threshold"] = tau
                metrics["threshold_criterion"] = threshold_criterion
                metrics["val_at_threshold"] = {
                    "accuracy": val_at_tau["accuracy"],
                    "f1": val_at_tau["f1"],
                    "f1_c0": val_at_tau.get("f1_c0"),
                    "f1_c1": val_at_tau.get("f1_c1"),
                }
                ckpt["metrics"] = metrics
                ckpt["threshold"] = tau
                torch.save(ckpt, best_path)

    payload = {
        "history": history,
        "best_epoch": best_epoch,
        "best_monitor": monitor,
        "best_score": best_score,
        "stopped_early": stopped_early,
        "val_best_argmax": val_m,
        "test_best_argmax": test_m,
        "threshold": threshold_info,
        # Back-compat keys used by prior diagnoses
        "test": test_at_tau if test_at_tau is not None else test_m,
        "config": cfg,
    }
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
