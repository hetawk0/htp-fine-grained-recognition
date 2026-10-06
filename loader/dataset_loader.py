"""Dataset factory — synthetic / quickdraw / clinical."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from loader.clinical import CLINICAL_LAYOUTS, HTPFolderDataset
from loader.quickdraw import QuickDrawHTPDataset
from loader.synthetic import SyntheticHTPDataset
from loader.transforms import cfg_transforms

# Paper-facing / research aliases → canonical layout key
DATA_NAME_ALIASES = {
    "htp_depression_dd": "dd",
    "htp-dd": "dd",
    "depression_drawings": "dd",
    "dd_htp": "dd",
    "soe_fbanet_dd": "dd",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_name(name: str) -> str:
    key = str(name).lower().strip()
    return DATA_NAME_ALIASES.get(key, key)


def build_dataset(cfg: Dict[str, Any], split: str = "train") -> Dataset:
    data = cfg.get("data", {})
    name = resolve_data_name(str(data.get("name", "synthetic")))
    transform = cfg_transforms(cfg, split)
    seed = int(cfg.get("seed", 42))
    image_size = int(data.get("image_size", 224))
    ratios = tuple(data.get("split_ratios", [0.7, 0.15, 0.15]))

    if name == "synthetic":
        n = int(data.get("num_samples", 64))
        split_n = {
            "train": max(8, int(n * ratios[0])),
            "val": max(4, int(n * ratios[1])),
            "test": max(4, int(n * ratios[2])),
        }.get(split, n)
        return SyntheticHTPDataset(
            num_samples=split_n,
            image_size=image_size,
            seed=seed + {"train": 0, "val": 1, "test": 2}.get(split, 0),
            transform=transform,
            small_object_ratio=float(data.get("small_object_ratio", 0.5)),
        )

    if name == "quickdraw":
        root = Path(data.get("root", _repo_root() / "datasets/raw/quickdraw/numpy_bitmap"))
        if not root.is_absolute():
            root = _repo_root() / root
        return QuickDrawHTPDataset(
            root=root,
            split=split,
            categories=tuple(data.get("categories", ["house", "tree", "face"])),
            samples_per_class=int(data.get("samples_per_class", 200)),
            image_size=image_size,
            transform=transform,
            invert_strokes=bool(data.get("invert", True)),
            binary=bool(data.get("binary", False)),
            seed=seed,
            split_ratios=ratios,
        )

    if name in CLINICAL_LAYOUTS or name in {"zhang", "pan", "bilgi", "dd", "clinical"}:
        key = "zhang" if name == "clinical" else name
        layout = CLINICAL_LAYOUTS[key]
        root = Path(data.get("root", _repo_root() / layout["root"]))
        if not root.is_absolute():
            root = _repo_root() / root
        split_file = data.get("split_file")
        if split_file:
            split_path = Path(split_file)
            if not split_path.is_absolute():
                split_path = _repo_root() / split_path
        else:
            split_path = None
        return HTPFolderDataset(
            root=root,
            split=split,
            transform=transform,
            class_dirs=layout.get("class_dirs"),
            split_ratios=ratios,
            seed=seed,
            require_exists=bool(data.get("require_exists", True)),
            parse_filename_labels=bool(
                data.get("parse_filename_labels", layout.get("parse_filename_labels", False))
            ),
            include_augmented=bool(
                data.get("include_augmented", layout.get("include_augmented_default", False))
            ),
            stratified=bool(data.get("stratified", True)),
            split_file=split_path,
            forbid_class_dirs=bool(data.get("forbid_class_dirs", key == "dd")),
        )

    raise ValueError(
        f"Unknown dataset '{name}'. Use: synthetic, quickdraw, zhang, pan, bilgi, dd "
        f"(alias: htp_depression_dd), …"
    )


def collate_htp(batch):
    import torch

    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    so_mask = torch.tensor([b["so_mask"] for b in batch], dtype=torch.float32)
    paths = [b["path"] for b in batch]
    return {"image": images, "label": labels, "so_mask": so_mask, "path": paths}


def _class_balanced_sampler(
    dataset: Dataset,
    de_boost: float = 1.0,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Inverse-frequency sampler; ``de_boost`` further upweights class 1 (De)."""
    if hasattr(dataset, "labels_list"):
        labels = list(dataset.labels_list())  # type: ignore[attr-defined]
    else:
        labels = [int(dataset[i]["label"]) for i in range(len(dataset))]  # type: ignore[arg-type]
    import torch

    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=2).float()
    counts = counts.clamp_min(1.0)
    weight_per_class = 1.0 / counts
    if de_boost != 1.0 and weight_per_class.numel() > 1:
        weight_per_class[1] = weight_per_class[1] * float(de_boost)
    sample_w = weight_per_class[torch.tensor(labels, dtype=torch.long)]
    n = int(num_samples) if num_samples is not None else len(labels)
    return WeightedRandomSampler(
        weights=sample_w.double(),
        num_samples=max(n, 1),
        replacement=True,
    )


def build_dataloaders(
    cfg: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))
    use_sampler = bool(train_cfg.get("class_balanced_sampler", False))
    de_boost = float(train_cfg.get("sampler_de_boost", 1.0))
    # Optional epoch length expansion (more De draws per epoch without leaking augs).
    sampler_num = train_cfg.get("sampler_num_samples")

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    test_ds = build_dataset(cfg, "test")

    sampler = None
    if use_sampler:
        n_samp = int(sampler_num) if sampler_num is not None else None
        sampler = _class_balanced_sampler(train_ds, de_boost=de_boost, num_samples=n_samp)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_htp,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_htp,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_htp,
    )
    return train_loader, val_loader, test_loader
