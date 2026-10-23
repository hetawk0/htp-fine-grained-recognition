from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train import _cli_config_overrides


class TrainCliOverrideTests(unittest.TestCase):
    def test_set_values_are_typed_and_direct_options_take_precedence(self):
        args = Namespace(
            set=[
                "data.split_ratios=[0.7, 0.15, 0.15]",
                "model.use_sofe=false",
                "train.epochs=12",
            ],
            data="zhang",
            arch="fbanet",
            backbone="resnet18",
            image_size=None,
            epochs=20,
            batch_size=8,
            lr=0.001,
            weight_decay=None,
            device="cpu",
            out_dir="outputs/cli-test",
            resume=None,
            seed=7,
            pretrained=False,
        )

        overrides = _cli_config_overrides(args)

        self.assertEqual(overrides["data"]["name"], "zhang")
        self.assertEqual(overrides["data"]["split_ratios"], [0.7, 0.15, 0.15])
        self.assertFalse(overrides["model"]["use_sofe"])
        self.assertEqual(overrides["model"]["name"], "fbanet")
        self.assertEqual(overrides["model"]["backbone"], "resnet18")
        self.assertFalse(overrides["model"]["pretrained"])
        self.assertEqual(overrides["train"]["epochs"], 20)
        self.assertEqual(overrides["train"]["batch_size"], 8)
        self.assertEqual(overrides["train"]["lr"], 0.001)
        self.assertEqual(overrides["train"]["device"], "cpu")
        self.assertEqual(overrides["train"]["out_dir"], "outputs/cli-test")
        self.assertEqual(overrides["seed"], 7)

    def test_set_requires_dotted_key_and_value(self):
        args = Namespace(
            set=["train.epochs"],
            data=None,
            arch=None,
            backbone=None,
            image_size=None,
            epochs=None,
            batch_size=None,
            lr=None,
            weight_decay=None,
            device=None,
            out_dir=None,
            resume=None,
            seed=None,
            pretrained=None,
        )

        with self.assertRaisesRegex(ValueError, "expected section.key=value"):
            _cli_config_overrides(args)


if __name__ == "__main__":
    unittest.main()
