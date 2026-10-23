from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _load_project_symbols():
    from model.model_loader import build_model
    from run_experiment_matrix import _load_matrix, _resolved_config

    return build_model, _load_matrix, _resolved_config


build_model, _load_matrix, _resolved_config = _load_project_symbols()


class ModelAblationTests(unittest.TestCase):
    def test_plain_backbone_classifier_has_no_so_modules(self):
        model = build_model({
            "model": {"name": "backbone_classifier", "backbone": "cnn_lite"}
        })
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output["logits"].shape), (2, 2))
        self.assertFalse(hasattr(model, "sofe"))
        self.assertFalse(hasattr(model, "sosm"))
        self.assertFalse(hasattr(model, "wff"))

    def test_all_sos_module_combinations_forward_and_match_flags(self):
        image = torch.randn(2, 3, 32, 32)
        for use_sofe in (False, True):
            for use_sosm in (False, True):
                for use_wff in (False, True):
                    with self.subTest(
                        sofe=use_sofe, sosm=use_sosm, wff=use_wff
                    ):
                        model = build_model({
                            "model": {
                                "name": "soe_fbanet",
                                "backbone": "cnn_lite",
                                "use_sofe": use_sofe,
                                "use_sosm": use_sosm,
                                "use_wff": use_wff,
                                "use_small_object_branch": use_wff,
                                "use_bilevel_attention": True,
                            }
                        }).eval()
                        self.assertEqual(model.use_sofe, use_sofe)
                        self.assertEqual(model.use_sosm, use_sosm)
                        self.assertEqual(model.use_wff, use_wff)
                        self.assertEqual(model.so_branch is not None, use_wff)
                        if not use_wff:
                            self.assertIsNone(model.wff)
                        with torch.no_grad():
                            output = model(image)
                        self.assertEqual(tuple(output["logits"].shape), (2, 2))

    def test_bilevel_attention_backward_works_with_deterministic_algorithms(self):
        if not torch.cuda.is_available():
            self.skipTest(
                "CUDA required for deterministic-backward regression")

        code = """
import os
import torch

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
try:
    torch.use_deterministic_algorithms(True, warn_only=False)
except TypeError:
    torch.use_deterministic_algorithms(True)

from model.modules.bilevel_attention import BiLevelAttention

x = torch.randn(2, 8, 16, 16, device='cuda', requires_grad=True)
model = BiLevelAttention(8).cuda().eval()
out = model(x)
loss = out.pow(2).mean()
loss.backward()
print('OK')
"""
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )


class ExperimentMatrixTests(unittest.TestCase):
    def test_initial_matrix_is_unique_and_uses_one_frozen_protocol(self):
        matrix_path = ROOT / "src/configs/archive/research/phase1_phase2_seed42.yaml"
        matrix = _load_matrix(matrix_path)
        self.assertEqual(len(matrix["experiments"]), 10)
        ids = [experiment["id"] for experiment in matrix["experiments"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [
            "baseline_resnet50_ce",
            "attention_control_ce",
            "ablation_sofe_ce",
            "ablation_sosm_ce",
            "ablation_wff_ce",
            "ablation_sofe_sosm_ce",
            "ablation_sofe_wff_ce",
            "ablation_sosm_wff_ce",
            "soefbanet_full_ce",
            "soefbanet_full_focal_gamma_0_5",
        ])
        self.assertEqual(
            matrix["experiments"][1]["legacy_id"],
            "a0_attention_control_ce",
        )

        resolved = [
            _resolved_config(matrix, experiment, ROOT /
                             "outputs" / experiment["id"])
            for experiment in matrix["experiments"]
        ]
        self.assertTrue(all(config["seed"] == 42 for config in resolved))
        self.assertTrue(all(config["data"]["split_file"].endswith(
            "seed42.json") for config in resolved))
        self.assertTrue(
            all(config["data"]["image_size"] == 512 for config in resolved))
        self.assertTrue(
            all(config["model"]["backbone"] == "resnet50" for config in resolved))
        self.assertTrue(all(config["reproducibility"]
                        ["deterministic"] for config in resolved))

        focal = next(
            config for config in resolved
            if config["experiment"]["id"] == "soefbanet_full_focal_gamma_0_5")
        self.assertEqual(focal["train"]["focal_gamma"], 0.5)
        self.assertIsNone(focal["train"]["class_weights"])


if __name__ == "__main__":
    unittest.main()
