"""No trained model, GPU or network needed for safe Forge candidate intake."""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from phiserve.forge_intake import ForgeIntakeError, inspect


class ForgeIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
        }))
        header = json.dumps({
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight":
                {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }).encode()
        (self.adapter / "adapter_model.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + b"abcd")
        self.receipt = self.root / "training.json"
        self.receipt.write_text(json.dumps({
            "schema": "phi.model-forge.training-receipt.v1",
            "status": "LOCALLY_TRAINED_UNVERIFIED_QUALITY",
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "dataset_sha256": "a" * 64,
            "production_authorized": False,
        }))

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_shape_is_not_production_approval(self):
        report = inspect(self.adapter, self.receipt, hash_weights=True)
        self.assertEqual(report["status"], "REVIEW_REQUIRED")
        self.assertEqual(report["adapter_tensors_observed"], 1)
        self.assertEqual(len(report["adapter_sha256"]), 64)
        self.assertFalse(report["gguf_compatible"])
        self.assertFalse(report["phi_serve_registered"])
        self.assertFalse(report["model_output_verified"])
        self.assertFalse(report["execution_authorized"])

    def test_receipt_base_mismatch_rejected(self):
        data = json.loads(self.receipt.read_text())
        data["base_model"] = "other"
        self.receipt.write_text(json.dumps(data))
        with self.assertRaisesRegex(ForgeIntakeError, "differs"):
            inspect(self.adapter, self.receipt)

    def test_bad_weights_offsets_rejected(self):
        header = json.dumps({"weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 900000],
        }}).encode()
        (self.adapter / "adapter_model.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + b"abcd")
        with self.assertRaisesRegex(ForgeIntakeError, "offsets"):
            inspect(self.adapter, self.receipt)

    def test_unapproved_receipt_and_unsupported_adapter_rejected(self):
        data = json.loads(self.receipt.read_text())
        data["production_authorized"] = True
        self.receipt.write_text(json.dumps(data))
        with self.assertRaisesRegex(ForgeIntakeError, "unpromoted"):
            inspect(self.adapter, self.receipt)
        data["production_authorized"] = False
        self.receipt.write_text(json.dumps(data))
        config_path = self.adapter / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["peft_type"] = "CUSTOM"
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ForgeIntakeError, "LORA"):
            inspect(self.adapter, self.receipt)


if __name__ == "__main__":
    unittest.main()
