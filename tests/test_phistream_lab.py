"""PhiStream bridge tests use stubs; no production weights or GPU are exercised."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phiserve.phistream_lab import (
    PhiStreamCatalog, PhiStreamLabError, parse_token_ids, run_ids,
)
from phiserve.registry import ModelRegistry


REVIEW = {
    "schema": "PHISERVE_PHISTREAM_PREFLIGHT_V1",
    "status": "METADATA_PREFLIGHT_ONLY",
    "format": "SAFETENSORS",
    "vocab_size": 100,
    "layers": 2,
    "tensor_count": 21,
    "weight_bytes": 1234,
    "shard_count": 1,
    "service_ready": False,
    "can_serve_chat": False,
    "tokenizer_available": False,
}


class PhiStreamCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.weights = self.home / "some-llama-safetensors"
        self.weights.mkdir()
        (self.weights / "config.json").write_text('{"model_type":"llama"}')
        self.catalog = PhiStreamCatalog(self.home / "catalog")

    def register(self):
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            return self.catalog.add("research-llama", self.weights,
                                    source="locally-held", revision="declared-rev",
                                    license_id="declared-license")

    def test_separate_catalog_does_not_make_fake_gguf_model(self):
        review = self.register()
        self.assertFalse(review["can_serve_chat"])
        self.assertIn("research-llama", self.catalog.list())
        self.assertEqual(ModelRegistry(self.home / "catalog").list_models(), {})
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            model, inspect = self.catalog.get("research-llama")
        self.assertEqual(model, self.weights.resolve())
        self.assertEqual(inspect["status"], "METADATA_PREFLIGHT_ONLY")
        self.assertTrue((self.weights / "config.json").exists())

    def test_source_declarations_and_duplicates(self):
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            with self.assertRaises(PhiStreamLabError):
                self.catalog.add("research", self.weights,
                    source="", revision="rev", license_id="MIT")
        self.register()
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            with self.assertRaises(PhiStreamLabError):
                self.catalog.add("research-llama", self.weights,
                    source="source", revision="rev", license_id="MIT")

    def test_remove_preserves_checkpoint_bytes(self):
        self.register()
        self.catalog.remove("research-llama")
        self.assertEqual(self.catalog.list(), {})
        self.assertTrue((self.weights / "config.json").is_file())

    def test_missing_checkpoint_rejected(self):
        self.register()
        (self.weights / "config.json").unlink()
        self.weights.rmdir()
        with self.assertRaises(PhiStreamLabError):
            self.catalog.get("research-llama")

    def test_malformed_catalog_not_overwritten(self):
        self.catalog.home.mkdir(parents=True)
        self.catalog.file.write_text("not-json")
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            with self.assertRaises(PhiStreamLabError):
                self.catalog.add("research", self.weights,
                    source="source", revision="rev", license_id="MIT")
        self.assertEqual(self.catalog.file.read_text(), "not-json")

    def test_ids_are_strict_and_bounded(self):
        self.assertEqual(parse_token_ids("0,1,99", 100), [0, 1, 99])
        for value in ("", "1, 2", "-1", "100", "1,,2", "01", "1;", "1.0"):
            with self.subTest(value=value), self.assertRaises(PhiStreamLabError):
                parse_token_ids(value, 100)

    def test_run_requires_explicit_approval_and_no_overwrite(self):
        self.register()
        out = self.home / "receipt.json"
        with self.assertRaisesRegex(PhiStreamLabError, "approve"):
            run_ids(self.catalog, "research-llama", prompt="1,2", tokens=4,
                    device="cpu", output=out, approved=False)
        out.write_text("do not overwrite")
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            with self.assertRaisesRegex(PhiStreamLabError, "already exists"):
                run_ids(self.catalog, "research-llama", prompt="1", tokens=4,
                        device="cpu", output=out, approved=True)
        self.assertEqual(out.read_text(), "do not overwrite")

    def test_run_uses_phi_stream_cli_without_shell_and_no_chat_switch(self):
        self.register()
        out = self.home / "receipt.json"
        def fake_subprocess(argv, **kwargs):
            self.assertEqual(argv[1:4], ["-m", "phistream.cli", "run-llama-ids"])
            self.assertIn("--prompt", argv)
            self.assertIn("--device", argv)
            self.assertFalse(kwargs["shell"])
            out.write_text(json.dumps({"schema": "PHISTREAM_LLAMA_TOKEN_IDS_V1"}))
            return SimpleNamespace(returncode=0)
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            result = run_ids(self.catalog, "research-llama", prompt="1,2",
                             tokens=2, device="cpu", output=out,
                             approved=True, runner=fake_subprocess)
        self.assertEqual(result["status"], "LOCAL_TOKEN_ID_EXECUTION_RECEIPT_CREATED")
        self.assertFalse(result["chat_endpoint_enabled"])
        self.assertFalse(result["gpu_execution_independently_verified"])

    def test_failed_subprocess_refuses_success_receipt(self):
        self.register()
        out = self.home / "receipt.json"
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            with self.assertRaises(PhiStreamLabError):
                run_ids(self.catalog, "research-llama", prompt="1",
                        tokens=1, device="cpu", output=out, approved=True,
                        runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=2))
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
