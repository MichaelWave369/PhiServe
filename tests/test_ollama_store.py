"""Read-only Ollama storage importer; uses synthetic manifests and tiny GGUF fixtures."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from phiserve.ollama_store import OllamaStoreError, discover, import_model
from phiserve.registry import ModelRegistry


class OllamaStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "ollama-models"
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True)
        self.model = b"GGUF" + b"test-model-content"
        self.digest = hashlib.sha256(self.model).hexdigest()
        self.blob = self.blobs / ("sha256-" + self.digest)
        self.blob.write_bytes(self.model)
        self.manifest = self.root / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "4b"
        self.manifest.parent.mkdir(parents=True)
        self.data = {"schemaVersion": 2, "layers": [
            {"mediaType": "application/vnd.ollama.image.model",
             "digest": "sha256:" + self.digest, "size": len(self.model)},
            {"mediaType": "application/vnd.ollama.image.template",
             "digest": "sha256:" + "0" * 64, "size": 1},
        ]}
        self.manifest.write_text(json.dumps(self.data))
        self.registry = ModelRegistry(Path(self.tmp.name) / "phiserve-state")

    def test_discover_then_import_reference_without_copying(self):
        models, warnings = discover(self.root)
        self.assertEqual(len(models), 1)
        self.assertEqual(warnings, [])
        self.assertEqual(models[0].name, "qwen3:4b")
        self.assertEqual(models[0].alias, "qwen3:4b")
        entry = import_model("qwen3:4b", root=self.root,
                             registry=self.registry, verify_sha256=True)
        self.assertEqual(self.registry.get(entry.alias), self.blob.resolve())
        self.assertEqual(self.blob.read_bytes(), self.model)
        self.assertEqual(len(list(self.blobs.iterdir())), 1)

    def test_mismatched_sha_blocks_verified_import(self):
        self.blob.write_bytes(b"GGUF" + b"modified-content")
        self.data["layers"][0]["size"] = self.blob.stat().st_size
        self.manifest.write_text(json.dumps(self.data))
        with self.assertRaises(OllamaStoreError):
            import_model("qwen3:4b", root=self.root, registry=self.registry,
                         verify_sha256=True)
        self.assertEqual(self.registry.list_models(), {})

    def test_invalid_digest_is_skipped(self):
        self.data["layers"][0]["digest"] = "sha256:../../secrets"
        self.manifest.write_text(json.dumps(self.data))
        found, warnings = discover(self.root)
        self.assertEqual(found, [])
        self.assertTrue(warnings)

    def test_missing_gguf_header_is_skipped(self):
        self.blob.write_bytes(b"NOTG" + self.model[4:])
        found, warnings = discover(self.root)
        self.assertEqual(found, [])
        self.assertTrue(warnings)

    def test_nonexistent_root_fails(self):
        with self.assertRaises(OllamaStoreError):
            discover(Path(self.tmp.name) / "no-models")


if __name__ == "__main__":
    unittest.main()
