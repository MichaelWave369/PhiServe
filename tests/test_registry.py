import json
import tempfile
import unittest
from pathlib import Path

from phiserve.registry import ModelRegistry, RegistryError


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.home = Path(self.directory.name) / "state"
        self.registry = ModelRegistry(self.home)
        self.model = Path(self.directory.name) / "tiny.gguf"
        self.model.write_bytes(b"GGUF" + b"test-bytes")

    def test_add_list_get_remove_preserves_model_file(self):
        self.assertEqual(self.registry.add("demo:1", self.model), self.model.resolve())
        self.assertEqual(self.registry.get("demo:1"), self.model.resolve())
        self.assertEqual(self.registry.list_models(), {"demo:1": str(self.model.resolve())})
        self.assertEqual(json.loads((self.home / "models.json").read_text())["schema_version"], 1)
        self.registry.remove("demo:1")
        self.assertEqual(self.registry.list_models(), {})
        self.assertTrue(self.model.is_file())

    def test_bad_magic_rejected(self):
        self.model.write_bytes(b"NOT-GGUF")
        with self.assertRaisesRegex(RegistryError, "GGUF magic"):
            self.registry.add("demo", self.model)

    def test_reject_alias_and_duplicate(self):
        with self.assertRaises(RegistryError):
            self.registry.add("../bad", self.model)
        self.registry.add("demo", self.model)
        with self.assertRaises(RegistryError):
            self.registry.add("demo", self.model)

    def test_moved_model_detected(self):
        self.registry.add("demo", self.model)
        self.model.unlink()
        with self.assertRaises(RegistryError):
            self.registry.get("demo")

    def test_malformed_registry_never_silently_overwritten(self):
        self.home.mkdir()
        (self.home / "models.json").write_text("{broken")
        with self.assertRaises(RegistryError):
            self.registry.add("demo", self.model)
        self.assertEqual((self.home / "models.json").read_text(), "{broken")


if __name__ == "__main__":
    unittest.main()
