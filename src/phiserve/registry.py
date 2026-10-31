"""Offline GGUF registry. We keep pointers, never copy or download weights."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path


_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class RegistryError(ValueError):
    """Invalid or unavailable local model metadata."""


class ModelRegistry:
    def __init__(self, home: str | Path | None = None):
        configured = home if home is not None else os.environ.get("PHISERVE_HOME", "~/.phiserve")
        self.home = Path(configured).expanduser().resolve()
        self.file = self.home / "models.json"

    @staticmethod
    def validate_alias(alias: str) -> None:
        if not isinstance(alias, str) or _ALIAS.fullmatch(alias) is None:
            raise RegistryError("Alias must be 1-128 characters: letters, digits, . _ : -; start with a letter or digit.")

    def _read(self) -> dict:
        if not self.file.exists():
            return {"schema_version": 1, "models": {}}
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"Cannot read model registry: {exc}") from exc
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != 1
            or not isinstance(data.get("models"), dict)
        ):
            raise RegistryError("Unsupported or malformed model registry")
        for alias, path in data["models"].items():
            self.validate_alias(alias)
            if not isinstance(path, str) or not path:
                raise RegistryError("Malformed model registry entry")
        return data

    def _write(self, data: dict) -> None:
        self.home.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".models-", suffix=".tmp", dir=self.home)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.file)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _check_gguf(path: Path) -> None:
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise RegistryError("Select an existing .gguf model file")
        try:
            with path.open("rb") as stream:
                magic = stream.read(4)
        except OSError as exc:
            raise RegistryError(f"Cannot read model header: {exc}") from exc
        if magic != b"GGUF":
            raise RegistryError("File does not begin with GGUF magic bytes")

    def add(self, alias: str, model_path: str | Path) -> Path:
        self.validate_alias(alias)
        try:
            resolved = Path(model_path).expanduser().resolve(strict=True)
        except OSError as exc:
            raise RegistryError(f"Model path cannot be resolved: {exc}") from exc
        self._check_gguf(resolved)
        data = self._read()
        if alias in data["models"]:
            raise RegistryError(f"Alias already registered: {alias}")
        data["models"][alias] = str(resolved)
        self._write(data)
        return resolved

    def list_models(self) -> dict[str, str]:
        return dict(sorted(self._read()["models"].items()))

    def get(self, alias: str) -> Path:
        self.validate_alias(alias)
        path = self._read()["models"].get(alias)
        if path is None:
            raise RegistryError(f"Unknown model alias: {alias}")
        result = Path(path)
        self._check_gguf(result)
        return result

    def remove(self, alias: str) -> None:
        self.validate_alias(alias)
        data = self._read()
        if alias not in data["models"]:
            raise RegistryError(f"Unknown model alias: {alias}")
        del data["models"][alias]
        self._write(data)
