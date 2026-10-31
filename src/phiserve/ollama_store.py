"""Explicit, read-only discovery/import of local Ollama GGUF layers.

This is a bridge for existing Ollama storage. We never modify the Ollama
manifest/blob tree, infer a model's format from its name, or download weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .registry import ModelRegistry, RegistryError


_DIGEST = re.compile(r"sha256:([a-f0-9]{64})\Z")
_MAX_MANIFEST = 1024 * 1024
_MODEL_MEDIA = "application/vnd.ollama.image.model"


class OllamaStoreError(ValueError):
    """The local Ollama store is invalid or a requested model is unsupported."""


@dataclass(frozen=True)
class OllamaEntry:
    name: str
    alias: str
    digest: str
    path: Path
    size: int


def default_root() -> Path:
    """Use the user-configured Ollama store, otherwise Ollama's user store."""
    return Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models").expanduser()


def _model_identity(parts: tuple[str, ...]) -> tuple[str, str]:
    if len(parts) < 4:
        raise OllamaStoreError("Manifest must be under host/namespace/model/tag")
    host, namespace, *rest = parts
    model = "/".join(rest[:-1])
    tag = rest[-1]
    if not host or not namespace or not model or not tag:
        raise OllamaStoreError("Invalid manifest path")
    name = f"{model}:{tag}" if (host, namespace) == ("registry.ollama.ai", "library") else f"{host}/{namespace}/{model}:{tag}"
    alias = name.replace("/", "-")
    ModelRegistry.validate_alias(alias)
    return name, alias


def _manifest_entry(file: Path, root: Path) -> OllamaEntry:
    resolved = file.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise OllamaStoreError("Manifest escapes the selected Ollama root")
    if file.stat().st_size > _MAX_MANIFEST:
        raise OllamaStoreError("Manifest too large")
    parts = file.relative_to(root / "manifests").parts
    name, alias = _model_identity(parts)
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OllamaStoreError(f"Cannot read manifest: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("layers"), list):
        raise OllamaStoreError("Malformed Ollama manifest")
    candidates = [layer for layer in data["layers"]
                  if isinstance(layer, dict) and layer.get("mediaType") == _MODEL_MEDIA]
    if len(candidates) != 1:
        raise OllamaStoreError("Expected one model-weight layer")
    layer = candidates[0]
    digest = layer.get("digest")
    match = _DIGEST.fullmatch(digest) if isinstance(digest, str) else None
    if match is None:
        raise OllamaStoreError("Invalid model layer SHA-256 digest")
    declared = layer.get("size")
    if type(declared) is not int or declared <= 0:
        raise OllamaStoreError("Invalid model layer size")
    blob = (root / "blobs" / f"sha256-{match.group(1)}").resolve(strict=False)
    if not blob.is_relative_to(root / "blobs"):
        raise OllamaStoreError("Model blob escapes the selected store")
    if not blob.is_file():
        raise OllamaStoreError("Model blob missing")
    if blob.stat().st_size != declared:
        raise OllamaStoreError("Model blob size differs from manifest")
    ModelRegistry._check_gguf(blob)
    return OllamaEntry(name=name, alias=alias, digest=match.group(1),
                       path=blob, size=declared)


def discover(root: str | Path | None = None) -> tuple[list[OllamaEntry], list[str]]:
    base = Path(root).expanduser().resolve() if root else default_root().resolve()
    manifests = base / "manifests"
    if not manifests.is_dir():
        raise OllamaStoreError(f"Ollama manifests directory not found: {manifests}")
    models: list[OllamaEntry] = []
    warnings: list[str] = []
    for index, file in enumerate(sorted(manifests.rglob("*"))):
        if index >= 10000:
            warnings.append("Store discovery stopped after 10,000 entries")
            break
        if not file.is_file():
            continue
        try:
            models.append(_manifest_entry(file, base))
        except (OllamaStoreError, RegistryError, OSError, ValueError) as exc:
            warnings.append(f"{file.relative_to(base)}: {exc}")
    return sorted(models, key=lambda item: item.name), warnings


def _verify_digest(entry: OllamaEntry) -> None:
    digest = hashlib.sha256()
    with entry.path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != entry.digest:
        raise OllamaStoreError("Ollama model blob SHA-256 differs from manifest")


def import_model(
    model_name: str,
    alias: str | None = None,
    root: str | Path | None = None,
    verify_sha256: bool = False,
    registry: ModelRegistry | None = None,
) -> OllamaEntry:
    entries, _warnings = discover(root)
    matches = [entry for entry in entries if entry.name == model_name]
    if len(matches) != 1:
        raise OllamaStoreError(f"Expected one matching local Ollama model for {model_name!r}; found {len(matches)}")
    entry = matches[0]
    alias = entry.alias if alias is None else alias
    ModelRegistry.validate_alias(alias)
    if verify_sha256:
        _verify_digest(entry)
    (registry or ModelRegistry()).add(alias, entry.path)
    return entry
