"""Optional, offline PhiStream safetensors research lane.

This does NOT register checkpoints as GGUF or advertise them on chat APIs.
All weights stay at their original path. Real token-ID execution requires
explicit approval and an independently installed PhiStream checkout.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .registry import ModelRegistry

_MAX_META = 256
_MAX_INPUT_IDS = 2048
_MAX_TOKENS = 128


class PhiStreamLabError(ValueError):
    """Invalid or unsupported experimental checkpoint request."""


def _declared(value: str, label: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= _MAX_META:
        raise PhiStreamLabError(f"{label} requires a nonempty declaration (max 256 characters)")
    if any(ch in value for ch in ("\x00", "\r", "\n")):
        raise PhiStreamLabError(f"{label} must be a single line")
    return value.strip()


def probe_checkpoint(path: str | Path) -> dict:
    """Recheck PhiStream's own strict supported layout, without reading tensors."""
    chosen = Path(path).expanduser()
    if chosen.is_symlink():
        raise PhiStreamLabError("Symlink-selected checkpoint paths are not supported")
    try:
        from phistream.safeweights import LocalCheckpoint
        from phistream.llama import LlamaSettings
    except ImportError as exc:
        raise PhiStreamLabError(
            "PhiStream is not installed in this Python environment. "
            "Install your trusted local PhiStream checkout separately, then retry."
        ) from exc
    ckpt = LocalCheckpoint(chosen)
    settings = LlamaSettings.from_checkpoint(ckpt)
    inspection = ckpt.inspect(llama_layout=True, hash_all=False)
    if inspection.get("status") != "STRUCTURE_VALID":
        raise PhiStreamLabError("PhiStream checkpoint structure is unsupported")
    return {
        "schema": "PHISERVE_PHISTREAM_PREFLIGHT_V1",
        "status": "METADATA_PREFLIGHT_ONLY",
        "format": "SAFETENSORS",
        "backend": "phistream",
        "service_ready": False,
        "tokenizer_available": False,
        "can_serve_chat": False,
        "tensor_count": inspection["tensor_count"],
        "weight_bytes": inspection["weight_bytes"],
        "shard_count": inspection["file_count"],
        "layers": settings.layers,
        "vocab_size": settings.vocab,
        "checkpoint_path_withheld": True,
        "independent_reference_parity_verified": False,
        "gpu_execution_verified": False,
    }


class PhiStreamCatalog:
    """Separate registry to prevent experimental checkpoints entering chat routing."""

    def __init__(self, home: str | Path | None = None):
        location = home if home is not None else os.environ.get("PHISERVE_HOME", "~/.phiserve")
        self.home = Path(location).expanduser().resolve()
        self.file = self.home / "phistream-checkpoints.json"

    def _read(self) -> dict:
        if not self.file.exists():
            return {"schema_version": 1, "checkpoints": {}}
        try:
            content = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PhiStreamLabError("PhiStream catalog is unreadable") from exc
        if (not isinstance(content, dict) or content.get("schema_version") != 1
                or not isinstance(content.get("checkpoints"), dict)):
            raise PhiStreamLabError("Invalid or unsupported PhiStream catalog schema")
        for alias, record in content["checkpoints"].items():
            ModelRegistry.validate_alias(alias)
            if (not isinstance(record, dict)
                    or set(record) != {"path", "source", "revision", "license_id"}
                    or not all(isinstance(value, str) and value for value in record.values())):
                raise PhiStreamLabError("Malformed PhiStream checkpoint record")
        return content

    def _write(self, content: dict) -> None:
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temp = tempfile.mkstemp(dir=self.home, prefix=".phistream-", suffix=".tmp")
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(content, output, sort_keys=True, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, self.file)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def list(self) -> dict:
        return dict(sorted(self._read()["checkpoints"].items()))

    def add(self, alias: str, path: str | Path, *, source: str,
            revision: str, license_id: str) -> dict:
        ModelRegistry.validate_alias(alias)
        source = _declared(source, "source")
        revision = _declared(revision, "revision")
        license_id = _declared(license_id, "license-id")
        if Path(path).expanduser().is_symlink():
            raise PhiStreamLabError("Symlink-selected checkpoint paths are not supported")
        target = Path(path).expanduser().resolve(strict=True)
        receipt = probe_checkpoint(target)
        data = self._read()
        if alias in data["checkpoints"]:
            raise PhiStreamLabError(f"Already registered: {alias}")
        data["checkpoints"][alias] = {
            "path": str(target), "source": source,
            "revision": revision, "license_id": license_id,
        }
        self._write(data)
        return receipt

    def get(self, alias: str) -> tuple[Path, dict]:
        ModelRegistry.validate_alias(alias)
        record = self._read()["checkpoints"].get(alias)
        if record is None:
            raise PhiStreamLabError(f"Unknown PhiStream research alias: {alias}")
        target = Path(record["path"])
        if target.is_symlink() or not target.exists():
            raise PhiStreamLabError("Registered checkpoint missing or symlinked")
        return target, probe_checkpoint(target)

    def remove(self, alias: str) -> None:
        ModelRegistry.validate_alias(alias)
        data = self._read()
        if alias not in data["checkpoints"]:
            raise PhiStreamLabError(f"Unknown PhiStream research alias: {alias}")
        del data["checkpoints"][alias]
        self._write(data)


def parse_token_ids(value: str, vocab_size: int) -> list[int]:
    if not isinstance(value, str) or not value or len(value) > 20000:
        raise PhiStreamLabError("Provide comma-separated token IDs")
    pieces = value.split(",")
    if not 1 <= len(pieces) <= _MAX_INPUT_IDS:
        raise PhiStreamLabError("Prompt must contain 1-2048 token IDs")
    if any(re.fullmatch(r"(0|[1-9][0-9]*)", x) is None for x in pieces):
        raise PhiStreamLabError("Token IDs must be canonical, nonnegative decimal integers")
    numbers = [int(x) for x in pieces]
    if any(x >= vocab_size for x in numbers):
        raise PhiStreamLabError("Prompt token ID outside checkpoint vocabulary")
    return numbers


def run_ids(catalog: PhiStreamCatalog, alias: str, *, prompt: str, tokens: int,
            device: str, output: str | Path, approved: bool,
            timeout_seconds: int = 900, runner=None) -> dict:
    """Execute explicit token-ID research command; never routed from HTTP."""
    if approved is not True:
        raise PhiStreamLabError("Explicit --approve-local-execution required")
    if type(tokens) is not int or not 1 <= tokens <= _MAX_TOKENS:
        raise PhiStreamLabError("tokens must be 1-128")
    if device not in {"cpu", "cuda"}:
        raise PhiStreamLabError("device must be cpu or cuda")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
        raise PhiStreamLabError("timeout must be 1-3600 seconds")
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise PhiStreamLabError("Receipt already exists; refusing overwrite")
    checkpoint, metadata = catalog.get(alias)
    ids = parse_token_ids(prompt, metadata["vocab_size"])
    cmd = [
        sys.executable, "-m", "phistream.cli", "run-llama-ids", str(checkpoint),
        "--prompt", ",".join(map(str, ids)), "--tokens", str(tokens),
        "--device", device, "--output", str(target),
    ]
    call = runner or subprocess.run
    try:
        result = call(cmd, shell=False, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise PhiStreamLabError("PhiStream token-ID execution timed out") from exc
    if result.returncode != 0 or not target.is_file():
        raise PhiStreamLabError("PhiStream execution failed or did not produce a receipt")
    return {
        "schema": "PHISERVE_PHISTREAM_LAB_HANDOFF_V1",
        "status": "LOCAL_TOKEN_ID_EXECUTION_RECEIPT_CREATED",
        "receipt_file": str(target),
        "model_alias": alias,
        "device_requested": device,
        "gpu_execution_independently_verified": False,
        "external_reference_parity_verified": False,
        "chat_endpoint_enabled": False,
    }
