"""Read-only Phi Model Forge adapter receipt intake. NOT a GGUF importer.

Confirms local file shapes and internal consistency, not source authenticity,
model quality or execution readiness. No weights loaded, changed or uploaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

MAX_JSON = 64 * 1024
MAX_ADAPTER = 512 * 1024 * 1024
SHA = re.compile(r"[a-f0-9]{64}\Z")


class ForgeIntakeError(ValueError):
    """A model candidate failed source/shape preflight."""


def _regular(path: str | Path, max_size: int) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ForgeIntakeError("Candidate must be an existing regular file, not a symlink")
    resolved = candidate.resolve(strict=True)
    if not 0 < resolved.stat().st_size <= max_size:
        raise ForgeIntakeError("File size exceeds bounded intake policy")
    return resolved


def _json_file(path: Path) -> dict[str, Any]:
    content = _regular(path, MAX_JSON).read_text(encoding="utf-8")
    def reject_duplicate(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ForgeIntakeError("Duplicate JSON field")
            result[key] = value
        return result
    try:
        value = json.loads(content, object_pairs_hook=reject_duplicate,
                           parse_constant=lambda _x: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ForgeIntakeError("Malformed JSON evidence") from exc
    if not isinstance(value, dict):
        raise ForgeIntakeError("Expected JSON object")
    return value


def _adapter_header(path: Path) -> tuple[int, int]:
    file = _regular(path, MAX_ADAPTER)
    with file.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ForgeIntakeError("Truncated safetensors header")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size == 0 or header_size > 1024 * 1024 or header_size + 8 >= file.stat().st_size:
            raise ForgeIntakeError("Invalid bounded safetensors header size")
        try:
            header = json.loads(stream.read(header_size))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ForgeIntakeError("Invalid safetensors metadata header") from exc
        if not isinstance(header, dict):
            raise ForgeIntakeError("Safetensors header must be an object")
        available = file.stat().st_size - header_size - 8
        tensors = 0
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not isinstance(entry, dict):
                raise ForgeIntakeError("Malformed tensor entry")
            offsets = entry.get("data_offsets")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(x) is not int for x in offsets)
                    or offsets[0] < 0 or offsets[1] <= offsets[0] or offsets[1] > available):
                raise ForgeIntakeError("Out-of-range tensor offsets")
            tensors += 1
            if tensors > 4096:
                raise ForgeIntakeError("Too many tensors for bounded intake")
        if tensors == 0:
            raise ForgeIntakeError("No tensor entries in adapter")
        return tensors, file.stat().st_size


def inspect(adapter: str | Path, receipt: str | Path, *, hash_weights: bool = False) -> dict[str, Any]:
    directory = Path(adapter).expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise ForgeIntakeError("Choose a regular local adapter directory, not a symlink")
    directory = directory.resolve(strict=True)
    adapter_config = _json_file(directory / "adapter_config.json")
    training = _json_file(receipt)
    if adapter_config.get("peft_type") != "LORA":
        raise ForgeIntakeError("Only PEFT LORA adapter candidates are recognized")
    if adapter_config.get("task_type") != "CAUSAL_LM":
        raise ForgeIntakeError("Adapter task must be CAUSAL_LM")
    if training.get("schema") != "phi.model-forge.training-receipt.v1":
        raise ForgeIntakeError("Unexpected Forge training receipt schema")
    if training.get("status") != "LOCALLY_TRAINED_UNVERIFIED_QUALITY":
        raise ForgeIntakeError("Receipt does not report a completed local training candidate")
    base = adapter_config.get("base_model_name_or_path")
    if not isinstance(base, str) or not base or base != training.get("base_model"):
        raise ForgeIntakeError("Training receipt base model differs from adapter config")
    digest = training.get("dataset_sha256")
    if not isinstance(digest, str) or not SHA.fullmatch(digest):
        raise ForgeIntakeError("Missing dataset fingerprint")
    if training.get("production_authorized") is not False:
        raise ForgeIntakeError("Intake expects explicitly unpromoted candidate")
    weights = directory / "adapter_model.safetensors"
    tensors, size = _adapter_header(weights)
    weights_hash = None
    if hash_weights:
        hasher = hashlib.sha256()
        with weights.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        weights_hash = hasher.hexdigest()
    return {
        "schema": "phiserve.forge-candidate-intake.v1",
        "status": "REVIEW_REQUIRED",
        "source": "local_peft_lora_candidate",
        "base_model_declared": base,
        "dataset_sha256_declared": digest,
        "adapter_tensors_observed": tensors,
        "adapter_bytes_observed": size,
        "adapter_sha256": weights_hash,
        "self_reported_training_only": True,
        "model_output_verified": False,
        "held_out_quality_verified": False,
        "gguf_compatible": False,
        "phi_serve_registered": False,
        "execution_authorized": False,
        "publishing_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="phiserve-forge-review",
        description="Inspect local Phi Model Forge adapter candidate (no serving or conversion)")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--hash-weights", action="store_true", help="Opt in to full adapter-file SHA256")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.adapter, args.receipt, hash_weights=args.hash_weights),
                         indent=2, sort_keys=True))
    except (ForgeIntakeError, OSError) as exc:
        parser.exit(2, f"Forge candidate held: {exc}\n")


if __name__ == "__main__":
    main()
