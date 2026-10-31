"""Fail-closed PhiStream reference gate for an explicitly chosen local tokenizer.

This is an offline research command, not a chat API or an independently
authenticated reference. The heavy comparison is delegated to PhiStream's
existing real-model-gate, with checkpoint hash and memory preflight.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from .phistream_lab import PhiStreamCatalog, PhiStreamLabError
from .phistream_text import encode_prompt

_MAX_TRACE_BYTES = 8 * 1024 * 1024
_MAX_REPORT_BYTES = 8 * 1024 * 1024
_MIN_REFERENCE_STEPS = 32


def _local_sha256(file: str | Path, size_limit: int) -> str:
    path = Path(file).expanduser()
    if path.is_symlink() or not path.is_file():
        raise PhiStreamLabError("Reference trace must be a local regular file, not a symlink")
    before = path.stat()
    if not 1 <= before.st_size <= size_limit:
        raise PhiStreamLabError("Reference trace has an invalid or oversized size")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    after = path.stat()
    if (after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ino != before.st_ino):
        raise PhiStreamLabError("Reference file changed during hashing")
    return digest.hexdigest()


def preflight_reference(
    catalog: PhiStreamCatalog, alias: str, *,
    tokenizer_path: str | Path, prompt: str, trace_path: str | Path,
    add_special_tokens: bool = False,
) -> tuple[dict, dict]:
    """Validate local trace, IDs and format; never execute tensors or hash weights.

    The reference declares its origin and fingerprint, but this phase does NOT
    verify weight bytes or the actual reference implementation's independence.
    """
    target = Path(trace_path).expanduser()
    trace_sha = _local_sha256(target, _MAX_TRACE_BYTES)
    checkpoint, _metadata = catalog.get(alias)
    try:
        from phistream.safeweights import LocalCheckpoint
        from phistream.reference import read_reference_trace, validate_reference_trace
    except ImportError as exc:
        raise PhiStreamLabError("PhiStream must be locally installed for reference validation") from exc
    try:
        trace = read_reference_trace(target)
        validated = validate_reference_trace(trace, LocalCheckpoint(checkpoint))
    except (ValueError, TypeError, OSError, KeyError) as exc:
        raise PhiStreamLabError("PhiStream rejected the reference trace or its local checkpoint metadata") from exc
    if trace_sha != _local_sha256(target, _MAX_TRACE_BYTES):
        raise PhiStreamLabError("Reference trace changed while validating")
    prompt_ids = validated["prompt"]
    reference_ids = validated["expected_ids"]
    if (not isinstance(prompt_ids, list) or not isinstance(reference_ids, list)
            or not 1 <= len(reference_ids) <= 128):
        raise PhiStreamLabError("Malformed reference token arrays")
    _, encoded_ids, token_report = encode_prompt(
        catalog, alias, tokenizer_path, prompt,
        tokens=len(reference_ids), add_special_tokens=add_special_tokens,
    )
    if encoded_ids != prompt_ids:
        raise PhiStreamLabError(
            "Tokenizer prompt IDs differ from the external reference; "
            "check tokenizer, normalization, special-token handling and prompt"
        )
    reference_origin = validated["origin"]
    if not isinstance(reference_origin, dict) or not isinstance(reference_origin.get("runtime"), str):
        raise PhiStreamLabError("Missing self-declared reference runtime")
    report = {
        "schema": "PHISERVE_PHISTREAM_REFERENCE_PREFLIGHT_V1",
        "status": "USER_REFERENCE_TOKENIZATION_MATCH_METADATA_ONLY",
        "model_alias": alias,
        "prompt_token_count": len(encoded_ids),
        "reference_steps": len(reference_ids),
        "reference_eligible_for_run": len(reference_ids) >= _MIN_REFERENCE_STEPS,
        "minimum_reference_steps": _MIN_REFERENCE_STEPS,
        "reference_runtime_declared": reference_origin["runtime"],
        "trace_sha256": trace_sha,
        "tokenizer_sha256": token_report["tokenizer_sha256"],
        "special_tokens_applied": add_special_tokens,
        "checkpoint_hashes_verified": False,
        "reference_independence_authenticated": False,
        "tokenizer_checkpoint_identity_verified": False,
        "model_weights_executed": False,
        "chat_service_enabled": False,
    }
    return report, trace


def _validate_execution_report(
    result: dict, trace: dict, preflight: dict, device: str, returncode: int,
) -> dict:
    if not isinstance(result, dict) or result.get("schema") != "PHISTREAM_REAL_MODEL_BRIDGE_V1":
        raise PhiStreamLabError("PhiStream returned an unsupported real-model report")
    status = result.get("status")
    expected = {
        "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED": 0,
        "PARITY_MISMATCH": 2,
    }
    if status not in expected or returncode != expected[status]:
        raise PhiStreamLabError("PhiStream did not complete an eligible parity execution")
    execution = result.get("execution")
    details = result.get("reference_trace")
    fingerprint = result.get("checkpoint_fingerprint")
    if (result.get("model_execution") is not True
            or result.get("real_model_run_authorized") is not True
            or result.get("device_requested") != device
            or not isinstance(execution, dict)
            or execution.get("schema") != "PHISTREAM_REFERENCE_COMPARISON_V1"
            or execution.get("device") != device
            or not isinstance(details, dict)
            or details.get("trace_sha256") != preflight["trace_sha256"]
            or details.get("steps") != preflight["reference_steps"]
            or not isinstance(fingerprint, dict)
            or fingerprint.get("algorithm") != "sha256"
            or fingerprint.get("files") != trace.get("checkpoint_fingerprint", {}).get("files")
            or execution.get("checkpoint_fingerprint") != fingerprint):
        raise PhiStreamLabError("PhiStream parity report failed trace/checkpoint/device binding")
    steps = preflight["reference_steps"]
    matched = execution.get("exact_greedy_tokens_matched")
    probes = execution.get("total_logit_probes")
    passed = execution.get("logit_probes_matched")
    if (type(matched) is not int or not 0 <= matched <= steps
            or type(probes) is not int or probes < steps
            or type(passed) is not int or not 0 <= passed <= probes
            or execution.get("steps") != steps):
        raise PhiStreamLabError("PhiStream parity report has invalid comparison counts")
    perfect = (matched == steps and passed == probes)
    if (status == "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED") != perfect:
        raise PhiStreamLabError("PhiStream status contradicts token/logit comparison counts")
    if execution.get("status") != (
        "MATCH_USER_REFERENCE_UNVERIFIED" if perfect else "MISMATCH"
    ):
        raise PhiStreamLabError("PhiStream nested comparison disagrees with parity status")
    return {
        "status": (
            "USER_ASSERTED_REFERENCE_MATCH_UNAUTHENTICATED"
            if perfect else "REFERENCE_MISMATCH"
        ),
        "exact_greedy_tokens_matched": matched,
        "total_reference_steps": steps,
        "sampled_logit_probes_matched": passed,
        "sampled_logit_probes_total": probes,
    }


def run_reference_gate(
    catalog: PhiStreamCatalog, alias: str, *,
    tokenizer_path: str | Path, prompt: str, trace_path: str | Path,
    device: str, capacity_mib: float, reserve_mib: float,
    output: str | Path, approved: bool,
    add_special_tokens: bool = False,
    timeout_seconds: int = 1800, runner: Callable | None = None,
) -> dict:
    """Compare local PhiStream against user-supplied external trace, opt-in only.

    A matched *user-asserted* trace is still not authenticated external
    independence or promotion to production chat.
    """
    if approved is not True:
        raise PhiStreamLabError("Explicit --approve-local-execution required")
    if device not in {"cpu", "cuda"}:
        raise PhiStreamLabError("Execution device must be cpu or cuda")
    if (type(capacity_mib) not in {float, int} or not math.isfinite(capacity_mib)
            or type(reserve_mib) not in {float, int} or not math.isfinite(reserve_mib)
            or not 0 < capacity_mib <= 1048576
            or not 0 <= reserve_mib < capacity_mib):
        raise PhiStreamLabError("A finite positive memory budget greater than its reserve is required")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7200:
        raise PhiStreamLabError("Timeout must be 1 to 7200 seconds")
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise PhiStreamLabError("Result receipt already exists; refusing overwrite")
    preflight, trace = preflight_reference(
        catalog, alias, tokenizer_path=tokenizer_path, prompt=prompt,
        trace_path=trace_path, add_special_tokens=add_special_tokens,
    )
    if not preflight["reference_eligible_for_run"]:
        raise PhiStreamLabError("Eligible real-model parity requires at least 32 reference steps")
    checkpoint, _metadata = catalog.get(alias)
    metadata = catalog.list()[alias]
    with tempfile.TemporaryDirectory(prefix="phiserve-reference-gate-") as scratch:
        temporary = Path(scratch) / "raw-parity.json"
        argv = [
            sys.executable, "-m", "phistream.cli", "real-model-gate",
            str(checkpoint),
            "--capacity-mib", str(capacity_mib),
            "--reserve-mib", str(reserve_mib),
            "--device", device,
            "--source", metadata["source"],
            "--revision", metadata["revision"],
            "--license-id", metadata["license_id"],
            "--trace", str(Path(trace_path).expanduser().resolve()),
            "--execute-parity", "--approve-local-execution",
            "--output", str(temporary),
        ]
        invoke = subprocess.run if runner is None else runner
        try:
            subprocess_result = invoke(
                argv, shell=False, capture_output=True, text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise PhiStreamLabError("PhiStream reference execution timed out") from exc
        if not temporary.is_file() or temporary.stat().st_size > _MAX_REPORT_BYTES:
            raise PhiStreamLabError("PhiStream produced no valid bounded parity receipt")
        try:
            raw = json.loads(temporary.read_text(encoding="utf-8"))
        except (UnicodeError, OSError, ValueError) as exc:
            raise PhiStreamLabError("PhiStream parity receipt is unreadable") from exc
        result = _validate_execution_report(
            raw, trace, preflight, device, subprocess_result.returncode,
        )
        # Prevent a trace/tokenizer being substituted between preflight and
        # execution; PhiStream independently hashes all checkpoint shards.
        if _local_sha256(trace_path, _MAX_TRACE_BYTES) != preflight["trace_sha256"]:
            raise PhiStreamLabError("Reference trace changed during execution")
        postflight, _ = preflight_reference(
            catalog, alias, tokenizer_path=tokenizer_path,
            prompt=prompt, trace_path=trace_path,
            add_special_tokens=add_special_tokens,
        )
        if (postflight["tokenizer_sha256"] != preflight["tokenizer_sha256"]
                or postflight["trace_sha256"] != preflight["trace_sha256"]):
            raise PhiStreamLabError("Tokenizer or reference changed during execution")
    receipt = {
        "schema": "PHISERVE_PHISTREAM_REFERENCE_GATE_V1",
        **result,
        "model_alias": alias,
        "device_requested": device,
        "tokenizer_sha256": preflight["tokenizer_sha256"],
        "reference_trace_sha256": preflight["trace_sha256"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_count": preflight["prompt_token_count"],
        "reference_runtime_declared": preflight["reference_runtime_declared"],
        "reference_independence_authenticated": False,
        "tokenizer_checkpoint_identity_verified": False,
        "tokenizer_special_tokens_applied": add_special_tokens,
        "full_checkpoint_hashes_compared_by_phistream": True,
        "model_execution_reported_by_phistream": True,
        "gpu_execution_independently_witnessed": False,
        "chat_service_enabled": False,
        "raw_prompt_tokens_or_model_response_stored": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output_file:
        json.dump(receipt, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return receipt
