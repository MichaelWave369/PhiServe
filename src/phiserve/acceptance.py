"""Local evidence collection and conservative acceptance summaries.

This module NEVER trains, runs a model, connects to a remote service, grants
model-serving permissions, or independently authenticates an operator's
receipts. CI may manufacture the same receipt formats; this is not attestation.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .phistream_lab import PhiStreamCatalog, PhiStreamLabError
from .registry import ModelRegistry, RegistryError


MAX_RECEIPT_BYTES = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MAX_IDENTITY_LEN = 256


class AcceptanceError(ValueError):
    """Receipt malformed, unbound, or unsuitable for honest evaluation."""


def _require(flag, message: str) -> None:
    if not flag:
        raise AcceptanceError(message)


def _identity(value, label: str) -> str:
    _require(isinstance(value, str) and 1 <= len(value) <= MAX_IDENTITY_LEN
             and not any(x in value for x in "\x00\r\n"),
             f"Invalid {label}")
    return value


def _digest(value, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"Invalid {label} SHA-256")
    return value


def _positive(value, label: str, minimum: int = 1):
    _require(type(value) is int and value >= minimum, f"Invalid {label}")
    return value


def _unique_fields(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise AcceptanceError(f"Duplicate JSON field {key!r}")
        data[key] = value
    return data


def _read_receipt(path: str | Path):
    selected = Path(path).expanduser()
    _require(not selected.is_symlink() and selected.is_file(),
             "Receipt must be an existing local regular non-symlink file")
    before = selected.stat()
    _require(2 <= before.st_size <= MAX_RECEIPT_BYTES,
             "Receipt is empty or exceeds 1 MiB")
    try:
        raw = selected.read_bytes()
        report = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_fields,
            parse_constant=lambda x: (_ for _ in ()).throw(
                AcceptanceError(f"Non-finite JSON value {x}")),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise AcceptanceError("Receipt JSON invalid or unreadable") from exc
    after = selected.stat()
    _require((len(raw) == before.st_size and before.st_size == after.st_size
              and before.st_mtime_ns == after.st_mtime_ns
              and before.st_ino == after.st_ino),
             "Receipt changed during reading")
    _require(isinstance(report, dict), "Receipt root must be a JSON object")
    return report, hashlib.sha256(raw).hexdigest()


def _write_exclusive(output: str | Path, payload: dict) -> Path:
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise AcceptanceError("Output receipt already exists; refusing overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with target.open("x", encoding="utf-8") as stream:
            created = True
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except Exception:
        # Never unlink a competing process's file if exclusive creation fails.
        if created:
            try:
                target.unlink()
            except OSError:
                pass
        raise
    return target


def _binary_found(binary: str) -> bool:
    if not isinstance(binary, str) or not binary.strip():
        return False
    return bool(shutil.which(binary)) or Path(binary).expanduser().is_file()


def _nvidia_probe(runner=None) -> dict:
    """Opt-in local nvidia-smi snapshot. Not process attribution or CUDA proof."""
    cmd = ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
           "--format=csv,noheader,nounits"]
    invoke = subprocess.run if runner is None else runner
    try:
        result = invoke(cmd, capture_output=True, text=True, timeout=8, shell=False)
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "NOT_AVAILABLE", "device_count": 0, "gpu_devices": []}
    if result.returncode != 0:
        return {"status": "NOT_AVAILABLE", "device_count": 0, "gpu_devices": []}
    devices = []
    try:
        for row in csv.reader(io.StringIO(result.stdout)):
            if not row or all(not part.strip() for part in row):
                continue
            if len(row) != 4 or len(devices) >= 16:
                raise ValueError("Malformed GPU output")
            index = int(row[0].strip())
            ram_mib = int(row[3].strip())
            name = _identity(row[1].strip(), "GPU name")
            driver = _identity(row[2].strip(), "driver version")
            if index < 0 or ram_mib < 1:
                raise ValueError("Invalid GPU metadata")
            devices.append({
                "index": index, "name": name, "driver_version": driver,
                "total_memory_mib_reported": ram_mib,
            })
    except (ValueError, AcceptanceError):
        return {"status": "UNREADABLE", "device_count": 0, "gpu_devices": []}
    return {
        "status": "LOCAL_TOOL_SELF_REPORT" if devices else "NOT_AVAILABLE",
        "device_count": len(devices), "gpu_devices": devices,
    }


def doctor(
    output: str | Path, *, llama_server: str = "llama-server",
    probe_nvidia: bool = False, registry: ModelRegistry | None = None,
    catalog: PhiStreamCatalog | None = None, probe_runner=None,
) -> dict:
    """Run a local, non-inference readiness inventory with no network I/O."""
    _require(type(probe_nvidia) is bool, "probe_nvidia must be boolean")
    if Path(output).expanduser().resolve().exists():
        raise AcceptanceError("Output receipt already exists; refusing overwrite")
    findings = []
    try:
        gguf_count = len((registry or ModelRegistry()).list_models())
    except (RegistryError, OSError, ValueError):
        gguf_count = None
        findings.append("GGUF_REGISTRY_REVIEW_REQUIRED")
    try:
        research_count = len((catalog or PhiStreamCatalog()).list())
    except (PhiStreamLabError, RegistryError, OSError, ValueError):
        research_count = None
        findings.append("PHISTREAM_CATALOG_REVIEW_REQUIRED")
    llama_found = _binary_found(llama_server)
    if not llama_found:
        findings.append("LLAMA_SERVER_BINARY_NOT_FOUND")
    if not gguf_count:
        findings.append("NO_GGUF_MODELS_FOUND_OR_REGISTRY_UNREADABLE")
    if not research_count:
        findings.append("NO_PHISTREAM_RESEARCH_CHECKPOINTS_OR_CATALOG_UNREADABLE")
    phi_installed = importlib.util.find_spec("phistream") is not None
    tokenizer_installed = importlib.util.find_spec("tokenizers") is not None
    if not phi_installed:
        findings.append("PHISTREAM_PACKAGE_NOT_INSTALLED")
    if not tokenizer_installed:
        findings.append("OPTIONAL_TOKENIZER_PACKAGE_NOT_INSTALLED")
    nvidia = (_nvidia_probe(probe_runner) if probe_nvidia else
              {"status": "NOT_REQUESTED", "device_count": 0, "gpu_devices": []})
    if probe_nvidia and nvidia["status"] != "LOCAL_TOOL_SELF_REPORT":
        findings.append("NVIDIA_SMI_LOCAL_PROBE_UNAVAILABLE")
    result = {
        "schema": "PHISERVE_LOCAL_READINESS_V1",
        "status": "LOCAL_ENVIRONMENT_OBSERVATION_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phiserve_version": __version__,
        "platform_family": platform.system(),
        "python_version": platform.python_version(),
        "llama_server_binary_found": llama_found,
        "gguf_registered_count": gguf_count,
        "phistream_research_registered_count": research_count,
        "phistream_python_installed": phi_installed,
        "tokenizers_python_installed": tokenizer_installed,
        "gpu_probe": nvidia,
        "findings": sorted(set(findings)),
        "network_requests_made": False,
        "model_weights_executed": False,
        "gpu_execution_verified": False,
        "machine_identity_authenticated": False,
        "physical_presence_witnessed": False,
    }
    _write_exclusive(output, result)
    return result


def _gguf(result: dict, kind: str) -> dict:
    _require(result.get("schema") == "PHISERVE_LOCAL_INFERENCE_RECEIPT_V1"
             and result.get("status") == "LOCAL_TEXT_OUTPUT_OBSERVED",
             f"{kind} must be a local inference receipt with observed text")
    _require(result.get("raw_prompt_or_output_stored") is False,
             f"{kind} cannot claim a private receipt with raw output stored")
    layers = result.get("gpu_layers_requested")
    _require(type(layers) is int and layers >= 0, f"Invalid {kind} GPU layers")
    _require(layers == 0 if kind == "cpu" else layers > 0,
             f"{kind} receipt has incorrect GPU-layer request")
    telemetry = result.get("gpu_process_telemetry")
    observation = None
    if telemetry is not None:
        _require(isinstance(telemetry, dict)
                 and telemetry.get("schema") == "PHISERVE_NVIDIA_PROCESS_SAMPLE_V1"
                 and telemetry.get("cuda_kernel_execution_independently_verified") is False
                 and telemetry.get("physical_hardware_witnessed") is False,
                 "Untrusted or contradictory GPU process telemetry receipt")
        allowed = {
            "PROCESS_MEMORY_DRIVER_SELF_REPORT", "PROCESS_PRESENT_MEMORY_UNAVAILABLE",
            "NVIDIA_SMI_UNAVAILABLE", "NVIDIA_SMI_QUERY_FAILED",
            "NO_MATCHING_COMPUTE_PROCESS_OBSERVED", "SAMPLER_DID_NOT_STOP",
        }
        status = telemetry.get("status")
        _require(status in allowed, "Unrecognized GPU process telemetry status")
        attempted = _positive(telemetry.get("sample_attempts"), "GPU sampling attempts", 0)
        matched = _positive(telemetry.get("matching_pid_samples"), "matching PID samples", 0)
        numeric = _positive(telemetry.get("numeric_memory_samples"), "numeric GPU memory samples", 0)
        peak = telemetry.get("peak_driver_reported_process_memory_mib")
        _require(matched <= attempted and numeric <= matched
                 and ((type(peak) is int and peak >= 0) if numeric else peak is None)
                 and (status == "PROCESS_MEMORY_DRIVER_SELF_REPORT") == (numeric > 0),
                 "Contradictory GPU process-memory observation counts")
        observation = {
            "status": status,
            "sampling_attempts": attempted,
            "matching_pid_samples": matched,
            "numeric_memory_samples": numeric,
            "peak_driver_reported_process_memory_mib": peak,
            "cuda_execution_independently_verified": False,
        }
    return {
        "gpu_process_observation": observation,
        "model_alias": _identity(result.get("model_alias"), f"{kind} model alias"),
        "model_filename": _identity(result.get("model_filename"), f"{kind} model filename"),
        "model_size_bytes": _positive(result.get("model_size_bytes"), "model size"),
        "prompt_sha256": _digest(result.get("prompt_sha256"), "prompt"),
        "output_sha256": _digest(result.get("output_sha256"), "output"),
        "output_characters": _positive(result.get("output_characters"), "output characters"),
        "gpu_layers_requested": layers,
        "runtime_observation": "USER_SUPPLIED_LOCAL_TEXT_RECEIPT",
        "gpu_execution_independently_verified": False,
    }


def _text(result: dict) -> dict:
    _require(result.get("schema") == "PHISERVE_PHISTREAM_TEXT_LAB_V1"
             and result.get("status") == "LOCAL_EXPERIMENTAL_TEXT_DECODED_UNVERIFIED",
             "Invalid PhiStream text-lab receipt")
    _require(result.get("raw_prompt_or_decoded_output_stored") is False,
             "Text-lab receipt cannot store raw generated text")
    _require(result.get("chat_service_enabled") is False,
             "Production chat cannot be enabled from research receipt")
    _require(type(result.get("tokenizer_special_tokens_applied")) is bool,
             "Invalid text tokenizer special-token setting")
    return {
        "model_alias": _identity(result.get("model_alias"), "research model alias"),
        "tokenizer_sha256": _digest(result.get("tokenizer_sha256"), "tokenizer"),
        "decoded_output_sha256": _digest(result.get("decoded_output_sha256"), "decoded output"),
        "decoded_output_characters": _positive(result.get("decoded_output_characters"), "decoded output length"),
        "tokenizer_special_tokens_applied": result.get("tokenizer_special_tokens_applied") is True,
        "result": "EXPERIMENTAL_DECODED_OUTPUT_UNVERIFIED",
    }


def _parity(result: dict) -> dict:
    _require(result.get("schema") == "PHISERVE_PHISTREAM_REFERENCE_GATE_V1",
             "Invalid PhiStream parity receipt schema")
    status = result.get("status")
    _require(status in {"USER_ASSERTED_REFERENCE_MATCH_UNAUTHENTICATED", "REFERENCE_MISMATCH"},
             "Unsupported PhiStream reference parity outcome")
    _require(result.get("reference_independence_authenticated") is False
             and result.get("chat_service_enabled") is False,
             "Cannot treat unverified reference as authenticated or a chat service")
    steps = _positive(result.get("total_reference_steps"), "reference steps")
    _require(32 <= steps <= 128, "Reference steps must be between 32 and 128")
    matches = result.get("exact_greedy_tokens_matched")
    total_probes = result.get("sampled_logit_probes_total")
    passing_probes = result.get("sampled_logit_probes_matched")
    _require(type(matches) is int and 0 <= matches <= steps, "Invalid matching-token count")
    _require(type(total_probes) is int and total_probes >= steps
             and type(passing_probes) is int and 0 <= passing_probes <= total_probes,
             "Invalid sampled-logit counts")
    all_pass = matches == steps and passing_probes == total_probes
    _require((status == "USER_ASSERTED_REFERENCE_MATCH_UNAUTHENTICATED") == all_pass,
             "Parity status contradicts its comparison counts")
    _require(type(result.get("tokenizer_special_tokens_applied")) is bool,
             "Invalid parity tokenizer special-token setting")
    return {
        "model_alias": _identity(result.get("model_alias"), "research model alias"),
        "tokenizer_sha256": _digest(result.get("tokenizer_sha256"), "tokenizer"),
        "reference_trace_sha256": _digest(result.get("reference_trace_sha256"), "reference trace"),
        "status": status, "reference_steps": steps,
        "greedy_tokens_matched": matches,
        "sampled_logit_probes_matched": passing_probes,
        "sampled_logit_probes_total": total_probes,
        "tokenizer_special_tokens_applied": result.get("tokenizer_special_tokens_applied") is True,
        "independence_authenticated": False,
    }


def _readiness(result: dict) -> dict:
    _require(result.get("schema") == "PHISERVE_LOCAL_READINESS_V1"
             and result.get("status") == "LOCAL_ENVIRONMENT_OBSERVATION_ONLY"
             and result.get("gpu_execution_verified") is False
             and result.get("model_weights_executed") is False,
             "Invalid local environment doctor receipt")
    gpu = result.get("gpu_probe")
    _require(isinstance(gpu, dict) and gpu.get("status") in {
        "NOT_REQUESTED", "NOT_AVAILABLE", "UNREADABLE", "LOCAL_TOOL_SELF_REPORT",
    }, "Invalid device observation")
    return {
        "gpu_probe_status": gpu["status"],
        "gpu_devices_reported": _positive(gpu.get("device_count"), "device count", 0),
        "llama_server_binary_found": result.get("llama_server_binary_found") is True,
        "model_execution_observed": False,
    }


def acceptance_report(
    output: str | Path, *, doctor_receipt=None, cpu_gguf=None,
    gpu_gguf=None, text_receipt=None, parity_receipt=None,
) -> dict:
    """Summarize optional existing local receipts, without ever claiming attestation."""
    paths = {
        "doctor": doctor_receipt, "gguf_cpu": cpu_gguf, "gguf_gpu": gpu_gguf,
        "phistream_text": text_receipt, "phistream_parity": parity_receipt,
    }
    given = {name: path for name, path in paths.items() if path is not None}
    _require(given, "At least one existing receipt is required")
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise AcceptanceError("Output receipt already exists; refusing overwrite")
    normalized = [Path(p).expanduser().resolve(strict=True) for p in given.values()]
    _require(len(set(normalized)) == len(normalized),
             "Each evidence input must be a distinct local receipt")
    _require(output_path not in normalized, "Summary output cannot be an input receipt")
    evidence = {}
    for kind, value in given.items():
        report, source_sha = _read_receipt(value)
        if kind == "doctor":
            minimal = _readiness(report)
        elif kind.startswith("gguf_"):
            minimal = _gguf(report, kind.split("_", 1)[1])
        elif kind == "phistream_text":
            minimal = _text(report)
        else:
            minimal = _parity(report)
        evidence[kind] = {"source_file_sha256": source_sha, **minimal}
    findings = []
    cpu = evidence.get("gguf_cpu")
    gpu = evidence.get("gguf_gpu")
    match = None
    if cpu and gpu:
        _require(
            all(cpu[field] == gpu[field] for field in (
                "model_alias", "model_filename", "model_size_bytes", "prompt_sha256"
            )),
            "CPU and GPU receipts are not comparable: different alias/model filename/size or prompt",
        )
        match = cpu["output_sha256"] == gpu["output_sha256"]
        if not match:
            findings.append("CPU_GPU_OBSERVED_OUTPUT_DIGESTS_DIFFER")
    else:
        findings.append("CPU_GPU_PAIRED_GGUF_OBSERVATION_INCOMPLETE")
    txt = evidence.get("phistream_text")
    parity = evidence.get("phistream_parity")
    if txt and parity:
        _require(txt["model_alias"] == parity["model_alias"]
                 and txt["tokenizer_sha256"] == parity["tokenizer_sha256"]
                 and (txt["tokenizer_special_tokens_applied"]
                      == parity["tokenizer_special_tokens_applied"]),
                 "PhiStream text and parity receipts use different research model/tokenizer settings")
    if parity is None:
        findings.append("PHISTREAM_REFERENCE_GATE_RECEIPT_MISSING")
    elif parity["status"] == "REFERENCE_MISMATCH":
        findings.append("PHISTREAM_REFERENCE_MISMATCH")
    findings.extend([
        "RECEIPTS_ARE_USER_SUPPLIED_NOT_INDEPENDENTLY_AUTHENTICATED",
        "GPU_EXECUTION_NOT_INDEPENDENTLY_WITNESSED",
        "PHISTREAM_PRODUCTION_CHAT_NOT_AUTHORIZED",
    ])
    result = {
        "schema": "PHISERVE_LOCAL_ACCEPTANCE_SUMMARY_V1",
        "status": ("DIAGNOSTIC_MISMATCH_REQUIRES_REVIEW"
                   if ("CPU_GPU_OBSERVED_OUTPUT_DIGESTS_DIFFER" in findings
                       or "PHISTREAM_REFERENCE_MISMATCH" in findings)
                   else "EVIDENCE_COLLECTED_REVIEW_REQUIRED"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phiserve_version": __version__,
        "evidence": evidence,
        "paired_gguf_output_sha256_equal": match,
        "paired_gguf_does_not_prove_same_weights": True,
        "independent_reference_authenticated": False,
        "gpu_execution_independently_verified": False,
        "production_phistream_chat_authorized": False,
        "physical_hardware_claim": "OPERATOR_SUPPLIED_RECEIPTS_UNWITNESSED",
        "findings": sorted(set(findings)),
        "raw_prompt_reference_logits_and_text_omitted": True,
    }
    _write_exclusive(output, result)
    return result
