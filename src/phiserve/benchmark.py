"""Opt-in, operator-controlled GGUF benchmark and fail-closed receipt reviewer.

No network beyond PhiServe's own loopback llama-server. No model download,
public telemetry, automatic promotion, inference scheduling, or GPU attestation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import sys
import time

from . import __version__
from .backend import LlamaBackend
from .field_telemetry import ProcessMemorySampler
from .registry import ModelRegistry

SCHEMA = "PHISERVE_BENCHMARK_FIELD_V1"
STATUS = "LOCAL_OPERATOR_OBSERVATION_UNATTESTED"
PROMPT = "Write exactly two short sentences explaining what a computer does."
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_STREAM_BYTES = 4 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
LIMITS = {
    "prompt_or_response_text_recorded": False,
    "cuda_execution_independently_verified": False,
    "physical_hardware_attested": False,
    "quality_evaluated": False,
    "performance_certified": False,
    "external_upload_performed": False,
}


class BenchmarkError(ValueError):
    """A benchmark or its receipt cannot be trusted for this narrow contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkError(message)


def _int(value, label: str, minimum: int, maximum: int) -> int:
    _require(type(value) is int and minimum <= value <= maximum, f"Invalid {label}")
    return value


def _number(value, label: str) -> float:
    _require(type(value) in (float, int) and math.isfinite(value)
             and value >= 0 and value <= 1e9, f"Invalid {label}")
    return float(value)


def _sha(value, label: str) -> str:
    _require(isinstance(value, str) and SHA.fullmatch(value) is not None,
             f"Invalid {label} SHA-256")
    return value


def _keys(obj, expected: set[str], label: str) -> None:
    _require(type(obj) is dict and set(obj) == expected, f"Invalid {label} fields")


def _hash_file(path: Path) -> str:
    before = path.stat()
    _require(path.is_file() and not path.is_symlink(), "Binary must be a regular, non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns, before.st_ino) ==
             (after.st_size, after.st_mtime_ns, after.st_ino),
             "Binary changed while hashing")
    return digest.hexdigest()


def _quantile(values: list[float], percentile: float) -> float:
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def _summary(trials: list[dict]) -> dict:
    walls = [t["wall_ms"] for t in trials]
    first = [t["first_content_ms"] for t in trials]
    known = [t["throughput_tps"] for t in trials if t["throughput_tps"] is not None]
    return {
        "p50_wall_ms": statistics.median(walls),
        "p95_wall_ms": _quantile(walls, .95),
        "p50_first_content_ms": statistics.median(first),
        "p95_first_content_ms": _quantile(first, .95),
        "p50_completion_tps": statistics.median(known) if known else None,
        "trials_with_reported_token_counts": len(known),
    }


def _request_stream(alias: str, port: int, max_tokens: int, timeout_seconds: int) -> dict:
    """Measure SSE first *content*, not a provable device-level first token.

    Completion token count is only accepted when the server reports usage;
    one SSE chunk is never assumed to equal one token.
    """
    body = json.dumps({
        "model": alias,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0, "seed": 42, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode("utf-8")
    start = time.perf_counter_ns()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_seconds)
    first = None
    usage_tokens = None
    char_count = 0
    digest = hashlib.sha256()
    seen_done = False
    bytes_read = 0
    event_count = 0
    try:
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise BenchmarkError(f"Local inference HTTP {response.status}")
        if "text/event-stream" not in (response.getheader("Content-Type", "").lower()):
            raise BenchmarkError("Backend did not return streaming SSE")
        while True:
            line = response.readline(65537)
            if not line:
                break
            now = time.perf_counter_ns()
            bytes_read += len(line)
            _require(bytes_read <= MAX_STREAM_BYTES and len(line) <= 65536,
                     "Streaming response exceeded limit")
            if not line.startswith(b"data:"):
                continue
            event_count += 1
            _require(event_count <= 32768, "Too many streamed events")
            payload = line[5:].strip()
            if payload == b"[DONE]":
                seen_done = True
                end = now
                break
            try:
                event = json.loads(payload)
            except (ValueError, UnicodeError) as exc:
                raise BenchmarkError("Invalid SSE JSON event") from exc
            _require(type(event) is dict, "Invalid SSE payload")
            usage = event.get("usage")
            if usage is not None:
                _require(type(usage) is dict, "Invalid usage")
                tokens = usage.get("completion_tokens")
                if tokens is not None:
                    _int(tokens, "reported completion tokens", 1, 100000)
                    _require(usage_tokens is None or usage_tokens == tokens,
                             "Conflicting completion token counts")
                    usage_tokens = tokens
            for choice in event.get("choices", []):
                _require(type(choice) is dict, "Invalid stream choice")
                delta = choice.get("delta", {})
                _require(type(delta) is dict, "Invalid stream delta")
                content = delta.get("content")
                if content is None or content == "":
                    continue
                _require(type(content) is str, "Non-text response")
                if first is None:
                    first = now
                char_count += len(content)
                _require(char_count <= 1000000, "Response content too long")
                digest.update(content.encode("utf-8"))
        _require(seen_done and first is not None and char_count > 0,
                 "No complete, nonempty SSE text response")
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        raise BenchmarkError("Local inference stream failed or timed out") from exc
    finally:
        conn.close()
    wall = (end - start) / 1e6
    first_ms = (first - start) / 1e6
    _require(wall > 0 and first_ms <= wall, "Nonmonotonic timing observation")
    return {
        "wall_ms": wall,
        "first_content_ms": first_ms,
        "generated_chars": char_count,
        "completion_tokens_reported": usage_tokens,
        "throughput_tps": (usage_tokens / (wall / 1000)) if usage_tokens else None,
        "response_sha256": digest.hexdigest(),
        "end_marker_observed": True,
    }


def _save_new(path: str | Path, receipt: dict) -> None:
    target = Path(path).expanduser()
    _require(not target.is_symlink() and not target.exists(),
             "Output exists or is a symlink; refusing overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def run_benchmark(alias: str, executable: str, output: str | Path, *,
                  gpu_layers: int = 0, ctx_size: int = 2048,
                  max_tokens: int = 64, warmups: int = 3, trials: int = 10,
                  backend_port: int = 11436, timeout_seconds: int = 180,
                  startup_timeout: int = 120, sample_nvidia: bool = False,
                  registry: ModelRegistry | None = None,
                  backend_cls=None, sampler=None, requester=None) -> dict:
    """Run one sequential, fixed-prompt local benchmark. No synthetic fallback."""
    ModelRegistry.validate_alias(alias)
    for value, name, lo, hi in (
        (gpu_layers, "GPU layers", 0, 1024), (ctx_size, "context", 128, 131072),
        (max_tokens, "max tokens", 8, 512), (warmups, "warmups", 0, 10),
        (trials, "trials", 3, 30), (backend_port, "port", 1, 65535),
        (timeout_seconds, "request timeout", 1, 900),
        (startup_timeout, "startup timeout", 1, 900),
    ):
        _int(value, name, lo, hi)
    _require(type(sample_nvidia) is bool, "Invalid sample_nvidia")
    _require(not Path(output).expanduser().exists() and
             not Path(output).expanduser().is_symlink(), "Receipt already exists")
    model = (registry or ModelRegistry()).get(alias)
    binary = shutil.which(executable)
    _require(binary is not None, "llama-server executable not found")
    binpath = Path(binary)
    model_digest = _hash_file(model)
    binary_digest = _hash_file(binpath)
    model_stat = model.stat()
    backend = (backend_cls or LlamaBackend)(
        executable=executable, model_path=model, alias=alias, port=backend_port,
        ctx_size=ctx_size, gpu_layers=gpu_layers, startup_timeout=startup_timeout,
    )
    gpu_sampler = sampler if sampler is not None else (
        ProcessMemorySampler() if sample_nvidia else None)
    request = requester or _request_stream
    measured = []
    telemetry = None
    load_begin = time.perf_counter_ns()
    with backend:
        startup_ms = (time.perf_counter_ns() - load_begin) / 1e6
        if gpu_sampler is not None:
            process = getattr(backend, "process", None)
            pid = getattr(process, "pid", None)
            _int(pid, "backend PID", 1, 2 ** 31 - 1)
            gpu_sampler.start(pid)
        try:
            for i in range(warmups + trials):
                trial = request(alias, backend_port, max_tokens, timeout_seconds)
                if i >= warmups:
                    measured.append({"index": i - warmups + 1, **trial})
        finally:
            if gpu_sampler is not None:
                telemetry = gpu_sampler.stop()
    _require(model.stat().st_size == model_stat.st_size and
             model.stat().st_mtime_ns == model_stat.st_mtime_ns,
             "Model file changed during benchmark")
    receipt = {
        "schema": SCHEMA, "status": STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {"name": "fixed-public-prompt-v1",
                       "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                       "warmups": warmups, "trials": trials},
        "model": {"alias": alias, "gguf_sha256": model_digest,
                  "size_bytes": model_stat.st_size},
        "runtime": {"binary_name": binpath.name, "binary_sha256": binary_digest,
                    "phiserve_version": __version__,
                    "platform_family": platform.system(),
                    "startup_ms": startup_ms},
        "requested": {"gpu_layers": gpu_layers, "ctx_size": ctx_size,
                      "max_tokens": max_tokens, "seed": 42, "temperature": 0,
                      "backend_port": backend_port},
        "measurements": measured,
        "summary": _summary(measured),
        "gpu_process_telemetry": telemetry,
        "unverified_limits": dict(LIMITS),
    }
    validate_receipt(receipt)
    _save_new(output, receipt)
    return receipt


def _no_dupes(pairs):
    out = {}
    for k, v in pairs:
        _require(k not in out, "Duplicate receipt JSON field")
        out[k] = v
    return out


def read_receipt(path: str | Path) -> dict:
    p = Path(path).expanduser()
    _require(p.is_file() and not p.is_symlink(), "Receipt is not a local regular file")
    before = p.stat()
    _require(2 <= before.st_size <= MAX_RECEIPT_BYTES, "Receipt size invalid")
    try:
        receipt = json.loads(p.read_text(encoding="utf-8"),
                             object_pairs_hook=_no_dupes,
                             parse_constant=lambda x: (_ for _ in ()).throw(
                                 BenchmarkError("Non-finite JSON value")))
    except (OSError, UnicodeError, ValueError) as exc:
        raise BenchmarkError("Receipt JSON invalid") from exc
    after = p.stat()
    _require((before.st_size, before.st_mtime_ns, before.st_ino) ==
             (after.st_size, after.st_mtime_ns, after.st_ino),
             "Receipt changed while reading")
    validate_receipt(receipt)
    return receipt


def validate_receipt(r: dict) -> None:
    _keys(r, {"schema", "status", "created_utc", "experiment", "model", "runtime",
              "requested", "measurements", "summary", "gpu_process_telemetry",
              "unverified_limits"}, "receipt")
    _require(r["schema"] == SCHEMA and r["status"] == STATUS, "Receipt is not a field observation")
    _require(isinstance(r["created_utc"], str) and
             len(r["created_utc"]) <= 50 and r["created_utc"].endswith("+00:00"),
             "Invalid UTC timestamp")
    _keys(r["experiment"], {"name", "prompt_sha256", "warmups", "trials"}, "experiment")
    e = r["experiment"]
    _require(e["name"] == "fixed-public-prompt-v1" and
             e["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest(),
             "Different or unrecognized benchmark prompt")
    _int(e["warmups"], "warmups", 0, 10)
    n = _int(e["trials"], "trials", 3, 30)
    _keys(r["model"], {"alias", "gguf_sha256", "size_bytes"}, "model")
    ModelRegistry.validate_alias(r["model"]["alias"])
    _sha(r["model"]["gguf_sha256"], "GGUF")
    _int(r["model"]["size_bytes"], "model size", 4, 2 ** 60)
    _keys(r["runtime"], {"binary_name", "binary_sha256", "phiserve_version",
                         "platform_family", "startup_ms"}, "runtime")
    rt = r["runtime"]
    _require(isinstance(rt["binary_name"], str) and 1 <= len(rt["binary_name"]) <= 128
             and rt["binary_name"] == Path(rt["binary_name"]).name
             and '\\' not in rt["binary_name"], "Invalid binary name")
    _sha(rt["binary_sha256"], "runtime binary")
    _number(rt["startup_ms"], "model startup duration")
    for field in ("phiserve_version", "platform_family"):
        _require(isinstance(rt[field], str) and 1 <= len(rt[field]) <= 128
                 and not any(c in rt[field] for c in "\r\n\0"), "Invalid runtime metadata")
    _keys(r["requested"], {"gpu_layers", "ctx_size", "max_tokens", "seed",
                           "temperature", "backend_port"}, "request")
    cfg = r["requested"]
    for field, lo, hi in (("gpu_layers", 0, 1024), ("ctx_size", 128, 131072),
                          ("max_tokens", 8, 512), ("backend_port", 1, 65535)):
        _int(cfg[field], field, lo, hi)
    _require(cfg["seed"] == 42 and type(cfg["seed"]) is int and
             cfg["temperature"] == 0 and type(cfg["temperature"]) is int,
             "Unexpected sampling settings")
    items = r["measurements"]
    _require(type(items) is list and len(items) == n, "Trial count mismatch")
    for i, t in enumerate(items, 1):
        _keys(t, {"index", "wall_ms", "first_content_ms", "generated_chars",
                  "completion_tokens_reported", "throughput_tps", "response_sha256",
                  "end_marker_observed"}, "trial")
        _require(t["index"] == i and type(t["index"]) is int, "Trial order mismatch")
        wall = _number(t["wall_ms"], "wall duration")
        first = _number(t["first_content_ms"], "first-content duration")
        _require(wall > 0 and first <= wall, "Contradictory timings")
        _int(t["generated_chars"], "output chars", 1, 1000000)
        _sha(t["response_sha256"], "response")
        _require(t["end_marker_observed"] is True, "Incomplete stream")
        tokens = t["completion_tokens_reported"]
        rate = t["throughput_tps"]
        if tokens is None:
            _require(rate is None, "Cannot calculate token rate without reported token count")
        else:
            _int(tokens, "reported completion tokens", 1, 100000)
            _require(abs(_number(rate, "tokens per second") - tokens * 1000 / wall)
                     <= max(1e-7, tokens * 1000 / wall * 1e-9),
                     "Token rate is inconsistent with observed duration")
    _keys(r["summary"], {"p50_wall_ms", "p95_wall_ms",
                          "p50_first_content_ms", "p95_first_content_ms",
                          "p50_completion_tps", "trials_with_reported_token_counts"},
          "summary")
    expected = _summary(items)
    for key, val in expected.items():
        actual = r["summary"][key]
        if val is None:
            _require(actual is None, f"Invalid summary {key}")
        elif key == "trials_with_reported_token_counts":
            _require(type(actual) is int and actual == val, f"Invalid summary {key}")
        else:
            _require(abs(_number(actual, key) - val) <= max(1e-7, abs(val) * 1e-9),
                     f"Invalid summary {key}")
    _keys(r["unverified_limits"], set(LIMITS), "evidence limits")
    _require(r["unverified_limits"] == LIMITS, "Unverified evidence falsely promoted")
    telemetry = r["gpu_process_telemetry"]
    if telemetry is not None:
        _require(type(telemetry) is dict and
                 telemetry.get("schema") == "PHISERVE_NVIDIA_PROCESS_SAMPLE_V1" and
                 telemetry.get("cuda_kernel_execution_independently_verified") is False and
                 telemetry.get("physical_hardware_witnessed") is False and
                 telemetry.get("gpu_model_weights_residency_verified") is False,
                 "GPU telemetry is not trustworthy")
        _require(telemetry.get("status") in {
            "PROCESS_MEMORY_DRIVER_SELF_REPORT", "PROCESS_PRESENT_MEMORY_UNAVAILABLE",
            "NVIDIA_SMI_UNAVAILABLE", "NVIDIA_SMI_QUERY_FAILED",
            "NO_MATCHING_COMPUTE_PROCESS_OBSERVED", "SAMPLER_DID_NOT_STOP",
        }, "Invalid GPU telemetry status")


def compare_receipts(first: dict, second: dict) -> dict:
    validate_receipt(first)
    validate_receipt(second)
    for field in ("gguf_sha256", "size_bytes"):
        _require(first["model"][field] == second["model"][field],
                 "Not comparable: different GGUF model bytes")
    for field in ("ctx_size", "max_tokens", "seed", "temperature"):
        _require(first["requested"][field] == second["requested"][field],
                 f"Not comparable: different {field}")
    _require(first["experiment"] == second["experiment"],
             "Not comparable: different experiment configuration")
    return {
        "status": "DESCRIPTIVE_COMPARISON_ONLY",
        "model_sha256": first["model"]["gguf_sha256"],
        "candidate_a": {"gpu_layers_requested": first["requested"]["gpu_layers"],
                        "platform_family": first["runtime"]["platform_family"],
                        **first["summary"]},
        "candidate_b": {"gpu_layers_requested": second["requested"]["gpu_layers"],
                        "platform_family": second["runtime"]["platform_family"],
                        **second["summary"]},
        "hardware_or_runtime_parity_attested": False,
        "quality_equivalence_established": False,
        "performance_improvement_certified": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="phiserve-benchmark", description="Opt-in local inference bench and receipt review")
    sub = p.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="Run controlled local benchmark against registered GGUF")
    run.add_argument("alias")
    run.add_argument("--llama-server", default=os.environ.get("PHISERVE_LLAMA_SERVER", "llama-server"))
    run.add_argument("--output", required=True)
    run.add_argument("--gpu-layers", type=int, default=0)
    run.add_argument("--ctx-size", type=int, default=2048)
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument("--warmups", type=int, default=3)
    run.add_argument("--trials", type=int, default=10)
    run.add_argument("--backend-port", type=int, default=11436)
    run.add_argument("--timeout-seconds", type=int, default=180)
    run.add_argument("--startup-timeout", type=int, default=120)
    run.add_argument("--sample-nvidia", action="store_true", help="Opt-in local driver self-report")
    review = sub.add_parser("review", help="Validate one bounded receipt without executing weights")
    review.add_argument("receipt")
    compare = sub.add_parser("compare", help="Descriptive matching-model comparison (no certification)")
    compare.add_argument("first")
    compare.add_argument("second")
    a = p.parse_args(argv)
    try:
        if a.action == "run":
            r = run_benchmark(a.alias, a.llama_server, a.output,
                              gpu_layers=a.gpu_layers, ctx_size=a.ctx_size,
                              max_tokens=a.max_tokens, warmups=a.warmups,
                              trials=a.trials, backend_port=a.backend_port,
                              timeout_seconds=a.timeout_seconds,
                              startup_timeout=a.startup_timeout,
                              sample_nvidia=a.sample_nvidia)
            print(json.dumps(r["summary"], indent=2))
            print("Local self-observation only; no GPU or performance attestation.")
        elif a.action == "review":
            r = read_receipt(a.receipt)
            print(json.dumps({"status": "STRUCTURALLY_VALID_NOT_ATTESTED",
                              "model_sha256": r["model"]["gguf_sha256"],
                              "summary": r["summary"],
                              "unverified_limits": r["unverified_limits"]}, indent=2))
        else:
            print(json.dumps(compare_receipts(read_receipt(a.first),
                                              read_receipt(a.second)), indent=2))
        return 0
    except (BenchmarkError, OSError, RuntimeError, ValueError) as exc:
        print(f"Benchmark refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
