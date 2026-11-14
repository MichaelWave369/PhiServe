"""R15: read-only, opt-in PhiStream/PhiServe hardware evidence bridge.

A hardware snapshot and a GGUF benchmark are observations, NOT permission to
start inference, proof of GPU execution, a model-quality score, or a resource
allocation order. This module never starts a model or connects to the network.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from .benchmark import BenchmarkError, _hash_file, read_receipt
from .registry import ModelRegistry, RegistryError

SCHEMA = "PHISERVE_HARDWARE_ADVICE_R15_V1"
SOURCE_SCHEMA = "PHISTREAM_HARDWARE_SNAPSHOT_V1"
STATUS = "REVIEW_ONLY_NOT_ROUTING_AUTHORITY"
FALSE_ASSURANCES = {
    "inference_started": False,
    "automatic_model_route": False,
    "execution_authorized": False,
    "gpu_execution_verified": False,
    "physical_hardware_attested": False,
    "performance_improvement_certified": False,
    "model_quality_evaluated": False,
    "remote_upload_performed": False,
}
SHA = re.compile(r"[0-9a-f]{64}\Z")


class AdviceError(ValueError):
    """Evidence cannot support the declared advisory comparison."""


def _need(predicate: bool, reason: str) -> None:
    if not predicate:
        raise AdviceError(reason)


def _whole(value, label: str, minimum=0, maximum=2 ** 60) -> int:
    _need(type(value) is int and minimum <= value <= maximum, f"Invalid {label}")
    return value


def _finite(value, label: str) -> float:
    _need(type(value) in (int, float) and math.isfinite(value) and value >= 0,
          f"Invalid {label}")
    return float(value)


def _fields(value, required: set[str], label: str) -> None:
    _need(type(value) is dict and set(value) == required, f"Invalid {label} fields")


def _duplicates(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def _read_bounded(path: str | Path, limit: int) -> tuple[dict, str]:
    p = Path(path).expanduser()
    _need(p.is_file() and not p.is_symlink(), "Input must be a local regular non-symlink file")
    before = p.stat()
    _need(2 <= before.st_size <= limit, "Input size outside configured bound")
    try:
        raw = p.read_bytes()
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_duplicates,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              AdviceError("Nonfinite JSON not accepted")))
    except (OSError, UnicodeError, ValueError) as exc:
        raise AdviceError("Input is not valid bounded UTF-8 JSON") from exc
    after = p.stat()
    _need((len(raw), before.st_size, before.st_ino, before.st_mtime_ns) ==
          (after.st_size, after.st_size, after.st_ino, after.st_mtime_ns),
          "Input changed during reading")
    _need(type(data) is dict, "JSON input must be an object")
    return data, hashlib.sha256(raw).hexdigest()


def _time(text: str, label: str) -> datetime:
    _need(type(text) is str and 20 <= len(text) <= 48, f"Invalid {label}")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise AdviceError(f"Invalid {label}") from exc
    _need(parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0),
          f"{label} must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc)


def _snapshot(raw: dict) -> dict:
    _fields(raw, {"schema", "status", "provenance", "observed_utc",
                  "platform", "logical_cpu_count", "ram", "gpu",
                  "gpu_requested", "drives", "limitations"}, "PhiStream snapshot")
    _need(raw["schema"] == SOURCE_SCHEMA
          and raw["status"] == "OBSERVED_READ_ONLY"
          and raw["provenance"] == "LOCAL_HARDWARE_COUNTERS_NOT_A_BENCHMARK",
          "Snapshot is not a supported local observation")
    _time(raw["observed_utc"], "hardware snapshot timestamp")
    _need(type(raw["platform"]) is str and 1 <= len(raw["platform"]) <= 256,
          "Invalid platform")
    cpu = raw["logical_cpu_count"]
    _need(cpu is None or (type(cpu) is int and 1 <= cpu <= 4096),
          "Invalid CPU count")
    _fields(raw["ram"], {"total_bytes", "available_bytes", "source"}, "RAM")
    ram = raw["ram"]
    total = _whole(ram["total_bytes"], "total RAM", 1)
    available = _whole(ram["available_bytes"], "available RAM")
    _need(available <= total, "Available RAM exceeds total")
    _need(ram["source"] in {"WIN32_GLOBAL_MEMORY_STATUS_EX",
                              "LINUX_PROC_MEMINFO_MEMAVAILABLE"},
          "Unexpected RAM counter source")
    _need(type(raw["drives"]) is list and len(raw["drives"]) <= 8,
          "Invalid drives metadata")
    for entry in raw["drives"]:
        _fields(entry, {"requested_path", "free_bytes", "total_bytes", "source",
                        "throughput_measured"}, "drive")
        _whole(entry["free_bytes"], "drive free bytes")
        _whole(entry["total_bytes"], "drive total bytes", 1)
        _need(entry["free_bytes"] <= entry["total_bytes"]
              and entry["source"] == "PYTHON_SHUTIL_DISK_USAGE"
              and entry["throughput_measured"] is False
              and type(entry["requested_path"]) is str,
              "Invalid drive observation")
    _need(type(raw["limitations"]) is list and len(raw["limitations"]) <= 20
          and all(type(v) is str and len(v) <= 1024 for v in raw["limitations"]),
          "Malformed snapshot limitations")
    requested = raw["gpu_requested"]
    _need(requested in {"none", "cuda"}, "Unknown GPU probe mode")
    gpu = raw["gpu"]
    if requested == "none":
        _need(gpu is None, "Unexpected GPU counters in non-CUDA snapshot")
        gpu_free = gpu_total = None
    else:
        _fields(gpu, {"device", "name", "device_index", "compute_capability",
                      "total_bytes", "free_bytes", "source", "torch_version",
                      "cuda_build"}, "GPU")
        _need(gpu["device"] == "cuda" and gpu["source"] == "TORCH_CUDA_MEM_GET_INFO",
              "Unsupported GPU observation origin")
        _whole(gpu["device_index"], "CUDA device index", 0, 64)
        for label in ("name", "compute_capability", "torch_version"):
            _need(type(gpu[label]) is str and 1 <= len(gpu[label]) <= 128,
                  f"Invalid GPU {label}")
        _need(gpu["cuda_build"] is None or
              (type(gpu["cuda_build"]) is str and len(gpu["cuda_build"]) <= 128),
              "Invalid CUDA build")
        gpu_total = _whole(gpu["total_bytes"], "total GPU memory", 1)
        gpu_free = _whole(gpu["free_bytes"], "free GPU memory")
        _need(gpu_free <= gpu_total, "Free GPU memory exceeds total")
    return {
        "observed_utc": raw["observed_utc"],
        "platform_family": ("Windows" if raw["platform"].startswith("Windows")
                            else "Linux" if raw["platform"].startswith("Linux")
                            else "UNKNOWN"),
        "available_ram_bytes": available,
        "total_ram_bytes": total,
        "free_cuda_bytes": gpu_free,
        "total_cuda_bytes": gpu_total,
        "cuda_probe": requested,
    }


def _age_state(when: datetime, now: datetime, limit: float) -> tuple[str, float]:
    age = (now - when).total_seconds()
    if age < -60:
        return "HOLD_FUTURE_TIMESTAMP", age
    if age > limit:
        return "HOLD_STALE_EVIDENCE", age
    return "RECENT_OPERATOR_ASSERTED", age


def advise(
    snapshot_file: str | Path, bench_files: list[str | Path],
    model_alias: str, output: str | Path, *,
    registry: ModelRegistry | None = None,
    ram_reserve_mib: int = 4096, gpu_reserve_mib: int = 1024,
    max_snapshot_age_seconds: int = 300,
    max_benchmark_age_seconds: int = 7 * 86400,
    now: datetime | None = None,
) -> dict:
    """Compare only already-observed local evidence, with no inference I/O."""
    _whole(ram_reserve_mib, "RAM reserve MiB", 0, 1024 * 1024)
    _whole(gpu_reserve_mib, "GPU reserve MiB", 0, 1024 * 1024)
    _whole(max_snapshot_age_seconds, "snapshot freshness window", 30, 3600)
    _whole(max_benchmark_age_seconds, "benchmark freshness window", 60, 30 * 86400)
    _need(type(bench_files) is list and 1 <= len(bench_files) <= 8,
          "Supply 1 to 8 existing benchmark receipts")
    target = Path(output).expanduser()
    _need(not target.exists() and not target.is_symlink(), "Output already exists")
    unique = [Path(p).expanduser().resolve(strict=True) for p in [snapshot_file, *bench_files]]
    _need(len(set(unique)) == len(unique), "Evidence input paths must be different")
    _need(target.resolve() not in unique, "Output cannot equal an input")
    registry = registry or ModelRegistry()
    try:
        model_path = registry.get(model_alias)
    except RegistryError as exc:
        raise AdviceError("Exact model alias must already be locally registered") from exc
    digest = _hash_file(model_path)
    size_bytes = model_path.stat().st_size
    snap, snap_digest = _read_bounded(snapshot_file, 256 * 1024)
    facts = _snapshot(snap)
    clock = now or datetime.now(timezone.utc)
    _need(clock.tzinfo is not None, "Comparison clock must be timezone aware")
    clock = clock.astimezone(timezone.utc)
    s_state, s_age = _age_state(_time(facts["observed_utc"], "snapshot time"),
                               clock, max_snapshot_age_seconds)
    ram_budget = max(0, facts["available_ram_bytes"] - ram_reserve_mib * 1024**2)
    gpu_budget = (None if facts["free_cuda_bytes"] is None else
                  max(0, facts["free_cuda_bytes"] - gpu_reserve_mib * 1024**2))
    profiles = []
    ref_experiment = ref_settings = None
    seen_layers: set[int] = set()
    all_recent = True
    for path in bench_files:
        try:
            bench = read_receipt(path)
        except (BenchmarkError, OSError, ValueError) as exc:
            raise AdviceError("Invalid PhiServe benchmark evidence") from exc
        model = bench["model"]
        _need(model["alias"] == model_alias
              and model["gguf_sha256"] == digest
              and model["size_bytes"] == size_bytes,
              "Benchmark model does not match actual registered GGUF bytes")
        experiment = bench["experiment"]
        settings = {k: bench["requested"][k]
                    for k in ("ctx_size", "max_tokens", "seed", "temperature")}
        if ref_experiment is None:
            ref_experiment, ref_settings = experiment, settings
        _need(experiment == ref_experiment and settings == ref_settings,
              "Benchmarks have different prompt, sample or context settings")
        _need(facts["platform_family"] != "UNKNOWN"
              and bench["runtime"]["platform_family"] == facts["platform_family"],
              "Benchmark/snapshot OS families disagree; no cross-device inference")
        layers = bench["requested"]["gpu_layers"]
        _need(layers not in seen_layers, "Duplicate GPU layer profiles are not compared")
        seen_layers.add(layers)
        state, age = _age_state(_time(bench["created_utc"], "benchmark time"),
                                clock, max_benchmark_age_seconds)
        all_recent = all_recent and state == "RECENT_OPERATOR_ASSERTED"
        profiles.append({
            "gpu_layers_requested": layers,
            "source_receipt_sha256": _hash_file(Path(path).expanduser()),
            "source_age_seconds": round(age, 3),
            "source_status": state,
            "wall_p50_ms": bench["summary"]["p50_wall_ms"],
            "wall_p95_ms": bench["summary"]["p95_wall_ms"],
            "startup_ms": bench["runtime"]["startup_ms"],
            "completion_tps_p50_reported": bench["summary"]["p50_completion_tps"],
            "reported_token_count_trials": bench["summary"]["trials_with_reported_token_counts"],
            "full_gpu_weight_file_floor_met": (
                None if gpu_budget is None else size_bytes <= gpu_budget),
            "gpu_execution_independently_verified": False,
        })
    if s_state != "RECENT_OPERATOR_ASSERTED":
        classification = "HOLD_HARDWARE_OBSERVATION_STALE_OR_FUTURE"
    elif not all_recent:
        classification = "HOLD_BENCHMARK_EVIDENCE_STALE_OR_FUTURE"
    elif any(p["gpu_layers_requested"] > 0 for p in profiles) and gpu_budget is None:
        classification = "HOLD_CUDA_COUNTERS_MISSING"
    else:
        classification = "REVIEW_REQUIRED_CAPACITY_NOT_CONFIRMED"
    advice = {
        "schema": SCHEMA, "status": STATUS,
        "evaluated_utc": clock.isoformat(),
        "classification": classification,
        "source": {
            "hardware_snapshot_schema": SOURCE_SCHEMA,
            "hardware_snapshot_file_sha256": snap_digest,
            "hardware_snapshot_age_seconds": round(s_age, 3),
            "hardware_snapshot_status": s_state,
            "model_alias": model_alias,
            "local_gguf_sha256": digest,
            "local_gguf_file_bytes": size_bytes,
        },
        "capacity_floor": {
            "ram_available_bytes_observed": facts["available_ram_bytes"],
            "ram_reserve_bytes_selected": ram_reserve_mib * 1024**2,
            "ram_budget_bytes": ram_budget,
            "gpu_free_bytes_observed": facts["free_cuda_bytes"],
            "gpu_reserve_bytes_selected": gpu_reserve_mib * 1024**2,
            "gpu_budget_bytes": gpu_budget,
            "full_cpu_weight_file_floor_met": size_bytes <= ram_budget,
            "full_gpu_weight_file_floor_met": (
                None if gpu_budget is None else size_bytes <= gpu_budget),
            "actual_model_working_set_estimated": False,
            "actual_model_fit_verified": False,
            "hybrid_offload_fit_evaluated": False,
        },
        "benchmark_profiles": sorted(profiles, key=lambda p: p["gpu_layers_requested"]),
        "comparison_limits": [
            "All input evidence is local and self-reported, not authenticated machine provenance.",
            "PhiStream counters are snapshots and may change before inference.",
            "GGUF file bytes are only a lower-bound heuristic for full weight residency.",
            "Partial GPU offload cannot be inferred from full-model file-size comparisons.",
            "Benchmark p50 throughput is end-to-end and server-token-count dependent.",
            "No power, quality, OS memory peak, independent GPU proof or model fit is measured.",
            "No live admission control, dynamic scheduling, or model promotion is enabled.",
        ],
        "assurances": dict(FALSE_ASSURANCES),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(advice, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return advice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phiserve-advisor",
        description="Offline PhiStream snapshot + PhiServe GGUF benchmark evidence review",
    )
    parser.add_argument("--snapshot", required=True, help="Existing PhiStream hardware snapshot JSON")
    parser.add_argument("--benchmark", required=True, action="append",
                        help="One to eight existing PhiServe benchmark JSON receipts")
    parser.add_argument("--model", required=True, help="Registered exact GGUF alias")
    parser.add_argument("--output", required=True, help="New local report file, never overwritten")
    parser.add_argument("--ram-reserve-mib", type=int, default=4096)
    parser.add_argument("--gpu-reserve-mib", type=int, default=1024)
    parser.add_argument("--max-snapshot-age-seconds", type=int, default=300)
    parser.add_argument("--max-benchmark-age-seconds", type=int, default=7 * 86400)
    args = parser.parse_args(argv)
    try:
        result = advise(args.snapshot, args.benchmark, args.model, args.output,
                        ram_reserve_mib=args.ram_reserve_mib,
                        gpu_reserve_mib=args.gpu_reserve_mib,
                        max_snapshot_age_seconds=args.max_snapshot_age_seconds,
                        max_benchmark_age_seconds=args.max_benchmark_age_seconds)
    except (AdviceError, OSError, ValueError, BenchmarkError) as exc:
        print(f"Advisory refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"classification": result["classification"],
                      "profiles": result["benchmark_profiles"],
                      "report": str(Path(args.output).expanduser()),
                      "execution_authorized": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
