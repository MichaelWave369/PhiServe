"""R18 opt-in, on-device field campaign for PhiStream -> PhiServe evidence.

Preview is read-only and never starts inference. Collect executes existing local
GGUF benchmarks ONLY with an explicit --execute flag, sequentially, then takes
a fresh PhiStream hardware snapshot and produces an R15 review. No automation,
network beyond benchmark's supervised loopback, model download, or upload.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from .benchmark import BenchmarkError, read_receipt, run_benchmark
from .hardware_advisor import AdviceError, advise
from .registry import ModelRegistry, RegistryError

SCHEMA = "PHISERVE_LOCAL_FIELD_CAMPAIGN_R18_V1"
STATUS = "OPERATOR_EXECUTED_UNATTESTED_REVIEW_ONLY"
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
LIMITS = {
    "actual_hardware_identity_authenticated": False,
    "gpu_execution_independently_verified": False,
    "model_working_set_fit_verified": False,
    "automatic_inference_routing": False,
    "execution_authorized_by_evidence": False,
    "model_quality_evaluated": False,
    "production_ready_certified": False,
    "external_upload_performed": False,
}


class CampaignHeld(ValueError):
    """An operator configuration or receipt cannot safely be used."""


def need(ok, why):
    if not ok:
        raise CampaignHeld(why)


def whole(value, lo, hi, label):
    need(type(value) is int and lo <= value <= hi, "INVALID_" + label)
    return value


def validate_options(*, alias, binary, out_dir, site, layers,
                     gpu_probe, ctx_size, max_tokens, trials, warmups,
                     ram_reserve_mib, gpu_reserve_mib, backend_port,
                     timeout_seconds, startup_timeout, sample_nvidia,
                     registry=None, check_inputs=True):
    ModelRegistry.validate_alias(alias)
    need(type(site) is str and LABEL.fullmatch(site) is not None,
         "INVALID_OPERATOR_SITE_LABEL")
    need(type(layers) in (tuple, list) and 1 <= len(layers) <= 4,
         "EXPECTED_1_TO_4_OFFLOAD_PROFILES")
    for layer in layers:
        whole(layer, 0, 1024, "GPU_LAYERS")
    need(layers[0] == 0 and len(set(layers)) == len(layers),
         "CPU_BASELINE_FIRST_AND_UNIQUE_GPU_LAYERS_REQUIRED")
    need(gpu_probe in ("none", "cuda"), "INVALID_GPU_PROBE")
    need(type(sample_nvidia) is bool, "INVALID_GPU_SAMPLER_CHOICE")
    for value, lo, hi, name in (
        (ctx_size, 128, 131072, "CONTEXT"),
        (max_tokens, 8, 512, "MAX_TOKENS"),
        (trials, 3, 30, "TRIALS"),
        (warmups, 0, 10, "WARMUPS"),
        (ram_reserve_mib, 0, 1024 * 1024, "RAM_RESERVE"),
        (gpu_reserve_mib, 0, 1024 * 1024, "GPU_RESERVE"),
        (backend_port, 1, 65535, "PORT"),
        (timeout_seconds, 1, 900, "TIMEOUT"),
        (startup_timeout, 1, 900, "STARTUP_TIMEOUT"),
    ):
        whole(value, lo, hi, name)
    need(type(binary) is str and len(binary) > 0 and len(binary) <= 2048,
         "INVALID_SERVER_EXECUTABLE")
    target = Path(out_dir).expanduser()
    need(not target.exists() and not target.is_symlink(), "OUTPUT_ALREADY_EXISTS")
    resolved = target.resolve()
    need(not any((parent / ".git").exists() for parent in (resolved, *resolved.parents)),
         "OUTPUT_WITHIN_GIT_CHECKOUT")
    need(target.parent.is_dir() and not target.parent.is_symlink(),
         "OUTPUT_PARENT_MUST_EXIST")
    if check_inputs:
        reg = registry or ModelRegistry()
        reg.get(alias)  # Existing local GGUF magic only; no download/hashing here.
        exe = shutil.which(binary)
        need(exe is not None and Path(exe).is_file() and not Path(exe).is_symlink(),
             "LLAMA_SERVER_NOT_AVAILABLE")
    return {
        "site_label_operator_declared": site,
        "model_alias": alias,
        "gpu_layers_requested": list(layers),
        "gpu_probe": gpu_probe,
        "nvidia_process_sampler_requested": sample_nvidia,
        "ctx_size": ctx_size,
        "max_tokens": max_tokens,
        "warmups": warmups,
        "trials": trials,
        "backend_port": backend_port,
        "request_timeout_seconds": timeout_seconds,
        "startup_timeout_seconds": startup_timeout,
        "ram_reserve_mib": ram_reserve_mib,
        "gpu_reserve_mib": gpu_reserve_mib,
        "status": "PLAN_ONLY_NOT_EXECUTED",
        "requires_explicit_execute": True,
        "source_model_path_exported": False,
        "local_server_path_exported": False,
        "automatic_install": False,
        "real_hardware_measurement_claimed": False,
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def collect(*, alias, binary, out_dir, site, layers=(0, 20), gpu_probe="none",
            ctx_size=2048, max_tokens=64, trials=10, warmups=3,
            ram_reserve_mib=4096, gpu_reserve_mib=1024, backend_port=11436,
            timeout_seconds=180, startup_timeout=120, sample_nvidia=False,
            execute=False, registry=None, benchmark_fn=run_benchmark,
            snapshot_fn=None, advisor_fn=advise):
    """Return a preview unless execute=True; no disk writes or model runs in preview."""
    info = validate_options(
        alias=alias, binary=binary, out_dir=out_dir, site=site, layers=layers,
        gpu_probe=gpu_probe, ctx_size=ctx_size, max_tokens=max_tokens,
        trials=trials, warmups=warmups, ram_reserve_mib=ram_reserve_mib,
        gpu_reserve_mib=gpu_reserve_mib, backend_port=backend_port,
        timeout_seconds=timeout_seconds, startup_timeout=startup_timeout,
        sample_nvidia=sample_nvidia, registry=registry,
    )
    need(type(execute) is bool, "INVALID_EXECUTE_FLAG")
    if not execute:
        return info
    # Fail before benchmark work when PhiStream is unavailable; no silent fake
    # hardware counters. Users intentionally install PhiStream on each host.
    if snapshot_fn is None:
        try:
            snapshot_fn = importlib.import_module("phistream.hardware").hardware_snapshot
        except (ImportError, AttributeError) as exc:
            raise CampaignHeld("PHISTREAM_HARDWARE_SNAPSHOT_UNAVAILABLE") from exc
    reg = registry or ModelRegistry()
    dest = Path(out_dir).expanduser()
    dest.mkdir(mode=0o700)
    completed = []
    try:
        for layer in layers:
            path = dest / ("benchmark-layers-" + str(layer) + ".json")
            observation = benchmark_fn(
                alias, binary, path, gpu_layers=layer, ctx_size=ctx_size,
                max_tokens=max_tokens, warmups=warmups, trials=trials,
                backend_port=backend_port, timeout_seconds=timeout_seconds,
                startup_timeout=startup_timeout,
                sample_nvidia=sample_nvidia and layer > 0, registry=reg,
            )
            # The actual benchmark contract saves a receipt only on success.
            receipt = read_receipt(path)
            need(receipt["summary"] == observation["summary"],
                 "BACKEND_RESULT_RECEIPT_MISMATCH")
            need(receipt["requested"]["gpu_layers"] == layer,
                 "OFFLOAD_PROFILE_RECEIPT_MISMATCH")
            completed.append(path)
        # Observe hardware AFTER potentially long benchmarks, to keep R15's
        # snapshot freshness window meaningful at review time.
        snapshot = snapshot_fn(gpu=gpu_probe, drives=())
        need(type(snapshot) is dict and snapshot.get("schema") ==
             "PHISTREAM_HARDWARE_SNAPSHOT_V1" and
             snapshot.get("status") == "OBSERVED_READ_ONLY",
             "HARDWARE_SNAPSHOT_CONTRACT_INVALID")
        snapshot_path = dest / "hardware-snapshot.json"
        _write_new(snapshot_path, snapshot)
        advice_path = dest / "capacity-advice.json"
        result = advisor_fn(
            snapshot_path, completed, alias, advice_path, registry=reg,
            ram_reserve_mib=ram_reserve_mib, gpu_reserve_mib=gpu_reserve_mib,
        )
        need(result["status"] == "REVIEW_ONLY_NOT_ROUTING_AUTHORITY" and
             result["assurances"]["execution_authorized"] is False,
             "ADVISORY_HAS_INVALID_AUTHORITY")
        receipts = [read_receipt(p) for p in completed]
        model_sha = receipts[0]["model"]["gguf_sha256"]
        model_size = receipts[0]["model"]["size_bytes"]
        for r in receipts[1:]:
            need(r["model"]["gguf_sha256"] == model_sha and
                 r["model"]["size_bytes"] == model_size and
                 r["runtime"]["binary_sha256"] == receipts[0]["runtime"]["binary_sha256"] and
                 r["experiment"] == receipts[0]["experiment"] and
                 r["requested"]["ctx_size"] == receipts[0]["requested"]["ctx_size"],
                 "SAME_SESSION_RECEIPTS_NOT_COMPARABLE")
        manifest = {
            "schema": SCHEMA, "status": STATUS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "site_label_operator_declared": site,
            "model_alias": alias,
            "model_gguf_sha256": model_sha,
            "model_file_size_bytes": model_size,
            "server_sha256_operator_observed": receipts[0]["runtime"]["binary_sha256"],
            "profiles": [{"gpu_layers_requested": layer,
                          "benchmark_file": p.name,
                          "benchmark_sha256": _sha(p)}
                         for layer, p in zip(layers, completed)],
            "snapshot_file": snapshot_path.name,
            "snapshot_sha256": _sha(snapshot_path),
            "advice_file": advice_path.name,
            "advice_sha256": _sha(advice_path),
            "r15_classification": result["classification"],
            "measurement_status": "LOCAL_OBSERVATION_NOT_ATTESTED",
            "claims": dict(LIMITS),
        }
        _write_new(dest / "campaign-manifest.json", manifest)
        return {
            "status": STATUS,
            "site_label_operator_declared": site,
            "profile_count": len(completed),
            "r15_classification": result["classification"],
            "manifest_name": "campaign-manifest.json",
            "all_sources_local_only": True,
            "hardware_identity_authenticated": False,
            "automatic_model_route": False,
            "production_approval": False,
        }
    except Exception:
        # Preserve already written source receipts for diagnosis. The manifest
        # is the sole completion marker and is written ONLY at the very end.
        raise


def main(argv=None):
    p = argparse.ArgumentParser(prog="phiserve-fieldkit",
        description="Preview or explicitly execute existing local PhiStream/PhiServe field evidence")
    sub = p.add_subparsers(dest="action", required=True)
    for cmd in ("preview", "collect"):
        a = sub.add_parser(cmd, help="No inference" if cmd=="preview" else "Operator-controlled measured inference")
        a.add_argument("--alias", required=True, help="Already-registered local GGUF alias")
        a.add_argument("--llama-server", required=True, help="Installed local llama-server executable")
        a.add_argument("--site", required=True, help="Operator-declared host label, not authenticated identity")
        a.add_argument("--out-dir", required=True, help="Fresh private directory outside a Git checkout")
        a.add_argument("--gpu-layers", type=int, action="append", required=True,
                       help="Repeatable: first 0, then unique offload profiles")
        a.add_argument("--gpu-probe", choices=("none","cuda"), default="none",
                       help="Opt-in PhiStream CUDA free-memory read")
        a.add_argument("--sample-nvidia", action="store_true",
                       help="Opt-in nvidia-smi process-memory sampler during GPU trials")
        a.add_argument("--ctx-size", type=int, default=2048)
        a.add_argument("--max-tokens", type=int, default=64)
        a.add_argument("--warmups", type=int, default=3)
        a.add_argument("--trials", type=int, default=10)
        a.add_argument("--ram-reserve-mib", type=int, default=4096)
        a.add_argument("--gpu-reserve-mib", type=int, default=1024)
        a.add_argument("--backend-port", type=int, default=11436)
        a.add_argument("--timeout-seconds", type=int, default=180)
        a.add_argument("--startup-timeout", type=int, default=120)
        if cmd == "collect":
            a.add_argument("--execute", action="store_true",
                           help="Explicit consent to launch sequential local GGUF inference")
    a = p.parse_args(argv)
    if a.action == "collect" and not a.execute:
        p.error("collect requires --execute; use preview for read-only inspection")
    try:
        result = collect(alias=a.alias, binary=a.llama_server,
                         out_dir=a.out_dir, site=a.site, layers=a.gpu_layers,
                         gpu_probe=a.gpu_probe, ctx_size=a.ctx_size,
                         max_tokens=a.max_tokens, warmups=a.warmups, trials=a.trials,
                         ram_reserve_mib=a.ram_reserve_mib,
                         gpu_reserve_mib=a.gpu_reserve_mib,
                         backend_port=a.backend_port,
                         timeout_seconds=a.timeout_seconds,
                         startup_timeout=a.startup_timeout,
                         sample_nvidia=a.sample_nvidia,
                         execute=a.action=="collect")
    except (CampaignHeld, BenchmarkError, AdviceError, RegistryError, OSError, RuntimeError, ValueError) as exc:
        print("Field campaign held: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
