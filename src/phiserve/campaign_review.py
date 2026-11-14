"""R19: fail-closed offline review of R18 field-campaign source bytes.

This is a *consistency* verifier, not a device attestation or benchmark runner.
It reads only operator-selected local JSON and never touches a GGUF or backend.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from .benchmark import validate_receipt, BenchmarkError
from .field_campaign import SCHEMA as CAMPAIGN_SCHEMA, STATUS as CAMPAIGN_STATUS
from .field_campaign import LABEL, LIMITS as CAMPAIGN_LIMITS
from .hardware_advisor import (
    SCHEMA as ADVICE_SCHEMA, STATUS as ADVICE_STATUS,
    FALSE_ASSURANCES, SOURCE_SCHEMA, _snapshot, _time, _age_state, AdviceError,
)

REPORT_SCHEMA = "PHISERVE_FIELD_CAMPAIGN_REVIEW_R19_V1"
MAX_MANIFEST = 65536
MAX_SOURCE = 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
BENCH_FIELDS = {
    "gpu_layers_requested", "source_receipt_sha256", "source_age_seconds",
    "source_status", "wall_p50_ms", "wall_p95_ms", "startup_ms",
    "completion_tps_p50_reported", "reported_token_count_trials",
    "full_gpu_weight_file_floor_met", "gpu_execution_independently_verified",
}
CAPACITY_FIELDS = {
    "ram_available_bytes_observed", "ram_reserve_bytes_selected",
    "ram_budget_bytes", "gpu_free_bytes_observed", "gpu_reserve_bytes_selected",
    "gpu_budget_bytes", "full_cpu_weight_file_floor_met",
    "full_gpu_weight_file_floor_met", "actual_model_working_set_estimated",
    "actual_model_fit_verified", "hybrid_offload_fit_evaluated",
}
MANIFEST_FIELDS = {
    "schema", "status", "created_utc", "site_label_operator_declared",
    "model_alias", "model_gguf_sha256", "model_file_size_bytes",
    "server_sha256_operator_observed", "profiles", "snapshot_file",
    "snapshot_sha256", "advice_file", "advice_sha256",
    "r15_classification", "measurement_status", "claims",
}
SOURCE_FIELDS = {
    "hardware_snapshot_schema", "hardware_snapshot_file_sha256",
    "hardware_snapshot_age_seconds", "hardware_snapshot_status",
    "model_alias", "local_gguf_sha256", "local_gguf_file_bytes",
}
CLASSIFICATIONS = {
    "REVIEW_REQUIRED_CAPACITY_NOT_CONFIRMED",
    "HOLD_HARDWARE_OBSERVATION_STALE_OR_FUTURE",
    "HOLD_BENCHMARK_EVIDENCE_STALE_OR_FUTURE",
    "HOLD_CUDA_COUNTERS_MISSING",
}


class CampaignReviewHeld(ValueError):
    """Untrusted, corrupted or incompatible campaign evidence was rejected."""


def check(ok, message):
    if not ok:
        raise CampaignReviewHeld(message)


def fields(value, expected, name):
    check(type(value) is dict and set(value) == expected, name + "_FIELDS_INVALID")


def whole(value, lo=0, hi=2**60):
    return type(value) is int and lo <= value <= hi


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(value, target):
    return number(value) and number(target) and math.isclose(
        value, target, abs_tol=0.002, rel_tol=1e-9)


def hash_ok(value):
    return type(value) is str and SHA.fullmatch(value) is not None


def _duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _no_constant(_value):
    raise CampaignReviewHeld("NONFINITE_JSON_CONSTANT")


def _safe_root(value):
    root = Path(value).expanduser()
    check(not root.is_symlink() and root.is_dir(), "CAMPAIGN_DIRECTORY_INVALID")
    # Do not accept a linked ancestor of the operator-selected local directory.
    check(not any(x.is_symlink() for x in (root, *root.parents)),
          "CAMPAIGN_ANCESTOR_SYMLINK")
    return root.resolve(strict=True)


def _named_file(root, filename, maximum):
    check(type(filename) is str and
          filename in {"campaign-manifest.json", "hardware-snapshot.json",
                       "capacity-advice.json"} or
          (type(filename) is str and re.fullmatch(
              r"benchmark-layers-(?:0|[1-9][0-9]{0,3})\.json", filename)),
          "CAMPAIGN_FILENAME_UNSAFE")
    file = root / filename
    check(file.parent == root and file.is_file() and not file.is_symlink(),
          "CAMPAIGN_SOURCE_MISSING_OR_SYMLINK")
    before = file.stat()
    check(2 <= before.st_size <= maximum, "CAMPAIGN_SOURCE_SIZE_INVALID")
    raw = file.read_bytes()
    after = file.stat()
    check(len(raw) == before.st_size and
          (before.st_size, before.st_mtime_ns, before.st_ino) ==
          (after.st_size, after.st_mtime_ns, after.st_ino),
          "CAMPAIGN_SOURCE_CHANGED_DURING_READ")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_duplicate_keys,
                          parse_constant=_no_constant)
    except CampaignReviewHeld:
        raise
    except (UnicodeError, ValueError) as exc:
        raise CampaignReviewHeld("CAMPAIGN_JSON_INVALID") from exc
    check(type(data) is dict, "CAMPAIGN_JSON_OBJECT_REQUIRED")
    return data, hashlib.sha256(raw).hexdigest()


def _verify_source_advice(advice, snapshot, receipts, hashes, manifest):
    fields(advice, {"schema", "status", "evaluated_utc", "classification",
                    "source", "capacity_floor", "benchmark_profiles",
                    "comparison_limits", "assurances"}, "ADVICE")
    check(advice["schema"] == ADVICE_SCHEMA and
          advice["status"] == ADVICE_STATUS and
          advice["classification"] in CLASSIFICATIONS,
          "R15_SCHEMA_OR_STATUS_INVALID")
    fields(advice["assurances"], set(FALSE_ASSURANCES), "ADVICE_ASSURANCES")
    check(advice["assurances"] == FALSE_ASSURANCES, "R15_FALSE_AUTHORITY")
    checked_at = _time(advice["evaluated_utc"], "R15 evaluated UTC")
    fields(advice["source"], SOURCE_FIELDS, "ADVICE_SOURCE")
    fields(advice["capacity_floor"], CAPACITY_FIELDS, "ADVICE_CAPACITY")
    s, c = advice["source"], advice["capacity_floor"]
    facts = _snapshot(snapshot)  # producer's existing strict snapshot contract
    check(s["hardware_snapshot_schema"] == SOURCE_SCHEMA and
          hash_ok(s["hardware_snapshot_file_sha256"]) and
          s["hardware_snapshot_file_sha256"] == hashes["snapshot"] and
          s["model_alias"] == manifest["model_alias"] and
          s["local_gguf_sha256"] == manifest["model_gguf_sha256"] and
          s["local_gguf_file_bytes"] == manifest["model_file_size_bytes"],
          "R15_SOURCE_IDENTITY_MISMATCH")
    age_state, age = _age_state(
        _time(snapshot["observed_utc"], "hardware snapshot time"),
        checked_at, 300)
    check(s["hardware_snapshot_status"] == age_state and
          close(s["hardware_snapshot_age_seconds"], round(age, 3)),
          "R15_SNAPSHOT_AGE_OR_STATUS_MISMATCH")
    for key in ("ram_available_bytes_observed", "ram_reserve_bytes_selected",
                "ram_budget_bytes", "gpu_reserve_bytes_selected"):
        check(whole(c[key]), "R15_RAM_OR_RESERVE_INVALID")
    check(c["ram_available_bytes_observed"] == facts["available_ram_bytes"] and
          c["ram_budget_bytes"] == max(
              0, facts["available_ram_bytes"] - c["ram_reserve_bytes_selected"]) and
          type(c["full_cpu_weight_file_floor_met"]) is bool and
          c["full_cpu_weight_file_floor_met"] ==
          (manifest["model_file_size_bytes"] <= c["ram_budget_bytes"]) and
          c["actual_model_working_set_estimated"] is False and
          c["actual_model_fit_verified"] is False and
          c["hybrid_offload_fit_evaluated"] is False,
          "R15_RAM_FLOOR_OR_FIT_MISMATCH")
    gpu_free = facts["free_cuda_bytes"]
    if gpu_free is None:
        check(c["gpu_free_bytes_observed"] is None and
              c["gpu_budget_bytes"] is None and
              c["full_gpu_weight_file_floor_met"] is None,
              "R15_CUDA_COUNTERS_MISSING_MISMATCH")
    else:
        budget = max(0, gpu_free - c["gpu_reserve_bytes_selected"])
        check(c["gpu_free_bytes_observed"] == gpu_free and
              whole(c["gpu_budget_bytes"]) and
              c["gpu_budget_bytes"] == budget and
              type(c["full_gpu_weight_file_floor_met"]) is bool and
              c["full_gpu_weight_file_floor_met"] ==
              (manifest["model_file_size_bytes"] <= budget),
              "R15_GPU_FLOOR_MISMATCH")
    limits = advice["comparison_limits"]
    check(type(limits) is list and 1 <= len(limits) <= 20 and
          all(type(x) is str and len(x) <= 1024 for x in limits),
          "R15_LIMITATIONS_INVALID")
    ps = advice["benchmark_profiles"]
    check(type(ps) is list and len(ps) == len(receipts), "R15_PROFILE_COUNT_INVALID")
    seen, states = set(), []
    for p in ps:
        fields(p, BENCH_FIELDS, "R15_PROFILE")
        layer = p["gpu_layers_requested"]
        check(whole(layer, 0, 1024) and layer not in seen and
              layer in receipts and hash_ok(p["source_receipt_sha256"]) and
              p["source_receipt_sha256"] == hashes[layer],
              "R15_PROFILE_SOURCE_MISMATCH")
        seen.add(layer)
        bench = receipts[layer]
        bench_state, bench_age = _age_state(
            _time(bench["created_utc"], "benchmark time"), checked_at, 7*86400)
        states.append(bench_state)
        check(p["source_status"] == bench_state and
              close(p["source_age_seconds"], round(bench_age, 3)) and
              close(p["wall_p50_ms"], bench["summary"]["p50_wall_ms"]) and
              close(p["wall_p95_ms"], bench["summary"]["p95_wall_ms"]) and
              close(p["startup_ms"], bench["runtime"]["startup_ms"]) and
              p["reported_token_count_trials"] ==
              bench["summary"]["trials_with_reported_token_counts"] and
              p["full_gpu_weight_file_floor_met"] is
              c["full_gpu_weight_file_floor_met"] and
              p["gpu_execution_independently_verified"] is False,
              "R15_PROFILE_AGGREGATES_MISMATCH")
        original_tps = bench["summary"]["p50_completion_tps"]
        if original_tps is None:
            check(p["completion_tps_p50_reported"] is None,
                  "R15_UNSUPPORTED_TOKEN_RATE")
        else:
            check(close(p["completion_tps_p50_reported"], original_tps),
                  "R15_TOKEN_RATE_MISMATCH")
    check(seen == set(receipts), "R15_MISSING_PROFILE")
    expected = ("HOLD_HARDWARE_OBSERVATION_STALE_OR_FUTURE"
                if age_state != "RECENT_OPERATOR_ASSERTED" else
                "HOLD_BENCHMARK_EVIDENCE_STALE_OR_FUTURE"
                if any(x != "RECENT_OPERATOR_ASSERTED" for x in states) else
                "HOLD_CUDA_COUNTERS_MISSING"
                if gpu_free is None and any(x > 0 for x in receipts) else
                "REVIEW_REQUIRED_CAPACITY_NOT_CONFIRMED")
    check(expected == advice["classification"] ==
          manifest["r15_classification"], "R15_CLASSIFICATION_MISMATCH")
    return checked_at, facts


def inspect_campaign(directory, *, now=None):
    """Verify one complete R18 campaign against all its original JSON bytes."""
    root = _safe_root(directory)
    manifest, manifest_sha = _named_file(root, "campaign-manifest.json", MAX_MANIFEST)
    fields(manifest, MANIFEST_FIELDS, "CAMPAIGN")
    check(manifest["schema"] == CAMPAIGN_SCHEMA and
          manifest["status"] == CAMPAIGN_STATUS and
          manifest["measurement_status"] == "LOCAL_OBSERVATION_NOT_ATTESTED",
          "CAMPAIGN_NOT_COMPLETE")
    fields(manifest["claims"], set(CAMPAIGN_LIMITS), "CAMPAIGN_CLAIMS")
    check(manifest["claims"] == CAMPAIGN_LIMITS, "CAMPAIGN_FALSE_AUTHORITY")
    check(type(manifest["site_label_operator_declared"]) is str and
          LABEL.fullmatch(manifest["site_label_operator_declared"]),
          "CAMPAIGN_SITE_LABEL_INVALID")
    check(type(manifest["model_alias"]) is str and
          1 <= len(manifest["model_alias"]) <= 128 and
          re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}",
                       manifest["model_alias"]) is not None and
          hash_ok(manifest["model_gguf_sha256"]) and
          whole(manifest["model_file_size_bytes"], 4) and
          hash_ok(manifest["server_sha256_operator_observed"]),
          "CAMPAIGN_MODEL_OR_BINARY_INVALID")
    manifest_time = _time(manifest["created_utc"], "manifest creation time")
    check(manifest["snapshot_file"] == "hardware-snapshot.json" and
          manifest["advice_file"] == "capacity-advice.json" and
          hash_ok(manifest["snapshot_sha256"]) and
          hash_ok(manifest["advice_sha256"]), "CAMPAIGN_FIXED_SOURCE_NAMES_INVALID")
    profiles = manifest["profiles"]
    check(type(profiles) is list and 1 <= len(profiles) <= 4,
          "CAMPAIGN_PROFILE_COUNT_INVALID")
    receipts, hashes, declared = {}, {}, []
    for p in profiles:
        fields(p, {"gpu_layers_requested", "benchmark_file",
                   "benchmark_sha256"}, "CAMPAIGN_PROFILE")
        layer = p["gpu_layers_requested"]
        check(whole(layer, 0, 1024) and layer not in receipts and
              p["benchmark_file"] == "benchmark-layers-" + str(layer) + ".json" and
              hash_ok(p["benchmark_sha256"]),
              "CAMPAIGN_PROFILE_NAME_OR_LAYER_INVALID")
        receipt, digest = _named_file(root, p["benchmark_file"], MAX_SOURCE)
        check(digest == p["benchmark_sha256"], "CAMPAIGN_BENCH_HASH_MISMATCH")
        validate_receipt(receipt)  # R13 strict trial/summary and limits validation
        check(receipt["requested"]["gpu_layers"] == layer and
              receipt["model"]["alias"] == manifest["model_alias"] and
              receipt["model"]["gguf_sha256"] == manifest["model_gguf_sha256"] and
              receipt["model"]["size_bytes"] == manifest["model_file_size_bytes"] and
              receipt["runtime"]["binary_sha256"] ==
              manifest["server_sha256_operator_observed"],
              "CAMPAIGN_BENCH_MODEL_RUNTIME_MISMATCH")
        declared.append(layer)
        receipts[layer], hashes[layer] = receipt, digest
    check(declared[0] == 0 and len(declared) == len(set(declared)),
          "CAMPAIGN_CPU_BASELINE_FIRST_REQUIRED")
    snapshot, snapshot_sha = _named_file(root, manifest["snapshot_file"], MAX_SOURCE)
    check(snapshot_sha == manifest["snapshot_sha256"],
          "CAMPAIGN_SNAPSHOT_HASH_MISMATCH")
    advice, advice_sha = _named_file(root, manifest["advice_file"], MAX_SOURCE)
    check(advice_sha == manifest["advice_sha256"], "CAMPAIGN_ADVICE_HASH_MISMATCH")
    evaluated, facts = _verify_source_advice(
        advice, snapshot, receipts, {**hashes, "snapshot": snapshot_sha}, manifest)
    check(manifest_time >= evaluated, "CAMPAIGN_MANIFEST_TIME_INVALID")
    baseline = receipts[0]
    for receipt in receipts.values():
        check(receipt["experiment"] == baseline["experiment"] and
              all(receipt["requested"][name] == baseline["requested"][name]
                  for name in ("ctx_size", "max_tokens", "seed", "temperature")) and
              receipt["runtime"]["platform_family"] == facts["platform_family"] and
              receipt["runtime"]["phiserve_version"] ==
              baseline["runtime"]["phiserve_version"],
              "CAMPAIGN_SAME_WORKLOAD_OR_OS_MISMATCH")
        check(_time(receipt["created_utc"], "benchmark time") <=
              _time(snapshot["observed_utc"], "hardware snapshot time"),
              "CAMPAIGN_SNAPSHOT_MUST_FOLLOW_BENCH")
    clock = now or datetime.now(timezone.utc)
    check(clock.tzinfo is not None and clock.utcoffset().total_seconds() == 0,
          "REVIEW_CLOCK_NOT_UTC")
    snapshot_age = (clock - _time(snapshot["observed_utc"], "snapshot time")).total_seconds()
    benchmark_age = [(clock - _time(x["created_utc"], "benchmark time")).total_seconds()
                     for x in receipts.values()]
    review_state = ("REVIEW_CLOCK_OR_SOURCE_FUTURE"
                    if snapshot_age < -60 or any(age < -60 for age in benchmark_age)
                    or (clock - evaluated).total_seconds() < -60 else
                    "STALE_HARDWARE_SNAPSHOT"
                    if snapshot_age > 300 else
                    "STALE_BENCHMARK_EVIDENCE"
                    if any(age > 7*86400 for age in benchmark_age) else
                    "SOURCE_AGES_WITHIN_DEFAULT_WINDOWS")
    return {
        "site_label_operator_declared": manifest["site_label_operator_declared"],
        "manifest_sha256": manifest_sha,
        "model_alias": manifest["model_alias"],
        "model_gguf_sha256": manifest["model_gguf_sha256"],
        "model_file_size_bytes": manifest["model_file_size_bytes"],
        "server_sha256_operator_observed": manifest["server_sha256_operator_observed"],
        "r15_classification": advice["classification"],
        "review_time_freshness": review_state,
        "source_file_count_locally_matched": len(receipts) + 2,
        "benchmark_protocol": {
            "experiment": dict(baseline["experiment"]),
            "settings": {k: baseline["requested"][k] for k in (
                "ctx_size", "max_tokens", "seed", "temperature")},
            "platform_family": baseline["runtime"]["platform_family"],
        },
        "requested_profiles": [{
            "gpu_layers_requested": layer,
            "p50_wall_ms": r["summary"]["p50_wall_ms"],
            "p95_wall_ms": r["summary"]["p95_wall_ms"],
            "startup_ms": r["runtime"]["startup_ms"],
            "end_to_end_tps_p50_if_reported": r["summary"]["p50_completion_tps"],
            "gpu_execution_independently_verified": False,
        } for layer, r in receipts.items()],
        "source_integrity_verified_only": True,
        "hardware_provenance_authenticated": False,
        "gguf_weights_rehashed_by_reviewer": False,
        "gpu_execution_verified": False,
        "deployment_or_routing_authorized": False,
    }


def review(directories, *, now=None):
    """Two campaigns get a protocol-check classification, never a speed winner."""
    check(type(directories) in (list, tuple) and 1 <= len(directories) <= 2,
          "ONE_OR_TWO_CAMPAIGNS_REQUIRED")
    roots = [_safe_root(x) for x in directories]
    check(len(set(roots)) == len(roots), "CAMPAIGNS_DUPLICATED")
    clock = now or datetime.now(timezone.utc)
    inspected = [inspect_campaign(root, now=clock) for root in roots]
    comparable = {"status": "SINGLE_CAMPAIGN_REVIEW", "common_gpu_layers": [],
                  "speed_winner_declared": False}
    if len(inspected) == 2:
        a, b = inspected
        shared = sorted(set(p["gpu_layers_requested"] for p in a["requested_profiles"]) &
                        set(p["gpu_layers_requested"] for p in b["requested_profiles"]))
        if (a["model_gguf_sha256"], a["model_file_size_bytes"]) != (
                b["model_gguf_sha256"], b["model_file_size_bytes"]):
            status = "HOLD_DIFFERENT_GGUF_BYTES"
        elif a["benchmark_protocol"] != b["benchmark_protocol"]:
            status = "HOLD_DIFFERENT_BENCHMARK_PROTOCOL"
        elif not shared or 0 not in shared:
            status = "HOLD_NO_COMMON_CPU_BASELINE"
        elif a["server_sha256_operator_observed"] != b["server_sha256_operator_observed"]:
            status = "DESCRIPTIVE_ONLY_DIFFERENT_SERVER_BUILDS"
        else:
            status = "DESCRIPTIVE_ONLY_MATCHING_REPORTED_PROTOCOL"
        comparable = {"status": status, "common_gpu_layers": shared,
                      "speed_winner_declared": False,
                      "same_physical_machine_proven": False,
                      "same_thermal_or_driver_conditions_proven": False}
    return {
        "schema": REPORT_SCHEMA,
        "status": "LOCAL_SOURCE_BYTES_MATCH_NOT_MACHINE_AUTHENTICITY",
        "reviewed_utc": clock.isoformat(),
        "campaigns": inspected,
        "cross_campaign_comparison": comparable,
        "notes": [
            "Local source byte hashes validate only consistency with this self-reported manifest.",
            "Original GGUF and llama-server bytes are not rehashed or authenticated by this review.",
            "R15 memory file-size floors do not prove runtime working-set fit or GPU residency.",
            "Reported p50/p95 are full HTTP-request times, not device-only decode speed.",
            "Other machine load, drivers, thermals and GPU execution remain unverified.",
            "No automatic model routing, permission grant, quality or deployment approval.",
        ],
        "physical_hardware_authenticated": False,
        "quality_evaluated": False,
        "model_fit_verified": False,
        "best_device_selected": False,
        "inference_executed_by_reviewer": False,
        "deployment_authorized": False,
        "remote_upload_performed": False,
    }


def write_review(directories, output, *, now=None):
    """Create only a new, private JSON file outside any Git checkout."""
    result = review(directories, now=now)
    target = Path(output).expanduser()
    check(not target.exists() and not target.is_symlink(), "OUTPUT_ALREADY_EXISTS")
    check(target.parent.exists() and target.parent.is_dir() and
          not target.parent.is_symlink(), "OUTPUT_PARENT_INVALID")
    check(not any((parent / ".git").exists()
                  for parent in (target.resolve(), *target.resolve().parents)),
          "OUTPUT_INSIDE_GIT_CHECKOUT")
    check(not any(target.resolve().is_relative_to(_safe_root(x))
                  for x in directories), "OUTPUT_CANNOT_MUTATE_SOURCE_CAMPAIGNS")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="phiserve-campaign-review",
        description="Read-only R19 verifier for 1-2 existing R18 local field campaigns")
    parser.add_argument("--campaign", type=Path, action="append", required=True,
                        help="Completed local R18 directory; repeat at most twice")
    parser.add_argument("--output", type=Path, required=True,
                        help="New private JSON file outside a Git checkout and source campaigns")
    a = parser.parse_args(argv)
    try:
        result = write_review(a.campaign, a.output)
    except (CampaignReviewHeld, AdviceError, BenchmarkError, OSError,
            ValueError, OverflowError) as exc:
        print("Campaign review held: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "campaign_count": len(result["campaigns"]),
        "cross_campaign_comparison": result["cross_campaign_comparison"]["status"],
        "machine_authenticated": False,
        "execution_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
