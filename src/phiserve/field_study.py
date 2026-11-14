"""R21 one-command local operator study runner, built entirely on R18-R20.

No new evidence format, no model install, no autonomous measurement.
'preview' is read-only. 'run --execute' is an explicit user action that
executes locally through existing R18, then checks with R19 and packs with R20.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .campaign_bundle import BundleHeld, pack_campaign
from .campaign_review import CampaignReviewHeld, write_review
from .field_campaign import (
    CampaignHeld, LABEL, collect, validate_options,
)
from .hardware_advisor import AdviceError
from .benchmark import BenchmarkError
from .registry import RegistryError


class StudyHeld(ValueError):
    """Failed closed before dispatch or at a known study stage."""


def need(test, message):
    if not test:
        raise StudyHeld(message)


def layout(workspace, site, session):
    """Derive only private, flat, collision-safe filenames from operator labels."""
    need(type(site) is str and LABEL.fullmatch(site) is not None,
         "INVALID_OPERATOR_SITE_LABEL")
    need(type(session) is str and LABEL.fullmatch(session) is not None,
         "INVALID_SESSION_LABEL")
    root = Path(workspace).expanduser()
    need(root.is_dir() and not any(x.is_symlink() for x in (root, *root.parents)),
         "WORKSPACE_MUST_BE_EXISTING_REAL_DIRECTORY")
    root = root.resolve(strict=True)
    need(not any((x / ".git").exists() for x in (root, *root.parents)),
         "WORKSPACE_INSIDE_GIT_CHECKOUT")
    stem = f"{site}-{session}"
    campaign = root / stem
    review = root / (stem + ".review.json")
    archive = root / (stem + ".zip")
    need(all(not p.exists() and not p.is_symlink() for p in
             (campaign, review, archive)), "STUDY_OUTPUT_ALREADY_EXISTS")
    return campaign, review, archive


def study(*, workspace, site, session, alias, binary, layers,
          gpu_probe="none", sample_nvidia=False, ctx_size=2048,
          max_tokens=64, warmups=3, trials=10, backend_port=11436,
          timeout_seconds=180, startup_timeout=120,
          ram_reserve_mib=4096, gpu_reserve_mib=1024,
          execute=False, registry=None,
          collector=collect, reviewer=write_review, packer=pack_campaign):
    """Preview or deliberately run sequential R18 + R19 + R20.

    Custom stage funcs are only for offline tests; CLI never exposes them.
    """
    need(type(execute) is bool, "INVALID_EXECUTION_GATE")
    campaign, review, archive = layout(workspace, site, session)
    options = dict(
        alias=alias, binary=binary, out_dir=campaign, site=site,
        layers=layers, gpu_probe=gpu_probe, sample_nvidia=sample_nvidia,
        ctx_size=ctx_size, max_tokens=max_tokens, warmups=warmups,
        trials=trials, backend_port=backend_port,
        timeout_seconds=timeout_seconds, startup_timeout=startup_timeout,
        ram_reserve_mib=ram_reserve_mib, gpu_reserve_mib=gpu_reserve_mib,
    )
    # Existing R18 local-only preflight. No model loading or CUDA probe.
    plan = validate_options(**options, registry=registry)
    display = {
        "site_label_operator_declared": site,
        "session_label": session,
        "model_alias": alias,
        "requested_gpu_layers": list(layers),
        "gpu_snapshot_mode": gpu_probe,
        "process_memory_sampler_requested": sample_nvidia,
        "campaign_folder_name": campaign.name,
        "review_file_name": review.name,
        "portable_zip_name": archive.name,
        "requires_explicit_execute": True,
        "local_model_and_binary_preflight": "PASSED",
        "physical_hardware_preflight": "NOT_PERFORMED",
        "gpu_execution_verified": False,
        "model_quality_verified": False,
        "execution_authorized_by_evidence": False,
        "external_upload_performed": False,
    }
    if not execute:
        return {"status": "READ_ONLY_PREVIEW", "inference_started": False,
                "files_created": False, **display}
    # R18 fails closed on missing PhiStream hardware module before benchmarks,
    # and leaves no success manifest on a partial inference failure.
    collected = collector(**options, execute=True, registry=registry)
    need(collected.get("status") == "OPERATOR_EXECUTED_UNATTESTED_REVIEW_ONLY",
         "R18_NOT_COMPLETE")
    need((campaign / "campaign-manifest.json").is_file(),
         "R18_COMPLETION_MANIFEST_ABSENT")
    # R19 reviews the actual original source files before we even pack them.
    verified = reviewer([campaign], review)
    need(verified.get("status") ==
         "LOCAL_SOURCE_BYTES_MATCH_NOT_MACHINE_AUTHENTICITY" and
         verified.get("deployment_authorized") is False,
         "R19_REVIEW_NOT_CONSISTENT")
    packaged = packer(campaign, archive)
    need(packaged.get("status") == "PRIVATE_LOCAL_BUNDLE_CREATED" and
         packaged.get("source_bytes_verified") is True and
         packaged.get("machine_identity_authenticated") is False,
         "R20_PORTABLE_BUNDLE_NOT_VERIFIED")
    return {
        "status": "LOCAL_OPERATOR_STUDY_FILES_COMPLETE",
        "r18_status": collected["status"],
        "r15_classification": collected["r15_classification"],
        "r19_status": verified["status"],
        "r20_status": packaged["status"],
        "inference_explicitly_requested": True,
        "source_files_locally_consistent": True,
        "gpu_execution_verified": False,
        "hardware_identity_authenticated": False,
        "automatic_model_routing": False,
        "deployment_authorized": False,
        "external_upload_performed": False,
        "site_label_operator_declared": site,
        "session_label": session,
        "campaign_folder_name": campaign.name,
        "review_file_name": review.name,
        "portable_zip_name": archive.name,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="phiserve-field-study",
        description="R21 one-command local R18 collection, R19 review and R20 ZIP")
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("preview", "run"):
        p = modes.add_parser(mode,
            help="Read-only local preflight" if mode == "preview" else
                 "Explicitly collect local measured GGUF evidence")
        p.add_argument("--workspace", required=True,
                       help="Existing private folder outside a Git checkout")
        p.add_argument("--site", required=True,
                       help="Operator-declared machine label, not attestation")
        p.add_argument("--session", required=True,
                       help="Fresh alphanumeric session ID for distinct outputs")
        p.add_argument("--alias", required=True,
                       help="Already registered licensed local GGUF alias")
        p.add_argument("--llama-server", required=True,
                       help="Existing local llama-server binary")
        p.add_argument("--gpu-layers", action="append", type=int, required=True,
                       help="Repeat 1-4 times; CPU-only 0 must come first")
        p.add_argument("--gpu-probe", choices=("none","cuda"), default="none",
                       help="Opt-in PhiStream CUDA free-memory snapshot")
        p.add_argument("--sample-nvidia", action="store_true",
                       help="Opt-in nvidia-smi process memory sampling")
        p.add_argument("--ctx-size", type=int, default=2048)
        p.add_argument("--max-tokens", type=int, default=64)
        p.add_argument("--warmups", type=int, default=3)
        p.add_argument("--trials", type=int, default=10)
        p.add_argument("--backend-port", type=int, default=11436)
        p.add_argument("--timeout-seconds", type=int, default=180)
        p.add_argument("--startup-timeout", type=int, default=120)
        p.add_argument("--ram-reserve-mib", type=int, default=4096)
        p.add_argument("--gpu-reserve-mib", type=int, default=1024)
        if mode == "run":
            p.add_argument("--execute", action="store_true",
                           help="Required to execute GGUF inference on this machine")
    args = parser.parse_args(argv)
    if args.mode == "run" and not args.execute:
        parser.error("run requires explicit --execute; use preview for safe inspection")
    try:
        result = study(
            workspace=args.workspace, site=args.site, session=args.session,
            alias=args.alias, binary=args.llama_server, layers=args.gpu_layers,
            gpu_probe=args.gpu_probe, sample_nvidia=args.sample_nvidia,
            ctx_size=args.ctx_size, max_tokens=args.max_tokens,
            warmups=args.warmups, trials=args.trials,
            backend_port=args.backend_port,
            timeout_seconds=args.timeout_seconds,
            startup_timeout=args.startup_timeout,
            ram_reserve_mib=args.ram_reserve_mib,
            gpu_reserve_mib=args.gpu_reserve_mib,
            execute=args.mode == "run",
        )
    except (StudyHeld, CampaignHeld, CampaignReviewHeld, BundleHeld,
            AdviceError, BenchmarkError, RegistryError, OSError,
            RuntimeError, ValueError) as exc:
        print("Field study held: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
