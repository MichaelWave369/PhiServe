"""R20: private portable R18 evidence bundles. Offline ZIP, no model or network.

Only the exact manifest-listed, bounded R18 JSON source bytes are transported.
The archive is not a signature and a hash match never authenticates a host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import zipfile

from .campaign_review import (
    CampaignReviewHeld, _named_file, _safe_root, inspect_campaign,
)

FORMAT = "PHISERVE_PORTABLE_CAMPAIGN_BUNDLE_R20_V1"
MAX_ZIP_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 65536
MAX_SOURCES = 7
RECEIPT = re.compile(r"benchmark-layers-(?:0|[1-9][0-9]{0,3})\.json\Z")
FIXED_FILES = frozenset({
    "campaign-manifest.json", "capacity-advice.json", "hardware-snapshot.json",
})


class BundleHeld(ValueError):
    """Untrusted, incompatible or unsafe transport evidence."""


def hold(unless: bool, reason: str):
    if not unless:
        raise BundleHeld(reason)


def _safe_output(path, *, is_dir, source=None):
    dest = Path(path).expanduser()
    hold(not dest.exists() and not dest.is_symlink(), "OUTPUT_ALREADY_EXISTS")
    hold(dest.parent.is_dir() and not dest.parent.is_symlink(),
         "OUTPUT_PARENT_INVALID")
    resolved = dest.resolve()
    hold(not any(parent.is_symlink() for parent in (dest.parent, *dest.parent.parents)),
         "OUTPUT_ANCESTOR_SYMLINK")
    hold(not any((ancestor / ".git").exists()
                 for ancestor in (resolved, *resolved.parents)),
         "OUTPUT_INSIDE_GIT_CHECKOUT")
    if source is not None:
        hold(not resolved.is_relative_to(source), "OUTPUT_INSIDE_SOURCE_CAMPAIGN")
    return dest


def _files_from_manifest(root: Path):
    """R19 already checked the manifest; identify only its exact source files."""
    manifest, _ = _named_file(root, "campaign-manifest.json", MAX_MANIFEST_BYTES)
    profiles = manifest["profiles"]
    names = ["campaign-manifest.json", "hardware-snapshot.json", "capacity-advice.json"]
    names.extend(p["benchmark_file"] for p in profiles)
    hold(4 <= len(names) <= MAX_SOURCES and len(names) == len(set(names)),
         "UNEXPECTED_SOURCE_FILE_COUNT")
    return names


def _validate_archive(archive):
    path = Path(archive).expanduser()
    hold(path.is_file() and not path.is_symlink() and
         not any(parent.is_symlink() for parent in path.parents),
         "ARCHIVE_NOT_REGULAR_LOCAL_FILE")
    hold(1 <= path.stat().st_size <= MAX_ZIP_BYTES,
         "ARCHIVE_SIZE_UNSAFE")
    return path


def _entries(handle):
    infos = handle.infolist()
    hold(4 <= len(infos) <= MAX_SOURCES, "ARCHIVE_FILE_COUNT_INVALID")
    seen, names, total = set(), [], 0
    for info in infos:
        name = info.filename
        hold(type(name) is str and
             (name in FIXED_FILES or RECEIPT.fullmatch(name) is not None) and
             "/" not in name and "\\" not in name and ":" not in name and
             name not in seen, "ARCHIVE_UNEXPECTED_OR_DUPLICATE_ENTRY")
        seen.add(name)
        names.append(name)
        # Only uncompressed entries. This avoids zip-bomb amplification and
        # silently changing extraction semantics between Python platforms.
        hold(info.compress_type == zipfile.ZIP_STORED and
             info.compress_size == info.file_size and
             info.flag_bits & 0x1 == 0 and not info.is_dir(),
             "ARCHIVE_COMPRESSION_OR_ENCRYPTION_UNSUPPORTED")
        mode = info.external_attr >> 16
        hold(stat.S_IFMT(mode) in (0, stat.S_IFREG),
             "ARCHIVE_LINK_OR_SPECIAL_FILE_REFUSED")
        cap = MAX_MANIFEST_BYTES if name == "campaign-manifest.json" else MAX_JSON_BYTES
        hold(2 <= info.file_size <= cap, "ARCHIVE_MEMBER_SIZE_UNSAFE")
        total += info.file_size
        hold(total <= 6 * MAX_JSON_BYTES + MAX_MANIFEST_BYTES,
             "ARCHIVE_TOTAL_UNCOMPRESSED_UNSAFE")
    hold(set(FIXED_FILES).issubset(seen) and
         len([name for name in names if RECEIPT.fullmatch(name)]) >= 1,
         "ARCHIVE_REQUIRED_FILES_MISSING")
    return infos


def _extract_checked(source: Path, dest: Path):
    """Defensive, bounded, flat extraction; caller owns private destination."""
    try:
        with zipfile.ZipFile(source, mode="r") as z:
            infos = _entries(z)
            for info in infos:
                # Read exactly the already bounded uncompressed size; ZIP
                # reader also checks CRC32. No arbitrary path is followed.
                with z.open(info, mode="r") as member:
                    data = member.read(info.file_size + 1)
                    hold(len(data) == info.file_size and member.read(1) == b"",
                         "ARCHIVE_MEMBER_LENGTH_CHANGED")
                with (dest / info.filename).open("xb") as f:
                    f.write(data)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError) as exc:
        raise BundleHeld("ARCHIVE_INVALID_OR_CORRUPT") from exc
    # The destination is a fresh private temp dir. R19 replays all original
    # R13/R15 checks AND manifest source hashes after extracting the bytes.
    return inspect_campaign(dest)


def inspect_bundle(archive):
    """Check an existing ZIP without a permanent import or any model execution."""
    path = _validate_archive(archive)
    with tempfile.TemporaryDirectory(prefix="phiserve-r20-verify-") as tmp:
        result = _extract_checked(path, Path(tmp))
    return {
        "format": FORMAT,
        "status": "LOCAL_SOURCE_BYTES_MATCH_UNAUTHENTICATED",
        "site_label_operator_declared": result["site_label_operator_declared"],
        "model_gguf_sha256": result["model_gguf_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "source_file_count": result["source_file_count_locally_matched"] + 1,
        "r15_classification": result["r15_classification"],
        "source_bytes_verified": True,
        "machine_identity_authenticated": False,
        "gpu_execution_verified": False,
        "model_fit_verified": False,
        "external_transfer_performed": False,
    }


def pack_campaign(campaign, output):
    """Refuse invalid R18 source BEFORE creating an exclusive ZIP output."""
    root = _safe_root(campaign)
    report = inspect_campaign(root)
    dest = _safe_output(output, is_dir=False, source=root)
    names = _files_from_manifest(root)
    created = False
    try:
        with dest.open("xb") as file:
            created = True
            with zipfile.ZipFile(file, mode="w", compression=zipfile.ZIP_STORED,
                                 allowZip64=False) as z:
                for name in names:
                    maximum = (MAX_MANIFEST_BYTES if name == "campaign-manifest.json"
                               else MAX_JSON_BYTES)
                    _, digest = _named_file(root, name, maximum)
                    data = (root / name).read_bytes()
                    hold(hashlib.sha256(data).hexdigest() == digest,
                         "SOURCE_CHANGED_WHILE_PACKING")
                    # writestr uses a top-level relative name only, with no
                    # absolute paths or original file system metadata.
                    z.writestr(name, data, compress_type=zipfile.ZIP_STORED)
        hold(dest.stat().st_size <= MAX_ZIP_BYTES, "OUTPUT_ARCHIVE_OVERSIZED")
        verified = inspect_bundle(dest)
        hold(verified["manifest_sha256"] == report["manifest_sha256"],
             "SOURCE_MANIFEST_CHANGED_DURING_PACKING")
        return {
            **verified,
            "status": "PRIVATE_LOCAL_BUNDLE_CREATED",
            "archive_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "archive_size_bytes": dest.stat().st_size,
            "source_bytes_verified": True,
            "external_transfer_performed": False,
        }
    except Exception:
        if created:
            dest.unlink(missing_ok=True)
        raise


def unpack_bundle(archive, output):
    """Extract into a fresh private stage; publish only after R19 accepts."""
    source = _validate_archive(archive)
    dest = _safe_output(output, is_dir=True)
    # Create staging in the selected parent so renaming stays on the same disk.
    stage = Path(tempfile.mkdtemp(prefix=".phiserve-r20-stage-", dir=dest.parent))
    try:
        result = _extract_checked(source, stage)
        hold(not dest.exists() and not dest.is_symlink(),
             "OUTPUT_ALREADY_EXISTS")
        stage.rename(dest)
        return {
            "format": FORMAT,
            "status": "PRIVATE_LOCAL_BUNDLE_IMPORTED_AND_RECHECKED",
            "site_label_operator_declared": result["site_label_operator_declared"],
            "manifest_sha256": result["manifest_sha256"],
            "source_file_count": result["source_file_count_locally_matched"] + 1,
            "r15_classification": result["r15_classification"],
            "source_bytes_verified": True,
            "machine_identity_authenticated": False,
            "gpu_execution_verified": False,
            "model_fit_verified": False,
            "external_transfer_performed": False,
        }
    finally:
        # On a successful rename the stage no longer exists. On any malformed
        # input this deletes only a new, internally allocated staging folder.
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    p = argparse.ArgumentParser(prog="phiserve-campaign-bundle",
                                description="Pack/check/unpack verified R18 JSON offline")
    sub = p.add_subparsers(dest="action", required=True)
    pack = sub.add_parser("pack", help="Create private ZIP only from verified R18 source")
    pack.add_argument("--campaign", required=True, type=Path)
    pack.add_argument("--output", required=True, type=Path)
    check = sub.add_parser("check", help="Inspect ZIP in a temporary local directory")
    check.add_argument("--archive", required=True, type=Path)
    unpack = sub.add_parser("unpack", help="Import into a new private directory")
    unpack.add_argument("--archive", required=True, type=Path)
    unpack.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args(argv)
    try:
        if args.action == "pack":
            result = pack_campaign(args.campaign, args.output)
        elif args.action == "check":
            result = inspect_bundle(args.archive)
        else:
            result = unpack_bundle(args.archive, args.out_dir)
    except (BundleHeld, CampaignReviewHeld, OSError, ValueError,
            zipfile.BadZipFile, RuntimeError) as exc:
        print("Portable campaign bundle held: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
