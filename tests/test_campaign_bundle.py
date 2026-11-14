"""R20 offline portable evidence tests. Synthetic R18 JSON, never real inference."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
import zipfile

from phiserve.campaign_bundle import (
    BundleHeld, FORMAT, inspect_bundle, main, pack_campaign, unpack_bundle,
)
from phiserve.campaign_review import CampaignReviewHeld, inspect_campaign, review
from test_field_campaign import FieldCampaignTests


class PortableEvidenceBundleTests(unittest.TestCase):
    def setUp(self):
        self.producer = FieldCampaignTests("test_preview_never_writes_anything_or_benchmarks")
        self.producer.setUp()
        self.addCleanup(self.producer.doCleanups)
        self.producer.run_fake(execute=True)
        self.root = self.producer.root
        self.campaign = self.producer.output
        self.archive = self.root / "campaign.zip"
        self.imported = self.root / "imported-campaign"

    def make_bundle(self):
        return pack_campaign(self.campaign, self.archive)

    def remake_zip(self, replace=None, add=None, compression=zipfile.ZIP_STORED):
        replace = replace or {}
        with zipfile.ZipFile(self.archive, "r") as zipped:
            rows = [(info.filename, zipped.read(info.filename)) for info in zipped.infolist()]
        with zipfile.ZipFile(self.archive, "w", compression=compression) as zipped:
            for name, content in rows:
                zipped.writestr(name, replace.get(name, content), compress_type=compression)
            for name, content in (add or []):
                zipped.writestr(name, content, compress_type=compression)

    def test_pack_then_check_then_import_then_r19_review(self):
        before = inspect_campaign(self.campaign)
        result = self.make_bundle()
        self.assertEqual(result["status"], "PRIVATE_LOCAL_BUNDLE_CREATED")
        self.assertEqual(result["format"], FORMAT)
        self.assertTrue(result["source_bytes_verified"])
        self.assertFalse(result["machine_identity_authenticated"])
        self.assertFalse(result["external_transfer_performed"])
        self.assertEqual(result["manifest_sha256"], before["manifest_sha256"])
        checked = inspect_bundle(self.archive)
        self.assertEqual(checked["status"], "LOCAL_SOURCE_BYTES_MATCH_UNAUTHENTICATED")
        imported = unpack_bundle(self.archive, self.imported)
        self.assertEqual(imported["status"], "PRIVATE_LOCAL_BUNDLE_IMPORTED_AND_RECHECKED")
        self.assertEqual(imported["manifest_sha256"], before["manifest_sha256"])
        self.assertEqual(
            inspect_campaign(self.imported)["manifest_sha256"], before["manifest_sha256"]
        )
        comparison = review([self.campaign, self.imported])
        self.assertFalse(comparison["best_device_selected"])
        self.assertFalse(comparison["physical_hardware_authenticated"])

    def test_archive_contains_exact_6_expected_files_no_extra_secrets(self):
        (self.campaign / "operator-private-secrets.txt").write_text("SECRET DO NOT EXPORT")
        self.make_bundle()
        with zipfile.ZipFile(self.archive) as z:
            self.assertEqual(set(z.namelist()), {
                "campaign-manifest.json",
                "hardware-snapshot.json",
                "capacity-advice.json",
                "benchmark-layers-0.json",
                "benchmark-layers-12.json",
            })
            # The count is five: manifest + two benchmark receipts + snapshot + R15.
            self.assertNotIn("SECRET DO NOT EXPORT", str(z.namelist()))
            self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED
                                for i in z.infolist()))
            self.assertTrue(all("/" not in n and "\\" not in n for n in z.namelist()))
        self.assertNotIn(str(self.root), self.archive.read_bytes().decode("latin1"))

    def test_pack_blocks_changed_source_before_creating_archive(self):
        path = self.campaign / "benchmark-layers-0.json"
        path.write_text(path.read_text(encoding="utf-8")+" ")
        with self.assertRaisesRegex(CampaignReviewHeld, "BENCH_HASH_MISMATCH"):
            self.make_bundle()
        self.assertFalse(self.archive.exists())

    def test_pack_refuses_partial_campaign(self):
        (self.campaign / "campaign-manifest.json").unlink()
        with self.assertRaises(CampaignReviewHeld):
            self.make_bundle()
        self.assertFalse(self.archive.exists())

    def test_pack_refuses_existing_output_and_never_overwrites(self):
        self.archive.write_text("keep-existing", encoding="utf-8")
        with self.assertRaisesRegex(BundleHeld, "OUTPUT_ALREADY_EXISTS"):
            self.make_bundle()
        self.assertEqual(self.archive.read_text(), "keep-existing")

    def test_pack_refuses_output_inside_source_and_git_repo(self):
        with self.assertRaisesRegex(BundleHeld, "OUTPUT_INSIDE_SOURCE_CAMPAIGN"):
            pack_campaign(self.campaign, self.campaign/"export.zip")
        repo = self.root/"repo"
        repo.mkdir()
        (repo/".git").mkdir()
        with self.assertRaisesRegex(BundleHeld, "OUTPUT_INSIDE_GIT_CHECKOUT"):
            pack_campaign(self.campaign, repo/"export.zip")

    def test_unpack_refuses_existing_destination(self):
        self.make_bundle()
        self.imported.mkdir()
        marker = self.imported/"do-not-delete.txt"
        marker.write_text("keep me")
        with self.assertRaisesRegex(BundleHeld, "OUTPUT_ALREADY_EXISTS"):
            unpack_bundle(self.archive, self.imported)
        self.assertEqual(marker.read_text(), "keep me")

    def test_duplicate_member_rejected_without_destination(self):
        self.make_bundle()
        self.remake_zip(add=[("benchmark-layers-0.json", b'{"junk":true}')])
        with self.assertRaisesRegex(BundleHeld, "DUPLICATE_ENTRY"):
            unpack_bundle(self.archive, self.imported)
        self.assertFalse(self.imported.exists())

    def test_zip_path_traversal_rejected_without_output(self):
        self.make_bundle()
        self.remake_zip(add=[("../escape.txt", b"hijack")])
        with self.assertRaisesRegex(BundleHeld, "UNEXPECTED_OR_DUPLICATE"):
            unpack_bundle(self.archive, self.imported)
        self.assertFalse(self.imported.exists())
        self.assertFalse((self.root/"escape.txt").exists())

    def test_nested_paths_and_unlisted_files_rejected(self):
        for name in ["private.txt", "private/hidden.json", "C:\\secret.txt",
                     "benchmark-layers-0001.json"]:
            with self.subTest(name=name):
                self.archive.unlink(missing_ok=True)
                self.make_bundle()
                self.remake_zip(add=[(name,b"PRIVATE")])
                with self.assertRaises(BundleHeld):
                    inspect_bundle(self.archive)
                self.assertFalse(self.imported.exists())

    def test_compressed_archive_rejected_for_bomb_resistance(self):
        self.make_bundle()
        self.remake_zip(compression=zipfile.ZIP_DEFLATED)
        with self.assertRaisesRegex(BundleHeld, "COMPRESSION_OR_ENCRYPTION"):
            unpack_bundle(self.archive,self.imported)
        self.assertFalse(self.imported.exists())

    def test_missing_member_rejected_even_if_zip_valid(self):
        self.make_bundle()
        with zipfile.ZipFile(self.archive,"r") as z:
            items={name:z.read(name) for name in z.namelist()
                   if name!="benchmark-layers-0.json"}
        with zipfile.ZipFile(self.archive,"w") as z:
            for name,data in items.items():
                z.writestr(name,data)
        with self.assertRaises(CampaignReviewHeld):
            unpack_bundle(self.archive,self.imported)
        self.assertFalse(self.imported.exists())

    def test_tampered_receipt_with_resealed_outer_zip_fails_r19(self):
        self.make_bundle()
        self.remake_zip(replace={"benchmark-layers-12.json":b'{"fake":"report"}'})
        with self.assertRaisesRegex(CampaignReviewHeld, "BENCH_HASH_MISMATCH"):
            unpack_bundle(self.archive,self.imported)
        self.assertFalse(self.imported.exists())

    def test_source_has_bogus_authority_but_bytes_match_still_refused(self):
        self.make_bundle()
        with zipfile.ZipFile(self.archive) as z:
            archive_files = {name:z.read(name) for name in z.namelist()}
        manifest=json.loads(archive_files["campaign-manifest.json"])
        advice=json.loads(archive_files["capacity-advice.json"])
        advice["assurances"]["execution_authorized"]=True
        amended=json.dumps(advice).encode()
        manifest["advice_sha256"]=hashlib.sha256(amended).hexdigest()
        archive_files["capacity-advice.json"]=amended
        archive_files["campaign-manifest.json"]=json.dumps(manifest).encode()
        with zipfile.ZipFile(self.archive,"w") as z:
            for name,data in archive_files.items():
                z.writestr(name,data)
        with self.assertRaisesRegex(CampaignReviewHeld, "FALSE_AUTHORITY"):
            unpack_bundle(self.archive, self.imported)
        self.assertFalse(self.imported.exists())

    def test_archive_symlink_rejected(self):
        self.make_bundle()
        link=self.root/"symlink.zip"
        try:
            link.symlink_to(self.archive)
        except OSError:
            self.skipTest("Symlink capability unavailable")
        with self.assertRaisesRegex(BundleHeld, "ARCHIVE_NOT_REGULAR"):
            inspect_bundle(link)

    def test_unexpected_zip_member_count_rejected(self):
        self.make_bundle()
        self.remake_zip(add=[
            ("benchmark-layers-1.json", b"{}"),
            ("benchmark-layers-2.json", b"{}"),
            ("benchmark-layers-3.json", b"{}"),
        ])
        with self.assertRaisesRegex(BundleHeld, "FILE_COUNT_INVALID"):
            inspect_bundle(self.archive)

    def test_size_ceiling_blocks_oversized_archive_without_writes(self):
        self.archive.write_bytes(b"0"*(8*1024*1024+1))
        with self.assertRaisesRegex(BundleHeld, "ARCHIVE_SIZE_UNSAFE"):
            unpack_bundle(self.archive,self.imported)
        self.assertFalse(self.imported.exists())

    def test_read_only_inspection_leaves_no_import_directory(self):
        self.make_bundle()
        checked = inspect_bundle(self.archive)
        self.assertTrue(checked["source_bytes_verified"])
        self.assertFalse(self.imported.exists())

    def test_import_output_within_git_checkout_refused(self):
        self.make_bundle()
        repo = self.root/"project"
        repo.mkdir()
        (repo/".git").mkdir()
        with self.assertRaisesRegex(BundleHeld, "OUTPUT_INSIDE_GIT_CHECKOUT"):
            unpack_bundle(self.archive,repo/"imported")
        self.assertFalse((repo/"imported").exists())

    def test_cli_pack_check_unpack_on_local_fixture(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["pack","--campaign",str(self.campaign),
                                   "--output",str(self.archive)]),0)
            self.assertEqual(main(["check","--archive",str(self.archive)]),0)
            self.assertEqual(main(["unpack","--archive",str(self.archive),
                                   "--out-dir",str(self.imported)]),0)
        self.assertTrue(self.imported.exists())
        self.assertEqual(self.imported.joinpath("campaign-manifest.json").read_bytes(),
                         self.campaign.joinpath("campaign-manifest.json").read_bytes())

    def test_cli_refuses_missing_archive_without_creating_output(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["unpack","--archive",str(self.archive),
                                   "--out-dir",str(self.imported)]),2)
        self.assertFalse(self.imported.exists())


if __name__ == "__main__":
    unittest.main()
