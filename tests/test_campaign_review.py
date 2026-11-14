"""R19 source-byte verification tests using R18's synthetic field campaign producer.

No external backend, GGUF inference, GPU, downloads or public uploads.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from phiserve.campaign_review import (
    CampaignReviewHeld, REPORT_SCHEMA, inspect_campaign, main, review,
    write_review,
)
from test_field_campaign import FieldCampaignTests


class CampaignReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = self.producer("primary")
        self.directory = self.fixture.output

    def producer(self, name, *, model_bytes=None, binary_bytes=None, settings=None):
        """Make actual R18 JSON byte files using its synthetic-only test producer."""
        producer = FieldCampaignTests("test_preview_never_writes_anything_or_benchmarks")
        producer.setUp()
        self.addCleanup(producer.doCleanups)
        if model_bytes is not None:
            producer.model.write_bytes(b"GGUF" + model_bytes)
        if binary_bytes is not None:
            producer.bin.write_bytes(binary_bytes)
        producer.run_fake(execute=True, site=name, **(settings or {}))
        return producer

    def mutate(self, filename, callback, *, root=None, reseal=True):
        """Edit JSON and optionally reseal only the outer unsigned R18 digest."""
        root = root or self.directory
        path = root / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        callback(value)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        if reseal:
            manifest_path = root / "campaign-manifest.json"
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if filename == "hardware-snapshot.json":
                m["snapshot_sha256"] = digest
            elif filename == "capacity-advice.json":
                m["advice_sha256"] = digest
            else:
                for profile in m["profiles"]:
                    if profile["benchmark_file"] == filename:
                        profile["benchmark_sha256"] = digest
            manifest_path.write_text(json.dumps(m), encoding="utf-8")

    def test_r18_real_fixture_sources_pass_r19_without_authenticating_hardware(self):
        result = inspect_campaign(self.directory)
        self.assertEqual(result["site_label_operator_declared"], "primary")
        self.assertEqual(result["source_file_count_locally_matched"], 4)
        self.assertEqual(result["review_time_freshness"],
                         "SOURCE_AGES_WITHIN_DEFAULT_WINDOWS")
        self.assertEqual([x["gpu_layers_requested"] for x in result["requested_profiles"]],
                         [0, 12])
        self.assertFalse(result["hardware_provenance_authenticated"])
        self.assertFalse(result["gpu_execution_verified"])
        self.assertFalse(result["gguf_weights_rehashed_by_reviewer"])
        self.assertFalse(result["deployment_or_routing_authorized"])
        self.assertFalse(review([self.directory])["best_device_selected"])

    def test_static_json_report_excludes_local_paths_and_gpu_labels(self):
        output = self.fixture.root / "report.json"
        result = write_review([self.directory], output)
        self.assertTrue(output.is_file())
        data = output.read_text(encoding="utf-8")
        self.assertEqual(result["schema"], REPORT_SCHEMA)
        self.assertNotIn(str(self.directory), data)
        self.assertNotIn(str(self.fixture.model), data)
        self.assertNotIn(str(self.fixture.bin), data)
        self.assertNotIn("synthetic GPU", data)
        self.assertNotIn("Synthetic fixture", data)
        self.assertFalse(result["inference_executed_by_reviewer"])
        with self.assertRaisesRegex(CampaignReviewHeld, "OUTPUT_ALREADY_EXISTS"):
            write_review([self.directory], output)

    def test_manifest_sha_mismatch_fails_closed(self):
        self.mutate("benchmark-layers-12.json", lambda d: None, reseal=False)
        with self.assertRaisesRegex(CampaignReviewHeld, "BENCH_HASH_MISMATCH"):
            inspect_campaign(self.directory)

    def test_missing_source_or_manifest_is_held(self):
        (self.directory / "benchmark-layers-12.json").unlink()
        with self.assertRaisesRegex(CampaignReviewHeld, "SOURCE_MISSING"):
            inspect_campaign(self.directory)
        (self.directory / "campaign-manifest.json").unlink()
        with self.assertRaisesRegex(CampaignReviewHeld, "SOURCE_MISSING"):
            inspect_campaign(self.directory)

    def test_resealed_benchmark_aggregate_tamper_detected_by_r13(self):
        self.mutate("benchmark-layers-0.json",
                    lambda d: d["summary"].__setitem__("p50_wall_ms", 1.0))
        with self.assertRaises(ValueError):
            inspect_campaign(self.directory)

    def test_resealed_snapshot_counter_mismatch_rejected(self):
        self.mutate("hardware-snapshot.json",
                    lambda d: d["ram"].__setitem__(
                        "available_bytes", d["ram"]["available_bytes"]//2))
        with self.assertRaisesRegex(CampaignReviewHeld, "SOURCE_IDENTITY"):
            inspect_campaign(self.directory)

    def test_resealed_fake_r15_execution_authority_rejected(self):
        self.mutate("capacity-advice.json",
                    lambda d: d["assurances"].__setitem__("execution_authorized", True))
        with self.assertRaisesRegex(CampaignReviewHeld, "FALSE_AUTHORITY"):
            inspect_campaign(self.directory)

    def test_resealed_r15_capacity_arithmetic_held(self):
        self.mutate("capacity-advice.json",
                    lambda d: d["capacity_floor"].__setitem__("ram_budget_bytes", 12))
        with self.assertRaisesRegex(CampaignReviewHeld, "RAM_FLOOR"):
            inspect_campaign(self.directory)

    def test_resealed_advisory_profile_latency_mismatch_rejected(self):
        self.mutate("capacity-advice.json",
                    lambda d: d["benchmark_profiles"][0].__setitem__("wall_p50_ms", 9999.0))
        with self.assertRaisesRegex(CampaignReviewHeld, "PROFILE_AGGREGATES"):
            inspect_campaign(self.directory)

    def test_resealed_original_r18_claim_false_authority_held(self):
        self.mutate("campaign-manifest.json",
                    lambda d: d["claims"].__setitem__("automatic_inference_routing", True))
        with self.assertRaisesRegex(CampaignReviewHeld, "FALSE_AUTHORITY"):
            inspect_campaign(self.directory)

    def test_offload_profile_path_traversal_blocked(self):
        self.mutate("campaign-manifest.json",
                    lambda d: d["profiles"][0].__setitem__("benchmark_file", "../secret.json"))
        with self.assertRaisesRegex(CampaignReviewHeld, "NAME_OR_LAYER_INVALID"):
            inspect_campaign(self.directory)

    def test_symlink_source_rejected(self):
        original = self.directory / "hardware-snapshot.json"
        original.rename(self.directory / "snapshot-underlying.json")
        try:
            original.symlink_to(self.directory / "snapshot-underlying.json")
        except OSError:
            self.skipTest("Symlink creation unavailable")
        with self.assertRaisesRegex(CampaignReviewHeld, "SYMLINK"):
            inspect_campaign(self.directory)

    def test_oversized_input_and_duplicate_keys_held(self):
        original = self.directory / "capacity-advice.json"
        original.write_bytes(b" " * (1024*1024+1))
        with self.assertRaisesRegex(CampaignReviewHeld, "SOURCE_SIZE_INVALID"):
            inspect_campaign(self.directory)
        original.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        with self.assertRaisesRegex(CampaignReviewHeld, "DUPLICATE_JSON_KEY"):
            inspect_campaign(self.directory)

    def test_snapshot_age_at_review_degrades_without_changing_r15_claim(self):
        result = inspect_campaign(self.directory,
                                  now=datetime.now(timezone.utc)+timedelta(minutes=16))
        self.assertEqual(result["review_time_freshness"], "STALE_HARDWARE_SNAPSHOT")
        self.assertEqual(result["r15_classification"],
                         "REVIEW_REQUIRED_CAPACITY_NOT_CONFIRMED")

    def test_source_hash_cannot_prove_same_machine_or_speed_winner(self):
        second = self.producer("backup")
        result = review([self.directory, second.output])
        c = result["cross_campaign_comparison"]
        self.assertEqual(c["status"], "DESCRIPTIVE_ONLY_MATCHING_REPORTED_PROTOCOL")
        self.assertEqual(c["common_gpu_layers"], [0,12])
        self.assertFalse(c["speed_winner_declared"])
        self.assertFalse(c["same_physical_machine_proven"])
        self.assertFalse(result["physical_hardware_authenticated"])
        self.assertFalse(result["best_device_selected"])

    def test_different_binary_is_only_descriptive(self):
        second = self.producer("backup", binary_bytes=b"other-synthetic-llama-server")
        result = review([self.directory, second.output])
        self.assertEqual(
            result["cross_campaign_comparison"]["status"],
            "DESCRIPTIVE_ONLY_DIFFERENT_SERVER_BUILDS")

    def test_different_gguf_blocks_cross_campaign_comparison(self):
        second = self.producer("backup", model_bytes=b"OTHER_SYNTHETIC_MODEL")
        result = review([self.directory, second.output])
        self.assertEqual(result["cross_campaign_comparison"]["status"],
                         "HOLD_DIFFERENT_GGUF_BYTES")
        self.assertFalse(result["best_device_selected"])

    def test_different_workload_blocks_cross_campaign_comparison(self):
        second = self.producer("backup", settings={"ctx_size":4096})
        result = review([self.directory, second.output])
        self.assertEqual(result["cross_campaign_comparison"]["status"],
                         "HOLD_DIFFERENT_BENCHMARK_PROTOCOL")

    def test_two_same_campaign_paths_rejected(self):
        with self.assertRaisesRegex(CampaignReviewHeld, "DUPLICATED"):
            review([self.directory,self.directory])

    def test_output_inside_git_and_inside_source_refused(self):
        git = self.fixture.root/"repo"
        git.mkdir()
        (git/".git").mkdir()
        with self.assertRaisesRegex(CampaignReviewHeld, "INSIDE_GIT"):
            write_review([self.directory],git/"report.json")
        with self.assertRaisesRegex(CampaignReviewHeld, "CANNOT_MUTATE_SOURCE"):
            write_review([self.directory],self.directory/"review.json")

    def test_missing_manifest_on_partial_campaign_held(self):
        (self.directory/"campaign-manifest.json").unlink()
        with self.assertRaises(CampaignReviewHeld):
            review([self.directory])
        self.assertFalse((self.directory/"review.json").exists())

    def test_cli_help(self):
        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(["--help"])
        self.assertEqual(result.exception.code,0)


if __name__ == "__main__":
    unittest.main()
