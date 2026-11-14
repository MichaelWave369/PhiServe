"""Synthetic R15 hardware-advice tests: no CUDA, weights or network execution."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy
import hashlib
import json
import platform
from pathlib import Path
import tempfile
import unittest

from phiserve.benchmark import LIMITS, PROMPT, _summary, _hash_file
from phiserve.hardware_advisor import AdviceError, advise
from phiserve.registry import ModelRegistry


class HardwareAdviceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 11, 13, 19, 0, tzinfo=timezone.utc)
        self.registry = ModelRegistry(self.root / "models")
        self.model = self.root / "tiny.gguf"
        self.model.write_bytes(b"GGUF" + b"fake-test-weights" * 100)
        self.registry.add("tiny", self.model)
        self.snap_path = self.root / "hardware.json"
        self.bench_path = self.root / "bench.json"
        self.bench2_path = self.root / "bench2.json"
        self.out_path = self.root / "advice.json"
        self.snapshot = {
            "schema": "PHISTREAM_HARDWARE_SNAPSHOT_V1",
            "status": "OBSERVED_READ_ONLY",
            "provenance": "LOCAL_HARDWARE_COUNTERS_NOT_A_BENCHMARK",
            "observed_utc": (self.now - timedelta(seconds=15)).isoformat(),
            "platform": platform.platform(),
            "logical_cpu_count": 16,
            "ram": {"total_bytes": 32 * 1024**3,
                    "available_bytes": 24 * 1024**3,
                    "source": "WIN32_GLOBAL_MEMORY_STATUS_EX"
                              if platform.system() == "Windows" else
                              "LINUX_PROC_MEMINFO_MEMAVAILABLE"},
            "gpu": {
                "device": "cuda",
                "name": "synthetic-RTX",
                "device_index": 0,
                "compute_capability": "8.9",
                "total_bytes": 12 * 1024**3,
                "free_bytes": 10 * 1024**3,
                "source": "TORCH_CUDA_MEM_GET_INFO",
                "torch_version": "synthetic",
                "cuda_build": "synthetic",
            },
            "gpu_requested": "cuda",
            "drives": [{
                "requested_path": "C:\\Users\\secret\\Downloads",
                "free_bytes": 900 * 1024**3,
                "total_bytes": 1024**4,
                "source": "PYTHON_SHUTIL_DISK_USAGE",
                "throughput_measured": False,
            }],
            "limitations": ["synthetic local-only observation"],
        }
        self.benchmark = self.make_benchmark(gpu_layers=0)
        self.flush()

    def make_benchmark(self, gpu_layers=0):
        trials = []
        for i in range(1, 4):
            wall = 100.0 + i
            trials.append({
                "index": i,
                "wall_ms": wall,
                "first_content_ms": 10.0 + i,
                "generated_chars": 32,
                "completion_tokens_reported": 5,
                "throughput_tps": 5000 / wall,
                "response_sha256": "a" * 64,
                "end_marker_observed": True,
            })
        return {
            "schema": "PHISERVE_BENCHMARK_FIELD_V1",
            "status": "LOCAL_OPERATOR_OBSERVATION_UNATTESTED",
            "created_utc": (self.now - timedelta(hours=2)).isoformat(),
            "experiment": {
                "name": "fixed-public-prompt-v1",
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "warmups": 3, "trials": 3,
            },
            "model": {"alias": "tiny", "gguf_sha256": _hash_file(self.model),
                      "size_bytes": self.model.stat().st_size},
            "runtime": {"binary_name": "llama-server", "binary_sha256": "b" * 64,
                        "phiserve_version": "0.10.0",
                        "platform_family": platform.system(), "startup_ms": 250.0},
            "requested": {"gpu_layers": gpu_layers, "ctx_size": 2048,
                          "max_tokens": 64, "seed": 42, "temperature": 0,
                          "backend_port": 11436},
            "measurements": trials,
            "summary": _summary(trials),
            "gpu_process_telemetry": None,
            "unverified_limits": dict(LIMITS),
        }

    def flush(self):
        self.snap_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
        self.bench_path.write_text(json.dumps(self.benchmark), encoding="utf-8")

    def advise(self, bench_files=None, **kwargs):
        return advise(self.snap_path, [self.bench_path] if bench_files is None else bench_files,
                      "tiny", self.out_path, registry=self.registry, now=self.now, **kwargs)

    def test_real_benchmark_receipt_shape_and_fresh_candidate(self):
        result = self.advise()
        self.assertEqual(result["classification"], "REVIEW_REQUIRED_CAPACITY_NOT_CONFIRMED")
        self.assertTrue(result["capacity_floor"]["full_cpu_weight_file_floor_met"])
        self.assertTrue(result["capacity_floor"]["full_gpu_weight_file_floor_met"])
        self.assertFalse(result["capacity_floor"]["actual_model_fit_verified"])
        self.assertEqual(result["benchmark_profiles"][0]["startup_ms"], 250)
        self.assertFalse(result["assurances"]["automatic_model_route"])
        self.assertFalse(result["assurances"]["execution_authorized"])

    def test_private_snapshot_drive_path_is_not_written_to_advice(self):
        self.advise()
        text = self.out_path.read_text()
        self.assertNotIn("secret", text)
        self.assertNotIn("Downloads", text)
        self.assertNotIn(str(self.model), text)
        self.assertNotIn("synthetic-RTX", text)

    def test_stale_snapshot_holds_instead_of_autorouting(self):
        self.snapshot["observed_utc"] = (self.now - timedelta(hours=1)).isoformat()
        self.flush()
        result = self.advise()
        self.assertEqual(result["classification"], "HOLD_HARDWARE_OBSERVATION_STALE_OR_FUTURE")

    def test_future_timestamp_holds(self):
        self.snapshot["observed_utc"] = (self.now + timedelta(minutes=5)).isoformat()
        self.flush()
        self.assertEqual(self.advise()["classification"],
                         "HOLD_HARDWARE_OBSERVATION_STALE_OR_FUTURE")

    def test_old_benchmark_holds(self):
        self.benchmark["created_utc"] = (self.now - timedelta(days=9)).isoformat()
        self.flush()
        self.assertEqual(self.advise()["classification"],
                         "HOLD_BENCHMARK_EVIDENCE_STALE_OR_FUTURE")

    def test_missing_gpu_counters_blocks_gpu_advice(self):
        self.snapshot["gpu_requested"] = "none"
        self.snapshot["gpu"] = None
        self.benchmark = self.make_benchmark(gpu_layers=20)
        self.flush()
        result = self.advise()
        self.assertEqual(result["classification"], "HOLD_CUDA_COUNTERS_MISSING")
        self.assertIsNone(result["capacity_floor"]["gpu_budget_bytes"])
        self.assertIsNone(result["capacity_floor"]["full_gpu_weight_file_floor_met"])

    def test_low_gpu_budget_never_proves_model_failure_or_fit(self):
        self.snapshot["gpu"]["free_bytes"] = 100
        self.benchmark = self.make_benchmark(gpu_layers=20)
        self.flush()
        result = self.advise()
        self.assertFalse(result["capacity_floor"]["full_gpu_weight_file_floor_met"])
        self.assertFalse(result["capacity_floor"]["hybrid_offload_fit_evaluated"])
        self.assertEqual(result["classification"], "REVIEW_REQUIRED_CAPACITY_NOT_CONFIRMED")

    def test_compare_multiple_distinct_GPU_profiles(self):
        second = self.make_benchmark(gpu_layers=10)
        self.bench2_path.write_text(json.dumps(second), encoding="utf-8")
        result = self.advise([self.bench_path, self.bench2_path])
        self.assertEqual([p["gpu_layers_requested"] for p in result["benchmark_profiles"]], [0, 10])
        self.assertEqual(len(result["benchmark_profiles"]), 2)

    def test_duplicate_gpu_offload_profile_is_refused(self):
        self.bench2_path.write_text(json.dumps(self.benchmark), encoding="utf-8")
        with self.assertRaises(AdviceError):
            self.advise([self.bench_path, self.bench2_path])
        self.assertFalse(self.out_path.exists())

    def test_changed_model_digest_is_refused(self):
        self.benchmark["model"]["gguf_sha256"] = "0" * 64
        self.flush()
        with self.assertRaises(AdviceError):
            self.advise()
        self.assertFalse(self.out_path.exists())

    def test_unmatched_context_is_refused(self):
        second = self.make_benchmark(gpu_layers=20)
        second["requested"]["ctx_size"] = 4096
        self.bench2_path.write_text(json.dumps(second), encoding="utf-8")
        with self.assertRaises(AdviceError):
            self.advise([self.bench_path, self.bench2_path])

    def test_fake_platform_provenance_is_refused(self):
        self.benchmark["runtime"]["platform_family"] = "Mars"
        self.flush()
        with self.assertRaises(AdviceError):
            self.advise()

    def test_duplicate_key_in_snapshot_is_refused(self):
        self.snap_path.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        with self.assertRaises(AdviceError):
            self.advise()

    def test_nonfinite_and_bool_memory_counters_are_refused(self):
        self.snapshot["ram"]["available_bytes"] = True
        self.flush()
        with self.assertRaises(AdviceError):
            self.advise()
        self.snapshot["ram"]["available_bytes"] = 4096
        self.flush()
        self.snap_path.write_text(self.snap_path.read_text().replace("4096", "NaN"), encoding="utf-8")
        with self.assertRaises(AdviceError):
            self.advise()

    def test_false_claimed_gpu_authentication_is_refused(self):
        self.benchmark["unverified_limits"]["cuda_execution_independently_verified"] = True
        self.flush()
        with self.assertRaises(AdviceError):
            self.advise()

    def test_overwrite_and_same_input_are_refused(self):
        self.out_path.write_text("do not overwrite", encoding="utf-8")
        with self.assertRaises(AdviceError):
            self.advise()
        self.assertEqual(self.out_path.read_text(), "do not overwrite")
        self.out_path.unlink()
        with self.assertRaises(AdviceError):
            self.advise([self.snap_path])

    def test_external_symlink_snapshot_is_rejected(self):
        other = self.root / "true.json"
        other.write_text(self.snap_path.read_text(), encoding="utf-8")
        self.snap_path.unlink()
        try:
            self.snap_path.symlink_to(other)
        except (OSError, NotImplementedError):
            self.skipTest("Local symlink creation restricted")
        with self.assertRaises(AdviceError):
            self.advise()

    def test_limits_and_empty_inputs_refused(self):
        for value in (-1, True, 1024 * 1024 + 1):
            with self.assertRaises(AdviceError):
                self.advise(ram_reserve_mib=value)
        with self.assertRaises(AdviceError):
            self.advise(bench_files=[])
        self.assertFalse(self.out_path.exists())


if __name__ == "__main__":
    unittest.main()
