"""GPU field receipts: synthetic nvidia-smi and local fake HTTP backend.

These test only process-telemetry plumbing and honest provenance labels.
They never execute model weights or provide independent CUDA evidence.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phiserve.acceptance import AcceptanceError, acceptance_report
from phiserve.cli import parser
from phiserve.field_telemetry import ProcessMemorySampler
from phiserve.registry import ModelRegistry
from phiserve.validation import validate_local


class FakeInference(BaseHTTPRequestHandler):
    empty = False

    def log_message(self, *_args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert request["model"] == "demo"
        assert request["stream"] is False
        body = json.dumps({
            "choices": [{"message": {
                "content": "" if self.empty else "synthetic-response",
                "role": "assistant",
            }}],
            "usage": {"completion_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class BackendStandIn:
    def __init__(self, **kwargs):
        self.process = SimpleNamespace(pid=24680)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class StubSampler:
    def __init__(self):
        self.calls = []

    def start(self, pid):
        self.calls.append(("start", pid))

    def stop(self):
        self.calls.append(("stop",))
        return {
            "schema": "PHISERVE_NVIDIA_PROCESS_SAMPLE_V1",
            "status": "PROCESS_MEMORY_DRIVER_SELF_REPORT",
            "subprocess_pid": 24680,
            "sample_attempts": 2,
            "matching_pid_samples": 1,
            "numeric_memory_samples": 1,
            "peak_driver_reported_process_memory_mib": 2048,
            "cuda_kernel_execution_independently_verified": False,
            "gpu_model_weights_residency_verified": False,
            "physical_hardware_witnessed": False,
        }


class NvidiaSamplerTests(unittest.TestCase):
    def fake(self, output, status=0):
        def runner(argv, **kwargs):
            self.assertEqual(argv[0], "nvidia-smi")
            self.assertFalse(kwargs["shell"])
            self.assertLessEqual(kwargs["timeout"], 3)
            return SimpleNamespace(returncode=status, stdout=output)
        return runner

    def test_driver_process_memory_matches_only_the_child_pid(self):
        sample = ProcessMemorySampler(runner=self.fake(
            "123, 3000\n24680, 1250\n456, 2048\n24680, 750\n"
        ))
        sample._pid = 24680
        sample.sample_once()
        observed = sample.stop()
        self.assertEqual(observed["status"], "PROCESS_MEMORY_DRIVER_SELF_REPORT")
        self.assertEqual(observed["sample_attempts"], 1)
        self.assertEqual(observed["matching_pid_samples"], 1)
        self.assertEqual(observed["peak_driver_reported_process_memory_mib"], 2000)
        self.assertFalse(observed["cuda_kernel_execution_independently_verified"])
        self.assertFalse(observed["gpu_model_weights_residency_verified"])

    def test_windows_wddm_na_is_not_claimed_as_gpu_memory(self):
        sample = ProcessMemorySampler(runner=self.fake("24680, N/A\n"))
        sample._pid = 24680
        sample.sample_once()
        result = sample.stop()
        self.assertEqual(result["status"], "PROCESS_PRESENT_MEMORY_UNAVAILABLE")
        self.assertIsNone(result["peak_driver_reported_process_memory_mib"])
        self.assertEqual(result["numeric_memory_samples"], 0)

    def test_missing_nvidia_tool_reports_unavailable(self):
        def missing(_args, **_kwargs):
            raise FileNotFoundError("nvidia-smi missing")
        sample = ProcessMemorySampler(runner=missing)
        sample._pid = 24680
        sample.sample_once()
        self.assertEqual(sample.stop()["status"], "NVIDIA_SMI_UNAVAILABLE")

    def test_query_failure_is_not_hidden_as_no_gpu(self):
        sample = ProcessMemorySampler(runner=self.fake("", status=1))
        sample._pid = 24680
        sample.sample_once()
        self.assertEqual(sample.stop()["status"], "NVIDIA_SMI_QUERY_FAILED")

    def test_no_matching_pid_is_distinct_from_unsupported_memory(self):
        sample = ProcessMemorySampler(runner=self.fake("77, 1024\n"))
        sample._pid = 24680
        sample.sample_once()
        result = sample.stop()
        self.assertEqual(result["status"], "NO_MATCHING_COMPUTE_PROCESS_OBSERVED")
        self.assertEqual(result["matching_pid_samples"], 0)

    def test_background_worker_stops_and_does_not_leak(self):
        sample = ProcessMemorySampler(interval=0.1, runner=self.fake("24680, 2048\n"))
        sample.start(24680)
        result = sample.stop()
        self.assertTrue(sample._stopped)
        self.assertGreaterEqual(result["sample_attempts"], 0)
        self.assertFalse(result["physical_hardware_witnessed"])
        with self.assertRaises(RuntimeError):
            sample.start(24680)

    def test_bad_sampler_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            ProcessMemorySampler(interval=0.01)
        with self.assertRaises(ValueError):
            ProcessMemorySampler().start(-1)
        with self.assertRaises(RuntimeError):
            ProcessMemorySampler().stop()


class OperatorFieldRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), FakeInference)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join(timeout=3)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.registry = ModelRegistry(self.base / "state")
        model = self.base / "dummy.gguf"
        model.write_bytes(b"GGUF" + b"fake-no-actual-weights")
        self.registry.add("demo", model)
        self.receipt = self.base / "gpu.json"
        self.summary = self.base / "summary.json"
        FakeInference.empty = False

    def test_local_inference_starts_monitor_with_supervised_child_pid(self):
        monitor = StubSampler()
        with patch("phiserve.validation.LlamaBackend", BackendStandIn):
            result = validate_local(
                "demo", "fake-llama-server", self.receipt,
                gpu_layers=12, backend_port=self.http.server_address[1],
                sampler=monitor, registry=self.registry,
            )
        self.assertEqual(monitor.calls, [("start", 24680), ("stop",)])
        self.assertEqual(result["gpu_layers_requested"], 12)
        self.assertEqual(result["gpu_process_telemetry"]["subprocess_pid"], 24680)
        self.assertFalse(result["gpu_execution_verified"])
        self.assertFalse(result["gpu_process_telemetry"]["cuda_kernel_execution_independently_verified"])
        self.assertNotIn("synthetic-response", self.receipt.read_text())
        self.assertNotIn("Write exactly one short greeting", self.receipt.read_text())

    def test_local_failure_stops_monitor_and_does_not_write_success_receipt(self):
        FakeInference.empty = True
        monitor = StubSampler()
        with patch("phiserve.validation.LlamaBackend", BackendStandIn):
            with self.assertRaises(RuntimeError):
                validate_local(
                    "demo", "fake-server", self.receipt,
                    backend_port=self.http.server_address[1],
                    sampler=monitor, registry=self.registry,
                )
        self.assertEqual(monitor.calls, [("start", 24680), ("stop",)])
        self.assertFalse(self.receipt.exists())

    def test_receipt_reviewer_preserves_unverified_gpu_status(self):
        monitor = StubSampler()
        with patch("phiserve.validation.LlamaBackend", BackendStandIn):
            validate_local(
                "demo", "fake-server", self.receipt,
                backend_port=self.http.server_address[1],
                gpu_layers=20, sampler=monitor, registry=self.registry,
            )
        summary = acceptance_report(self.summary, gpu_gguf=self.receipt)
        observation = summary["evidence"]["gguf_gpu"]["gpu_process_observation"]
        self.assertEqual(observation["peak_driver_reported_process_memory_mib"], 2048)
        self.assertFalse(observation["cuda_execution_independently_verified"])
        self.assertFalse(summary["gpu_execution_independently_verified"])

    def test_reviewer_rejects_fabricated_telemetry_success(self):
        monitor = StubSampler()
        with patch("phiserve.validation.LlamaBackend", BackendStandIn):
            validate_local(
                "demo", "fake-server", self.receipt,
                gpu_layers=20, backend_port=self.http.server_address[1],
                sampler=monitor, registry=self.registry,
            )
        payload = json.loads(self.receipt.read_text())
        payload["gpu_process_telemetry"]["cuda_kernel_execution_independently_verified"] = True
        self.receipt.write_text(json.dumps(payload))
        with self.assertRaises(AcceptanceError):
            acceptance_report(self.summary, gpu_gguf=self.receipt)
        self.assertFalse(self.summary.exists())

    def test_parser_has_explicit_field_run_telemetry_flag(self):
        args = parser().parse_args([
            "acceptance", "field-run", "demo", "--llama-server", "llama-server.exe",
            "--gpu-layers", "20", "--sample-nvidia", "--output", "gpu.json",
        ])
        self.assertEqual(args.action, "field-run")
        self.assertTrue(args.sample_nvidia)
        self.assertEqual(args.gpu_layers, 20)
        self.assertEqual(args.alias, "demo")


if __name__ == "__main__":
    unittest.main()
