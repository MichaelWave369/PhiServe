"""Benchmark contract tests: synthetic HTTP SSE only, no checkpoint inference."""
from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phiserve.benchmark import (
    BenchmarkError, PROMPT, _request_stream, compare_receipts, main,
    read_receipt, run_benchmark, validate_receipt,
)
from phiserve.registry import ModelRegistry


class SSEStandIn(BaseHTTPRequestHandler):
    mode = "valid"

    def log_message(self, *args):
        pass

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert data["stream"] is True
        assert data["temperature"] == 0
        assert data["seed"] == 42
        assert data["messages"][0]["content"] == PROMPT
        if self.mode == "http_error":
            self.send_response(422)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if self.mode != "plain" else "application/json")
        self.end_headers()
        if self.mode == "plain":
            self.wfile.write(b'{"reply":"fake"}')
            return
        items = [{"choices":[{"delta":{"content":"synthetic "}}]},
                 {"choices":[{"delta":{"content":"text"}}]}]
        if self.mode != "no_usage":
            items.append({"choices":[], "usage":{"completion_tokens":3}})
        for item in items:
            self.wfile.write(b"data: " + json.dumps(item).encode() + b"\n\n")
            self.wfile.flush()
        if self.mode != "incomplete":
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()


class BackendStandIn:
    def __init__(self, **kwargs):
        self.process = SimpleNamespace(pid=10101)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class SamplerStandIn:
    def __init__(self):
        self.calls = []

    def start(self, pid):
        self.calls.append(("start", pid))

    def stop(self):
        self.calls.append(("stop",))
        return {
            "schema":"PHISERVE_NVIDIA_PROCESS_SAMPLE_V1",
            "status":"PROCESS_PRESENT_MEMORY_UNAVAILABLE",
            "cuda_kernel_execution_independently_verified":False,
            "gpu_model_weights_residency_verified":False,
            "physical_hardware_witnessed":False,
        }


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SSEStandIn)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.model = self.directory / "dummy.gguf"
        self.model.write_bytes(b"GGUFsynthetic fake bytes")
        self.binary = self.directory / "fake-llama-server"
        self.binary.write_bytes(b"fake binary")
        self.registry = ModelRegistry(self.directory / "home")
        self.registry.add("test-model", self.model)
        self.path = self.directory / "field.json"
        SSEStandIn.mode = "valid"

    def run_field(self, **options):
        with patch("phiserve.benchmark.shutil.which", return_value=str(self.binary)):
            return run_benchmark(
                "test-model", str(self.binary), self.path,
                registry=self.registry, backend_cls=BackendStandIn,
                backend_port=self.server.server_address[1], warmups=1, trials=3,
                **options,
            )

    def test_real_sse_transport_records_first_content_and_reported_usage(self):
        t = _request_stream("test-model", self.server.server_address[1], 32, 3)
        self.assertGreater(t["wall_ms"], 0)
        self.assertLessEqual(t["first_content_ms"], t["wall_ms"])
        self.assertEqual(t["completion_tokens_reported"], 3)
        self.assertEqual(t["generated_chars"], 14)
        self.assertTrue(t["throughput_tps"] > 0)
        self.assertNotIn("synthetic text", str(t))

    def test_absent_token_usage_never_invents_tokens_per_second(self):
        SSEStandIn.mode = "no_usage"
        t = _request_stream("test-model", self.server.server_address[1], 32, 3)
        self.assertIsNone(t["throughput_tps"])
        self.assertIsNone(t["completion_tokens_reported"])

    def test_no_done_is_failure(self):
        SSEStandIn.mode = "incomplete"
        with self.assertRaises(BenchmarkError):
            _request_stream("test-model", self.server.server_address[1], 32, 3)

    def test_http_failure_and_non_stream_are_rejected(self):
        for mode in ("http_error", "plain"):
            SSEStandIn.mode = mode
            with self.assertRaises(BenchmarkError):
                _request_stream("test-model", self.server.server_address[1], 32, 3)

    def test_warmup_is_discarded_and_review_recalculates_summary(self):
        receipt = self.run_field()
        self.assertEqual(receipt["experiment"]["warmups"], 1)
        self.assertEqual(len(receipt["measurements"]), 3)
        self.assertEqual([t["index"] for t in receipt["measurements"]], [1,2,3])
        self.assertEqual(read_receipt(self.path)["model"]["gguf_sha256"], receipt["model"]["gguf_sha256"])
        self.assertFalse(receipt["unverified_limits"]["cuda_execution_independently_verified"])
        self.assertFalse(receipt["unverified_limits"]["performance_certified"])
        self.assertNotIn("synthetic text", self.path.read_text())
        self.assertNotIn(PROMPT, self.path.read_text())

    def test_sampling_is_opt_in_and_is_never_execution_attestation(self):
        monitor = SamplerStandIn()
        receipt = self.run_field(sampler=monitor)
        self.assertEqual(monitor.calls, [("start", 10101), ("stop",)])
        self.assertFalse(receipt["gpu_process_telemetry"]["cuda_kernel_execution_independently_verified"])

    def test_overwrite_refused_without_mutation(self):
        self.path.write_text("existing")
        with self.assertRaises(BenchmarkError):
            self.run_field()
        self.assertEqual(self.path.read_text(), "existing")

    def test_error_leaves_no_success_receipt(self):
        SSEStandIn.mode = "incomplete"
        with self.assertRaises(BenchmarkError):
            self.run_field()
        self.assertFalse(self.path.exists())

    def test_input_bounds_refuse_before_running(self):
        for key, value in (("trials", 2), ("ctx_size", 0), ("gpu_layers", -1),
                           ("max_tokens", 100000), ("backend_port", 0)):
            kwargs = {key:value}
            if key in ("trials", "backend_port"):
                with patch("phiserve.benchmark.shutil.which", return_value=str(self.binary)):
                    with self.assertRaises(BenchmarkError):
                        run_benchmark("test-model", str(self.binary), self.path,
                                      registry=self.registry, backend_cls=BackendStandIn, **kwargs)
            else:
                with self.assertRaises(BenchmarkError):
                    self.run_field(**kwargs)
            self.assertFalse(self.path.exists())

    def test_aggregate_tampering_is_refused(self):
        receipt = self.run_field()
        receipt["summary"]["p50_wall_ms"] += 99
        with self.assertRaises(BenchmarkError):
            validate_receipt(receipt)

    def test_forged_gpu_proof_and_unknown_fields_refused(self):
        receipt = self.run_field()
        r = copy.deepcopy(receipt)
        r["unverified_limits"]["cuda_execution_independently_verified"] = True
        with self.assertRaises(BenchmarkError):
            validate_receipt(r)
        r = copy.deepcopy(receipt)
        r["quality_score"] = 1.0
        with self.assertRaises(BenchmarkError):
            validate_receipt(r)

    def test_token_rate_tamper_or_claim_without_usage_refused(self):
        receipt = self.run_field()
        r = copy.deepcopy(receipt)
        r["measurements"][0]["throughput_tps"] *= 100
        with self.assertRaises(BenchmarkError):
            validate_receipt(r)
        r = copy.deepcopy(receipt)
        r["measurements"][0]["completion_tokens_reported"] = None
        with self.assertRaises(BenchmarkError):
            validate_receipt(r)

    def test_duplicate_keys_in_json_refused(self):
        self.path.write_text('{"schema":1,"schema":2}', encoding="utf-8")
        with self.assertRaises(BenchmarkError):
            read_receipt(self.path)

    def test_descriptive_comparison_requires_same_model_and_workload(self):
        a = self.run_field()
        b = copy.deepcopy(a)
        b["requested"]["gpu_layers"] = 12
        compared = compare_receipts(a,b)
        self.assertEqual(compared["status"], "DESCRIPTIVE_COMPARISON_ONLY")
        self.assertFalse(compared["performance_improvement_certified"])
        b["model"]["gguf_sha256"] = 'a' * 64
        with self.assertRaises(BenchmarkError):
            compare_receipts(a,b)
        b = copy.deepcopy(a)
        b["requested"]["ctx_size"] = 4096
        with self.assertRaises(BenchmarkError):
            compare_receipts(a,b)

    def test_cli_review_does_not_execute_model(self):
        self.run_field()
        self.assertEqual(main(["review", str(self.path)]), 0)


if __name__ == "__main__":
    unittest.main()
