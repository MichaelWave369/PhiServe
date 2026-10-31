"""Simulated receipt tests. These do NOT certify hardware inference."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from phiserve.registry import ModelRegistry
from phiserve.validation import validate_local


class FakeInference(BaseHTTPRequestHandler):
    empty = False

    def log_message(self, *_args):
        pass

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert request["stream"] is False
        result = {"choices": [{"message": {
            "role": "assistant",
            "content": "" if self.empty else "Synthetic local test",
        }}], "usage": {"completion_tokens": 4}}
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class StandInBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class ValidationTests(unittest.TestCase):
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
        self.registry = ModelRegistry(Path(self.tmp.name) / "state")
        self.model = Path(self.tmp.name) / "dummy.gguf"
        self.model.write_bytes(b"GGUF" + b"not-a-real-checkpoint")
        self.registry.add("demo", self.model)
        self.output = Path(self.tmp.name) / "receipt.json"
        FakeInference.empty = False

    def test_observation_receipt_does_not_save_prompt_or_output(self):
        with patch("phiserve.validation.LlamaBackend", StandInBackend):
            receipt = validate_local(
                "demo", "fake-llama-server", self.output,
                backend_port=self.http.server_address[1],
                registry=self.registry)
        self.assertEqual(receipt["schema"], "PHISERVE_LOCAL_INFERENCE_RECEIPT_V1")
        self.assertEqual(receipt["status"], "LOCAL_TEXT_OUTPUT_OBSERVED")
        self.assertFalse(receipt["gpu_execution_verified"])
        self.assertFalse(receipt["raw_prompt_or_output_stored"])
        self.assertEqual(receipt["completion_tokens_reported"], 4)
        self.assertEqual(receipt["output_characters"], len("Synthetic local test"))
        raw = self.output.read_text()
        self.assertNotIn("Synthetic local test", raw)
        self.assertNotIn("Write exactly one short greeting.", raw)

    def test_refuse_overwrite(self):
        self.output.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "already exists"):
            validate_local("demo", "llama-server", self.output, registry=self.registry)
        self.assertEqual(self.output.read_text(), "preserve")

    def test_no_receipt_when_text_empty(self):
        FakeInference.empty = True
        with patch("phiserve.validation.LlamaBackend", StandInBackend):
            with self.assertRaisesRegex(RuntimeError, "did not return"):
                validate_local("demo", "fake-llama-server", self.output,
                               backend_port=self.http.server_address[1],
                               registry=self.registry)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
