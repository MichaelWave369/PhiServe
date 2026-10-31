"""Protocol-level checks for the deliberately limited Ollama API adapter."""

import http.client
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from phiserve.ollama_compat import translate
from phiserve.server import make_server


class StubLlama(BaseHTTPRequestHandler):
    last_request = None

    def log_message(self, *_args):
        pass

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubLlama.last_request = (self.path, data)
        if data.get("stream"):
            if self.path == "/v1/chat/completions":
                frames = [
                    {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ]
            else:
                frames = [
                    {"choices": [{"text": "Hel", "finish_reason": None}]},
                    {"choices": [{"text": "lo", "finish_reason": None}]},
                    {"choices": [{"text": "", "finish_reason": "stop"}]},
                ]
            frames.append({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}})
            blob = b"".join(b"data: " + json.dumps(x).encode() + b"\n\n" for x in frames) + b"data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            choice = (
                {"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                if self.path == "/v1/chat/completions"
                else {"text": "Hello", "finish_reason": "stop"}
            )
            blob = json.dumps({
                "choices": [choice],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            }).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


class OllamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        model = Path(cls.tmp.name) / "real-file.gguf"
        model.write_bytes(b"GGUFmetadata")
        cls.backend = ThreadingHTTPServer(("127.0.0.1", 0), StubLlama)
        cls.front = make_server("local:1", cls.backend.server_address[1], 0, model_path=model)
        cls.base = f"http://127.0.0.1:{cls.front.server_address[1]}"
        cls.threads = [
            threading.Thread(target=cls.backend.serve_forever, daemon=True),
            threading.Thread(target=cls.front.serve_forever, daemon=True),
        ]
        for t in cls.threads:
            t.start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.front, cls.backend):
            server.shutdown()
            server.server_close()
        for t in cls.threads:
            t.join(timeout=3)
        cls.tmp.cleanup()

    def post(self, path, body):
        return urllib.request.urlopen(urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST"), timeout=3)

    def test_tags_real_metadata_and_no_fake_digest(self):
        with urllib.request.urlopen(self.base + "/api/tags") as response:
            data = json.load(response)["models"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "local:1")
        self.assertEqual(data[0]["size"], len(b"GGUFmetadata"))
        self.assertNotIn("digest", data[0])
        self.assertEqual(data[0]["details"]["format"], "gguf")

    def test_chat_defaults_to_ndjson_stream(self):
        with self.post("/api/chat", {"model": "local:1", "messages": [{"role": "user", "content": "hi"}]}) as r:
            self.assertEqual(r.headers.get_content_type(), "application/x-ndjson")
            frames = [json.loads(line) for line in r.readlines()]
        self.assertEqual([x["message"]["content"] for x in frames[:-1]], ["Hel", "lo"])
        self.assertTrue(all(not x["done"] for x in frames[:-1]))
        self.assertTrue(frames[-1]["done"])
        self.assertEqual(frames[-1]["done_reason"], "stop")
        self.assertEqual(frames[-1]["prompt_eval_count"], 7)
        self.assertEqual(frames[-1]["eval_count"], 2)
        self.assertIn("total_duration", frames[-1])
        self.assertEqual(StubLlama.last_request[0], "/v1/chat/completions")
        self.assertTrue(StubLlama.last_request[1]["stream"])

    def test_chat_nonstream_and_options(self):
        body = {"model": "local:1", "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.6, "num_predict": 12, "seed": 3}}
        with self.post("/api/chat", body) as r:
            data = json.load(r)
        self.assertTrue(data["done"])
        self.assertEqual(data["message"]["content"], "Hello")
        upstream = StubLlama.last_request[1]
        self.assertEqual(upstream["max_tokens"], 12)
        self.assertEqual(upstream["seed"], 3)
        self.assertEqual(upstream["temperature"], 0.6)
        self.assertEqual(data["eval_count"], 2)

    def test_generate_uses_chat_template_unless_raw(self):
        with self.post("/api/generate", {"model": "local:1", "prompt": "Tell me", "system": "Be brief", "stream": False}) as r:
            data = json.load(r)
        self.assertEqual(data["response"], "Hello")
        self.assertEqual(StubLlama.last_request[0], "/v1/chat/completions")
        self.assertEqual(StubLlama.last_request[1]["messages"][0]["role"], "system")
        with self.post("/api/generate", {"model": "local:1", "prompt": "RAW", "raw": True, "stream": False}) as r:
            self.assertEqual(json.load(r)["response"], "Hello")
        self.assertEqual(StubLlama.last_request[0], "/v1/completions")
        self.assertEqual(StubLlama.last_request[1]["prompt"], "RAW")

    def test_generate_streaming_shape(self):
        with self.post("/api/generate", {"model": "local:1", "prompt": "Hello"}) as r:
            frames = [json.loads(line) for line in r.readlines()]
        self.assertEqual([e["response"] for e in frames[:-1]], ["Hel", "lo"])
        self.assertTrue(frames[-1]["done"])
        with self.post("/api/generate", {"model": "local:1", "prompt": "Hello", "raw": True}) as r:
            raw_frames = [json.loads(line) for line in r.readlines()]
        self.assertEqual([e["response"] for e in raw_frames[:-1]], ["Hel", "lo"])
        self.assertTrue(raw_frames[-1]["done"])

    def test_unsupported_and_invalid_rejected_not_ignored(self):
        invalid = [
            ("/api/chat", {"model": "local:1", "messages": [{"role": "user", "content": "hi"}], "tools": []}),
            ("/api/generate", {"model": "local:1", "prompt": "hi", "options": {"num_ctx": 4096}}),
            ("/api/generate", {"model": "local:1", "prompt": "hi", "keep_alive": 0}),
            ("/api/chat", {"model": "local:1", "messages": [{"role": "user", "content": "hi", "images": ["..."]}]}),
            ("/api/generate", {"model": "local:1", "prompt": "hi", "raw": True, "system": "x"}),
        ]
        for path, request in invalid:
            with self.subTest(request=request), self.assertRaises(urllib.error.HTTPError) as cm:
                self.post(path, request)
            self.assertEqual(cm.exception.code, 422)

    def test_wrong_model_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("/api/chat", {"model": "not-active", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(cm.exception.code, 400)

    def test_translate_rejects_bad_options(self):
        with self.assertRaises(ValueError):
            translate("/api/generate", {"model": "local:1", "prompt": "x", "options": {"temperature": -1}})


if __name__ == "__main__":
    unittest.main()
