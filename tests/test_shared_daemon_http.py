"""End-to-end daemon HTTP protocol tests with synthetic model processes.

The backend is a fake HTTP server; these never claim physical inference.
"""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from phiserve.cli import parser
from phiserve.manager import ModelManager
from phiserve.registry import ModelRegistry
from phiserve.server import make_server


class FakeProcess:
    def __init__(self):
        self.alive = True

    def poll(self):
        return None if self.alive else 1


class LocalStubBackend:
    instances = []

    def __init__(self, **kwargs):
        self.process = FakeProcess()
        self.alias = kwargs["alias"]
        self.stopped = False
        LocalStubBackend.instances.append(self)

    def start(self):
        pass

    def stop(self):
        self.process.alive = False
        self.stopped = True


class LocalStubAPI(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        alias = body["model"]
        if body.get("stream"):
            frames = [
                {"choices": [{"delta": {"content": "answer-" + alias}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            blob = (b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in frames)
                    + b"data: [DONE]\n\n")
            kind = "text/event-stream"
        else:
            blob = json.dumps({
                "choices": [{"message": {"content": "answer-" + alias}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3},
            }).encode()
            kind = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


class SharedDaemonHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        reg = ModelRegistry(Path(cls.tmp.name) / "reg")
        for alias in ("alpha", "beta"):
            model = Path(cls.tmp.name) / (alias + ".gguf")
            model.write_bytes(b"GGUF" + alias.encode())
            reg.add(alias, model)
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), LocalStubAPI)
        cls.manager = ModelManager(reg, "simulated-server",
                                   cls.upstream.server_address[1],
                                   backend_factory=LocalStubBackend,
                                   idle_seconds=0)
        cls.server = make_server("", cls.upstream.server_address[1], 0, manager=cls.manager)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.threads = [threading.Thread(target=cls.upstream.serve_forever, daemon=True),
                       threading.Thread(target=cls.server.serve_forever, daemon=True)]
        for t in cls.threads:
            t.start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.server, cls.upstream):
            server.shutdown()
            server.server_close()
        for t in cls.threads:
            t.join(timeout=3)
        cls.manager.close(grace_seconds=0)
        cls.tmp.cleanup()

    def post(self, path, payload):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req, timeout=5)

    def test_list_registered_models_and_status(self):
        with urllib.request.urlopen(self.url + "/api/tags") as response:
            tags = json.load(response)["models"]
        self.assertEqual([e["name"] for e in tags], ["alpha", "beta"])
        with urllib.request.urlopen(self.url + "/v1/models") as response:
            self.assertEqual([x["id"] for x in json.load(response)["data"]], ["alpha", "beta"])
        with urllib.request.urlopen(self.url + "/health") as response:
            self.assertEqual(json.load(response)["status"], "ok")

    def test_chat_loads_selected_model_and_switches(self):
        for alias in ("alpha", "beta"):
            with self.post("/api/chat", {
                "model": alias, "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            }) as response:
                reply = json.load(response)
            self.assertEqual(reply["model"], alias)
            self.assertEqual(reply["message"]["content"], "answer-" + alias)
            self.assertTrue(reply["done"])
            with urllib.request.urlopen(self.url + "/api/ps") as response:
                active = json.load(response)["models"]
            self.assertEqual([m["name"] for m in active], [alias])

    def test_streaming_keeps_ollama_ndjson(self):
        with self.post("/api/chat", {
            "model": "alpha", "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }) as response:
            self.assertEqual(response.headers.get_content_type(), "application/x-ndjson")
            events = [json.loads(line) for line in response.readlines()]
        self.assertEqual(events[0]["message"]["content"], "answer-alpha")
        self.assertTrue(events[-1]["done"])

    def test_busy_switch_returns_409_without_unloading_model(self):
        with self.manager.acquire("alpha"):
            resident = self.manager.status()["active_model"]
            with self.assertRaises(urllib.error.HTTPError) as result:
                self.post("/api/chat", {
                    "model": "beta", "messages": [{"role": "user", "content": "hello"}],
                })
            self.assertEqual(result.exception.code, 409)
            self.assertEqual(self.manager.status()["active_model"], resident)

    def test_unknown_alias_is_404_and_no_download(self):
        with self.assertRaises(urllib.error.HTTPError) as result:
            self.post("/api/generate", {"model": "not-registered", "prompt": "hello"})
        self.assertEqual(result.exception.code, 404)

    def test_invalid_ollama_request_is_422_before_loading(self):
        previous = self.manager.status()["active_model"]
        with self.assertRaises(urllib.error.HTTPError) as result:
            self.post("/api/chat", {
                "model": "beta", "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
            })
        self.assertEqual(result.exception.code, 422)
        self.assertEqual(self.manager.status()["active_model"], previous)

    def test_parser_has_new_daemon_mode(self):
        args = parser().parse_args(["daemon", "--idle-seconds", "0", "--max-concurrent", "2"])
        self.assertEqual(args.command, "daemon")
        self.assertEqual(args.idle_seconds, 0)
        self.assertEqual(args.max_concurrent, 2)


if __name__ == "__main__":
    unittest.main()
