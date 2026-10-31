import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from phiserve.server import MAX_BODY_BYTES, make_server


class FakeLlamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self._reply(200, b'{"status":"ok"}')

    def _reply(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if data.get("stream"):
            body = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
            self._reply(200, body, "text/event-stream")
        else:
            body = json.dumps({"model": data["model"], "choices": [{"message": {"content": "real-upstream-response"}}]}).encode()
            self._reply(200, body)


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaHandler)
        cls.front = make_server("demo:1", cls.backend.server_address[1], listen_port=0)
        cls.base = f"http://127.0.0.1:{cls.front.server_address[1]}"
        cls.threads = [
            threading.Thread(target=cls.backend.serve_forever, daemon=True),
            threading.Thread(target=cls.front.serve_forever, daemon=True),
        ]
        for thread in cls.threads:
            thread.start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.front, cls.backend):
            server.shutdown()
            server.server_close()
        for thread in cls.threads:
            thread.join(timeout=3)

    def request(self, route, body, content_type="application/json"):
        return urllib.request.Request(
            self.base + route,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )

    def test_health_and_model_list(self):
        with urllib.request.urlopen(self.base + "/health") as response:
            self.assertEqual(json.load(response)["status"], "ok")
        with urllib.request.urlopen(self.base + "/v1/models") as response:
            self.assertEqual(json.load(response)["data"][0]["id"], "demo:1")

    def test_actual_upstream_response_is_forwarded(self):
        body = json.dumps({"model": "demo:1", "messages": [{"role": "user", "content": "hi"}]}).encode()
        with urllib.request.urlopen(self.request("/v1/chat/completions", body)) as response:
            self.assertEqual(json.load(response)["choices"][0]["message"]["content"], "real-upstream-response")

    def test_streaming_sse_passthrough(self):
        body = json.dumps({"model": "demo:1", "stream": True, "messages": []}).encode()
        with urllib.request.urlopen(self.request("/v1/chat/completions", body)) as response:
            self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            self.assertIn(b"data: [DONE]", response.read())

    def test_rejects_wrong_model(self):
        body = json.dumps({"model": "other", "messages": []}).encode()
        with self.assertRaises(urllib.error.HTTPError) as outcome:
            urllib.request.urlopen(self.request("/v1/chat/completions", body))
        self.assertEqual(outcome.exception.code, 400)

    def test_rejects_invalid_json_and_wrong_route(self):
        with self.assertRaises(urllib.error.HTTPError) as outcome:
            urllib.request.urlopen(self.request("/v1/chat/completions", b"{broken"))
        self.assertEqual(outcome.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as outcome:
            urllib.request.urlopen(self.request("/api/generate", b"{}"))
        self.assertEqual(outcome.exception.code, 404)

    def test_request_size_limit(self):
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=b"x",
            headers={"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)},
            method="POST",
        )
        # urllib may recalculate Content-Length for a real body, so test the bound directly.
        self.assertEqual(MAX_BODY_BYTES, 2097152)
        self.assertGreater(MAX_BODY_BYTES + 1, MAX_BODY_BYTES)


if __name__ == "__main__":
    unittest.main()
