"""Opt-in CORS and bearer-token boundary tests; no real model execution."""

import http.client
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from phiserve.security import BrowserAccess, create_token_file, load_access, validate_origin
from phiserve.server import make_server

TRUSTED = "https://michaelwave369.github.io"
TOKEN = "private-local-test-token-that-is-not-saved-anywhere"


class FakeLlamaAPI(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if data.get("stream"):
            body = (b'data: {"choices":[{"delta":{"content":"simulated"},"finish_reason":null}]}\n\n'
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    b'data: [DONE]\n\n')
            kind = "text/event-stream"
        else:
            body = b'{"choices":[{"message":{"content":"simulated"},"finish_reason":"stop"}]}'
            kind = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class BrowserBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaAPI)
        cls.front = make_server("demo", cls.backend.server_address[1], 0,
                                browser_access=BrowserAccess(TRUSTED, TOKEN))
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

    def request(self, path, method="GET", token=TOKEN, origin=TRUSTED, body=None):
        headers = {}
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        if origin is not None:
            headers["Origin"] = origin
        if body is not None:
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(self.base + path, method=method, headers=headers,
                                      data=json.dumps(body).encode() if body is not None else None)

    def test_unauthenticated_requests_are_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as result:
            urllib.request.urlopen(self.request("/api/tags", token=None), timeout=3)
        self.assertEqual(result.exception.code, 401)
        self.assertEqual(result.exception.headers["Access-Control-Allow-Origin"], TRUSTED)
        with self.assertRaises(urllib.error.HTTPError) as result:
            urllib.request.urlopen(self.request("/api/chat", method="POST", token=None,
                body={"model":"demo", "messages":[{"role":"user", "content":"hi"}]}), timeout=3)
        self.assertEqual(result.exception.code, 401)

    def test_origin_restriction_with_and_without_correct_token(self):
        with self.assertRaises(urllib.error.HTTPError) as result:
            urllib.request.urlopen(self.request("/api/tags", origin="https://untrusted.example"), timeout=3)
        self.assertEqual(result.exception.code, 403)
        self.assertIsNone(result.exception.headers.get("Access-Control-Allow-Origin"))
        with urllib.request.urlopen(self.request("/api/tags"), timeout=3) as response:
            self.assertEqual(json.load(response)["models"][0]["name"], "demo")
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], TRUSTED)

    def test_valid_same_host_nonbrowser_client_with_token(self):
        with urllib.request.urlopen(self.request("/api/tags", origin=None), timeout=3) as response:
            self.assertEqual(response.status, 200)
            self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

    def test_preflight_only_allows_expected_origin_and_headers(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.front.server_address[1], timeout=3)
        try:
            connection.request("OPTIONS", "/api/chat", headers={
                "Origin": TRUSTED,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
                "Access-Control-Request-Private-Network": "true",
            })
            response = connection.getresponse()
            self.assertEqual(response.status, 204)
            self.assertEqual(response.getheader("Access-Control-Allow-Private-Network"), "true")
            self.assertEqual(response.getheader("Access-Control-Allow-Origin"), TRUSTED)
            response.read()
        finally:
            connection.close()
        connection = http.client.HTTPConnection("127.0.0.1", self.front.server_address[1], timeout=3)
        try:
            connection.request("OPTIONS", "/api/chat", headers={
                "Origin": TRUSTED,
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "authorization",
            })
            self.assertEqual(connection.getresponse().status, 403)
        finally:
            connection.close()

    def test_ollama_ndjson_stream_and_openai_sse_include_allowed_cors(self):
        body = {"model": "demo", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        with urllib.request.urlopen(self.request("/api/chat", method="POST", body=body), timeout=3) as response:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], TRUSTED)
            self.assertEqual(response.headers.get_content_type(), "application/x-ndjson")
            events = [json.loads(line) for line in response.readlines()]
            self.assertTrue(events[-1]["done"])
        with urllib.request.urlopen(self.request("/v1/chat/completions", method="POST", body=body), timeout=3) as response:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], TRUSTED)
            self.assertEqual(response.headers.get_content_type(), "text/event-stream")
            self.assertIn(b"data: [DONE]", response.read())

    def test_wrong_token_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as result:
            urllib.request.urlopen(self.request("/health", token="wrong"), timeout=3)
        self.assertEqual(result.exception.code, 401)


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "private" / "token.txt"

    def test_create_secret_exclusively_and_load(self):
        create_token_file(self.file)
        self.assertTrue(self.file.is_file())
        key = self.file.read_text().strip()
        self.assertGreaterEqual(len(key), 32)
        self.assertEqual(load_access(TRUSTED, self.file), BrowserAccess(TRUSTED, key))
        with self.assertRaises(FileExistsError):
            create_token_file(self.file)
        self.assertEqual(self.file.read_text().strip(), key)
        if os.name != "nt":
            self.assertEqual(self.file.stat().st_mode & 0o077, 0)

    def test_origin_must_be_https_or_localhost(self):
        for invalid in (
            "*", "null", "http://evil.example", "https://site.example/a",
            "https://site.example/", "https://site.example?x=1",
            "https://user:pass@site.example", "http://192.168.0.4:11435",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_origin(invalid)
        self.assertEqual(validate_origin("http://localhost:5173"), "http://localhost:5173")
        self.assertEqual(validate_origin(TRUSTED), TRUSTED)

    def test_origin_and_file_are_paired(self):
        create_token_file(self.file)
        with self.assertRaises(ValueError):
            load_access(TRUSTED, None)
        with self.assertRaises(ValueError):
            load_access(None, self.file)
        self.assertIsNone(load_access(None, None))


if __name__ == "__main__":
    unittest.main()
