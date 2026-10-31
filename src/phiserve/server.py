"""Minimal loopback-only API gateway, forwarding actual llama.cpp output."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


MAX_BODY_BYTES = 2 * 1024 * 1024
POST_ROUTES = {"/v1/chat/completions", "/v1/completions"}


class PhiServeHTTP(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], model_alias: str, backend_port: int):
        super().__init__(address, APIHandler)
        self.model_alias = model_alias
        self.backend_port = backend_port


class APIHandler(BaseHTTPRequestHandler):
    server: PhiServeHTTP
    protocol_version = "HTTP/1.0"

    def _json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{
                    "id": self.server.model_alias,
                    "object": "model",
                    "owned_by": "local",
                    "created": 0,
                }],
            })
        elif path == "/health":
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.server.backend_port}/health", timeout=2
                ) as response:
                    state = json.load(response)
                    if response.status != 200 or state.get("status") not in {"ok", "ready"}:
                        raise ValueError("Backend not ready")
                self._json(200, {"status": "ok", "runtime": "local", "backend": "llama.cpp", "model": self.server.model_alias})
            except (OSError, ValueError, urllib.error.HTTPError):
                self._json(503, {"status": "unavailable", "backend": "llama.cpp"})
        else:
            self._json(404, {"error": {"message": "Unknown route"}})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in POST_ROUTES:
            self._json(404, {"error": {"message": "Unknown route"}})
            return
        length_header = self.headers.get("Content-Length")
        if self.headers.get("Transfer-Encoding") is not None or length_header is None:
            self._json(411, {"error": {"message": "Content-Length required; chunked requests unsupported"}})
            return
        try:
            length = int(length_header)
        except ValueError:
            self._json(400, {"error": {"message": "Invalid Content-Length"}})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(413, {"error": {"message": "Request body must be 1 byte to 2 MiB"}})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": {"message": "Content-Type must be application/json"}})
            return
        try:
            request = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            self._json(400, {"error": {"message": "Invalid JSON"}})
            return
        if not isinstance(request, dict) or request.get("model") != self.server.model_alias:
            self._json(400, {"error": {"message": "Model must match the active alias"}})
            return
        if "stream" in request and not isinstance(request["stream"], bool):
            self._json(400, {"error": {"message": "stream must be boolean"}})
            return
        payload = json.dumps(request, separators=(",", ":")).encode("utf-8")
        self._forward(path, payload)

    def _forward(self, path: str, payload: bytes) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.server.backend_port, timeout=600)
        headers_sent = False
        try:
            conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
            upstream = conn.getresponse()
            self.send_response(upstream.status)
            content_type = upstream.getheader("Content-Type", "application/json")
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            headers_sent = True
            # read1 allows SSE chunks through promptly without buffering full generations.
            while True:
                chunk = upstream.read1(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            if not headers_sent:
                self._json(502, {"error": {"message": "Local inference backend unavailable"}})
            # Once streaming headers are sent, the only honest action is to close.
        finally:
            self.close_connection = True
            conn.close()


def make_server(model_alias: str, backend_port: int, listen_port: int = 11435) -> PhiServeHTTP:
    # Intentionally no bind address argument until authentication is implemented.
    return PhiServeHTTP(("127.0.0.1", listen_port), model_alias, backend_port)
