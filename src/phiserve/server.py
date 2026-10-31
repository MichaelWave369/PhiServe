"""Minimal loopback-only API gateway, forwarding actual llama.cpp output."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .ollama_compat import handle_ollama, model_tags, translate, UnsupportedRequest
from .manager import ModelManagerError
from .security import BrowserAccess


MAX_BODY_BYTES = 2 * 1024 * 1024
POST_ROUTES = {"/v1/chat/completions", "/v1/completions", "/api/chat", "/api/generate"}


class PhiServeHTTP(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], model_alias: str, backend_port: int, model_path=None, manager=None, browser_access: BrowserAccess | None = None):
        super().__init__(address, APIHandler)
        self.model_alias = model_alias
        self.backend_port = backend_port
        self.model_path = model_path
        self.manager = manager
        self.browser_access = browser_access


class APIHandler(BaseHTTPRequestHandler):
    server: PhiServeHTTP
    protocol_version = "HTTP/1.0"

    def _cors_headers(self) -> None:
        access = self.server.browser_access
        if access is not None and self.headers.get("Origin") == access.origin:
            self.send_header("Access-Control-Allow-Origin", access.origin)
            self.send_header("Vary", "Origin")

    def _security_gate(self) -> bool:
        access = self.server.browser_access
        if access is None:
            return True
        origin = self.headers.get("Origin")
        if origin is not None and origin != access.origin:
            self._json(403, {"error": "Browser origin not permitted"})
            return False
        if not access.check(self.headers.get("Authorization", "")):
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.send_header("Cache-Control", "no-store")
            body = b'{"error":"Valid local bridge bearer token required"}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False
        return True

    def do_OPTIONS(self) -> None:
        access = self.server.browser_access
        origin = self.headers.get("Origin")
        method = self.headers.get("Access-Control-Request-Method", "").upper()
        requested = self.headers.get("Access-Control-Request-Headers", "")
        names = {value.strip().lower() for value in requested.split(",") if value.strip()}
        if (access is None or origin != access.origin or method not in {"GET", "POST"}
                or not names.issubset({"authorization", "content-type"})):
            self._json(403, {"error": "CORS preflight denied"})
            return
        self.send_response(204)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if not self._security_gate():
            return
        path = urlsplit(self.path).path
        manager = self.server.manager
        if path == "/api/tags":
            if manager is None:
                self._json(200, model_tags(self.server.model_alias, self.server.model_path))
            else:
                entries = []
                for alias, model_path in sorted(manager.catalog().items()):
                    entries.extend(model_tags(alias, model_path)["models"])
                self._json(200, {"models": entries})
        elif path == "/api/ps":
            if manager is None:
                models = model_tags(self.server.model_alias, self.server.model_path)["models"]
            else:
                state = manager.status()
                models = (model_tags(state["active_model"], state["model_path"])["models"]
                          if state["active_model"] is not None and state["backend_alive"] else [])
            self._json(200, {"models": models})
        elif path == "/v1/models":
            aliases = list(manager.catalog()) if manager is not None else [self.server.model_alias]
            self._json(200, {
                "object": "list",
                "data": [{"id": alias, "object": "model", "owned_by": "local", "created": 0}
                         for alias in sorted(aliases)],
            })
        elif path == "/health":
            if manager is not None:
                state = manager.status()
                ready = not state["closed"] and not state["transitioning"]
                if state["active_model"] and not state["backend_alive"]:
                    ready = False
                self._json(200 if ready else 503, {
                    "status": "ok" if ready else "unavailable",
                    "runtime": "local", "manager": "single-resident",
                    "active_model": state["active_model"],
                    "inflight_requests": state["inflight_requests"],
                })
                return
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
        if not self._security_gate():
            return
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
        if not isinstance(request, dict):
            self._json(400, {"error": {"message": "Request must be a JSON object"}})
            return
        alias = request.get("model")
        manager = self.server.manager
        if not isinstance(alias, str) or (manager is None and alias != self.server.model_alias):
            self._json(400, {"error": {"message": "Model must match a valid active alias"}})
            return
        if "stream" in request and not isinstance(request["stream"], bool):
            self._json(400, {"error": {"message": "stream must be boolean"}})
            return
        # Validate Ollama subset before an expensive cold model load.
        if path in {"/api/chat", "/api/generate"}:
            try:
                translate(path, request)
            except UnsupportedRequest as exc:
                self._json(422, {"error": str(exc)})
                return
        if manager is None:
            self._dispatch(path, request, alias)
            return
        try:
            lease = manager.acquire(alias)
        except ModelManagerError as exc:
            self._json(exc.status, {"error": str(exc)})
            return
        with lease:
            self._dispatch(path, request, alias)

    def _dispatch(self, path: str, request: dict, alias: str) -> None:
        if path in {"/api/chat", "/api/generate"}:
            handle_ollama(self, path, request, model_alias=alias)
        else:
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
            self._cors_headers()
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


def make_server(model_alias: str, backend_port: int, listen_port: int = 11435, model_path=None, manager=None, browser_access=None) -> PhiServeHTTP:
    # Intentionally no bind address argument until authentication is implemented.
    return PhiServeHTTP(("127.0.0.1", listen_port), model_alias, backend_port, model_path, manager, browser_access)
