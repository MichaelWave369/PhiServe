"""Explicit, loopback-only browser access contract.

A browser Origin header is not authentication. Every endpoint requires a
bearer token when browser mode is enabled. Default mode remains local and
does not enable CORS or remote listeners.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class BrowserAccess:
    origin: str
    token: str

    def check(self, authorization: str) -> bool:
        scheme, sep, supplied = authorization.partition(" ")
        if not sep or scheme.lower() != "bearer":
            return False
        return secrets.compare_digest(supplied, self.token)


def validate_origin(origin: str) -> str:
    if not isinstance(origin, str) or not origin or origin.strip() != origin:
        raise ValueError("Browser origin must be an explicit complete origin")
    try:
        p = urlsplit(origin)
        port = p.port
    except ValueError as exc:
        raise ValueError("Browser origin has an invalid port") from exc
    host = p.hostname
    loopback = host in {"localhost", "127.0.0.1", "::1"}
    if (not host or p.username or p.password or p.path or p.query or p.fragment
            or p.scheme not in {"http", "https"}
            or (p.scheme != "https" and not loopback)
            or (port is not None and not (1 <= port <= 65535))
            or p.netloc.endswith(":")):
        raise ValueError("Use an exact HTTPS origin, or HTTP loopback for local development; no path or credentials")
    if p.geturl() != origin:
        raise ValueError("Origin must be canonical without path, query or fragment")
    return origin


def load_access(origin: str | None, token_path: str | Path | None) -> BrowserAccess | None:
    if origin is None and token_path is None:
        return None
    if not origin or not token_path:
        raise ValueError("--browser-origin and --api-key-file are required together")
    allowed = validate_origin(origin)
    path = Path(token_path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 256:
        raise ValueError("Browser API key file missing or oversized")
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 32 or any(ch.isspace() for ch in token):
        raise ValueError("Browser API key must be a single strong token of at least 32 characters")
    return BrowserAccess(allowed, token)


def create_token_file(output: str | Path) -> Path:
    """Generate a secret file exclusively; never print the token to logs."""
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(32) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path
