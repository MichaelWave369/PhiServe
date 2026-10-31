"""Honest, limited Ollama text API compatibility over the local llama.cpp backend.

Ollama NDJSON is *not* the OpenAI SSE protocol: the adapter converts the two
formats and explicitly rejects unsupported request semantics instead of
silently changing them.
"""

from __future__ import annotations

import http.client
import json
import time
from datetime import datetime, timezone
from pathlib import Path


MAX_REPLY_BYTES = 8 * 1024 * 1024
MAX_EVENT_LINE = 64 * 1024


class UnsupportedRequest(ValueError):
    """A request uses an Ollama feature not implemented in this adapter."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def model_tags(alias: str, model_path: str | Path | None) -> dict:
    """Only advertise the loaded model, not every registered but unloaded alias.

    Unknown metadata is omitted. We never invent a checksum or model family.
    """
    info = {"name": alias, "model": alias, "details": {"format": "gguf"}}
    if model_path is not None:
        try:
            stat = Path(model_path).stat()
            info["size"] = stat.st_size
            info["modified_at"] = datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat().replace("+00:00", "Z")
        except OSError:
            pass
    return {"models": [info]}


def _reject_unknown(data: dict, allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise UnsupportedRequest("Unsupported Ollama fields: " + ", ".join(unknown))


def _options(request: dict) -> dict:
    opts = request.get("options", {})
    if not isinstance(opts, dict):
        raise UnsupportedRequest("options must be an object")
    supported = {"temperature", "top_p", "seed", "num_predict", "stop"}
    _reject_unknown(opts, supported)
    result = {}
    for key in ("temperature", "top_p"):
        if key in opts:
            value = opts[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise UnsupportedRequest(f"options.{key} must be numeric")
            if (key == "top_p" and not (0 <= value <= 1)) or (key == "temperature" and not (0 <= value <= 2)):
                raise UnsupportedRequest(f"options.{key} out of supported range")
            result[key] = value
    for key, target in (("seed", "seed"), ("num_predict", "max_tokens")):
        if key in opts:
            value = opts[key]
            if type(value) is not int or (key == "num_predict" and value <= 0):
                raise UnsupportedRequest(f"options.{key} must be a valid integer")
            result[target] = value
    if "stop" in opts:
        stops = opts["stop"]
        if isinstance(stops, str):
            stops = [stops]
        if not isinstance(stops, list) or not stops or not all(isinstance(x, str) and x for x in stops):
            raise UnsupportedRequest("options.stop must be a nonempty string/list of strings")
        result["stop"] = stops
    return result


def translate(route: str, request: dict) -> tuple[str, dict, bool]:
    """Validate a strict supported subset and translate to OpenAI-style calls."""
    stream = request.get("stream", True)
    if not isinstance(stream, bool):
        raise UnsupportedRequest("stream must be boolean")
    common = {"model", "stream", "options"}
    params = {"model": request["model"], "stream": stream}
    params.update(_options(request))
    if route == "/api/chat":
        _reject_unknown(request, common | {"messages"})
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise UnsupportedRequest("messages must be a nonempty list")
        for item in messages:
            if (not isinstance(item, dict) or set(item) != {"role", "content"}
                    or item["role"] not in {"system", "user", "assistant"}
                    or not isinstance(item["content"], str)):
                raise UnsupportedRequest("Only text messages with system, user or assistant role are supported")
        params["messages"] = messages
        target = "/v1/chat/completions"
    else:
        _reject_unknown(request, common | {"prompt", "system", "raw"})
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise UnsupportedRequest("prompt must be a nonempty string")
        system = request.get("system")
        raw = request.get("raw", False)
        if not isinstance(raw, bool) or (system is not None and not isinstance(system, str)):
            raise UnsupportedRequest("raw must be boolean and system must be text")
        if raw:
            if system:
                raise UnsupportedRequest("raw with system is unsupported")
            params["prompt"] = prompt
            target = "/v1/completions"
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            params["messages"] = messages
            target = "/v1/chat/completions"
    if stream:
        # Some llama.cpp builds provide usage in the final OpenAI SSE chunk.
        params["stream_options"] = {"include_usage": True}
    return target, params, stream


def _usage(payload: dict) -> dict:
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    result = {}
    for source, dest in (("prompt_tokens", "prompt_eval_count"), ("completion_tokens", "eval_count")):
        value = usage.get(source)
        if type(value) is int and value >= 0:
            result[dest] = value
    return result


def _content(choice: dict, is_chat: bool, stream: bool) -> str:
    if not isinstance(choice, dict):
        raise ValueError("Malformed inference choice")
    if stream:
        part = choice.get("delta") if is_chat else choice.get("text", "")
    else:
        part = choice.get("message") if is_chat else choice.get("text", "")
    if is_chat:
        if not isinstance(part, dict):
            raise ValueError("Missing assistant content")
        if part.get("tool_calls"):
            raise ValueError("Tool-call responses are not supported")
        part = part.get("content") or ""
    if not isinstance(part, str):
        raise ValueError("Non-text model response is unsupported")
    return part


def _convert_payload(route: str, alias: str, content: str, done: bool,
                     reason: str | None = None, elapsed_ns: int | None = None,
                     usage: dict | None = None) -> dict:
    result = {"model": alias, "created_at": _timestamp(), "done": done}
    if route == "/api/chat":
        result["message"] = {"role": "assistant", "content": content}
    else:
        result["response"] = content
    if done:
        if reason is not None:
            result["done_reason"] = reason
        if elapsed_ns is not None:
            result["total_duration"] = elapsed_ns
        if usage:
            result.update(usage)
    return result


def _sse_frames(response):
    """Read bounded SSE frames without buffering a full generation."""
    data = []
    size = 0
    while True:
        line = response.readline(MAX_EVENT_LINE + 1)
        if len(line) > MAX_EVENT_LINE:
            raise ValueError("Oversized backend SSE line")
        if not line:
            if data:
                yield "\n".join(data)
            return
        line = line.rstrip(b"\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
                size = 0
            continue
        if line.startswith(b"data:"):
            chunk = line[5:].lstrip(b" ")
            size += len(chunk)
            if size > MAX_REPLY_BYTES:
                raise ValueError("Oversized backend SSE frame")
            data.append(chunk.decode("utf-8"))


def _send_stream(handler, route: str, alias: str, response, started_ns: int, is_chat: bool) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "application/x-ndjson")
    handler.send_header("Cache-Control", "no-store")
    handler._cors_headers()
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True

    def emit(obj: dict) -> None:
        handler.wfile.write(json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n")
        handler.wfile.flush()

    usage = {}
    reason = None
    completed = False
    try:
        for frame in _sse_frames(response):
            if frame == "[DONE]":
                completed = True
                break
            event = json.loads(frame)
            if not isinstance(event, dict) or event.get("error"):
                raise ValueError("Backend returned an invalid SSE event")
            usage.update(_usage(event))
            choices = event.get("choices", [])
            if not isinstance(choices, list):
                raise ValueError("Backend returned invalid choices")
            if choices:
                choice = choices[0]
                content = _content(choice, is_chat, True)
                if content:
                    emit(_convert_payload(route, alias, content, False))
                if choice.get("finish_reason") is not None:
                    reason = choice["finish_reason"]
        if not completed:
            raise ValueError("Backend stream ended without a completion marker")
        emit(_convert_payload(route, alias, "", True, reason, time.monotonic_ns() - started_ns, usage))
    except (ValueError, UnicodeDecodeError, OSError, json.JSONDecodeError):
        # Headers have been sent. An explicit NDJSON error is safer than a fake success.
        try:
            emit({"error": "Local inference stream failed or ended prematurely"})
        except OSError:
            pass


def _send_single(handler, route: str, alias: str, response, started_ns: int, is_chat: bool) -> None:
    body = response.read(MAX_REPLY_BYTES + 1)
    if len(body) > MAX_REPLY_BYTES:
        handler._json(502, {"error": "Local backend response exceeded size limit"})
        return
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("Backend returned non-object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Backend returned no choices")
        first = choices[0]
        content = _content(first, is_chat, False)
        reason = first.get("finish_reason")
    except (ValueError, TypeError, AttributeError):
        handler._json(502, {"error": "Local inference backend returned an incompatible response"})
        return
    handler._json(200, _convert_payload(route, alias, content, True,
                                      reason, time.monotonic_ns() - started_ns, _usage(data)))


def handle_ollama(handler, route: str, request: dict, model_alias: str | None = None) -> None:
    try:
        target, body, stream = translate(route, request)
    except UnsupportedRequest as exc:
        handler._json(422, {"error": str(exc)})
        return

    selected_alias = handler.server.model_alias if model_alias is None else model_alias
    connection = http.client.HTTPConnection("127.0.0.1", handler.server.backend_port, timeout=600)
    started_ns = time.monotonic_ns()
    try:
        connection.request("POST", target, body=json.dumps(body).encode("utf-8"),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            handler._json(502, {"error": f"Local inference backend HTTP {response.status}"})
            return
        if stream:
            _send_stream(handler, route, selected_alias, response, started_ns, target == "/v1/chat/completions")
        else:
            _send_single(handler, route, selected_alias, response, started_ns, target == "/v1/chat/completions")
    except (OSError, http.client.HTTPException):
        if not handler.wfile.closed:
            try:
                handler._json(502, {"error": "Local inference backend unavailable"})
            except OSError:
                pass
    finally:
        connection.close()
