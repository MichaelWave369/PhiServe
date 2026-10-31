"""Operator-run real inference smoke test and privacy-conscious receipt.

CI may mock the backend; a receipt is a local operator observation, not an
independently witnessed GPU benchmark, CUDA proof, or model parity certificate.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .backend import LlamaBackend
from .registry import ModelRegistry

_MAX_RESPONSE_BYTES = 1024 * 1024
_TEST_PROMPT = "Write exactly one short greeting."


def validate_local(
    alias: str, executable: str, output: str | Path, *,
    backend_port: int = 11436, ctx_size: int = 2048,
    gpu_layers: int = 0, startup_timeout: float = 120,
    registry: ModelRegistry | None = None,
) -> dict:
    """Launch real llama-server, require nonempty text, write receipt exclusively."""
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError(f"Receipt already exists and will not be overwritten: {target}")
    model = (registry or ModelRegistry()).get(alias)
    request = {
        "model": alias,
        "messages": [{"role": "user", "content": _TEST_PROMPT}],
        "max_tokens": 24,
        "temperature": 0,
        "stream": False,
    }
    started = time.monotonic_ns()
    backend = LlamaBackend(executable=executable, model_path=model,
                           alias=alias, port=backend_port, ctx_size=ctx_size,
                           gpu_layers=gpu_layers, startup_timeout=startup_timeout)
    with backend:
        connection = http.client.HTTPConnection("127.0.0.1", backend_port, timeout=180)
        try:
            connection.request("POST", "/v1/chat/completions",
                               body=json.dumps(request).encode("utf-8"),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            body = response.read(_MAX_RESPONSE_BYTES + 1)
            if response.status != 200:
                raise RuntimeError(f"Local backend HTTP status {response.status}")
            if len(body) > _MAX_RESPONSE_BYTES:
                raise RuntimeError("Model response exceeded receipt-test limit")
            data = json.loads(body)
        finally:
            connection.close()
    try:
        reply = data["choices"][0]["message"]["content"]
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("Empty assistant response")
        usage = data.get("usage") or {}
        token_count = usage.get("completion_tokens") if isinstance(usage, dict) else None
    except (IndexError, KeyError, TypeError, AttributeError, ValueError) as exc:
        raise RuntimeError(f"Model did not return an acceptable text response: {exc}") from exc
    if type(token_count) is not int or token_count < 0:
        token_count = None
    receipt = {
        "schema": "PHISERVE_LOCAL_INFERENCE_RECEIPT_V1",
        "status": "LOCAL_TEXT_OUTPUT_OBSERVED",
        "attestation": "operator_supplied_not_independently_witnessed",
        "model_alias": alias,
        "model_filename": model.name,
        "model_size_bytes": model.stat().st_size,
        "server_binary_name": Path(executable).name,
        "gpu_layers_requested": gpu_layers,
        "gpu_execution_verified": False,
        "ctx_size_requested": ctx_size,
        "prompt_sha256": hashlib.sha256(_TEST_PROMPT.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(reply.encode()).hexdigest(),
        "output_characters": len(reply),
        "completion_tokens_reported": token_count,
        "wall_duration_ns": time.monotonic_ns() - started,
        "utc_created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "phiserve_version": __version__,
        "raw_prompt_or_output_stored": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt
