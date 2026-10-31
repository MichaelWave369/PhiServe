"""Explicit local tokenizer research for PhiStream's token-ID executor.

This is not an HTTP text backend or proof that a tokenizer matches weights.
Only a local tokenizer.json is parsed; no remote files are fetched.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

from .phistream_lab import (
    PhiStreamCatalog, PhiStreamLabError, parse_token_ids, run_ids,
)

MAX_TOKENIZER_BYTES = 32 * 1024 * 1024
MAX_TEXT_CHARS = 4096
MAX_PROMPT_TOKENS = 2048


def load_local_tokenizer(tokenizer_file: str | Path, model_vocab: int):
    """Load a selected local Hugging Face tokenizers JSON using identical hashed bytes."""
    chosen = Path(tokenizer_file).expanduser()
    if chosen.is_symlink():
        raise PhiStreamLabError("Symlink-selected tokenizers are not supported")
    try:
        path = chosen.resolve(strict=True)
    except OSError as exc:
        raise PhiStreamLabError("Tokenizer JSON file not found") from exc
    if path.name != "tokenizer.json" or not path.is_file():
        raise PhiStreamLabError("Supply a local tokenizer.json file (not a model path)")
    size = path.stat().st_size
    if not 2 <= size <= MAX_TOKENIZER_BYTES:
        raise PhiStreamLabError("Tokenizer JSON exceeds local size limits")
    raw = path.read_bytes()
    if len(raw) != size:
        raise PhiStreamLabError("Tokenizer file changed while being read")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise PhiStreamLabError(
            "Optional tokenizer engine missing. Install PhiServe's [phistream-text] extra "
            "in your trusted local environment. No download is performed by PhiServe."
        ) from exc
    try:
        tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
        vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    except (UnicodeError, ValueError, RuntimeError, Exception) as exc:
        raise PhiStreamLabError("Tokenizer JSON cannot be parsed by the local tokenizers engine") from exc
    if (not isinstance(vocabulary, dict) or not vocabulary
            or any(type(index) is not int or index < 0 or index >= model_vocab
                   for index in vocabulary.values())):
        raise PhiStreamLabError(
            "Tokenizer vocabulary IDs exceed the selected PhiStream checkpoint vocabulary"
        )
    return tokenizer, digest, len(vocabulary)


def encode_prompt(catalog: PhiStreamCatalog, alias: str, tokenizer_path: str | Path,
                  prompt: str, *, tokens: int, add_special_tokens: bool = False):
    """Tokenizer/architecture preflight; never executes checkpoint weights."""
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= MAX_TEXT_CHARS or not prompt.strip():
        raise PhiStreamLabError("Prompt must be nonempty text at most 4096 characters")
    if type(tokens) is not int or not 1 <= tokens <= 128:
        raise PhiStreamLabError("tokens must be from 1 through 128")
    if type(add_special_tokens) is not bool:
        raise PhiStreamLabError("add_special_tokens must be explicitly boolean")
    _, metadata = catalog.get(alias)
    vocab = metadata["vocab_size"]
    context = metadata["max_context"]
    if type(vocab) is not int or type(context) is not int:
        raise PhiStreamLabError("Checkpoint metadata must include vocabulary and context bounds")
    tokenizer, fingerprint, vocabulary_entries = load_local_tokenizer(tokenizer_path, vocab)
    try:
        ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens).ids
    except Exception as exc:
        raise PhiStreamLabError("Tokenizer could not encode the provided text") from exc
    if not 1 <= len(ids) <= MAX_PROMPT_TOKENS:
        raise PhiStreamLabError("Encoded prompt must contain 1-2048 token IDs")
    # Reuse the Rung 6 strict validator to reject invalid and out-of-range IDs.
    parse_token_ids(",".join(map(str, ids)), vocab)
    if len(ids) + tokens - 1 > context:
        raise PhiStreamLabError("Prompt and requested tokens exceed checkpoint maximum context")
    return tokenizer, ids, {
        "schema": "PHISERVE_PHISTREAM_TOKENIZER_PREFLIGHT_V1",
        "status": "LOCAL_TOKENIZER_PREFLIGHT_ONLY",
        "backend": "phistream_research",
        "model_alias": alias,
        "tokenizer_sha256": fingerprint,
        "tokenizer_vocab_entries": vocabulary_entries,
        "checkpoint_vocab_size": vocab,
        "checkpoint_max_context": context,
        "prompt_tokens": len(ids),
        "requested_generated_tokens": tokens,
        "tokenizer_special_tokens_applied": add_special_tokens,
        "source_tokenizer_match_verified": False,
        "independent_reference_parity_verified": False,
        "can_serve_chat": False,
    }


def text_lab(
    catalog: PhiStreamCatalog, alias: str, *, tokenizer_path: str | Path,
    prompt: str, tokens: int, device: str, capacity_mib: float,
    reserve_mib: float, output: str | Path, approved: bool,
    add_special_tokens: bool = False, timeout_seconds: int = 900,
    runner=None,
) -> tuple[dict, str]:
    """Explicit local execution via the existing validated PhiStream ID runner.

    The token-ID intermediate lives only in a temporary directory.
    The durable receipt contains hashes/metadata, not prompt or generated text.
    """
    if approved is not True:
        raise PhiStreamLabError("Explicit --approve-local-execution required")
    final_path = Path(output).expanduser().resolve()
    if final_path.exists():
        raise PhiStreamLabError("Receipt already exists; refusing overwrite")
    tokenizer, ids, preflight = encode_prompt(
        catalog, alias, tokenizer_path, prompt,
        tokens=tokens, add_special_tokens=add_special_tokens,
    )
    with tempfile.TemporaryDirectory(prefix="phiserve-phistream-text-") as scratch:
        temporary_receipt = Path(scratch) / "ids.json"
        handoff = run_ids(
            catalog, alias, prompt=",".join(map(str, ids)),
            tokens=tokens, device=device, capacity_mib=capacity_mib,
            reserve_mib=reserve_mib, output=temporary_receipt,
            approved=True, timeout_seconds=timeout_seconds, runner=runner,
        )
        if handoff.get("status") != "LOCAL_TOKEN_ID_EXECUTION_RECEIPT_CREATED":
            raise PhiStreamLabError("PhiStream did not report the required token-ID handoff")
        content = temporary_receipt.read_bytes()
        if len(content) > 1024 * 1024:
            raise PhiStreamLabError("PhiStream intermediate receipt too large")
        try:
            generated = json.loads(content)["generated_token_ids"]
            decoded = tokenizer.decode(generated, skip_special_tokens=False)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise PhiStreamLabError("Unable to decode the verified PhiStream generated token IDs") from exc
        if not isinstance(decoded, str) or not decoded.strip():
            raise PhiStreamLabError("Model returned no usable decoded text (possibly only special tokens)")
        receipt = {
            "schema": "PHISERVE_PHISTREAM_TEXT_LAB_V1",
            "status": "LOCAL_EXPERIMENTAL_TEXT_DECODED_UNVERIFIED",
            "model_alias": alias,
            "device_requested": device,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "decoded_output_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
            "decoded_output_characters": len(decoded),
            "prompt_tokens": len(ids),
            "generated_tokens": len(generated),
            "tokenizer_sha256": preflight["tokenizer_sha256"],
            "tokenizer_special_tokens_applied": add_special_tokens,
            "raw_prompt_or_decoded_output_stored": False,
            "tokenizer_weights_match_verified": False,
            "chat_template_verified": False,
            "external_reference_parity_verified": False,
            "gpu_execution_independently_verified": False,
            "chat_service_enabled": False,
        }
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt, decoded
