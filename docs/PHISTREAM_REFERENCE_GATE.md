# Rung 8: governed PhiStream external-reference parity gate

This milestone **does not add a production PhiStream chat model**. It provides a reproducible, offline bridge between PhiServe's Rung 7 local tokenizer and PhiStream's existing `real-model-gate` for user-supplied independent reference comparisons.

## Prerequisites

PhiServe v0.8.0 with its `phistream-text` optional dependency, PhiStream installed into the same trusted Python environment, a compatible licensed local canonical Llama safetensors checkpoint already registered as a PhiStream *research* alias, and a local `PHISTREAM_REFERENCE_TRACE_V1` created by a separately run reference implementation.

The trace must contain **all** checkpoint config/shard SHA256 digests, a prompt token-ID array, 32 to 128 expected greedy token IDs, one nonempty sampled logit-probe map per step, finite bounded tolerances, and explicit self-declared reference runtime/device metadata. See [PhiStream reference trace contract](https://github.com/MichaelWave369/PhiStream/blob/main/docs/REFERENCE_TRACE.md).

The reference origin is **self-declared and not independently authenticated**. A matching result is not a license audit, third-party attestation, full-vocabulary logit parity, benchmark, or model service authorization.

## Phase A: tokenizer/reference preflight, without model execution

~~~powershell
phiserve phistream parity-preflight research-llama --tokenizer "D:\\Models\\Llama\\tokenizer.json" --prompt "Hello" --trace ".\\reference-trace.json"
~~~

The prompt is encoded by the explicit tokenizer with `--add-special-tokens` **off by default** (opt in only if your reference trace used those exact added tokens). If the resulting IDs differ from the trace's `prompt_token_ids`, the operation fails. This is crucial: a logit comparison using different prompt tokens would be meaningless. The preflight does not hash the checkpoint bytes and therefore **does not prove weight identity**. It flags short traces as ineligible rather than pretending they support a real parity run.

## Phase B: actual local parity execution, explicit permission

~~~powershell
phiserve phistream parity-run research-llama --tokenizer "D:\\Models\\Llama\\tokenizer.json" --prompt "Hello" --trace ".\\reference-trace.json" --device cpu --capacity-mib 8192 --reserve-mib 1024 --approve-local-execution --output ".\\receipts\\parity-run.json"
~~~

This requires at least 32 reference steps and explicit approval. It delegates via a fixed argument-array subprocess call to the installed PhiStream `real-model-gate`. Its own safeguards perform memory preflight and SHA-256 verification of **every** referenced checkpoint shard and config before running, then compare greedy token IDs and sparse sampled logits against the user-supplied reference with teacher forcing. PhiServe validates the returned report schema, status, matching checkpoint hashes, reference SHA256, device and counts before producing a **new** JSON summary. Any mismatch is reported and exits nonzero. Blocked memory policy and invalid/partial reports never produce a success receipt.

The durable PhiServe summary intentionally omits raw reference logits, prompt text, token arrays and generated response. The underlying raw PhiStream comparison receipt is temporary and removed at completion. The trace itself is never changed or uploaded. Paths and model/source/license declarations remain local. Plain-text prompts passed on a shell command line may be visible in shell history or OS process lists; don't use sensitive prompts.

## Promotion remains blocked

This is a **research acceptance gate**, not a certified deployment. Matching tokens in a **user-declared** trace cannot prove that the reference actually ran independently, nor does it authenticate the checkpoint's origin or tokenizer identity. Only sampled logits are compared. PhiServe continues to serve chat with llama.cpp/GGUF only. Research aliases are never added to `/api/tags` or `/v1/models`.

Before enabling a native PhiStream chat backend: independent reference execution evidence, a verified model/tokenizer/chat-template identity, supported stop/EOS and streaming semantics, actual hardware measurements on the operator's RTX 5070, robust cancellation, memory accounting and new backend conformance tests are required.

CI checks fake metadata/reference input and a mocked subprocess. **It does not run a real pretrained checkpoint or prove CUDA execution.**
