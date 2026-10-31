# Φ PhiServe ↔ ΦStream checkpoint bridge, Rung 6

This is a **separate, opt-in, token-ID research lane**. It does not make PhiStream a general chat engine. PhiStream itself ships no pretrained weights: you must possess an appropriate, licensed, **local safetensors checkpoint**. Registering it stores only its existing absolute path and *user-declared* source, revision and license identifier. Those declarations are **not authenticated**.

## Setup

Install PhiServe and your trusted [PhiStream](https://github.com/MichaelWave369/PhiStream) checkout into the **same Python environment**. PhiStream brings PyTorch dependencies; install the appropriate CUDA-capable build yourself if you intend to run on an NVIDIA GPU. No package is downloaded by this bridge. Do not use an Ollama GGUF blob here.

### Metadata-only registration

~~~powershell
phiserve phistream inspect "D:\\Models\\CanonicalLlamaSafetensors"
phiserve phistream register research-llama "D:\\Models\\CanonicalLlamaSafetensors" --source "USER_DECLARED_SOURCE" --revision "PINNED_REVISION" --license-id "CHECKPOINT_LICENSE"
phiserve phistream list
~~~

The inspect/register calls use PhiStream's own `LocalCheckpoint` and `LlamaSettings` validation. They inspect local configuration and header metadata. They do not stage model tensors, perform model inference, convert weights or authenticate licenses. PhiServe keeps registrations in `~/.phiserve/phistream-checkpoints.json`, **separate** from GGUF `models.json`.

### Explicit, opt-in token-ID research execution

~~~powershell
phiserve phistream run-ids research-llama --prompt 1,2,3 --tokens 4 --device cpu --capacity-mib 4096 --reserve-mib 1024 --approve-local-execution --output "benchmarks\\local\\phiids.json"
~~~

The explicit capacity/reserve values trigger PhiStream's metadata memory preflight before execution (not a hard RAM/VRAM cap). PhiStream already supports these two memory-policy flags on its existing `run-llama-ids` command. This delegates to the **installed PhiStream implementation** through a fixed argument-array subprocess call, no shell. It accepts **token IDs**, not text. The operator must supply correctly tokenized IDs from a suitably licensed tokenizer. The command requests no network access or downloads and refuses to overwrite an existing receipt. Setting `--device cuda` requests CUDA explicitly but is not independent proof that real GPU execution occurred.

Use `phiserve phistream remove research-llama` to remove only the PhiServe catalog entry; the original checkpoint files stay untouched.

## Important boundaries

- Neither `GET /api/tags` nor `GET /v1/models` advertises a PhiStream research checkpoint as a chat-capable model.
- `/api/chat` and `/v1/chat/completions` remain llama.cpp/GGUF routes. No PhiStream research execution is remotely accessible.
- PhiStream currently supports a narrow canonical Llama safetensors format. It rejects architectural variants, GGUF, unsupported RoPE, MoE, and other incompatible weight layouts.
- Full SHA-256 checkpoint fingerprinting and independent reference parity are **separate existing PhiStream tools**, not silently inferred from a valid metadata check.
- The optional bridge cannot certify CUDA speedups, training origin, permissions, or text accuracy. CI tests simulate the subprocess and checkpoint inspector; no real model weights or GPU are executed in CI.
- Metadata registry writes are atomic but do not yet use cross-process locks. Only trusted local operators should edit registries or run the experimental model code.

## Promotion gate

Before PhiServe can expose PhiStream as a production chat backend, PhiStream needs a tokenizer and chat-template contract, robust text streaming and cancellation, model/license provenance, real checkpoint parity against an independently executed reference, memory/latency receipts on operator hardware, and concurrency/process lifecycle handling. Add that as a **new validated backend**, rather than routing users to an experimental token-ID executor.

## Rung 7: optional text research lane

This checkpoint catalog also supports opt-in [local tokenizer and decoded text experiments](PHISTREAM_TEXT_LAB.md). `tokenizer-check` validates local token IDs without executing weights. `text-lab` invokes the same governed PhiStream token-ID runner as above and decodes locally. This does **not** introduce any natural-language chat API, model auto-selection, or reference parity assertion.
