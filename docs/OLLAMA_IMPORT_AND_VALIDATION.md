# Rung 3: reuse existing Ollama model blobs and test real inference

PhiServe can now **discover and register existing Ollama GGUF blobs without copying model weights**. It reads manifests only and never edits the Ollama store. If you have used `OLLAMA_MODELS` to move the store to another drive, discovery respects it; you can also pass `--root` explicitly. The default user-home location is `~/.ollama/models`.

## Commands

~~~powershell
phiserve models discover-ollama
phiserve models discover-ollama --root "D:\OllamaModels"
phiserve models import-ollama qwen3:4b --verify-sha256
phiserve models list
phiserve serve qwen3:4b --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 20
~~~

The name in `import-ollama` must exactly match discovery output. `--alias` optionally overrides the PhiServe alias. Without `--verify-sha256`, PhiServe checks manifest structure, declared byte length and GGUF magic bytes, **but does not verify the SHA-256 checksum**. The verification flag reads the entire local weight blob and may take time for multi-gigabyte models. Imported models point to a blob inside the existing Ollama store; don't delete that store while PhiServe depends on it.

**Not all Ollama models are GGUF-compatible.** Unsupported blobs and manifests are skipped with a warning, rather than silently registered. We don't install `llama-server`, auto-detect actual GPU support, map Ollama templates/adapters, or promise any particular model is compatible with the separate llama.cpp binary.

## Physical model test / local receipt

~~~powershell
phiserve validate qwen3:4b --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 20 --output ".\receipts\qwen3-smoke.json"
~~~

This command actually launches the local llama.cpp server with the selected model, waits for readiness, performs a non-streaming chat-completion request and requires nonempty assistant text. It creates one JSON receipt exclusively (never overwrites an existing file). The receipt records version, elapsed time, requested GPU layers, real model file size, and SHA-256 fingerprints of the fixed prompt and returned text; it **does not store prompt or generated content**. It also explicitly states that GPU execution has **not been independently witnessed**.

For a baseline check, run first with `--gpu-layers 0` (CPU) then compare with an explicit supported GPU layer count if your binary was built for CUDA. A successful response with GPU layers requested alone does not prove actual CUDA residency: that requires backend GPU logs and device-level evidence.

## Acceptance gaps

- Automated tests build fake Ollama manifests and tiny GGUF headers. These are **not usable model weights**.
- CI does not have a physical RTX 5070 or a real checkpoint.
- No shared-memory scheduler, downloads, new GGUF parser, multiprocess locking or mandatory PhiStream integration.
- This bridge is **read-only** for Ollama's store; PhiServe writes only to its own model registry and explicitly requested receipt files.
