# Φ PhiServe

**Sovereign local model serving, without accounts, telemetry, or mandatory cloud services.**

PhiServe Rung 1 is a **working local HTTP serving layer** around a separately installed [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` binary. It registers an existing local GGUF model, supervises a real inference process, and exposes OpenAI-style chat and text-completion endpoints on loopback. PhiServe does not ship model weights or download anything automatically.

**Status: v0.3.0 experimental, Ollama text-API subset and read-only model reuse.** Rung 2 adds compatibility for text-only `/api/tags`, `/api/chat`, and `/api/generate` with Ollama-style newline-delimited JSON streams. It is not a complete Ollama replacement. Automatic scheduling, a GUI, downloads, multi-model serving, multimodal requests, PhiStream integration, and LAN access are future work. CI uses a fake backend; real checkpoint execution must be validated on operator hardware.

## Quick start

Requirements: Python 3.11+; a trusted, independently installed `llama-server` executable; and a legitimate local GGUF model whose license permits your use.

~~~bash
git clone https://github.com/MichaelWave369/PhiServe.git
cd PhiServe
python -m pip install -e .
python -m unittest discover -s tests -v
phiserve models add local-assistant /absolute/path/to/model.gguf
phiserve models list
phiserve serve local-assistant --llama-server /path/to/llama-server --gpu-layers 0
~~~

On Windows PowerShell, replace the model and binary paths, for example:

~~~powershell
py -m pip install -e .
phiserve models add local-assistant "D:\Models\assistant.gguf"
phiserve serve local-assistant --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 20
~~~

The `--gpu-layers` value is **explicit**. `0` means CPU-only; increase it only if your llama.cpp build has a suitable GPU backend and your model fits. PhiServe does not auto-detect GPU memory or promise automatic offloading in Rung 1.

The PhiServe API binds to `127.0.0.1:11435`; its private llama.cpp child binds to `127.0.0.1:11436`. Both ports are configurable and must be different. Use Ctrl+C to stop the service and its child.

Check health:

~~~bash
curl http://127.0.0.1:11435/health
curl http://127.0.0.1:11435/v1/models
~~~

Chat against a **real model** (replace its alias as needed):

~~~bash
curl -N http://127.0.0.1:11435/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"local-assistant","messages":[{"role":"user","content":"Hello from PhiServe!"}],"stream":true}'
~~~

The API forwards both streaming and non-streaming requests to the actual local `llama-server`; it never generates canned model answers.

## Ollama compatibility (Rung 2)

Point a compatible text-only Ollama client at `http://127.0.0.1:11435` (not Ollama\'s default port). The following examples use the **active** model alias, not an Ollama model you have not registered:

~~~bash
curl http://127.0.0.1:11435/api/tags
curl -N http://127.0.0.1:11435/api/chat -H "Content-Type: application/json" -d \'{"model":"local-assistant","messages":[{"role":"user","content":"Hello!"}]}\'
curl http://127.0.0.1:11435/api/generate -H "Content-Type: application/json" -d \'{"model":"local-assistant","prompt":"Hello!","stream":false}\'
~~~

Default streaming produces **NDJSON** (one Ollama-format JSON object per line), not OpenAI SSE. `stream:false` returns a single Ollama-format JSON object. Usage counts are reported only when the upstream supplies them. `total_duration` is measured from PhiServe\'s local upstream request start, not a claim about model-only execution time.

**Supported request subset:** text messages and prompt, optional `system` for generation, `raw` generation, `stream`, and selected `options`: `temperature`, `top_p`, `seed`, `num_predict`, `stop`. Unsupported fields, including images, tools, `format`, `think`, `keep_alive`, and `num_ctx`, return HTTP 422 rather than being silently ignored. In particular `/api/generate` uses model chat-template formatting by default; `raw:true` uses the literal completion endpoint. Compatibility is intentionally limited; not all Ollama clients work yet. `/api/tags` advertises only the actively served model, with real file size and modified time when available; it does not invent digest, family, or quantization metadata.

## Reuse your Ollama models and validate real inference (Rung 3)

PhiServe can discover supported **GGUF model-weight layers already downloaded by Ollama**, register their existing blobs by reference (no duplicate downloads), and run an operator-requested real inference smoke test. The importer never modifies the Ollama model store. See [Rung 3 guide](docs/OLLAMA_IMPORT_AND_VALIDATION.md) for commands, integrity checks, and hardware-evidence limits.

~~~powershell
phiserve models discover-ollama
phiserve models import-ollama qwen3:4b --verify-sha256
phiserve validate qwen3:4b --llama-server "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe" --gpu-layers 20 --output ".\\receipts\\smoke.json"
~~~

Import only works for a compatible single GGUF model layer. Inference still requires a separately installed, compatible `llama-server` binary; tests do not certify your physical GPU.

## Command reference

~~~text
phiserve models add ALIAS PATH
phiserve models list
phiserve models remove ALIAS
phiserve models discover-ollama [--root ROOT]
phiserve models import-ollama MODEL [--alias ALIAS] [--root ROOT] [--verify-sha256]
phiserve validate ALIAS --llama-server PATH --output NEW_JSON [--gpu-layers N]
phiserve serve ALIAS [--llama-server PATH] [--port 11435] [--backend-port 11436] [--ctx-size 4096] [--gpu-layers 0] [--startup-timeout 120]
~~~

You can set `PHISERVE_HOME` to move its metadata directory (default: `~/.phiserve`). Registry entries refer to existing GGUF files: they are not copied or downloaded.

## HTTP surface (Rungs 1-2)

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/health` | PhiServe + child health, 503 when unavailable |
| GET | `/v1/models` | Currently selected model alias only |
| POST | `/v1/chat/completions` | Validated, streaming-safe local proxy |
| POST | `/v1/completions` | Validated, streaming-safe local proxy |
| GET | `/api/tags` | Active GGUF model alias and available file metadata |
| POST | `/api/chat` | Ollama text chat; NDJSON streaming or one JSON result |
| POST | `/api/generate` | Ollama text generation; NDJSON streaming or one JSON result |

Requests must be JSON with a matching `model` alias. Maximum request body is 2 MiB; unsupported routes are rejected. Only the Ollama text API subset above is claimed in this release.

## Safety and limitations

- Loopback-only. No TLS, network listener, or remote authentication in this rung; **do not expose these ports to untrusted networks**.
- No model downloads, external APIs, telemetry, account, or model weights. Local model file paths remain local.
- Spawned child commands use argument arrays rather than a shell; no arbitrary user-provided command-line flags are passed through the HTTP API.
- The model registry uses atomic writes, but does **not** yet implement cross-process write locking.
- Only **one** active model process per PhiServe instance. The user must supply a trusted `llama-server` and GGUF. Model-file header validation is not a substitute for trusting the source.
- Streaming is relayed from llama.cpp, not synthesized. If the child crashes mid-stream, the client connection may end early.
- Automated tests exercise the registry, HTTP boundaries, and a simulated upstream, **not actual GPU inference**.

See [architecture](docs/ARCHITECTURE.md) and [roadmap](docs/ROADMAP.md).

## Ecosystem design

PhiServe is the runtime/service boundary. [PhiStream](https://github.com/MichaelWave369/PhiStream) is a separate experimental inference research effort, currently without a general GGUF chat-serving contract. Integrate it only when a tested real-token inference adapter exists. PhiOS, PocketBrain, GAIA, and other clients can use PhiServe's local API as it matures.

**License:** MIT for PhiServe code; model weights and llama.cpp are governed by their respective licenses.
