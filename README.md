# Φ PhiServe

**Sovereign local model serving, without accounts, telemetry, or mandatory cloud services.**

PhiServe Rung 1 is a **working local HTTP serving layer** around a separately installed [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` binary. It registers an existing local GGUF model, supervises a real inference process, and exposes OpenAI-style chat and text-completion endpoints on loopback. PhiServe does not ship model weights or download anything automatically.

**Status: v0.1.0 experimental.** This is an intentionally small foundation, not an Ollama replacement yet. Ollama API compatibility, automatic scheduling, a GUI, downloads, multi-model serving, PhiStream inference integration, and LAN access are future work. CI uses a fake backend; real checkpoint execution must be validated on operator hardware.

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

## Command reference

~~~text
phiserve models add ALIAS PATH
phiserve models list
phiserve models remove ALIAS
phiserve serve ALIAS [--llama-server PATH] [--port 11435] [--backend-port 11436] [--ctx-size 4096] [--gpu-layers 0] [--startup-timeout 120]
~~~

You can set `PHISERVE_HOME` to move its metadata directory (default: `~/.phiserve`). Registry entries refer to existing GGUF files: they are not copied or downloaded.

## HTTP surface (Rung 1)

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/health` | PhiServe + child health, 503 when unavailable |
| GET | `/v1/models` | Currently selected model alias only |
| POST | `/v1/chat/completions` | Validated, streaming-safe local proxy |
| POST | `/v1/completions` | Validated, streaming-safe local proxy |

Requests must be JSON with a matching `model` alias. Maximum request body is 2 MiB; unsupported routes are rejected. No Ollama-compatible routes are claimed in this release.

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
