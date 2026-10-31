# Rung 4: shared single-resident PhiServe manager

The `phiserve daemon` command lets multiple local clients address many registered model aliases through one stable port **without launching one process per application**. The first request for a model starts a real llama.cpp server on the private loopback backend port. The daemon supervises no more than one active child at a time.

## Correctness model

- A request acquires a lease *before* it connects to the backend. The lease stays active through the entire OpenAI proxy/SSE stream or Ollama NDJSON conversion, and is released on success, error, or disconnect.
- Concurrent requests for **the same loaded model** may proceed up to `--max-concurrent` (default 4). Excess requests receive **429**; llama.cpp itself may have smaller practical capacity.
- A request for a different model **while any lease is outstanding** receives **409 Conflict**. Nothing is killed. Clients retry after the active requests finish. This is not a priority/fairness scheduler.
- When no leases remain, a request can switch models. A transition stops the prior child and starts the new one. While loading/evicting, incoming requests receive **503** and can retry.
- Idle eviction occurs only once the **last** request lease has been released and the configurable `--idle-seconds` interval has elapsed. Setting 0 disables idle eviction.
- Ctrl+C initiates shutdown and waits up to 30 seconds for active requests. Any still-active request may be interrupted after that grace period. Startup in progress observes the shutdown flag and stops its child once startup returns.
- A child that crashes is not declared healthy. A subsequent request can attempt a new startup when no leases are active.

## API behavior

- `GET /api/tags`: registered models, whether loaded or not; no invented checksum, family, or GPU metrics.
- `GET /v1/models`: registered aliases.
- `GET /api/ps`: loaded model only, or `{"models":[]}`. This is a **limited subset** of Ollama's process-list structure and omits unknown fields.
- `GET /health`: manager readiness; an idle daemon with no loaded weights is healthy.
- `POST /api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/completions`: model-aware routing using existing strict text protocols. Unsupported API fields remain unsupported.

The original `phiserve serve ALIAS` mode retains its fixed model and original local endpoints, with `/api/ps` added. The daemon will never auto-download weights or execute an API-supplied path.

## Usage

~~~powershell
phiserve models list
phiserve daemon --llama-server "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe" --gpu-layers 20 --idle-seconds 300 --max-concurrent 4
curl.exe http://127.0.0.1:11435/api/tags
curl.exe http://127.0.0.1:11435/api/ps
~~~

This is one subprocess **per daemon**, not a machine-wide GPU allocator. Running multiple daemons or other inference programs could still overcommit GPU resources. CUDA residency, performance, crash recovery, and real inference remain separate operator validation gates. Automated tests use fake processes and fake inference upstreams. Not hardened for network or multi-user serving.
