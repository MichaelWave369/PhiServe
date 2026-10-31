# Development gates

## Rung 1 (this PR): real local inference foundation

- [x] Model aliases point to explicit, local GGUF files; no automatic download
- [x] Supervise a real llama.cpp child and check readiness
- [x] Bind private backend and local API to loopback
- [x] Forward chat/completions, including streaming SSE
- [x] Expose selected model and honest health
- [x] Test registry and API using a simulated upstream
- [ ] Operator hardware receipt with an actual GGUF and llama-server binary

## Rung 2: text-compatibility and stability (in progress)

- [x] Ollama text subset: `/api/tags`, `/api/chat`, `/api/generate` with NDJSON and simulated-upstream protocol tests
- [ ] Full Ollama feature and request/response compatibility: tools, images, structured output, embeddings, `/api/show`, all options, accurate model metadata
- OpenAI API error and streaming conformance, model metadata, optional embeddings when backend supports it
- Graceful cancellation, timeouts, supervised restarts, request concurrency and backpressure
- Cross-process registry locks, race-safe port reservations, clear error reporting
- Isolated binary and checkpoint provenance checks

## Rung 3: model lifecycle (in progress)

- [x] Read-only Ollama GGUF model manifest discovery and reference import
- [x] Optional full SHA-256 check for imported blob
- [x] Operator-run local inference smoke receipt (hardware execution still pending)
- Integrity-checked downloads with explicit consent, resume and per-model license display
- Disk quotas, remove/import, GGUF metadata parsing, configuration profiles
- Hardware-aware resource estimates, tested GPU fallback and scheduler

## Rung 4: shared model lifecycle (experimental)

- [x] Single-resident lazy model manager for multiple registered aliases
- [x] Lease protection for in-flight streaming responses; busy switch returns 409
- [x] Configurable idle unload, explicit request concurrency ceiling, active-model status
- [x] Loopback-only daemon command; no remote control surface
- [ ] Physical GPU/model validation, true VRAM accounting, model swap benchmark, and crash recovery receipts
- [ ] Shared multi-process scheduling, fairness, queuing, cancellation, and memory-aware admission

## Rung 5: authenticated browser-to-loopback bridge (experimental)

- [x] Explicit trusted browser origin plus bearer-token file
- [x] Protected loopback endpoints and restricted preflight/NDJSON/SSE headers
- [x] Exclusive private token-file creation; no token in daemon logs
- [ ] Real browser acceptance on the operator's machine with actual model weights
- [ ] Separately designed secure remote-device gateway (no direct LAN listener)

## Rung 6: PhiStream experimental weight bridge

- [x] Isolated research checkpoint catalog for user-supplied canonical Llama safetensors
- [x] Reuse PhiStream's own strict metadata layout inspection; no weight duplication
- [x] Explicit, locally approved token-ID runner with bounded input and receipts
- [x] Keep research aliases out of GGUF / Ollama / OpenAI chat catalogs
- [ ] Validate with a real licensed local checkpoint and independent reference parity
- [ ] Tokenizer, EOS/chat-template, text streaming and safe subprocess lifecycle before enabling a native chat backend

## Rung 7: ecosystem adapters

- Stable backend interface; only integrate PhiStream after real chat inference with parity receipts
- PhiOS trusted gateway and local service manifests
- PocketBrain, GAIA and game clients
- Optional permissioned LAN serving after authentication and transport security

**Acceptance rule:** Documentation, CI, synthetic fixtures, or an upstream HTTP stub never count as proof that a real model generated tokens on physical hardware.
