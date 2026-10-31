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

## Rung 3: model lifecycle

- Integrity-checked downloads with explicit consent, resume and per-model license display
- Disk quotas, remove/import, GGUF metadata parsing, configuration profiles
- Hardware-aware resource estimates, tested GPU fallback and scheduler

## Rung 4: ecosystem adapters

- Stable backend interface; only integrate PhiStream after real chat inference with parity receipts
- PhiOS trusted gateway and local service manifests
- PocketBrain, GAIA and game clients
- Optional permissioned LAN serving after authentication and transport security

**Acceptance rule:** Documentation, CI, synthetic fixtures, or an upstream HTTP stub never count as proof that a real model generated tokens on physical hardware.
