# PhiServe Rung 1 Architecture

~~~text
Local client (PhiOS, PocketBrain, CLI, etc.)
          |
          | HTTP OpenAI-style JSON or SSE
          v
PhiServe 127.0.0.1:11435
  - request-size and active-model validation
  - /health, /v1/models
  - /v1/chat/completions, /v1/completions
          |
          | HTTP loopback
          v
llama.cpp llama-server 127.0.0.1:11436
          |
          v
User-supplied local .gguf file
~~~

The process manager starts a real llama.cpp server, waits for readiness, and terminates it when PhiServe shuts down. The HTTP gateway never claims to implement inference. It forwards upstream SSE chunks without collecting the full response.

**Trust boundaries.** Client requests cannot set a backend URL, binary, file path, or subprocess options. The daemon only listens on loopback, with a separate private loopback port for its backend. Registry paths are selected by the local CLI operator and verified for file existence and GGUF magic. That is **not** a proof a checkpoint is benign, nor are these routes safe for public exposure.

**No cloud requirements.** Requests and weights stay on the local host; there are no download calls, accounts, telemetry collectors, or remote API integrations.

**Important current limits.** One registered model selected per process; no model hot-swap; no persistent GPU scheduler; no streaming cancellation protocol to child on client disconnect beyond closing the local connection; no cross-process registry locking; no real-device inference proof in CI.

**Why PhiStream is not a backend yet.** PhiStream currently supports experimental narrow local safetensors token-ID execution and verification, but does not provide production chat-tokenization or generic GGUF inference. Bridging it requires an independently tested adapter, not optimistic routing.
