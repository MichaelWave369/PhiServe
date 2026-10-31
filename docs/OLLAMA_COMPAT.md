# Ollama compatibility, Rung 2 (experimental)

PhiServe now implements a **text-only subset** of Ollama's HTTP API:
- `GET /api/tags`: the active model only; truthful file size/mtime when available.
- `POST /api/chat`: text-only chat messages with `role` and `content`.
- `POST /api/generate`: text prompt (chat-template path by default), optional system message, or literal prompt with `raw:true`.

Default responses are Ollama-style **NDJSON**; `stream:false` returns a single object. We translate to the locally supervised llama.cpp OpenAI-format server. The Ollama NDJSON adapter is not used for PhiServe's existing OpenAI SSE passthrough.

Supported sampling options are `temperature`, `top_p`, `seed`, `num_predict`, `stop`. Other options and features (tool calls, images, JSON schemas, embeddings, `think`, context override, `keep_alive`, pull, push, delete, `/api/show`, etc.) are not implemented and are **rejected** with 422 rather than ignored.

Stream final markers include measured gateway elapsed time and real upstream usage counts when provided. They do not invent missing per-token timing metrics. A missing model digest or GGUF family is omitted rather than made up.

The `model` must be the loaded PhiServe alias; PhiServe currently serves one GGUF per process. There is no network or authentication support. Bind only to loopback.

Acceptance: simulated protocol integration tests on Linux/Windows. Physical GPU GGUF inference is a separate validation gate.
