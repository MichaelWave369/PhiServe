# PhiServe R14: opt-in bounded inference queue

**Status: experimental, single-daemon and loopback only.** A fixed-capacity
FIFO admission queue lets explicitly queued OpenAI-style clients wait for a
model lease instead of immediately receiving legacy 409/429/503 responses.
Existing nonqueued endpoints preserve their original behavior when no queued
requests are pending. There is still one resident llama.cpp subprocess per
daemon, no parallel GPU residency, no cross-node dispatch, no cloud fallback,
and no adaptive inference scheduler.

## Start the opt-in queue

From a locally installed PhiServe on Windows PowerShell:

~~~powershell
phiserve daemon --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 20 --queue-capacity 8 --queue-timeout-seconds 30
~~~

Default queue capacity is **zero**, so existing callers do not implicitly
start waiting. Capacity can be 0 to 32, and wait timeout 0.1 to 120 seconds.
This bounds queued server request threads but does not speed up llama.cpp.
Use a client HTTP timeout longer than the selected server queue timeout,
with room for model startup and generation.

## Explicit request paths

Use ordinary supported OpenAI-compatible JSON and call:

- POST /v1/queued/chat/completions (forwarded to /v1/chat/completions)
- POST /v1/queued/completions (forwarded to /v1/completions)

Example from same-device PowerShell with registered alias local-assistant:

~~~powershell
$body = @{ model = "local-assistant"; messages = @(@{ role = "user"; content = "Hello" }); stream = $false } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:11435/v1/queued/chat/completions" -Method Post -ContentType "application/json" -Body $body
~~~

The implementation does not add another public listener. Existing strict
JSON, content-length, model validation, bearer-token, origin, and streaming
guards remain in force before queue admission. GET /health adds
queued_requests and queue_capacity, protected by the same browser token gate
when enabled. The queue only stores anonymous in-memory tickets, not prompt
text, user identities, response bodies or raw model paths.

## Semantics

- Strict FIFO **admission** among opt-in queued requests. Simultaneous
  same-model requests can complete out of order.
- Queue capacity counts waiting requests, separate from active model leases.
- Admission requires a healthy same-model backend with an available
  concurrency slot, or no active leases before an atomic model transition.
- A queued request never evicts an active model. Legacy requests remain
  nonblocking; while a queue is waiting they get HTTP 429 to prevent
  starvation of queued clients.
- HTTP 429: queue full; 504: wait timed out; 503: queue disabled,
  stopping daemon, or failed model startup; 404: unavailable model alias.
  Legacy request codes remain unchanged when no queue is pending.
- Shutdown wakes waiting clients for immediate refusal. Timeouts remove
  the relevant ticket. Idle eviction skips models with waiting requests.
  Streaming keeps the lease through its response lifecycle.
- No prioritization, predictive prefetching, energy optimization, remote
  dispatch, persistence, in-flight generation cancellation, or detection
  of a queued client's TCP disconnect. Queue expiry is a server-side
  bound, not proof that client-observed latency improves.

## Verification

~~~powershell
py -m unittest discover -s tests -v
~~~

Automated tests use synthetic local HTTP and fake model processes. They
exercise FIFO model switches, queue capacity, expiry, shutdown, startup
failures, same-model reuse and existing nonblocking behavior.
They do not benchmark physical CPU/GPU inference.

Next operator field gate: use one licensed registered GGUF and the existing
phiserve-benchmark R13 tool to collect measured CPU/GPU receipts, then test
bounded contention. Review p95 total request latency and model startup costs
before increasing the queue size.
