# PhiServe Hardware Benchmark V1: local, measured, never attested

**Status:** Opt-in operator-run field benchmark for a **registered GGUF** using an
independently installed `llama-server`, plus strict local receipt review.
This is not a production benchmark leaderboard, an independent GPU witness,
a model-quality evaluation, a hosted endpoint, a memory scheduler, or a
proof that offloaded layers executed CUDA kernels.

## Windows local first run

1. Install PhiServe from this branch (`py -m pip install -e .`) and separately
   install a trusted compatible `llama-server`. Keep it on the same machine.
2. Register your own licensed GGUF if not already registered:
   `phiserve models add my-local-model "D:\Models\my-model.gguf"`
3. Run a controlled CPU measurement (a **new** output path):

```powershell
phiserve-benchmark run my-local-model --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 0 --ctx-size 2048 --warmups 3 --trials 10 --output ".\receipts\benchmark-cpu.json"
```

4. Run a GPU/hybrid *requested* offload configuration **sequentially**, after
   ensuring no other process uses the private backend port:

```powershell
phiserve-benchmark run my-local-model --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 20 --ctx-size 2048 --warmups 3 --trials 10 --sample-nvidia --output ".\receipts\benchmark-gpu.json"
phiserve-benchmark review ".\receipts\benchmark-gpu.json"
phiserve-benchmark compare ".\receipts\benchmark-cpu.json" ".\receipts\benchmark-gpu.json"
```

This uses an identical built-in **public** prompt, temperature 0, seed 42,
streaming SSE and requested token/context settings. It hashes the entire local
GGUF and selected binary before benchmarking, which can take time for very
large files. Then it launches one local `llama-server`, runs warmups that are
not included in results, and runs measured requests sequentially. It neither
downloads a model nor changes the registered model or its weights.

Output is a new JSON file created exclusively. Raw prompts, completions,
URLs, credentials, private paths and model contents are **not** included.
The public fixed prompt's hash and generated reply hashes are retained.
Do not publish receipts without reviewing model aliases and hardware labels.

## Measurement semantics

| Field | Meaning | Important limitation |
|---|---|---|
| `runtime.startup_ms` | Launch supervised local backend to ready condition | Includes process startup and health checks, excludes file hashing |
| `wall_ms` | Request start to observed SSE `[DONE]` | Includes local HTTP/serving overhead; not just decode |
| `first_content_ms` | Request start to first nonempty assistant text chunk | **Not** independently measured first model token |
| `completion_tokens_reported` | Optional usage supplied by `llama-server` SSE | Missing/null is honest; chunks are **never** counted as tokens |
| `throughput_tps` | Reported completion tokens divided by request wall duration | End-to-end text throughput, **not** decode-only tokens/sec |
| `p50_*`, `p95_*` | Median and nearest-rank p95 over measured trials | Descriptive, not significance tests |
| `gpu_process_telemetry` | Optional `nvidia-smi` process self-report | WDDM can report N/A; no CUDA attestation |
| `gguf_sha256`, `binary_sha256` | Hashes of selected local files | Locally computed hashes, not external authenticity |

All unknown data stay unknown: there is no invented host process peak RAM,
GPU layer residency, energy use, model-quality score, or inference speed when
upstream omits completion-token counts. The `compare` command refuses different
GGUF bytes, fixed workloads, context sizes or sampling settings. Even a
comparable result is **descriptive only**, and different machines, drivers,
thermal states, and binaries remain potential confounders.

If the server does not support OpenAI-compatible SSE with
`stream_options.include_usage`, update the trusted `llama-server` or run
existing `phiserve acceptance field-run` instead. **Do not silently fall back to
an incomparable non-streaming result**.

## Evidence and safeguards

- The benchmark never binds a network listener and contacts only its own
  supervised backend at fixed loopback `127.0.0.1:<selected-port>`.
- A preexisting port is rejected by `LlamaBackend`; output is never overwritten.
- Input bounds limit prompt, response, requests, context and run duration.
- Invalid, nonfinite, duplicate-key, contradictory, or falsely promoted JSON
  receipts are rejected by the review command.
- Every receipt explicitly records `cuda_execution_independently_verified: false`,
  `quality_evaluated: false`, `physical_hardware_attested: false` and
  `performance_certified: false`.
- CI tests replace inference with a fake HTTP stream. Passing CI demonstrates
  only contract/parser behavior. **Real model execution requires running the
  command on your own machine.**

## Recommended study order

1. Choose **one** installed GGUF, freeze model + runtime version.
2. Record CPU-only and offload variants sequentially, with the same context,
   prompt, max tokens, temperature, warmups and measured trial counts.
3. Repeat runs across multiple sessions, documenting the operating system,
   driver, background load and cooling separately.
4. Require a separate answer-quality evaluation before promoting a route.
5. Pass reviewed results to PhiRepoNerve / PhiMadaPrime only in a future,
   separately authorized integration. No automatic transfer or promotion.
