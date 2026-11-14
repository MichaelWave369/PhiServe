# PhiServe R15: hardware evidence advisor

**Status: offline / read-only / review-only.** This bridge compares an existing
PhiStream hardware snapshot with one to eight already-completed PhiServe R13
benchmark receipts for **one exact locally registered GGUF**. It refuses
incompatible evidence, computes conservative memory-capacity *floors*, and
writes a new local report with no prompts or private paths.

It does **not** choose or switch a live model, alter R14 queue admission,
reserve GPU or CPU memory, start inference, modify PhiStream, download weights,
contact an external provider, prove actual model fit, or certify speedups.

## Prerequisites

1. In PhiStream, opt in to a recent observation, ideally without personal drive
   paths unless capacity information is needed. A CUDA snapshot requires
   compatible local PyTorch; a RAM-only snapshot is allowed.
2. In PhiServe, use phiserve-benchmark (R13) to create genuine receipts for the
   **same locally registered GGUF** with consistent context and sampling
   settings. Run CPU and GPU profiles sequentially with a compatible
   llama-server; do not infer GPU execution from the selected --gpu-layers.
3. Store the inputs locally and ensure the referenced model bytes remain
   exactly the same. R15 hashes the complete registered GGUF and matches it
   against each R13 receipt. Large weights take time to hash.

## Run on Windows

Example from a PhiServe checkout, after a PhiStream snapshot was already
collected and a model was registered as local-assistant:

~~~powershell
py -m pip install -e .
phiserve-advisor --model local-assistant --snapshot "D:\PhiLab\private\phistream-hardware.json" --benchmark "D:\PhiLab\private\bench-cpu.json" --benchmark "D:\PhiLab\private\bench-gpu.json" --ram-reserve-mib 4096 --gpu-reserve-mib 1024 --output "D:\PhiLab\private\capacity-advice.json"
~~~

A snapshot is fresh by policy only within 300 seconds by default. Re-run
PhiStream's read-only hardware-snapshot command immediately before comparing.
Benchmarks are held if older than seven days by default. Both windows can be
overridden *explicitly* within bounded ranges.

Missing CUDA counters cannot be replaced by a guessed RTX VRAM size, even if
the GPU was selected for a benchmark. For CPU-only evidence, no GPU counter
is required.

## Report interpretation

- REVIEW_REQUIRED_CAPACITY_NOT_CONFIRMED: all basic evidence gates cleared;
  further human review, real placement, and correctness tests are required.
- HOLD_HARDWARE_OBSERVATION_STALE_OR_FUTURE: snapshot not recent or its
  timestamp is materially in the future.
- HOLD_BENCHMARK_EVIDENCE_STALE_OR_FUTURE: at least one benchmark is stale
  or future-dated.
- HOLD_CUDA_COUNTERS_MISSING: GPU layers were requested, but the source
  snapshot never observed CUDA free memory.

The RAM/GPU floor simply compares the GGUF **file size** to observed free
memory minus the selected reserve. A false value can rule out straightforward
full weight residency in that tier under the chosen reserve; a true value
**never proves fit**. It excludes KV cache, allocator overhead, runtime
buffers, context length effects, OS contention, transfer costs and
quantization execution constraints. GPU offload can still be viable when a
full-GPU file-size floor fails.

Per-profile p50/p95 wall times and R13-reported token throughput are
descriptive field observations, not decode-only benchmarks, model quality
comparisons, energy measurements or independent hardware provenance.

The report always records false for automatic routing, execution approval,
GPU execution verification, performance certification and model-quality
verification. It intentionally omits source drive paths, device names,
private model paths, raw prompts and generated text. Model alias and file
hashes may still be identifying; inspect reports before publishing.

The reviewer checks structural/source contracts, evidence freshness,
matching actual GGUF bytes, model alias, OS family, workload settings and
unique GPU-layer profile counts. Matching OS families are **not** proof
that benchmarks came from the same physical computer. Do not combine
receipts from separate machines into an asserted same-machine comparison.

## Security / reproducibility

- Only existing local JSON/GGUF files are read. No network or inference APIs
  are contacted. The output is created exclusively, never overwritten.
- Duplicate JSON keys, nonfinite fields, unexpected schemas, contradictory
  counters, bogus verification claims and unsupported sources are refused.
- Source receipts are bounded and must not be symlinks. The newly written
  report includes SHA-256 hashes of source JSON and GGUF bytes as local
  binding references, not authenticated signatures.
- R15 remains isolated from PhiOS authorization and the R14 live queue.
  Integration into live workload admission requires a separately reviewed
  allowlisted authority handoff and fresh hardware rechecks immediately
  before execution.

## CI

~~~powershell
py -m unittest discover -s tests -v
~~~

CI exercises deliberately synthetic snapshots and benchmark fixtures. It
does not claim to have observed the user's RTX 5070, RTX 3060, physical
RAM, actual CUDA kernels or real end-to-end throughput.
