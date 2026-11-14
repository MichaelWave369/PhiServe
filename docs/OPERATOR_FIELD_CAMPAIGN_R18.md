# Φ Inference Fabric R18: Operator-controlled physical GGUF field campaign

**Purpose:** collect real CPU and requested GPU-layer GGUF measurements on a
Windows workstation in one repeatable, private session. This CLI only
orchestrates existing PhiServe R13 and R15 components and PhiStream's existing
hardware snapshot. It does **not** simulate or forge measurements. GitHub CI
tests use synthetic receipts and cannot evaluate RTX 5070 / RTX 3060 speed.

## Prerequisites on EACH computer

1. On Windows 11, use Python 3.11+ with PhiServe and PhiStream installed in
   the **same active Python environment**. PhiStream requires compatible
   NumPy/PyTorch. For the CUDA snapshot, use a matching CUDA-enabled PyTorch
   build already tested on that computer. None is installed automatically.
2. Have an independent, trusted, local llama.cpp **llama-server.exe** with the
   intended CPU/CUDA backend. It must already exist; this kit never installs
   models, tools, libraries or drivers.
3. Register **the same licensed GGUF bytes** on both machines with
   phiserve models add local-inference "D:\Models\your-model.gguf".
   Use the same exact alias on both machines. Receipts contain GGUF SHA-256.
   A matching hash is local byte consistency, not independent origin proof.
4. Close other model servers using the selected loopback backend port; ensure
   enough CPU RAM and GPU VRAM and suitable power/cooling.
5. Pick a **new private output folder** in an existing parent directory and
   outside all Git checkouts. Do not upload the resulting JSON by default.

## Step 1: preview, no model execution

From the main machine (operator label is NOT authenticated machine identity):

~~~powershell
phiserve-fieldkit preview --alias local-inference --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --site main-5070 --out-dir "D:\PhiPrivate\main-run-01" --gpu-layers 0 --gpu-layers 20 --gpu-probe cuda --sample-nvidia
~~~

Preview verifies local model registry/binary existence, chosen profile ordering,
bounds and safe new output location. It **does not** create any files, connect
to a model, read CUDA counters or launch inference. It prints no local paths
or model weights. The chosen model alias and site label are operator inputs.

Before collecting CUDA data, verify your chosen local Python environment can
collect it at all (no model execution):

~~~powershell
phistream hardware-snapshot --gpu cuda
~~~

This may print GPU names, which are private local observations, not attestation.

## Step 2: explicitly run the same controlled workload

On the main workstation:

~~~powershell
phiserve-fieldkit collect --execute --alias local-inference --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --site main-5070 --out-dir "D:\PhiPrivate\main-run-01" --gpu-layers 0 --gpu-layers 20 --gpu-probe cuda --sample-nvidia
~~~

On the backup workstation, with an independently installed GGUF and runtime:

~~~powershell
phiserve-fieldkit preview --alias local-inference --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --site backup-3060 --out-dir "D:\PhiPrivate\backup-run-01" --gpu-layers 0 --gpu-layers 20 --gpu-probe cuda --sample-nvidia
phiserve-fieldkit collect --execute --alias local-inference --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --site backup-3060 --out-dir "D:\PhiPrivate\backup-run-01" --gpu-layers 0 --gpu-layers 20 --gpu-probe cuda --sample-nvidia
~~~

These **collect commands intentionally execute real inference** on the local
machines, run CPU-only first and then request 20 GPU layers. Choosing 20 is a
starting candidate, not a guarantee of fit or that offload succeeded. Adjust
the GPU-layer count for the actual model/backends. There are 3 warmups and
10 measured requests per configuration by default. Each measured profile
uses the existing fixed public prompt, identical sampling settings, bounded
SSE transport, and its own supervised loopback llama-server lifecycle.

If CUDA memory observation is unavailable, do not guess GPU capacity. Use
--gpu-probe none to obtain an honestly **held** review for GPU profiles,
or fix the local CUDA environment and rerun in a *new* output directory.
--sample-nvidia is an additional optional process-memory driver query, not
independent proof of CUDA execution.

## Output directory structure

If and **only if** all requested operations finish:

- benchmark-layers-0.json: real CPU-run R13 receipt
- benchmark-layers-20.json: real requested GPU/hybrid R13 receipt
- hardware-snapshot.json: PhiStream's live RAM/optional CUDA snapshot taken
  **after** the benchmarks, immediately before the R15 evaluation
- capacity-advice.json: PhiServe R15 human-review-only capacity report
- campaign-manifest.json: completion marker; local source SHA-256 bindings,
  operator-declared site label and all false automatic-routing assertions

A failure may leave already completed receipts in the private directory for
debugging. **No completion manifest is written on failure**; do not treat a
partially collected folder as a completed experiment. A fresh new output
directory is required to rerun safely; existing files are never overwritten.

The R15 hardware observation is collected after inference because its
freshness window is only five minutes. Hashing large GGUF files may itself
cause the snapshot to age beyond that limit: if the result is held as stale,
collect fresh evidence and rerun R15 intentionally. No automatic stale
override occurs.

## Review the real data locally

~~~powershell
phiserve-benchmark review "D:\PhiPrivate\main-run-01\benchmark-layers-0.json"
phiserve-benchmark review "D:\PhiPrivate\main-run-01\benchmark-layers-20.json"
phiserve-benchmark compare "D:\PhiPrivate\main-run-01\benchmark-layers-0.json" "D:\PhiPrivate\main-run-01\benchmark-layers-20.json"
~~~

To view detailed source-hash consistency, open the merged
**PhiMadaPrime → ThinkTank → Inference Observatory** and manually select the
R15 JSON together with its source snapshot and benchmarks. To create a
review-task ledger, run **PhiRepoNerve R17** locally against the capacity
advice JSON. Neither viewer can authenticate GPU execution or promote a
model into automatic routing.

For fair cross-machine interpretation, compare only receipts with the same
GGUF SHA-256, prompts, context, token limits, settings, trials and comparable
runtime binaries. Independent thermal load, driver, clocks and CUDA backend
still matter. The site labels main-5070 and backup-3060 are human-declared,
not independently observed device identities.

## Guarantees and non-guarantees

- Running preview never initiates inference, samples hardware or writes
  files. Collect refuses to start without the literal --execute flag.
- CPU/GPU requested profiles execute in sequence, never concurrently.
- GGUF and llama-server are local and preinstalled; model text is not copied
  into the resulting JSON. The benchmark contacts only supervised loopback.
- No network download, telemetry upload, auto-launch, model promotion,
  GPU routing, queue switching, merge or user credential access.
- SHA-256 references provide local file consistency only. Model fit, physical
  GPU execution, quality, cross-host identity and best device remain unknown.
- Site label, model alias and hashes may reveal private project information.
  Keep the output folder private and review it before manual sharing.

Automated CI validates contract and safe behavior with synthetic files only.
**Real field acceptance starts only when the operator runs the CLI on their
physical machines with an actual licensed GGUF.**
