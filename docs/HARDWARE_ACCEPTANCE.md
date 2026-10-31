# PhiServe Rung 9: hardware acceptance kit (local, no automatic promotion)

This kit helps the operator prepare honest hardware and parity evidence. It does not certify that an NVIDIA GPU executed weights, authenticate external reference independence, or install any software/models.

## Phase 1: inspect readiness, without executing weights

~~~powershell
phiserve acceptance doctor --llama-server "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe" --probe-nvidia --output ".\\receipts\\doctor.json"
~~~

This reads the local PhiServe GGUF and research catalog counts, checks whether the selected llama.cpp binary is present, and reports whether the optional PhiStream and tokenizer Python packages can be found. With `--probe-nvidia`, it also calls `nvidia-smi` **on the current local computer** and records GPU model, driver version and reported total memory. The probe is optional and does not check CUDA kernels, query the network, read weights or launch a model. All results are **operator-machine observations**, not external hardware attestation.

## Phase 2: real GGUF inference receipts, sequentially

~~~powershell
phiserve validate local-assistant --llama-server "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe" --gpu-layers 0 --output ".\\receipts\\gguf-cpu.json"
phiserve validate local-assistant --llama-server "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe" --gpu-layers 20 --output ".\\receipts\\gguf-gpu.json"
~~~

`phiserve validate` already exists from Rung 3. It launches the local llama.cpp backend against a registered, existing GGUF and requires a non-empty model text response. Run CPU and GPU requests sequentially because they use the same local backend port. Choose `--gpu-layers N` intentionally; requesting an offload count does not prove actual GPU residency. Comparing identical output SHA digests on the fixed prompt is a useful diagnostic, but even exact equality does not prove identical checkpoint bytes or true GPU execution. Version differences, nondeterministic kernels or unsupported settings can lead to legitimate mismatches.

**Evidence not yet collected by this kit:** process-attributed CUDA memory residency, physical hardware witness, wall-time benchmark fairness, sampled logits, validated tokenizer/weight provenance. If GPU validation matters, manually inspect a real runtime log and system GPU telemetry, keeping private data private. Do not publish receipts blindly: model aliases and GPU model names may be sensitive.

## Phase 3: optional PhiStream experimental evidence

If you already hold a supported licensed canonical Llama safetensors checkpoint and the matching trusted local tokenizer and reference trace:

1. Use `phiserve phistream tokenizer-check` and `parity-preflight` to verify the chosen prompt and reference token IDs.
2. Use `phiserve phistream text-lab` and `parity-run` *only with explicit local execution approval* and appropriate memory preflight.
3. Keep their **existing receipts** locally. The kit will check compatible research aliases, tokenizer SHA-256, and special-token flags.

A perfect parity receipt still originates from a **user-supplied, unauthenticated reference**. It has no automatic authority to enable PhiStream in production chat.

## Phase 4: summarize evidence conservatively

~~~powershell
phiserve acceptance report --doctor-receipt ".\\receipts\\doctor.json" --cpu-gguf ".\\receipts\\gguf-cpu.json" --gpu-gguf ".\\receipts\\gguf-gpu.json" --output ".\\receipts\\acceptance-summary.json"
~~~

Add `--phistream-text FILE` and/or `--phistream-parity FILE` if you have them. Any subset of existing receipts can be reviewed. The reviewer refuses incompatible CPU/GPU aliases/prompts, research model aliases/tokenizer fingerprints, malformed/duplicate-key or oversized JSON, non-local symlink input files, inconsistent parity success counts, and contradictory evidence. The report is a **new JSON file, never overwriting existing files**, and embeds hashes of input JSON bytes but not their raw content, local paths, full logits, prompt text or generated responses.

- `EVIDENCE_COLLECTED_REVIEW_REQUIRED`: parsed data is internally consistent. **Not independent verification.**
- `DIAGNOSTIC_MISMATCH_REQUIRES_REVIEW`: CPU/GPU output hashes disagree or PhiStream parity explicitly mismatched. CLI exits 2 while preserving the summary.
- `production_phistream_chat_authorized: false` and `gpu_execution_independently_verified: false` **always**, regardless of other signals.

GitHub CI runs fake GGUF and PhiStream receipt fixtures and never supplies physical proof. The next acceptance milestone is an operator-run real model experiment with independently observed GPU execution, independently executed reference trace and strict matching model/tokenizer provenance.
