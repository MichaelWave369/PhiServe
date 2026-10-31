# Rung 10: Windows GPU field run (operator-executed, no automatic install)

The objective is to finally **run a real local GGUF checkpoint** on your own hardware and attach optional process telemetry. GitHub Actions still uses fake models and fake telemetry, so green CI does not establish that a physical RTX 5070 ran any weights.

## Prerequisites

- Windows 11 with a locally installed, trusted `llama-server.exe` built with your intended CPU/CUDA backend; PhiServe does **not** download one.
- A real, licensed GGUF file already registered with `phiserve models add ALIAS PATH` or imported by reference from Ollama with `phiserve models import-ollama ...`.
- Use a model that fits your actual memory/context limits; shut down other model servers occupying the chosen backend port.
- `nvidia-smi` on PATH is optional. It is only used if you explicitly pass `--sample-nvidia`.
- This command deliberately runs locally through `127.0.0.1`, not over PocketBrain, LAN, remote desktop or GitHub-hosted CI.

## Real field-run commands

~~~powershell
phiserve acceptance doctor --probe-nvidia --output ".\\receipts\\environment.json"
phiserve acceptance field-run YOUR_GGUF_ALIAS --llama-server "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe" --gpu-layers 0 --output ".\\receipts\\gguf-cpu.json"
phiserve acceptance field-run YOUR_GGUF_ALIAS --llama-server "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe" --gpu-layers 20 --sample-nvidia --output ".\\receipts\\gguf-gpu.json"
phiserve acceptance report --doctor-receipt ".\\receipts\\environment.json" --cpu-gguf ".\\receipts\\gguf-cpu.json" --gpu-gguf ".\\receipts\\gguf-gpu.json" --output ".\\receipts\\comparison.json"
~~~

Use **different output paths** each time; receipts are never overwritten. Both runs use the same fixed prompt from `phiserve validate`. Requests are sequential; the child `llama-server` starts and stops around each attempt. It never silently installs or downloads model files. If model startup, model generation or output parsing fails, there is **no new success receipt**.

## What the GPU sampler does

After llama.cpp reports local readiness, PhiServe identifies its direct subprocess PID. If `--sample-nvidia` was requested, a background thread polls:

`nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits`

The query is local, shell-free, subprocess-time-bounded and stops before the child process exits. Reported per-process GPU memory for the matching PID is recorded as an observed *peak MiB*. It never stores model text, argument lists, GPU-process information about other PIDs, or full command output.

You may see these statuses:
- `PROCESS_MEMORY_DRIVER_SELF_REPORT`: at least one numeric memory sample was reported for the exact child PID. **Not independently verified CUDA inference.**
- `PROCESS_PRESENT_MEMORY_UNAVAILABLE`: the child PID was seen but Windows WDDM reported `N/A` or unsupported process memory.
- `NO_MATCHING_COMPUTE_PROCESS_OBSERVED`: the query worked but did not list the child; it may have missed a short inference or lacked visibility.
- `NVIDIA_SMI_UNAVAILABLE` / `NVIDIA_SMI_QUERY_FAILED`: optional tool missing or tool output failed.
- `SAMPLER_DID_NOT_STOP`: sampling thread shutdown did not finish promptly; treat telemetry as suspect.

**Limitations:** PID reuse, Windows WDDM GPU accounting, driver self-reporting, short generations, background GPU tasks and subprocess timing mean that this is an *observation*, not physical attestation. Model weights staying resident is not proven by process-memory data alone. GPU layers requested do not certify CUDA offload. PhiServe does not enable GPU scheduling, privileged tracing, telemetry upload, secret network listeners, or remote model control.

## Evidence promotion gate

A passing receipt can be reviewed using `phiserve acceptance report`; the reviewer checks the shape and consistency of optional process-memory telemetry. It will **never** mark independent GPU execution or production PhiStream chat as verified. The next phase is to compare real driver activity, backend offload logs, model/hash identity and independently produced PhiStream reference traces on the operator's own device.

Do not include personal information in the test prompt. CLI receipt filenames and model aliases can be revealing. Keep receipts private unless you choose to share a reviewed redacted copy.
