# Φ Inference Fabric R21 · One-command operator field study

R21 is the missing **hands-on operator launch point** for the merged R18
collector, R19 source verifier and R20 private ZIP packager. It adds no new
evidence schema, model runner, new daemon, physical GPU attestation or ranking.

## What must already be on each PC

- Python 3.11+ with **PhiServe and PhiStream installed in the same active
  Python environment**. The optional CUDA snapshot needs an already-working
  CUDA-enabled PyTorch install; no dependency is installed automatically.
- A trusted local llama.cpp llama-server executable, built for that PC.
- An existing licensed GGUF registered in PhiServe under a simple alias.
  Use the **same exact GGUF bytes** and public fixed-prompt protocol on the
  two PCs if you intend to compare them.
- An already created private workspace folder (example D:\PhiPrivate) **not
  inside any Git checkout** and not behind a symbolic-link path.
- An unused short session label, such as trial01. Each session has exclusive
  output names and no existing files are overwritten.

R21 never downloads model weights, installs drivers, discovers remote hosts,
copies private files over the network or launches a service without a user
command. It is an on-device CLI, not a GitHub Pages cloud runner.

Check the existing GGUF registrations first:

~~~powershell
phiserve models list
~~~

If needed, register a licensed GGUF you already have (actual path varies):

~~~powershell
phiserve models add field-model "D:\Models\your-local-model.gguf"
~~~

## Main PC (example operator label main-5070)

Start with a read-only preview. The path below is illustrative, not a
promise your executable lives there:

~~~powershell
phiserve-field-study preview --workspace "D:\PhiPrivate" --site main-5070 --session trial01 --alias field-model --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 0 --gpu-layers 20 --gpu-probe cuda --sample-nvidia
~~~

Preview checks the registered GGUF's local header, presence of the chosen
llama-server executable, accepted CPU-first offload profiles, output-name
collision and private workspace boundary. **It does not start models, probe
CUDA, write files or measure latency.** These checks are not physical GPU
readiness proof. If local PhiStream or CUDA is missing, the real run will
refuse or the R15 advice will be held; no fake GPU report is substituted.

When **you choose to perform actual inference** on this machine, change
preview to run and explicitly supply --execute:

~~~powershell
phiserve-field-study run --execute --workspace "D:\PhiPrivate" --site main-5070 --session trial01 --alias field-model --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 0 --gpu-layers 20 --gpu-probe cuda --sample-nvidia
~~~

The run does R18 **sequential local CPU + requested offload measurements**,
then captures a fresh PhiStream hardware snapshot and obtains its R15
review-only advice. It immediately reopens and validates the original JSON
with R19, then creates an R20 private transport ZIP. The tool returns a
success status only if all three stages completed. It does not authorize
model routing, deployment, or future unattended inference.

## Backup PC (example operator label backup-3060)

On the backup machine, install/register the same independently licensed GGUF
and its own trusted llama-server, then run the corresponding local commands:

~~~powershell
phiserve-field-study preview --workspace "D:\PhiPrivate" --site backup-3060 --session trial01 --alias field-model --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 0 --gpu-layers 20 --gpu-probe cuda --sample-nvidia
phiserve-field-study run --execute --workspace "D:\PhiPrivate" --site backup-3060 --session trial01 --alias field-model --llama-server "C:\llama.cpp\build\bin\Release\llama-server.exe" --gpu-layers 0 --gpu-layers 20 --gpu-probe cuda --sample-nvidia
~~~

The --site label is *declared by the human operator*, not an authenticated
device identity. Requesting 20 GPU layers is a suggested test configuration,
not a guarantee those layers fit or actually executed CUDA kernels. Select
a smaller offload count if the chosen GGUF and GPU cannot accommodate it.
For a CPU-only first session use --gpu-layers 0 and --gpu-probe none, but
this will not provide GPU comparisons.

## The three outputs (one session)

For the main-PC example, the workspace contains:

- main-5070-trial01/ : private original R18 JSON, including manifest,
  benchmark receipts, hardware snapshot, and R15 capacity advisory
- main-5070-trial01.review.json : R19 original-source evidence review
- main-5070-trial01.zip : R20 verified private offline ZIP for manual transport

If R18 inference fails, **there is no complete campaign manifest and no R19
review or R20 ZIP**. Previously completed source files may remain in the
private folder solely for diagnosis. If R19 or R20 holds, the command does
not claim the overall study succeeded. Select a **new session label** after
fixing the issue; do not overwrite partial or previous results.

The portable ZIP is NOT encrypted or signed. Transfer only via an operator-
approved private offline channel, and treat aliases/hashes as potentially
sensitive. The program never uploads evidence or ranks the fastest machine.

## Offline two-PC review after the operator transfers the backup ZIP

On the main PC, after transferring the ZIP privately:

~~~powershell
phiserve-campaign-bundle check --archive "D:\PhiPrivate\backup-3060-trial01.zip"
phiserve-campaign-bundle unpack --archive "D:\PhiPrivate\backup-3060-trial01.zip" --out-dir "D:\PhiPrivate\imported-backup-trial01"
phiserve-campaign-review --campaign "D:\PhiPrivate\main-5070-trial01" --campaign "D:\PhiPrivate\imported-backup-trial01" --output "D:\PhiPrivate\comparison-trial01.json"
~~~

R19 only reports protocol compatibility and measured descriptive summaries,
NOT a speed winner. Physical authenticity, GPU execution, actual model fit,
quality, driver/thermal parity, user consent for routing and any subsequent
deployment remain separate gates.

## Testing and evidence boundaries

~~~powershell
python -m unittest discover -s tests -p test_field_study.py -v
python -m unittest discover -s tests -v
~~~

CI's runner uses a fake minimal GGUF and synthetic benchmark/snapshot data.
It exercises the actual R18, R19 and R20 orchestration with injected fixture
measurements, but **cannot measure your RTX 5070 or RTX 3060**. The first
real benchmark numbers arise only when the operator runs the command locally
on each computer with actual hardware and an actual GGUF.
