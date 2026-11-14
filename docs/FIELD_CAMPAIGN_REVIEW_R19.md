# Φ Inference Fabric R19: Offline Field Campaign Evidence Review

**Local consistency checker, NOT a model runner, GPU witness or scheduler.**

R18 in PhiServe can collect observed CPU and requested GPU-layer benchmark JSON,
then a PhiStream hardware snapshot, a PhiServe R15 capacity-advice report, and
a final campaign-manifest.json. R19 independently reads those **already existing
operator-selected local files** and checks that the completed campaign is
internally consistent. It does not open or hash the original GGUF model or
llama-server executable, and cannot attest physical machine identity.

## Review one completed campaign, after physically running R18

~~~powershell
phiserve-campaign-review --campaign "D:\PhiPrivate\main-run-01" --output "D:\PhiPrivate\review-main-01.json"
~~~

To check *two* campaigns with the same GGUF and comparable configurations,
copy the private R18 receipt folder off the backup workstation through your
own trusted offline/private channel, then explicitly select both folders:

~~~powershell
phiserve-campaign-review --campaign "D:\PhiPrivate\main-run-01" --campaign "D:\PhiPrivate\backup-run-01" --output "D:\PhiPrivate\review-pair-01.json"
~~~

All receipt paths are placeholders, not proof the files exist. Run this only
after R18 produced a final manifest on **each** physical PC. No network
transport, auto-discovery, host polling, API tokens, model inference,
automatic merging or queue modifications occur in R19.

The program refuses to overwrite existing output, write into a tracked Git
checkout or mutate source campaign directories. Reports are private and
contain only bounded model aliases, recorded SHA-256 digests, selected
measurements and false authority assurances. They omit original paths,
GPU device names, raw prompts, generated text and model weights. Even an
alias or hash can identify private work; never publish your report blindly.

## Evidence inspection

Every selected campaign must include the R18 completion manifest and all
the exact referenced source files:

- campaign-manifest.json
- benchmark-layers-0.json and up to three additional named GPU-layer receipts
- hardware-snapshot.json
- capacity-advice.json

R19 rejects absent, symlinked, oversized, duplicate-key and nonfinite JSON
sources, path traversal, mismatching raw-file SHA-256, forged authority flags,
mismatching GGUF hashes/sizes, and changed llama-server binary hashes **within
a single campaign**. Each R13 benchmark is independently revalidated,
including original trial measurements, reported tokens, and p50/p95 summary.

R19 also verifies the actual PhiStream snapshot shape/counters; repeats the
R15 source-hash, memory budget, file-size floor, evidence-age, benchmark-profile
aggregate and classification checks; checks that the source snapshot came
after the same-session benchmarks; and compares the bounded fixed-prompt
sampling/context settings across all requested offload profiles.

The review time matters: a R15 report marked REVIEW_REQUIRED in its own
historical classification may be STALE_HARDWARE_SNAPSHOT *today*, since RAM
and VRAM counters can change in seconds. The program never silently extends
the 300-second snapshot or seven-day benchmark default windows.

## Two-campaign interpretation

R19 compares reported workload identity and file sizes/hashes before showing
a descriptive comparison classification:

- HOLD_DIFFERENT_GGUF_BYTES: no same-model claim can be made.
- HOLD_DIFFERENT_BENCHMARK_PROTOCOL: the controlled prompt or sampling settings differ.
- HOLD_NO_COMMON_CPU_BASELINE: no matching CPU baseline.
- DESCRIPTIVE_ONLY_DIFFERENT_SERVER_BUILDS: same model/protocol, but reported
  llama-server SHA-256 differs. Do not infer a hardware-only difference.
- DESCRIPTIVE_ONLY_MATCHING_REPORTED_PROTOCOL: same claimed model bytes,
  workload and server binary hashes; independent hardware and equal thermal
  conditions are STILL NOT proven.

Identical source bytes can be copied between two folders; a local SHA-256 match
does not identify separate physical devices. Different GPU layer settings,
drivers, thermals, OS loads, throttling and CUDA execution need separately
reviewed evidence. **R19 never ranks the fastest GPU, auto-selects a host,
authenticates a physical machine or authorizes deployment.** Its review output
includes no speed winner, no verified model fit and no GPU execution witness.

## Automated testing

~~~powershell
python -m unittest discover -s tests -p 'test_campaign_review.py' -v
python -m unittest discover -s tests -v
~~~

CI exercises synthetic R18 campaign files and adversarial mutations. The
Windows and Ubuntu matrix does not run RTX 5070 or RTX 3060 weights and its
passing result cannot substitute for operator-run field receipts.

**Next acceptance:** physically run R18 with a licensed GGUF on both systems
and use this R19 verifier. Real device benchmarking, careful data collection
and human judgment remain required before a future consent-gated router.
