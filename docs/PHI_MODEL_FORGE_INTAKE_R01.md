# PhiServe ↔ Φ Model Forge candidate intake · R0.1

**Not a serving or conversion feature.** This isolated read-only intake inspects
a local PEFT LoRA adapter produced by
[PhiStream Φ Model Forge](https://github.com/MichaelWave369/PhiStream).
It checks the reported base model against `adapter_config.json`, expected
training receipt schema, basic safetensors envelope bounds, and a dataset SHA256
that the operator supplied. Optionally it hashes the *entire* adapter file.

```sh
phiserve-forge-review \
  --adapter /home/operator/phi-model-forge/adapter-r01 \
  --receipt /home/operator/phi-model-forge/adapter-r01/phi-training-receipt.json \
  --hash-weights
```

The output is `REVIEW_REQUIRED`, with `gguf_compatible: false`,
`phi_serve_registered: false` and `execution_authorized: false`.
The tool never registers a model, opens a network listener, executes model
code, downloads weights, or writes to the source directory. All input bytes
are operator-supplied; the script does NOT authenticate Hugging Face provenance,
verify a training run took place, measure intelligence, or establish licensing.

For future service:

1. Retain the original licensed base checkpoint, local LoRA adapter and
   hashed training/evaluation receipts.
2. Manually review outputs against the unmodified base with independent
   held-out data; report regressions as well as successes.
3. Separately merge/convert, license-check, and validate a **real GGUF**
   using a trusted external converter before considering `phiserve models add`.
4. Operator decides whether to register a distinct alias and run a local
   inference smoke. No automatic model replacement, port opening, or
   production promotion.
5. Browser, Mobile and XR clients never receive direct GPU control or
   automatic LAN model access; use explicit trust/authorization gateways.

Code is MIT. Model adapter licensing must be reviewed separately.
Heretic's AGPL-3.0 source is not embedded in PhiServe or PhiStream.
