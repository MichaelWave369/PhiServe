# PhiServe ecosystem integration map (Rung 12)

This is a **source-reviewed inventory**, not a live device scan, authority grant, or a claim that six applications are currently connected.

## What each project actually provides

| Repo | Type | How it relates to PhiServe | Current status |
| --- | --- | --- | --- |
| [PhiStream](https://github.com/MichaelWave369/PhiStream) | Experimental safetensors inference engine | Possible future model-execution backend | Research-only; token-ID and reference validation, not production chat |
| [PocketBrain](https://github.com/MichaelWave369/PocketBrain) | Browser assistant, local memory, WebLLM and Tiny browser model | Calls PhiServe | Dedicated `src/providers/phiServeProvider.ts` exists; real browser acceptance pending |
| [BrainC](https://github.com/MichaelWave369/BrainC) | Assistant, prompts, tools, memory | Future consumer, reuse compatible GGUF weights | Modelfile references `qwen2.5:14b`; weights not in repository |
| [SuperPhiVessel](https://github.com/MichaelWave369/SuperPhiVessel) | Governed orchestration, permissions and model routing | Future client | Not wired; governance stays in Vessie |
| [GAIA](https://github.com/MichaelWave369/GypsyAI) | Symbolic assistant and browser model research | Future optional client | Existing experiments, but no reviewed PhiServe client |
| [Think Tank](https://github.com/MichaelWave369/think-tank) | Governed multi-agent deliberation | Local Node provider calls PhiServe | [PR #55](https://github.com/MichaelWave369/think-tank/pull/55) implements adapter; pending merge/physical field test |

**Runtime directions matter:** PhiServe does **not** need to import memories, tools, identities or orchestration code from these apps. It should serve locally held, separately licensed GGUF checkpoints; clients supply their own prompts, memory scope, decisions and authorization. PhiStream is distinct from downloaded pretrained weights and cannot simply be treated as another GGUF chat model. PocketBrain's WebLLM/ONNX browser models are not GGUF binaries the Python daemon can import.

```text
Local GGUF -> PhiServe (one resident llama.cpp backend, 127.0.0.1)
                  |
        text-only API + explicit auth
          /       |       \
   PocketBrain  Think Tank   Future BrainC/GAIA/Vessie
   browser      Node bridge  opt-in scoped clients

PhiStream safetensors research remains separate from production chat.
```

## Important boundaries

- **One resident model:** concurrent clients asking for different models may receive HTTP 409; transitions may return 503. Never silently fall back to cloud.
- **Same-device browser:** localhost points to the machine running the browser, not another person's workstation. Browsers may require local-network access approval. For phones and Quest, design a separate TLS-authenticated gateway instead of exposing 11435.
- **Tokens:** keep bearer tokens in local server-side processes or ephemeral browser memory, never commit them, put them in URLs or broad logs, or transfer them with an unrelated app's memory.
- **Authority:** Vessie/Think Tank govern their own effects; inference text is not authorization.
- **BrainC:** `model/Modelfile` is a recipe including a system prompt, not Qwen weight bytes; use a legal compatible local GGUF and preserve persona through system messages.
- **PhiStream:** requires tokenizer provenance, independently executed reference parity and real hardware evidence before any production backend.
- **Inventory:** the new React ecosystem view is static code-reviewed information. It does not probe app endpoints or claim any connection succeeded.

## Recommended order

1. Merge and run [Think Tank PR #55](https://github.com/MichaelWave369/think-tank/pull/55) locally against PhiServe.
2. Perform same-machine acceptance of PocketBrain's existing PhiServe adapter.
3. Add BrainC's opt-in local model backend while keeping its memory/tool authorization.
4. Add GAIA scoped inference only with explicit user consent to transmit selected text to localhost.
5. Add Vessie's capability-scoped model adapter under its existing Reality Gate.
6. Finish real PhiStream backend parity, tokenizer/chat-template, cancellation and performance acceptance before it can join the serving registry.

**Status date:** 2026-10-31. Code inventory is not runtime telemetry, external attestation or deployment status.
