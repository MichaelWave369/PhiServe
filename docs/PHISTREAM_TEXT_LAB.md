# Rung 7: local tokenizer → PhiStream IDs → decoded text (research)

PhiServe supports a deliberately limited **local-only text lab** using the existing PhiStream research checkpoint catalog. This is a genuine path from plain text through a user-provided `tokenizer.json` into PhiStream's *experimental* decoder, followed by token ID decoding. It is neither a production inference backend nor a certified match between tokenizer and checkpoint.

## Dependencies

- PhiServe v0.7.0 installed with optional `phistream-text` extra, which adds the offline Hugging Face `tokenizers` parser.
- A separately installed, trusted PhiStream package in the same Python environment.
- An explicitly registered local, canonical Llama safetensors checkpoint, with user-declared model source/revision/license. This tool does not download or copy weights.
- An explicitly chosen local `tokenizer.json` file. The application does not fetch it or assume it belongs to the selected model. It reads/hash-binds the exact JSON bytes it parses.

## Two phases

1. **Preflight only**: `phiserve phistream tokenizer-check research-llama --tokenizer "D:\\Models\\Llama\\tokenizer.json" --prompt "Hello world" --tokens 4`. Returns vocabulary size, hashed tokenizer bytes, input token count and context-fit decision. No tensor staging or inference.
2. **Explicit execution**: `phiserve phistream text-lab research-llama --tokenizer "D:\\Models\\Llama\\tokenizer.json" --prompt "Hello world" --tokens 4 --device cpu --capacity-mib 4096 --reserve-mib 1024 --approve-local-execution --output ".\\receipts\\text-lab.json" --show-output`.

The required capacity/reserve arguments activate PhiStream's memory-policy preflight; **this is not a guaranteed VRAM/RAM cap**. The CLI `--prompt` argument can appear in OS process inspection/history, so don't use sensitive prompts. The optional `--add-special-tokens` applies only tokenizer-defined specials: it does **not** construct an instruct chat template. Readable output doesn't prove that a tokenizer matches the checkpoint.

The executor writes an intermediate numeric-token receipt into a temporary directory, validates its schema and requested prompt/device/output-token count, decodes the generated IDs, then deletes the intermediate. The durable JSON receipt is created exclusively and contains only hashes/lengths/explicit uncertainty flags, not raw prompts or generated text. `--show-output` prints decoded text to the local console only if requested. A failed run does not produce a success receipt.

## Limits and promotion gates

- Not exposed as `/api/chat`, `/api/tags`, `/v1/models`, or `/v1/chat/completions`. PhiServe's live chat remains llama.cpp/GGUF only.
- No automatic model weights, tokenizer downloads or token conversion from GGUF.
- No verified weight/tokenizer pair identity, BOS/EOS policy, chat template, stop behavior, streaming, concurrency, long-context validation, GPU measurements, or independent model-reference parity.
- CI uses a synthetic local tokenizer and simulated PhiStream subprocess. It demonstrates **software plumbing only**, not physical checkpoint execution.
- The optional tokenizer loader is a local JSON parser from a separately installed dependency. Treat both the tokenizer file and checkpoint as trusted operator input.
- Promotion to a service backend requires independently comparable token/logit/reference receipts, proven tokenizer/chat-template identity, real-machine hardware and memory benchmarks, safe cancellation and streaming, and a separate backend contract.
