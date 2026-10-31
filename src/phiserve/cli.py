"""Small CLI for a genuinely local, single-model server."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .backend import LlamaBackend
from .manager import ModelManager
from .registry import ModelRegistry, RegistryError
from .ollama_store import OllamaStoreError, discover, import_model
from .validation import validate_local
from .server import make_server
from .security import create_token_file, load_access
from .phistream_lab import PhiStreamCatalog, PhiStreamLabError, probe_checkpoint, run_ids
from .phistream_text import encode_prompt, text_lab
from .phistream_parity import preflight_reference, run_reference_gate
from .acceptance import AcceptanceError, doctor, acceptance_report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="phiserve", description="Sovereign local inference service")
    root.add_argument("--version", action="version", version=f"PhiServe {__version__}")
    commands = root.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="Manage local GGUF model references")
    model_commands = models.add_subparsers(dest="action", required=True)
    add = model_commands.add_parser("add")
    add.add_argument("alias")
    add.add_argument("path")
    model_commands.add_parser("list")
    remove = model_commands.add_parser("remove")
    remove.add_argument("alias")

    discover_cmd = model_commands.add_parser("discover-ollama", help="Read existing Ollama manifests without modifying them")
    discover_cmd.add_argument("--root", help="Ollama models directory (otherwise OLLAMA_MODELS / user default)")
    import_cmd = model_commands.add_parser("import-ollama", help="Register one existing GGUF blob without copying weights")
    import_cmd.add_argument("model", help="Exact name shown by discover-ollama, such as qwen2.5:7b")
    import_cmd.add_argument("--alias", help="Override automatically derived PhiServe alias")
    import_cmd.add_argument("--root", help="Ollama models directory")
    import_cmd.add_argument("--verify-sha256", action="store_true", help="Read/hash entire model blob against manifest")

    research = commands.add_parser("phistream", help="Optional, research-only local safetensors checkpoint lane")
    actions = research.add_subparsers(dest="action", required=True)
    inspect = actions.add_parser("inspect", help="PhiStream metadata/strict Llama preflight; no inference")
    inspect.add_argument("path")
    register = actions.add_parser("register", help="Register one local compatible checkpoint by path, no weight copies")
    register.add_argument("alias")
    register.add_argument("path")
    register.add_argument("--source", required=True, help="User-declared weight source")
    register.add_argument("--revision", required=True, help="User-declared checkpoint revision")
    register.add_argument("--license-id", required=True, help="User-declared model license (not verified)")
    actions.add_parser("list", help="Show research-only checkpoint aliases")
    remove_research = actions.add_parser("remove", help="Delete a research alias, preserve weights")
    remove_research.add_argument("alias")
    execute = actions.add_parser("run-ids", help="Explicit, offline PhiStream token-ID execution (not chat)")
    execute.add_argument("alias")
    execute.add_argument("--prompt", required=True, help="Comma-separated token IDs, not text")
    execute.add_argument("--tokens", type=int, default=4)
    execute.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    execute.add_argument("--approve-local-execution", action="store_true")
    execute.add_argument("--capacity-mib", type=float, required=True,
                         help="Explicit memory preflight budget; never a guaranteed VRAM cap")
    execute.add_argument("--reserve-mib", type=float, default=1024.0)
    execute.add_argument("--output", required=True, help="New local JSON receipt; refuse overwrite")
    execute.add_argument("--timeout-seconds", type=int, default=900)

    token_check = actions.add_parser("tokenizer-check", help="Offline text tokenizer preflight; no model execution")
    token_check.add_argument("alias")
    token_check.add_argument("--tokenizer", required=True, help="Existing local tokenizer.json file")
    token_check.add_argument("--prompt", required=True, help="Plain-text prompt; may appear in local process arguments")
    token_check.add_argument("--tokens", type=int, default=4)
    token_check.add_argument("--add-special-tokens", action="store_true",
                             help="Apply tokenizer's defined added specials; not an automatic chat template")

    text_command = actions.add_parser("text-lab", help="Explicitly execute and decode experimental PhiStream token IDs")
    text_command.add_argument("alias")
    text_command.add_argument("--tokenizer", required=True, help="Existing local tokenizer.json file")
    text_command.add_argument("--prompt", required=True)
    text_command.add_argument("--tokens", type=int, default=4)
    text_command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    text_command.add_argument("--capacity-mib", type=float, required=True)
    text_command.add_argument("--reserve-mib", type=float, default=1024.0)
    text_command.add_argument("--approve-local-execution", action="store_true")
    text_command.add_argument("--add-special-tokens", action="store_true")
    text_command.add_argument("--show-output", action="store_true", help="Explicitly print decoded text locally")
    text_command.add_argument("--output", required=True, help="New private metadata receipt; no raw text")
    text_command.add_argument("--timeout-seconds", type=int, default=900)

    parity_preflight = actions.add_parser(
        "parity-preflight",
        help="Validate prompt token IDs against a user-provided PhiStream reference trace, no inference",
    )
    parity_run = actions.add_parser(
        "parity-run",
        help="Governed local PhiStream reference parity (never promotes to chat)",
    )
    for option in (parity_preflight, parity_run):
        option.add_argument("alias")
        option.add_argument("--tokenizer", required=True, help="Trusted local tokenizer.json")
        option.add_argument("--prompt", required=True, help="Exact reference prompt text (visible in process arguments)")
        option.add_argument("--trace", required=True, help="Existing local PHISTREAM_REFERENCE_TRACE_V1 JSON")
        option.add_argument("--add-special-tokens", action="store_true")
    parity_run.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parity_run.add_argument("--capacity-mib", type=float, required=True)
    parity_run.add_argument("--reserve-mib", type=float, default=1024.0)
    parity_run.add_argument("--approve-local-execution", action="store_true")
    parity_run.add_argument("--timeout-seconds", type=int, default=1800)
    parity_run.add_argument("--output", required=True, help="New JSON summary only, refuse overwrite")

    serve = commands.add_parser("serve", help="Start supervised llama.cpp and the loopback HTTP API")
    serve.add_argument("alias")
    serve.add_argument("--llama-server", default=os.environ.get("PHISERVE_LLAMA_SERVER", "llama-server"))
    serve.add_argument("--port", type=int, default=11435)
    serve.add_argument("--backend-port", type=int, default=11436)
    serve.add_argument("--ctx-size", type=int, default=4096)
    serve.add_argument("--gpu-layers", type=int, default=0)
    serve.add_argument("--startup-timeout", type=float, default=120)
    auth = commands.add_parser("auth", help="Local browser-bridge token management")
    auth_commands = auth.add_subparsers(dest="action", required=True)
    create_key = auth_commands.add_parser("create", help="Write a new bearer token to a private file")
    create_key.add_argument("--output", required=True, help="New file; existing files are not overwritten")

    daemon = commands.add_parser("daemon", help="One shared local daemon; switch models after in-flight requests finish")
    daemon.add_argument("--llama-server", default=os.environ.get("PHISERVE_LLAMA_SERVER", "llama-server"))
    daemon.add_argument("--port", type=int, default=11435)
    daemon.add_argument("--backend-port", type=int, default=11436)
    daemon.add_argument("--ctx-size", type=int, default=4096)
    daemon.add_argument("--gpu-layers", type=int, default=0)
    daemon.add_argument("--startup-timeout", type=float, default=120)
    daemon.add_argument("--idle-seconds", type=float, default=300, help="Unload after idle interval (0 disables)")
    daemon.add_argument("--max-concurrent", type=int, default=4, help="Maximum simultaneous requests for loaded model")
    daemon.add_argument("--browser-origin", help="Exact trusted browser origin; opt-in CORS with token")
    daemon.add_argument("--api-key-file", help="Local token file; required with browser origin")
    validate = commands.add_parser("validate", help="Run an actual local model and write an observation receipt")
    validate.add_argument("alias")
    validate.add_argument("--llama-server", default=os.environ.get("PHISERVE_LLAMA_SERVER", "llama-server"))
    validate.add_argument("--output", required=True, help="New JSON file; existing files are never overwritten")
    validate.add_argument("--backend-port", type=int, default=11436)
    validate.add_argument("--ctx-size", type=int, default=2048)
    validate.add_argument("--gpu-layers", type=int, default=0)
    validate.add_argument("--startup-timeout", type=float, default=120)

    acceptance = commands.add_parser("acceptance", help="Local hardware readiness and conservative receipt review")
    acceptance_modes = acceptance.add_subparsers(dest="action", required=True)
    check = acceptance_modes.add_parser("doctor", help="Local non-inference environment check (no network)")
    check.add_argument("--llama-server", default=os.environ.get("PHISERVE_LLAMA_SERVER", "llama-server"))
    check.add_argument("--probe-nvidia", action="store_true",
                       help="Opt-in local nvidia-smi GPU metadata query; not CUDA proof")
    check.add_argument("--output", required=True, help="New JSON readiness report; never overwrite")
    review = acceptance_modes.add_parser("report", help="Review existing local GGUF/PhiStream evidence, no model execution")
    review.add_argument("--doctor-receipt", help="Existing acceptance doctor receipt")
    review.add_argument("--cpu-gguf", help="Existing phiserve validate receipt with 0 GPU layers")
    review.add_argument("--gpu-gguf", help="Existing phiserve validate receipt with >0 GPU layers")
    review.add_argument("--phistream-text", help="Existing PhiStream text-lab research receipt")
    review.add_argument("--phistream-parity", help="Existing PhiStream parity-run reference receipt")
    review.add_argument("--output", required=True, help="New minimal JSON report; never overwrite")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    registry = ModelRegistry()
    try:
        if args.command == "auth":
            path = create_token_file(args.output)
            print(f"Created private token file: {path} (token not printed)")
            return 0

        if args.command == "phistream":
            research_catalog = PhiStreamCatalog()
            if args.action == "inspect":
                print(json.dumps(probe_checkpoint(args.path), indent=2, sort_keys=True))
            elif args.action == "register":
                result = research_catalog.add(
                    args.alias, args.path, source=args.source,
                    revision=args.revision, license_id=args.license_id,
                )
                print(f"Registered PhiStream research alias {args.alias} (metadata-only, not chat-ready).")
                print(json.dumps(result, indent=2, sort_keys=True))
            elif args.action == "list":
                entries = research_catalog.list()
                if not entries:
                    print("No PhiStream research checkpoints registered.")
                for alias, metadata in entries.items():
                    print(f"{alias}\t{metadata['source']} @ {metadata['revision']}\tRESEARCH_ONLY")
            elif args.action == "remove":
                research_catalog.remove(args.alias)
                print(f"Removed research alias {args.alias}; checkpoint preserved.")
            elif args.action == "run-ids":
                result = run_ids(
                    research_catalog, args.alias, prompt=args.prompt,
                    tokens=args.tokens, device=args.device,
                    output=args.output, approved=args.approve_local_execution,
                    capacity_mib=args.capacity_mib, reserve_mib=args.reserve_mib,
                    timeout_seconds=args.timeout_seconds,
                )
                print(json.dumps(result, indent=2, sort_keys=True))
            elif args.action == "tokenizer-check":
                _, _, report = encode_prompt(
                    research_catalog, args.alias, args.tokenizer, args.prompt,
                    tokens=args.tokens, add_special_tokens=args.add_special_tokens,
                )
                print(json.dumps(report, indent=2, sort_keys=True))
            elif args.action == "text-lab":
                report, generated_text = text_lab(
                    research_catalog, args.alias, tokenizer_path=args.tokenizer,
                    prompt=args.prompt, tokens=args.tokens, device=args.device,
                    capacity_mib=args.capacity_mib, reserve_mib=args.reserve_mib,
                    output=args.output, approved=args.approve_local_execution,
                    add_special_tokens=args.add_special_tokens,
                    timeout_seconds=args.timeout_seconds,
                )
                print(json.dumps(report, indent=2, sort_keys=True))
                if args.show_output:
                    print("Local experimental decoded text (NOT validated chat):")
                    print(generated_text)
            elif args.action == "parity-preflight":
                report, _trace = preflight_reference(
                    research_catalog, args.alias, tokenizer_path=args.tokenizer,
                    prompt=args.prompt, trace_path=args.trace,
                    add_special_tokens=args.add_special_tokens,
                )
                print(json.dumps(report, indent=2, sort_keys=True))
                if not report["reference_eligible_for_run"]:
                    print("Reference too short for execution gate: 32 steps required.", file=sys.stderr)
                    return 2
            elif args.action == "parity-run":
                report = run_reference_gate(
                    research_catalog, args.alias, tokenizer_path=args.tokenizer,
                    prompt=args.prompt, trace_path=args.trace,
                    device=args.device, capacity_mib=args.capacity_mib,
                    reserve_mib=args.reserve_mib, output=args.output,
                    approved=args.approve_local_execution,
                    add_special_tokens=args.add_special_tokens,
                    timeout_seconds=args.timeout_seconds,
                )
                print(json.dumps(report, indent=2, sort_keys=True))
                if report["status"] == "REFERENCE_MISMATCH":
                    return 2
            return 0

        if args.command == "acceptance":
            if args.action == "doctor":
                result = doctor(
                    args.output, llama_server=args.llama_server,
                    probe_nvidia=args.probe_nvidia, registry=registry,
                )
            else:
                result = acceptance_report(
                    args.output, doctor_receipt=args.doctor_receipt,
                    cpu_gguf=args.cpu_gguf, gpu_gguf=args.gpu_gguf,
                    text_receipt=args.phistream_text,
                    parity_receipt=args.phistream_parity,
                )
            print(json.dumps(result, indent=2, sort_keys=True))
            print("Receipt is local operator evidence, NOT independently witnessed GPU or parity proof.")
            return 2 if result["status"] == "DIAGNOSTIC_MISMATCH_REQUIRES_REVIEW" else 0

        if args.command == "models":
            if args.action == "add":
                model = registry.add(args.alias, args.path)
                print(f"Registered {args.alias} -> {model}")
            elif args.action == "list":
                entries = registry.list_models()
                if not entries:
                    print("No models registered.")
                for alias, path in entries.items():
                    print(f"{alias}\t{path}")
            elif args.action == "remove":
                registry.remove(args.alias)
                print(f"Removed model alias {args.alias}; model file preserved.")
            elif args.action == "discover-ollama":
                models, warnings = discover(args.root)
                for entry in models:
                    print(f"{entry.name}\t{entry.size} bytes\talias={entry.alias}")
                print(f"Discovered {len(models)} supported local Ollama GGUF model(s).")
                for warning in warnings:
                    print(f"Skipped: {warning}", file=sys.stderr)
            elif args.action == "import-ollama":
                entry = import_model(args.model, alias=args.alias, root=args.root,
                                     verify_sha256=args.verify_sha256, registry=registry)
                print(f"Registered Ollama {entry.name} as {args.alias or entry.alias} (read-only, no weight copy).")
                if not args.verify_sha256:
                    print("Note: manifest checksum was not recalculated; use --verify-sha256 to verify all model bytes.")
            return 0

        if args.command == "validate":
            if not (1 <= args.backend_port <= 65535) or args.ctx_size <= 0 or args.gpu_layers < 0 or args.startup_timeout <= 0:
                raise RegistryError("Invalid validation backend port, context, layer count, or timeout")
            receipt = validate_local(args.alias, args.llama_server, args.output,
                                     backend_port=args.backend_port, ctx_size=args.ctx_size,
                                     gpu_layers=args.gpu_layers, startup_timeout=args.startup_timeout,
                                     registry=registry)
            print(f"Local generation observed; receipt saved: {args.output}")
            print(f"Result: {receipt['status']}. GPU execution is NOT independently verified.")
            return 0

        if args.command == "daemon":
            if (not (1 <= args.port <= 65535 and 1 <= args.backend_port <= 65535)
                    or args.port == args.backend_port or args.ctx_size <= 0
                    or args.gpu_layers < 0 or args.startup_timeout <= 0
                    or args.idle_seconds < 0 or args.max_concurrent < 1):
                raise RegistryError("Invalid daemon ports, context, layer count, timeout, idle interval, or concurrency")
            access = load_access(args.browser_origin, args.api_key_file)
            manager = ModelManager(registry=registry, executable=args.llama_server,
                                   backend_port=args.backend_port, ctx_size=args.ctx_size,
                                   gpu_layers=args.gpu_layers, startup_timeout=args.startup_timeout,
                                   idle_seconds=args.idle_seconds, max_concurrent=args.max_concurrent)
            server = make_server("", args.backend_port, args.port, manager=manager, browser_access=access)
            server.timeout = 1
            try:
                print(f"PhiServe shared manager listening on http://127.0.0.1:{args.port}; no model loaded yet.", flush=True)
                while True:
                    server.handle_request()
                    manager.unload_idle()
            except KeyboardInterrupt:
                print("\\nStopping shared PhiServe.", flush=True)
            finally:
                server.server_close()
                manager.close(grace_seconds=30)
            return 0

        if not (1 <= args.port <= 65535 and 1 <= args.backend_port <= 65535):
            raise RegistryError("Ports must be in the range 1-65535")
        if args.port == args.backend_port:
            raise RegistryError("Public and backend ports must be different")
        if args.ctx_size <= 0 or args.gpu_layers < 0 or args.startup_timeout <= 0:
            raise RegistryError("Context, GPU layers, and startup timeout must be valid positive/nonnegative values")
        model_path = registry.get(args.alias)
        backend = LlamaBackend(
            executable=args.llama_server,
            model_path=model_path,
            alias=args.alias,
            port=args.backend_port,
            ctx_size=args.ctx_size,
            gpu_layers=args.gpu_layers,
            startup_timeout=args.startup_timeout,
        )
        with backend:
            server = make_server(args.alias, args.backend_port, args.port, model_path=model_path)
            try:
                print(f"PhiServe {__version__}: {args.alias} on http://127.0.0.1:{args.port}", flush=True)
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nStopping PhiServe.", flush=True)
            finally:
                server.server_close()
        return 0
    except (RegistryError, OllamaStoreError, PhiStreamLabError, AcceptanceError, RuntimeError, OSError, ValueError) as exc:
        print(f"PhiServe error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
