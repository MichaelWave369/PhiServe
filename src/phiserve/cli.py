"""Small CLI for a genuinely local, single-model server."""

from __future__ import annotations

import argparse
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
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    registry = ModelRegistry()
    try:
        if args.command == "auth":
            path = create_token_file(args.output)
            print(f"Created private token file: {path} (token not printed)")
            return 0

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
    except (RegistryError, OllamaStoreError, RuntimeError, OSError, ValueError) as exc:
        print(f"PhiServe error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
