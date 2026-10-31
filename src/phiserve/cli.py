"""Small CLI for a genuinely local, single-model server."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .backend import LlamaBackend
from .registry import ModelRegistry, RegistryError
from .server import make_server


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

    serve = commands.add_parser("serve", help="Start supervised llama.cpp and the loopback HTTP API")
    serve.add_argument("alias")
    serve.add_argument("--llama-server", default=os.environ.get("PHISERVE_LLAMA_SERVER", "llama-server"))
    serve.add_argument("--port", type=int, default=11435)
    serve.add_argument("--backend-port", type=int, default=11436)
    serve.add_argument("--ctx-size", type=int, default=4096)
    serve.add_argument("--gpu-layers", type=int, default=0)
    serve.add_argument("--startup-timeout", type=float, default=120)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    registry = ModelRegistry()
    try:
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
    except (RegistryError, RuntimeError, OSError) as exc:
        print(f"PhiServe error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
