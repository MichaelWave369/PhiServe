"""Supervise one real local llama.cpp server process."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class LlamaBackend:
    def __init__(
        self,
        executable: str,
        model_path: Path,
        alias: str,
        port: int,
        ctx_size: int = 4096,
        gpu_layers: int = 0,
        startup_timeout: float = 120,
    ):
        self.executable = executable
        self.model_path = model_path
        self.alias = alias
        self.port = port
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        executable = shutil.which(self.executable)
        if executable is None:
            raise RuntimeError(f"llama-server executable not found: {self.executable}")
        # Refuse to attach silently to an unrelated process already occupying the port.
        with socket.socket() as probe:
            probe.settimeout(0.3)
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"Backend port {self.port} is already in use")
        args = [
            executable,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--model", str(self.model_path),
            "--alias", self.alias,
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.gpu_layers),
        ]
        self.process = subprocess.Popen(args, shell=False)
        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with code {self.process.returncode}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/health", timeout=1
                    ) as response:
                        if response.status == 200:
                            payload = json.load(response)
                            if payload.get("status") in {"ok", "ready"}:
                                return
                except (OSError, ValueError, urllib.error.HTTPError):
                    pass
                time.sleep(0.25)
            raise RuntimeError("llama-server did not become healthy before startup timeout")
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

    def __enter__(self) -> "LlamaBackend":
        self.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.stop()
