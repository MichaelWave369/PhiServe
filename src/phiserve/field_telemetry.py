"""Opt-in process-attributed nvidia-smi sampling around a local inference run.

These are *driver self-reported* process-memory observations. Windows WDDM
can report N/A for compute-process memory. PID/memory observation cannot prove
that a CUDA kernel performed model inference, or independently witness VRAM.
"""

from __future__ import annotations

import csv
import io
import subprocess
import threading
from typing import Callable


class ProcessMemorySampler:
    """One short-lived sampler; attach only to a PhiServe-launched child PID."""

    def __init__(self, *, interval: float = 0.75,
                 runner: Callable | None = None, max_samples: int = 4096):
        if not 0.1 <= interval <= 10 or not 1 <= max_samples <= 10000:
            raise ValueError("Invalid GPU telemetry interval or sample limit")
        self.interval = interval
        self.max_samples = max_samples
        self._runner = subprocess.run if runner is None else runner
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pid: int | None = None
        self._attempted = 0
        self._process_seen = 0
        self._memory_samples = 0
        self._peak_memory_mib: int | None = None
        self._unavailable = 0
        self._failed_queries = 0
        self._unsupported_memory = 0
        self._stopped = False

    def start(self, pid: int) -> None:
        if type(pid) is not int or pid <= 0:
            raise ValueError("Cannot attach GPU sampler without a positive child PID")
        if self._thread is not None or self._pid is not None:
            raise RuntimeError("GPU sampler already started")
        self._pid = pid
        self._thread = threading.Thread(target=self._poll, name="phiserve-nvidia-smi",
                                        daemon=True)
        self._thread.start()

    def _poll(self):
        for _ in range(self.max_samples):
            if self._stop.is_set():
                break
            self.sample_once()
            if self._stop.wait(self.interval):
                break

    def sample_once(self) -> None:
        """One bounded local query. Never trusts the tool to attest execution."""
        if self._pid is None:
            raise RuntimeError("GPU sampler needs a selected process PID")
        with self._lock:
            self._attempted += 1
        args = ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits"]
        try:
            result = self._runner(args, shell=False, capture_output=True,
                                  text=True, timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            with self._lock:
                self._unavailable += 1
            return
        if result.returncode != 0:
            with self._lock:
                self._failed_queries += 1
            return

        present = False
        values = []
        unsupported = False
        try:
            for fields in csv.reader(io.StringIO(result.stdout)):
                if not fields or all(not item.strip() for item in fields):
                    continue
                if len(fields) != 2:
                    raise ValueError("Malformed nvidia-smi CSV")
                pid_s, memory_s = (item.strip() for item in fields)
                if pid_s.startswith("[") or not pid_s.isdecimal():
                    raise ValueError("Unsupported PID field")
                if int(pid_s) != self._pid:
                    continue
                present = True
                if memory_s.isdecimal():
                    values.append(int(memory_s))
                elif memory_s.lower() in {"n/a", "[not supported]", "not supported"}:
                    unsupported = True
                else:
                    raise ValueError("Unexpected memory field")
        except (ValueError, csv.Error, AttributeError):
            with self._lock:
                self._failed_queries += 1
            return

        with self._lock:
            if present:
                self._process_seen += 1
                if values and all(value >= 0 for value in values) and not unsupported:
                    # Sum matching per-device rows in one query, without claiming
                    # which GPU performed any work. This is a driver observation.
                    memory_mib = sum(values)
                    self._memory_samples += 1
                    self._peak_memory_mib = max(
                        self._peak_memory_mib or 0, memory_mib,
                    )
                else:
                    self._unsupported_memory += 1

    def stop(self) -> dict:
        """Stop before the supervised backend terminates; return limited metadata."""
        if self._pid is None:
            raise RuntimeError("GPU sampler never started")
        if not self._stopped:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._stopped = True
        with self._lock:
            hanging = self._thread is not None and self._thread.is_alive()
            if hanging:
                status = "SAMPLER_DID_NOT_STOP"
            elif self._memory_samples:
                status = "PROCESS_MEMORY_DRIVER_SELF_REPORT"
            elif self._process_seen:
                status = "PROCESS_PRESENT_MEMORY_UNAVAILABLE"
            elif self._unavailable and self._unavailable == self._attempted:
                status = "NVIDIA_SMI_UNAVAILABLE"
            elif self._failed_queries and self._failed_queries == self._attempted:
                status = "NVIDIA_SMI_QUERY_FAILED"
            else:
                status = "NO_MATCHING_COMPUTE_PROCESS_OBSERVED"
            return {
                "schema": "PHISERVE_NVIDIA_PROCESS_SAMPLE_V1",
                "status": status,
                "subprocess_pid": self._pid,
                "sample_attempts": self._attempted,
                "matching_pid_samples": self._process_seen,
                "numeric_memory_samples": self._memory_samples,
                "peak_driver_reported_process_memory_mib": self._peak_memory_mib,
                "sampling_interval_seconds": self.interval,
                "nvidia_smi_process_memory_may_be_unavailable_on_windows_wddm": True,
                "cuda_kernel_execution_independently_verified": False,
                "gpu_model_weights_residency_verified": False,
                "physical_hardware_witnessed": False,
            }
