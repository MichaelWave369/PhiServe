"""Single-resident, loopback-only model manager.

One llama.cpp subprocess at a time. Requests take a lease on the current
backend; a model switch or idle eviction cannot terminate a leased backend.
There is intentionally no unauthenticated remote management or GPU claim.
"""

from __future__ import annotations

from collections import deque
import threading
import time
from pathlib import Path
from typing import Callable

from .backend import LlamaBackend
from .registry import ModelRegistry, RegistryError


class ModelManagerError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class ModelLease:
    """One in-flight request, valid until its entire HTTP response is finished."""

    def __init__(self, manager: "ModelManager", alias: str):
        self.manager = manager
        self.alias = alias
        self._released = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.release()

    def release(self):
        if not self._released:
            self._released = True
            self.manager._release()


class ModelManager:
    def __init__(
        self,
        registry: ModelRegistry,
        executable: str,
        backend_port: int,
        *,
        ctx_size: int = 4096,
        gpu_layers: int = 0,
        startup_timeout: float = 120,
        idle_seconds: float = 300,
        max_concurrent: int = 4,
        queue_capacity: int = 0,
        queue_timeout_seconds: float = 30,
        backend_factory: Callable = LlamaBackend,
        clock: Callable[[], float] = time.monotonic,
    ):
        if idle_seconds < 0 or max_concurrent < 1:
            raise ValueError("idle_seconds must be nonnegative and max_concurrent positive")
        if (type(queue_capacity) is not int or not 0 <= queue_capacity <= 32
                or type(queue_timeout_seconds) not in (int, float)
                or not 0.1 <= queue_timeout_seconds <= 120):
            raise ValueError("Queue capacity must be 0..32 and timeout must be 0.1..120 seconds")
        self.registry = registry
        self.executable = executable
        self.backend_port = backend_port
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.startup_timeout = startup_timeout
        self.idle_seconds = idle_seconds
        self.max_concurrent = max_concurrent
        self.queue_capacity = queue_capacity
        self.queue_timeout_seconds = float(queue_timeout_seconds)
        self._waiters = deque()
        self._factory = backend_factory
        self._clock = clock
        self._condition = threading.Condition()
        self._backend = None
        self._alias = None
        self._model_path = None
        self._inflight = 0
        self._changing = False
        self._closed = False
        self._last_idle = self._clock()

    def catalog(self) -> dict[str, str]:
        return self.registry.list_models()

    def _alive(self) -> bool:
        if self._backend is None:
            return False
        process = getattr(self._backend, "process", None)
        return process is not None and process.poll() is None

    def status(self) -> dict:
        with self._condition:
            return {
                "active_model": self._alias,
                "model_path": str(self._model_path) if self._model_path is not None else None,
                "inflight_requests": self._inflight,
                "queued_requests": len(self._waiters),
                "queue_capacity": self.queue_capacity,
                "transitioning": self._changing,
                "backend_alive": self._alive(),
                "closed": self._closed,
            }

    def _resolve(self, alias: str) -> Path:
        try:
            return self.registry.get(alias)
        except RegistryError as exc:
            # An API-selected model cannot be used as a filesystem path.
            raise ModelManagerError(404, "Model unavailable") from exc

    def _activate(self, alias: str, model_path: Path, previous) -> ModelLease:
        """Start a model after an atomic transition reservation under condition.

        The reservation is installed before releasing the condition lock, so
        neither queued nor legacy calls can start a competing backend.
        """
        replacement = None
        try:
            if previous is not None:
                previous.stop()
            replacement = self._factory(
                executable=self.executable, model_path=model_path, alias=alias,
                port=self.backend_port, ctx_size=self.ctx_size,
                gpu_layers=self.gpu_layers, startup_timeout=self.startup_timeout,
            )
            replacement.start()
        except Exception as exc:
            if replacement is not None:
                try:
                    replacement.stop()
                except Exception:
                    pass
            with self._condition:
                self._changing = False
                self._condition.notify_all()
            raise ModelManagerError(503, "Local model could not start; inspect PhiServe logs") from exc

        with self._condition:
            if not self._closed:
                self._backend = replacement
                self._alias = alias
                self._model_path = model_path
                self._inflight = 1
                self._changing = False
                self._condition.notify_all()
                return ModelLease(self, alias)
            self._changing = False
            self._condition.notify_all()
        replacement.stop()
        raise ModelManagerError(503, "PhiServe is stopping")

    def _reserve_transition(self):
        """Caller holds condition lock and has already established no leases."""
        previous = self._backend
        self._changing = True
        self._backend = None
        self._alias = None
        self._model_path = None
        return previous

    def acquire(self, alias: str) -> ModelLease:
        """Legacy nonblocking admission. The default wire behavior is unchanged."""
        model_path = self._resolve(alias)
        with self._condition:
            if self._closed:
                raise ModelManagerError(503, "PhiServe is stopping")
            # A queued client must not be starved by a storm of legacy requests.
            if self._waiters:
                raise ModelManagerError(429, "Queued requests take precedence; retry")
            if self._changing:
                raise ModelManagerError(503, "Model transition in progress; retry")
            if self._alias == alias and self._alive():
                if self._inflight >= self.max_concurrent:
                    raise ModelManagerError(429, "Active model request limit reached")
                self._inflight += 1
                return ModelLease(self, alias)
            if self._inflight:
                raise ModelManagerError(409, "Different model is busy; retry after its requests finish")
            previous = self._reserve_transition()
        return self._activate(alias, model_path, previous)

    def acquire_queued(self, alias: str) -> ModelLease:
        """Bounded FIFO opt-in admission. Never cancel or evict a live lease.

        The only queued state is a request's anonymous position, not its prompt,
        response, model path, or identity. A timed-out waiter is removed.
        """
        model_path = self._resolve(alias)
        deadline = time.monotonic() + self.queue_timeout_seconds
        with self._condition:
            if self.queue_capacity == 0:
                raise ModelManagerError(503, "Opt-in inference queue disabled")
            if self._closed:
                raise ModelManagerError(503, "PhiServe is stopping")
            if len(self._waiters) >= self.queue_capacity:
                raise ModelManagerError(429, "Inference queue full")
            ticket = object()
            self._waiters.append(ticket)
            self._condition.notify_all()
            try:
                while True:
                    if self._closed:
                        raise ModelManagerError(503, "PhiServe is stopping")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ModelManagerError(504, "Inference queue wait timed out")
                    if self._waiters[0] is ticket and not self._changing:
                        if self._alias == alias and self._alive():
                            if self._inflight < self.max_concurrent:
                                self._inflight += 1
                                self._waiters.popleft()
                                self._condition.notify_all()
                                return ModelLease(self, alias)
                        elif self._inflight == 0:
                            previous = self._reserve_transition()
                            self._waiters.popleft()
                            self._condition.notify_all()
                            break
                    self._condition.wait(timeout=remaining)
            finally:
                # This is a no-op for a ticket already admitted.
                if ticket in self._waiters:
                    self._waiters.remove(ticket)
                    self._condition.notify_all()
        return self._activate(alias, model_path, previous)

    def _release(self) -> None:
        with self._condition:
            if self._inflight < 1:
                raise RuntimeError("Model lease underflow")
            self._inflight -= 1
            if self._inflight == 0:
                self._last_idle = self._clock()
            self._condition.notify_all()

    def unload_idle(self) -> bool:
        """Evict only after a completed idle interval, never mid-generation."""
        if self.idle_seconds == 0:
            return False
        with self._condition:
            if (self._changing or self._closed or self._backend is None
                    or self._inflight or self._waiters
                    or self._clock() - self._last_idle < self.idle_seconds):
                return False
            backend = self._backend
            self._changing = True
            self._backend = None
            self._alias = None
            self._model_path = None
        try:
            backend.stop()
        finally:
            with self._condition:
                self._changing = False
                self._condition.notify_all()
        return True

    def close(self, grace_seconds: float = 30) -> None:
        """Gracefully drain HTTP leases, then stop the child process.

        Explicit shutdown may interrupt a generation after the grace deadline.
        A load already in progress is allowed to finish its startup and is
        immediately stopped by acquire() when it observes the closed flag.
        """
        deadline = time.monotonic() + max(0, grace_seconds)
        with self._condition:
            self._closed = True
            self._condition.notify_all()  # Wake queued waiters to fail closed immediately.
            while (self._changing or self._inflight) and time.monotonic() < deadline:
                self._condition.wait(timeout=min(0.25, max(0, deadline - time.monotonic())))
            backend = self._backend
            self._backend = None
            self._alias = None
            self._model_path = None
        if backend is not None:
            backend.stop()
