"""Manager concurrency is exercised with fake subprocesses, not GPU weights."""

import tempfile
import threading
import unittest
from pathlib import Path

from phiserve.manager import ModelManager, ModelManagerError
from phiserve.registry import ModelRegistry


class FakeProcess:
    def __init__(self):
        self.alive = True

    def poll(self):
        return None if self.alive else 1


class FakeBackend:
    instances = []
    failure = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.process = FakeProcess()
        self.started = False
        self.stopped = False
        FakeBackend.instances.append(self)

    def start(self):
        if self.failure:
            raise RuntimeError("Synthetic startup failure")
        self.started = True

    def stop(self):
        self.process.alive = False
        self.stopped = True


class ManagerTests(unittest.TestCase):
    def setUp(self):
        FakeBackend.instances = []
        FakeBackend.failure = False
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.reg = ModelRegistry(Path(self.tmp.name) / "registry")
        for alias in ("alpha", "beta"):
            model = Path(self.tmp.name) / (alias + ".gguf")
            model.write_bytes(b"GGUF" + alias.encode())
            self.reg.add(alias, model)
        self.now = [100.0]
        self.manager = ModelManager(self.reg, "fake-server", 19444,
                                    backend_factory=FakeBackend, clock=lambda: self.now[0],
                                    idle_seconds=20, max_concurrent=2)
        self.addCleanup(lambda: self.manager.close(grace_seconds=0))

    def test_lazy_load_then_reuse_and_switch(self):
        self.assertIsNone(self.manager.status()["active_model"])
        with self.manager.acquire("alpha"):
            with self.manager.acquire("alpha"):
                self.assertEqual(self.manager.status()["inflight_requests"], 2)
                self.assertEqual(len(FakeBackend.instances), 1)
        self.assertEqual(self.manager.status()["inflight_requests"], 0)
        first = FakeBackend.instances[0]
        with self.manager.acquire("beta"):
            self.assertTrue(first.stopped)
            self.assertEqual(self.manager.status()["active_model"], "beta")
        self.assertEqual(len(FakeBackend.instances), 2)

    def test_lease_blocks_model_switch(self):
        with self.manager.acquire("alpha"):
            first = FakeBackend.instances[0]
            with self.assertRaises(ModelManagerError) as err:
                self.manager.acquire("beta")
            self.assertEqual(err.exception.status, 409)
            self.assertFalse(first.stopped)
            self.assertEqual(self.manager.status()["active_model"], "alpha")
        with self.manager.acquire("beta"):
            pass
        self.assertTrue(first.stopped)

    def test_request_limit_rejects_without_killing_backend(self):
        with self.manager.acquire("alpha"), self.manager.acquire("alpha"):
            with self.assertRaises(ModelManagerError) as err:
                self.manager.acquire("alpha")
            self.assertEqual(err.exception.status, 429)
            self.assertFalse(FakeBackend.instances[0].stopped)

    def test_idle_eviction_only_after_last_release(self):
        lease = self.manager.acquire("alpha")
        self.now[0] = 10000
        self.assertFalse(self.manager.unload_idle())
        lease.release()
        self.assertFalse(self.manager.unload_idle())
        self.now[0] += 21
        self.assertTrue(self.manager.unload_idle())
        self.assertTrue(FakeBackend.instances[0].stopped)
        self.assertIsNone(self.manager.status()["active_model"])

    def test_failure_cannot_publish_fake_healthy_backend(self):
        FakeBackend.failure = True
        with self.assertRaises(ModelManagerError) as err:
            self.manager.acquire("alpha")
        self.assertEqual(err.exception.status, 503)
        self.assertIsNone(self.manager.status()["active_model"])
        self.assertFalse(self.manager.status()["transitioning"])
        FakeBackend.failure = False
        with self.manager.acquire("alpha"):
            self.assertTrue(self.manager.status()["backend_alive"])

    def test_unknown_model_is_not_started_or_downloaded(self):
        with self.assertRaises(ModelManagerError) as err:
            self.manager.acquire("unknown")
        self.assertEqual(err.exception.status, 404)
        self.assertEqual(FakeBackend.instances, [])

    def test_explicit_close_stops_and_refuses_new_requests(self):
        with self.manager.acquire("alpha"):
            pass
        self.manager.close()
        self.assertTrue(FakeBackend.instances[0].stopped)
        with self.assertRaises(ModelManagerError) as err:
            self.manager.acquire("beta")
        self.assertEqual(err.exception.status, 503)

    def test_parallel_request_cannot_interrupt_active_lease(self):
        in_flight = threading.Event()
        release = threading.Event()
        def worker():
            with self.manager.acquire("alpha"):
                in_flight.set()
                release.wait(timeout=3)
        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(in_flight.wait(timeout=3))
            with self.assertRaises(ModelManagerError) as err:
                self.manager.acquire("beta")
            self.assertEqual(err.exception.status, 409)
            self.assertFalse(FakeBackend.instances[0].stopped)
        finally:
            release.set()
            thread.join(timeout=3)
        with self.manager.acquire("beta"):
            self.assertEqual(self.manager.status()["active_model"], "beta")


if __name__ == "__main__":
    unittest.main()
