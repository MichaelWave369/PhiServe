"""Opt-in queue tests use simulated local inference only, not GGUF/GPU weights."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from phiserve.cli import parser
from phiserve.manager import ModelManager, ModelManagerError
from phiserve.registry import ModelRegistry
from phiserve.server import make_server


class Process:
    def __init__(self):
        self.alive = True

    def poll(self):
        return None if self.alive else 1


class Backend:
    instances = []
    fail_alias = None

    def __init__(self, **kwargs):
        self.alias = kwargs["alias"]
        self.process = Process()
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        if self.alias == self.fail_alias:
            raise RuntimeError("Synthetic startup failure")

    def stop(self):
        self.stopped = True
        self.process.alive = False


class LocalUpstream(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        data = json.dumps({"choices": [{"message": {"content": "synthetic-" + request["model"]}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class AdmissionQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.registry = ModelRegistry(Path(self.temp.name) / "registry")
        for alias in ("alpha", "beta", "gamma"):
            model = Path(self.temp.name) / (alias + ".gguf")
            model.write_bytes(b"GGUF" + alias.encode())
            self.registry.add(alias, model)
        Backend.instances = []
        Backend.fail_alias = None
        self.manager = self.new_manager()
        self.addCleanup(lambda: self.manager.close(grace_seconds=0))

    def new_manager(self, capacity=2, timeout=2.0, concurrent=1):
        return ModelManager(
            self.registry, "fake", 11111, backend_factory=Backend,
            queue_capacity=capacity, queue_timeout_seconds=timeout,
            max_concurrent=concurrent, idle_seconds=0.1)

    def wait_count(self, amount, manager=None):
        manager = manager or self.manager
        end = time.monotonic() + 2
        while time.monotonic() < end:
            if manager.status()["queued_requests"] == amount:
                return
            time.sleep(0.005)
        self.fail(f"Expected {amount} queued, observed {manager.status()}")

    def run_worker(self, alias, entered, release, outcomes):
        def work():
            try:
                with self.manager.acquire_queued(alias):
                    outcomes.append(("admitted", alias))
                    entered.set()
                    release.wait(3)
            except ModelManagerError as exc:
                outcomes.append(("error", alias, exc.status))
                entered.set()
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self.addCleanup(lambda: thread.join(3))
        return thread

    def test_default_queue_disabled_without_changing_legacy(self):
        disabled = self.new_manager(capacity=0)
        self.addCleanup(lambda: disabled.close(grace_seconds=0))
        with self.assertRaises(ModelManagerError) as caught:
            disabled.acquire_queued("alpha")
        self.assertEqual(caught.exception.status, 503)
        with disabled.acquire("alpha"):
            self.assertEqual(disabled.status()["inflight_requests"], 1)

    def test_fifo_across_model_switches_and_no_active_eviction(self):
        first_active = self.manager.acquire("alpha")
        original = Backend.instances[-1]
        outcomes = []
        entered_b, release_b = threading.Event(), threading.Event()
        entered_a, release_a = threading.Event(), threading.Event()
        thread_b = self.run_worker("beta", entered_b, release_b, outcomes)
        self.wait_count(1)
        thread_a = self.run_worker("alpha", entered_a, release_a, outcomes)
        self.wait_count(2)
        self.assertFalse(original.stopped)
        self.assertFalse(self.manager.unload_idle())
        with self.assertRaises(ModelManagerError) as result:
            self.manager.acquire("alpha")
        self.assertEqual(result.exception.status, 429)
        first_active.release()
        try:
            self.assertTrue(entered_b.wait(3))
            self.assertEqual(outcomes[0], ("admitted", "beta"))
            self.assertFalse(entered_a.is_set())
            self.assertTrue(original.stopped)
            self.assertEqual(self.manager.status()["active_model"], "beta")
        finally:
            release_b.set()
        self.assertTrue(entered_a.wait(3))
        release_a.set()
        thread_b.join(3)
        thread_a.join(3)
        self.assertEqual(outcomes[:2], [("admitted", "beta"), ("admitted", "alpha")])
        self.assertEqual(self.manager.status()["queued_requests"], 0)

    def test_queue_capacity_refusal_while_active_model_safe(self):
        manager = self.new_manager(capacity=1)
        self.addCleanup(lambda: manager.close(grace_seconds=0))
        first = manager.acquire("alpha")
        entered, release = threading.Event(), threading.Event()
        results = []
        def worker():
            try:
                with manager.acquire_queued("beta"):
                    results.append("admitted")
                    entered.set()
                    release.wait(2)
            except ModelManagerError as err:
                results.append(err.status)
                entered.set()
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        self.wait_count(1, manager)
        with self.assertRaises(ModelManagerError) as exc:
            manager.acquire_queued("gamma")
        self.assertEqual(exc.exception.status, 429)
        self.assertFalse(Backend.instances[0].stopped)
        first.release()
        self.assertTrue(entered.wait(3))
        release.set()
        worker_thread.join(3)
        self.assertEqual(results, ["admitted"])

    def test_timeout_clears_ticket_and_preserves_running_model(self):
        manager = self.new_manager(timeout=0.15)
        self.addCleanup(lambda: manager.close(grace_seconds=0))
        first = manager.acquire("alpha")
        outcomes = []
        def wait():
            try:
                with manager.acquire_queued("beta"):
                    outcomes.append(200)
            except ModelManagerError as err:
                outcomes.append(err.status)
        thread = threading.Thread(target=wait, daemon=True)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcomes, [504])
        self.assertEqual(manager.status()["queued_requests"], 0)
        self.assertFalse(Backend.instances[0].stopped)
        first.release()
        with manager.acquire("alpha"):
            pass

    def test_shutdown_wakes_all_waiters_without_starting_backend(self):
        first = self.manager.acquire("alpha")
        outcomes = []
        entered = threading.Event()
        def pending():
            try:
                self.manager.acquire_queued("beta")
            except ModelManagerError as err:
                outcomes.append(err.status)
                entered.set()
        thread = threading.Thread(target=pending, daemon=True)
        thread.start()
        self.wait_count(1)
        self.manager.close(grace_seconds=0)
        self.assertTrue(entered.wait(3))
        thread.join(3)
        first.release()
        self.assertEqual(outcomes, [503])
        self.assertEqual(self.manager.status()["queued_requests"], 0)
        self.assertEqual(len(Backend.instances), 1)

    def test_unknown_model_fails_before_queue_and_never_downloads(self):
        with self.assertRaises(ModelManagerError) as caught:
            self.manager.acquire_queued("nonexistent")
        self.assertEqual(caught.exception.status, 404)
        self.assertEqual(self.manager.status()["queued_requests"], 0)
        self.assertEqual(len(Backend.instances), 0)

    def test_same_model_waiter_admitted_when_concurrency_slot_frees(self):
        first = self.manager.acquire("alpha")
        outcomes = []
        entered, release = threading.Event(), threading.Event()
        thread = self.run_worker("alpha", entered, release, outcomes)
        self.wait_count(1)
        self.assertFalse(entered.is_set())
        first.release()
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(outcomes[0], ("admitted", "alpha"))
            self.assertEqual(len(Backend.instances), 1)
        finally:
            release.set()
        thread.join(3)

    def test_failed_model_start_is_not_reported_as_success(self):
        first = self.manager.acquire("alpha")
        Backend.fail_alias = "beta"
        outcomes = []
        entered, release = threading.Event(), threading.Event()
        thread = self.run_worker("beta", entered, release, outcomes)
        self.wait_count(1)
        first.release()
        self.assertTrue(entered.wait(3))
        release.set()
        thread.join(3)
        self.assertEqual(outcomes, [("error", "beta", 503)])
        self.assertFalse(self.manager.status()["transitioning"])
        self.assertIsNone(self.manager.status()["active_model"])
        Backend.fail_alias = None
        with self.manager.acquire("alpha"):
            self.assertTrue(self.manager.status()["backend_alive"])

    def test_invalid_queue_config_rejected(self):
        for capacity, timeout in ((-1, 2), (33, 2), (True, 2), (2, -1),
                                  (2, float("nan")), (2, 121)):
            with self.assertRaises(ValueError):
                self.new_manager(capacity, timeout)

    def test_daemon_flags_are_explicit_and_opt_in(self):
        a = parser().parse_args(["daemon"])
        self.assertEqual(a.queue_capacity, 0)
        b = parser().parse_args(["daemon", "--queue-capacity", "3",
                                  "--queue-timeout-seconds", "4.5"])
        self.assertEqual((b.queue_capacity, b.queue_timeout_seconds), (3, 4.5))


class AdmissionHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        reg = ModelRegistry(Path(self.temp.name) / "registry")
        for alias in ("alpha", "beta"):
            path = Path(self.temp.name) / (alias + ".gguf")
            path.write_bytes(b"GGUF" + alias.encode())
            reg.add(alias, path)
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), LocalUpstream)
        self.manager = ModelManager(reg, "fake", self.upstream.server_address[1],
                                    queue_capacity=2, queue_timeout_seconds=2,
                                    backend_factory=Backend, max_concurrent=1,
                                    idle_seconds=0)
        self.http = make_server("", self.upstream.server_address[1], 0, manager=self.manager)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"
        self.threads = [threading.Thread(target=s.serve_forever, daemon=True)
                        for s in (self.upstream, self.http)]
        for thread in self.threads:
            thread.start()
        self.addCleanup(self.stop_servers)

    def stop_servers(self):
        for server in (self.http, self.upstream):
            server.shutdown()
            server.server_close()
        for t in self.threads:
            t.join(3)
        self.manager.close(grace_seconds=0)

    def post(self, path, model="alpha"):
        req = urllib.request.Request(self.base + path,
                                     data=json.dumps({"model":model, "messages":[
                                         {"role":"user", "content":"synthetic test"}],
                                         "stream":False}).encode(),
                                     headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.load(resp)

    def test_explicit_queued_route_rewrites_to_backend_and_reports_counts(self):
        with self.manager.acquire("alpha"):
            completed = []
            def queued():
                completed.append(self.post("/v1/queued/chat/completions", "beta"))
            thread = threading.Thread(target=queued, daemon=True)
            thread.start()
            deadline = time.monotonic() + 2
            while self.manager.status()["queued_requests"] == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(self.manager.status()["queued_requests"], 1)
            with urllib.request.urlopen(self.base + "/health") as r:
                status = json.load(r)
            self.assertEqual(status["queued_requests"], 1)
            self.assertEqual(status["queue_capacity"], 2)
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(completed[0][0], 200)
        self.assertEqual(completed[0][1]["choices"][0]["message"]["content"], "synthetic-beta")

    def test_queued_route_disabled_on_legacy_mode(self):
        self.manager.queue_capacity = 0
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/v1/queued/chat/completions")
        self.assertEqual(caught.exception.code, 503)
        self.assertEqual(self.post("/v1/chat/completions")[0], 200)


if __name__ == "__main__":
    unittest.main()
