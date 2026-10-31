"""Rung 8 research gate: synthetic tokenizer, user trace and fake PhiStream runner.

These tests do not load actual model weights, execute CUDA, or authenticate a
reference implementation. PhiStream's own comparator has separate tests.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from phiserve.cli import parser
from phiserve.phistream_lab import PhiStreamCatalog, PhiStreamLabError
from phiserve.phistream_parity import preflight_reference, run_reference_gate


REVIEW = {
    "schema": "PHISERVE_PHISTREAM_PREFLIGHT_V1",
    "status": "METADATA_PREFLIGHT_ONLY",
    "format": "SAFETENSORS",
    "vocab_size": 10,
    "max_context": 512,
    "layers": 2,
    "tensor_count": 21,
    "weight_bytes": 256,
    "shard_count": 1,
    "service_ready": False,
    "can_serve_chat": False,
    "tokenizer_available": False,
}


class ReferenceGateTests(unittest.TestCase):
    def setUp(self):
        self.working = tempfile.TemporaryDirectory()
        self.addCleanup(self.working.cleanup)
        self.root = Path(self.working.name)
        self.weights = self.root / "weights"
        self.weights.mkdir()
        (self.weights / "config.json").write_text('{"model_type":"llama"}')
        self.catalog = PhiStreamCatalog(self.root / "catalog")
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            self.catalog.add(
                "test-llama", self.weights,
                source="user-declared-test", revision="rev1", license_id="test-license",
            )
        tok = Tokenizer(WordLevel(
            vocab={"[UNK]": 0, "hello": 1, "world": 2, "answer": 3, "moon": 4},
            unk_token="[UNK]",
        ))
        tok.pre_tokenizer = Whitespace()
        self.tokenizer = self.root / "tokenizer.json"
        tok.save(str(self.tokenizer))
        self.trace = self.root / "reference.json"
        self.digest = "a" * 64
        self.data = self._trace(32)
        self._save()
        self.output = self.root / "summary.json"

    def _trace(self, steps: int):
        return {
            "schema": "PHISTREAM_REFERENCE_TRACE_V1",
            "reference": {
                "runtime": "Operator-declared separate runtime",
                "version": "user-claimed",
                "execution_device": "cpu",
                "asserted_independent": True,
            },
            "checkpoint_fingerprint": {
                "algorithm": "sha256",
                "files": {"config.json": self.digest, "model.safetensors": "b" * 64},
            },
            "prompt_token_ids": [1, 2],
            "expected_generated_token_ids": [3] * steps,
            "logit_probes": [{"3": 1.0} for _ in range(steps)],
        }

    def _save(self):
        self.trace.write_text(json.dumps(self.data), encoding="utf-8")

    @contextmanager
    def mocked_reference(self):
        fake_pkg = types.ModuleType("phistream")
        fake_pkg.__path__ = []
        safe = types.ModuleType("phistream.safeweights")
        reference = types.ModuleType("phistream.reference")
        safe.LocalCheckpoint = lambda path: SimpleNamespace(location=path)

        def read(file):
            return json.loads(Path(file).read_text(encoding="utf-8"))

        def validate(data, checkpoint):
            if data.get("schema") != "PHISTREAM_REFERENCE_TRACE_V1":
                raise ValueError("Bad reference schema")
            if not data["reference"]["asserted_independent"]:
                raise ValueError("No asserted reference")
            return {
                "prompt": data["prompt_token_ids"],
                "expected_ids": data["expected_generated_token_ids"],
                "origin": data["reference"],
            }

        reference.read_reference_trace = read
        reference.validate_reference_trace = validate
        with patch.dict(sys.modules, {
            "phistream": fake_pkg,
            "phistream.safeweights": safe,
            "phistream.reference": reference,
        }):
            with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
                yield

    def preflight(self):
        return preflight_reference(
            self.catalog, "test-llama", tokenizer_path=self.tokenizer,
            prompt="hello world", trace_path=self.trace,
        )

    def run_gate(self, runner, **overrides):
        args = {
            "tokenizer_path": self.tokenizer, "prompt": "hello world",
            "trace_path": self.trace, "device": "cpu",
            "capacity_mib": 4096, "reserve_mib": 512,
            "output": self.output, "approved": True, "runner": runner,
        }
        args.update(overrides)
        return run_reference_gate(self.catalog, "test-llama", **args)

    def fake_runner(self, *, match=True, tamper=None):
        def fake(argv, **kwargs):
            self.assertEqual(argv[1:4], ["-m", "phistream.cli", "real-model-gate"])
            self.assertIn("--execute-parity", argv)
            self.assertIn("--approve-local-execution", argv)
            self.assertEqual(argv[argv.index("--capacity-mib")+1], "4096")
            self.assertEqual(argv[argv.index("--device")+1], "cpu")
            self.assertFalse(kwargs["shell"])
            trace_hash = hashlib.sha256(self.trace.read_bytes()).hexdigest()
            fp = self.data["checkpoint_fingerprint"]
            count = len(self.data["expected_generated_token_ids"])
            probes = count
            result = {
                "schema": "PHISTREAM_REAL_MODEL_BRIDGE_V1",
                "status": "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED" if match else "PARITY_MISMATCH",
                "model_execution": True,
                "real_model_run_authorized": True,
                "device_requested": "cpu",
                "reference_trace": {"trace_sha256": trace_hash, "steps": count},
                "checkpoint_fingerprint": fp,
                "execution": {
                    "schema": "PHISTREAM_REFERENCE_COMPARISON_V1",
                    "status": "MATCH_USER_REFERENCE_UNVERIFIED" if match else "MISMATCH",
                    "device": "cpu",
                    "steps": count,
                    "exact_greedy_tokens_matched": count if match else count - 1,
                    "total_logit_probes": probes,
                    "logit_probes_matched": probes if match else probes - 1,
                    "checkpoint_fingerprint": fp,
                },
            }
            if tamper:
                tamper(result)
            report_path = Path(argv[argv.index("--output")+1])
            report_path.write_text(json.dumps(result), encoding="utf-8")
            return SimpleNamespace(returncode=0 if match else 2)
        return fake

    def test_local_preflight_matches_exact_prompt_ids_without_inference(self):
        with self.mocked_reference():
            report, original = self.preflight()
        self.assertEqual(report["status"], "USER_REFERENCE_TOKENIZATION_MATCH_METADATA_ONLY")
        self.assertTrue(report["reference_eligible_for_run"])
        self.assertEqual(report["prompt_token_count"], 2)
        self.assertEqual(report["reference_steps"], 32)
        self.assertFalse(report["checkpoint_hashes_verified"])
        self.assertFalse(report["reference_independence_authenticated"])
        self.assertEqual(original["prompt_token_ids"], [1, 2])
        self.assertFalse(self.output.exists())

    def test_mismatched_prompt_ids_rejected(self):
        self.data["prompt_token_ids"] = [2, 1]
        self._save()
        with self.mocked_reference(), self.assertRaisesRegex(PhiStreamLabError, "Tokenizer prompt IDs differ"):
            self.preflight()

    def test_short_reference_is_reported_ineligible_and_refused_for_execution(self):
        self.data = self._trace(4)
        self._save()
        with self.mocked_reference():
            report, _ = self.preflight()
            self.assertFalse(report["reference_eligible_for_run"])
            with self.assertRaisesRegex(PhiStreamLabError, "32 reference steps"):
                self.run_gate(runner=lambda *_a, **_k: self.fail("must not execute"))
        self.assertFalse(self.output.exists())

    def test_matching_user_reference_writes_only_minimal_unverified_receipt(self):
        with self.mocked_reference():
            result = self.run_gate(self.fake_runner())
        self.assertEqual(result["status"], "USER_ASSERTED_REFERENCE_MATCH_UNAUTHENTICATED")
        self.assertEqual(result["exact_greedy_tokens_matched"], 32)
        self.assertEqual(result["sampled_logit_probes_matched"], 32)
        self.assertFalse(result["reference_independence_authenticated"])
        self.assertFalse(result["chat_service_enabled"])
        content = self.output.read_text()
        self.assertNotIn("hello world", content)
        self.assertNotIn("expected_generated_token_ids", content)
        self.assertNotIn("logit_probes", json.loads(content))
        self.assertNotIn("model.safetensors", content)

    def test_genuine_mismatch_is_a_diagnostic_not_success(self):
        with self.mocked_reference():
            result = self.run_gate(self.fake_runner(match=False))
        self.assertEqual(result["status"], "REFERENCE_MISMATCH")
        self.assertFalse(result["chat_service_enabled"])
        self.assertTrue(self.output.exists())

    def test_inconsistent_pass_claim_is_rejected(self):
        with self.mocked_reference(), self.assertRaisesRegex(PhiStreamLabError, "contradicts"):
            self.run_gate(self.fake_runner(tamper=lambda d: d["execution"].update(
                {"exact_greedy_tokens_matched": 1}
            )))
        self.assertFalse(self.output.exists())

    def test_forged_checkpoint_fingerprint_is_rejected(self):
        with self.mocked_reference(), self.assertRaisesRegex(PhiStreamLabError, "binding"):
            self.run_gate(self.fake_runner(tamper=lambda d: d["checkpoint_fingerprint"]["files"].update(
                {"config.json": "z" * 64}
            )))
        self.assertFalse(self.output.exists())

    def test_modified_trace_during_execution_cannot_pass(self):
        delegate = self.fake_runner()
        def mutate(argv, **kwargs):
            output = delegate(argv, **kwargs)
            self.data["reference"]["version"] = "changed-mid-run"
            self._save()
            return output
        with self.mocked_reference(), self.assertRaisesRegex(PhiStreamLabError, "changed"):
            self.run_gate(mutate)
        self.assertFalse(self.output.exists())

    def test_approval_invalid_budget_and_existing_receipt_fail_before_run(self):
        with self.mocked_reference(), self.assertRaisesRegex(PhiStreamLabError, "approve"):
            self.run_gate(lambda *_a, **_k: self.fail("unexpected run"), approved=False)
        with self.mocked_reference(), self.assertRaisesRegex(PhiStreamLabError, "finite"):
            self.run_gate(lambda *_a, **_k: self.fail("unexpected run"), capacity_mib=float("nan"))
        self.output.write_text("existing", encoding="utf-8")
        with self.mocked_reference(), self.assertRaisesRegex(PhiStreamLabError, "already exists"):
            self.run_gate(lambda *_a, **_k: self.fail("unexpected run"))
        self.assertEqual(self.output.read_text(), "existing")

    def test_cli_registers_both_parity_commands(self):
        check = parser().parse_args([
            "phistream", "parity-preflight", "research",
            "--tokenizer", "tokenizer.json", "--trace", "reference.json",
            "--prompt", "hello",
        ])
        self.assertEqual(check.action, "parity-preflight")
        run = parser().parse_args([
            "phistream", "parity-run", "research",
            "--tokenizer", "tokenizer.json", "--trace", "reference.json",
            "--prompt", "hello", "--device", "cpu", "--capacity-mib", "4096",
            "--approve-local-execution", "--output", "out.json",
        ])
        self.assertEqual(run.action, "parity-run")
        self.assertTrue(run.approve_local_execution)


if __name__ == "__main__":
    unittest.main()
