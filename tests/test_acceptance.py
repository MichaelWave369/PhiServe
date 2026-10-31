"""Local acceptance checks use synthetic receipts and fake device observations.

These tests prove parser/report behavior, not any actual hardware inference,
checkpoint provenance, or independent reference execution.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from phiserve.acceptance import AcceptanceError, acceptance_report, doctor
from phiserve.cli import parser

H = "a" * 64
OUT = "b" * 64
DIFFERENT = "c" * 64
TOKENIZER = "d" * 64


class StubGGUFRegistry:
    def list_models(self):
        return {"gguf-demo": "/redacted/a.gguf"}


class StubResearchCatalog:
    def list(self):
        return {"research-demo": {"path": "/redacted/safetensors"}}


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.target = self.base / "summary.json"

    def write(self, name, value):
        path = self.base / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def gguf(self, layers, output=OUT, model="gguf-demo", prompt=H):
        return {
            "schema": "PHISERVE_LOCAL_INFERENCE_RECEIPT_V1",
            "status": "LOCAL_TEXT_OUTPUT_OBSERVED",
            "attestation": "operator_supplied_not_independently_witnessed",
            "model_alias": model,
            "model_filename": "existing.gguf",
            "model_size_bytes": 123456,
            "gpu_layers_requested": layers,
            "gpu_execution_verified": False,
            "prompt_sha256": prompt,
            "output_sha256": output,
            "output_characters": 27,
            "raw_prompt_or_output_stored": False,
        }

    def text_lab(self, model="research-demo", tokenizer=TOKENIZER):
        return {
            "schema": "PHISERVE_PHISTREAM_TEXT_LAB_V1",
            "status": "LOCAL_EXPERIMENTAL_TEXT_DECODED_UNVERIFIED",
            "model_alias": model,
            "tokenizer_sha256": tokenizer,
            "decoded_output_sha256": OUT,
            "decoded_output_characters": 12,
            "tokenizer_special_tokens_applied": False,
            "raw_prompt_or_decoded_output_stored": False,
            "chat_service_enabled": False,
        }

    def parity(self, match=True, model="research-demo", tokenizer=TOKENIZER):
        return {
            "schema": "PHISERVE_PHISTREAM_REFERENCE_GATE_V1",
            "status": ("USER_ASSERTED_REFERENCE_MATCH_UNAUTHENTICATED"
                       if match else "REFERENCE_MISMATCH"),
            "model_alias": model,
            "tokenizer_sha256": tokenizer,
            "reference_trace_sha256": H,
            "total_reference_steps": 32,
            "exact_greedy_tokens_matched": 32 if match else 31,
            "sampled_logit_probes_total": 128,
            "sampled_logit_probes_matched": 128 if match else 127,
            "tokenizer_special_tokens_applied": False,
            "reference_independence_authenticated": False,
            "chat_service_enabled": False,
        }

    def test_doctor_is_local_readiness_only_no_model_execution(self):
        receipt = doctor(
            self.target, llama_server="not-installed-gpu-binary",
            registry=StubGGUFRegistry(), catalog=StubResearchCatalog(),
            probe_nvidia=False,
        )
        self.assertEqual(receipt["status"], "LOCAL_ENVIRONMENT_OBSERVATION_ONLY")
        self.assertEqual(receipt["gpu_probe"]["status"], "NOT_REQUESTED")
        self.assertEqual(receipt["gguf_registered_count"], 1)
        self.assertEqual(receipt["phistream_research_registered_count"], 1)
        self.assertFalse(receipt["model_weights_executed"])
        self.assertFalse(receipt["gpu_execution_verified"])
        self.assertFalse(receipt["network_requests_made"])
        output = self.target.read_text()
        self.assertNotIn("/redacted", output)
        self.assertNotIn("not-installed-gpu-binary", output)

    def test_optional_gpu_probe_is_only_local_tool_metadata(self):
        def invoke(argv, **kwargs):
            self.assertEqual(argv[0], "nvidia-smi")
            self.assertFalse(kwargs["shell"])
            return SimpleNamespace(
                returncode=0,
                stdout="0, NVIDIA RTX Test GPU, 565.77, 12288\n",
            )
        report = doctor(
            self.target, probe_nvidia=True, probe_runner=invoke,
            registry=StubGGUFRegistry(), catalog=StubResearchCatalog(),
        )
        self.assertEqual(report["gpu_probe"]["status"], "LOCAL_TOOL_SELF_REPORT")
        self.assertEqual(report["gpu_probe"]["gpu_devices"][0]["total_memory_mib_reported"], 12288)
        self.assertFalse(report["gpu_execution_verified"])
        self.assertFalse(report["machine_identity_authenticated"])

    def test_complete_cpu_gpu_and_phistream_receipts_do_not_grant_serving(self):
        cpu = self.write("cpu.json", self.gguf(0))
        gpu = self.write("gpu.json", self.gguf(20))
        text = self.write("text.json", self.text_lab())
        parity = self.write("parity.json", self.parity())
        report = acceptance_report(self.target, cpu_gguf=cpu, gpu_gguf=gpu,
                                   text_receipt=text, parity_receipt=parity)
        self.assertTrue(report["paired_gguf_output_sha256_equal"])
        self.assertFalse(report["gpu_execution_independently_verified"])
        self.assertFalse(report["production_phistream_chat_authorized"])
        self.assertFalse(report["independent_reference_authenticated"])
        self.assertEqual(report["status"], "EVIDENCE_COLLECTED_REVIEW_REQUIRED")
        raw = self.target.read_text()
        self.assertNotIn("raw response text", raw)
        self.assertNotIn('"expected_generated_token_ids"', raw)
        self.assertNotIn('"logit_probes"', raw)
        self.assertEqual(report["evidence"]["gguf_cpu"]["model_alias"], "gguf-demo")
        self.assertEqual(len(report["evidence"]["gguf_cpu"]["source_file_sha256"]), 64)

    def test_output_mismatch_is_diagnostic_not_exception_or_success(self):
        cpu = self.write("cpu.json", self.gguf(0))
        gpu = self.write("gpu.json", self.gguf(20, output=DIFFERENT))
        result = acceptance_report(self.target, cpu_gguf=cpu, gpu_gguf=gpu)
        self.assertEqual(result["status"], "DIAGNOSTIC_MISMATCH_REQUIRES_REVIEW")
        self.assertFalse(result["paired_gguf_output_sha256_equal"])
        self.assertFalse(result["production_phistream_chat_authorized"])

    def test_reference_mismatch_is_diagnostic_and_not_promotion(self):
        parity = self.write("parity.json", self.parity(match=False))
        result = acceptance_report(self.target, parity_receipt=parity)
        self.assertEqual(result["status"], "DIAGNOSTIC_MISMATCH_REQUIRES_REVIEW")
        self.assertIn("PHISTREAM_REFERENCE_MISMATCH", result["findings"])
        self.assertFalse(result["independent_reference_authenticated"])

    def test_reject_incomparable_cpu_gpu_model_or_prompt(self):
        for case in ("model", "prompt"):
            with self.subTest(case=case):
                cpu = self.write("cpu.json", self.gguf(0))
                gpu = self.write("gpu.json", self.gguf(
                    20, model="other" if case == "model" else "gguf-demo",
                    prompt=DIFFERENT if case == "prompt" else H,
                ))
                with self.assertRaisesRegex(AcceptanceError, "not comparable"):
                    acceptance_report(self.target, cpu_gguf=cpu, gpu_gguf=gpu)
                self.assertFalse(self.target.exists())

    def test_reject_incompatible_phi_stream_alias_tokenizer_settings(self):
        text = self.write("text.json", self.text_lab())
        bad = self.write("parity.json", self.parity(tokenizer=DIFFERENT))
        with self.assertRaisesRegex(AcceptanceError, "different research"):
            acceptance_report(self.target, text_receipt=text, parity_receipt=bad)
        self.assertFalse(self.target.exists())

    def test_reject_fake_success_counts(self):
        p = self.parity(match=True)
        p["exact_greedy_tokens_matched"] = 29
        trace = self.write("parity.json", p)
        with self.assertRaisesRegex(AcceptanceError, "contradicts"):
            acceptance_report(self.target, parity_receipt=trace)
        self.assertFalse(self.target.exists())

    def test_reject_wrong_gpu_layers_without_relabeling(self):
        bogus = self.write("cpu.json", self.gguf(20))
        with self.assertRaisesRegex(AcceptanceError, "GPU-layer"):
            acceptance_report(self.target, cpu_gguf=bogus)

    def test_malformed_duplicate_key_nonfinite_and_oversized_receipts(self):
        file = self.base / "bad.json"
        samples = [
            '{"schema":"a","schema":"b"}',
            '{"schema":"a","value":NaN}',
            '{"schema":',
            "x" * (1024 * 1024 + 1),
        ]
        for contents in samples:
            with self.subTest(length=len(contents)):
                file.write_text(contents, encoding="utf-8")
                with self.assertRaises(AcceptanceError):
                    acceptance_report(self.target, cpu_gguf=file)
                self.assertFalse(self.target.exists())

    def test_existing_output_preserved_and_same_input_refused(self):
        cpu = self.write("cpu.json", self.gguf(0))
        self.target.write_text("do not touch", encoding="utf-8")
        with self.assertRaisesRegex(AcceptanceError, "already exists"):
            acceptance_report(self.target, cpu_gguf=cpu)
        self.assertEqual(self.target.read_text(), "do not touch")
        self.target.unlink()
        with self.assertRaisesRegex(AcceptanceError, "distinct"):
            acceptance_report(self.target, cpu_gguf=cpu, gpu_gguf=cpu)
        self.assertFalse(self.target.exists())

    def test_local_symlinked_input_rejected(self):
        cpu = self.write("cpu.json", self.gguf(0))
        link = self.base / "linked.json"
        try:
            link.symlink_to(cpu)
        except OSError:
            return  # Windows platform symlink permission may block creation.
        with self.assertRaises(AcceptanceError):
            acceptance_report(self.target, cpu_gguf=link)
        self.assertFalse(self.target.exists())

    def test_cli_parses_doctor_and_receipt_review(self):
        doctor_args = parser().parse_args([
            "acceptance", "doctor", "--output", "ready.json", "--probe-nvidia",
        ])
        self.assertEqual(doctor_args.action, "doctor")
        self.assertTrue(doctor_args.probe_nvidia)
        summary_args = parser().parse_args([
            "acceptance", "report", "--cpu-gguf", "cpu.json",
            "--phistream-parity", "parity.json", "--output", "summary.json",
        ])
        self.assertEqual(summary_args.action, "report")
        self.assertEqual(summary_args.phistream_parity, "parity.json")


if __name__ == "__main__":
    unittest.main()
