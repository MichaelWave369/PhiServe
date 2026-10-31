"""Optional local tokenizer and PhiStream text-lab contract tests.

WordLevel is a small synthetic tokenizer, not a pretrained LLM tokenizer.
No physical GPU or model checkpoint is loaded by this suite.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece

from phiserve.cli import parser
from phiserve.phistream_lab import PhiStreamCatalog, PhiStreamLabError
from phiserve.phistream_text import encode_prompt, text_lab

REVIEW = {
    "schema": "PHISERVE_PHISTREAM_PREFLIGHT_V1",
    "status": "METADATA_PREFLIGHT_ONLY",
    "format": "SAFETENSORS",
    "vocab_size": 10,
    "max_context": 32,
    "layers": 2,
    "tensor_count": 21,
    "weight_bytes": 256,
    "shard_count": 1,
    "service_ready": False,
    "can_serve_chat": False,
    "tokenizer_available": False,
}


class LocalTextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        model_dir = self.home / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type":"llama"}')
        self.catalog = PhiStreamCatalog(self.home / "catalog")
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            self.catalog.add(
                "research", model_dir, source="declared", revision="test",
                license_id="unknown-test-license",
            )
        tok = Tokenizer(WordLevel(vocab={
            "[UNK]": 0, "hello": 1, "world": 2, "answer": 3, "moon": 4,
        }, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        tok.decoder = WordPiece(prefix="##")
        self.tokenizer = self.home / "tokenizer.json"
        tok.save(str(self.tokenizer))
        self.receipt = self.home / "result.json"

    def inspect(self, prompt="hello world", tokens=2, **kwargs):
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            return encode_prompt(self.catalog, "research", self.tokenizer, prompt,
                                 tokens=tokens, **kwargs)

    def test_real_local_wordlevel_tokenizer_no_model_execution(self):
        _, ids, report = self.inspect()
        self.assertEqual(ids, [1, 2])
        self.assertEqual(report["status"], "LOCAL_TOKENIZER_PREFLIGHT_ONLY")
        self.assertEqual(report["prompt_tokens"], 2)
        self.assertEqual(report["tokenizer_vocab_entries"], 5)
        self.assertEqual(len(report["tokenizer_sha256"]), 64)
        self.assertFalse(report["source_tokenizer_match_verified"])
        self.assertFalse(report["can_serve_chat"])
        self.assertFalse(self.receipt.exists())

    def test_tokenizer_file_must_be_explicit_and_local(self):
        fake = self.home / "other.json"
        fake.write_bytes(self.tokenizer.read_bytes())
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            with self.assertRaisesRegex(PhiStreamLabError, "tokenizer.json"):
                encode_prompt(self.catalog, "research", fake, "hello", tokens=1)
        if hasattr(Path, "symlink_to"):
            link = self.home / "tokenizer-link.json"
            try:
                link.symlink_to(self.tokenizer)
            except OSError:
                return  # Windows privilege policy can disallow symlink creation
            with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
                with self.assertRaises(PhiStreamLabError):
                    encode_prompt(self.catalog, "research", link, "hello", tokens=1)

    def test_bad_vocab_and_context_refused(self):
        incompatible = dict(REVIEW, vocab_size=2)
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=incompatible):
            with self.assertRaisesRegex(PhiStreamLabError, "vocabulary"):
                encode_prompt(self.catalog, "research", self.tokenizer, "hello", tokens=1)
        short = dict(REVIEW, max_context=2)
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=short):
            with self.assertRaisesRegex(PhiStreamLabError, "maximum context"):
                encode_prompt(self.catalog, "research", self.tokenizer, "hello world", tokens=3)

    def test_empty_or_unsupported_prompt_rejected(self):
        for prompt in ("", "  ", "x" * 4097):
            with self.subTest(prompt_length=len(prompt)), self.assertRaises(PhiStreamLabError):
                self.inspect(prompt=prompt)

    def test_text_execution_keeps_raw_tokens_temporary_and_receipt_private(self):
        holder = {}
        def fake_run(argv, **kwargs):
            self.assertFalse(kwargs["shell"])
            self.assertEqual(argv[1:4], ["-m", "phistream.cli", "run-llama-ids"])
            self.assertIn("--capacity-mib", argv)
            self.assertIn("--reserve-mib", argv)
            self.assertEqual(argv[argv.index("--prompt") + 1], "1,2")
            temporary = Path(argv[argv.index("--output") + 1])
            holder["intermediate"] = temporary
            temporary.write_text(json.dumps({
                "schema": "PHISTREAM_LLAMA_TOKEN_IDS_V1",
                "status": "EXECUTED_UNVERIFIED_REFERENCE",
                "device": "cpu",
                "prompt_token_ids": [1, 2],
                "generated_token_ids": [3, 2],
                "generated_token_count": 2,
            }))
            return SimpleNamespace(returncode=0)

        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            report, text = text_lab(
                self.catalog, "research", tokenizer_path=self.tokenizer,
                prompt="hello world", tokens=2, device="cpu",
                capacity_mib=2048, reserve_mib=128,
                output=self.receipt, approved=True, runner=fake_run,
            )
        self.assertEqual(text, "answer world")
        self.assertEqual(report["status"], "LOCAL_EXPERIMENTAL_TEXT_DECODED_UNVERIFIED")
        self.assertFalse(report["chat_service_enabled"])
        self.assertFalse(report["tokenizer_weights_match_verified"])
        self.assertFalse(report["external_reference_parity_verified"])
        self.assertFalse(holder["intermediate"].exists())
        content = self.receipt.read_text()
        self.assertNotIn("hello world", content)
        self.assertNotIn("answer world", content)
        self.assertNotIn("generated_token_ids", content)
        self.assertEqual(json.loads(content)["decoded_output_characters"], 12)

    def test_approval_and_receipt_collision_fail_before_inference(self):
        with self.assertRaisesRegex(PhiStreamLabError, "approve"):
            text_lab(self.catalog, "research", tokenizer_path=self.tokenizer,
                     prompt="hello", tokens=1, device="cpu", capacity_mib=4096,
                     reserve_mib=128, output=self.receipt, approved=False)
        self.receipt.write_text("existing")
        with self.assertRaisesRegex(PhiStreamLabError, "already exists"):
            text_lab(self.catalog, "research", tokenizer_path=self.tokenizer,
                     prompt="hello", tokens=1, device="cpu", capacity_mib=4096,
                     reserve_mib=128, output=self.receipt, approved=True)
        self.assertEqual(self.receipt.read_text(), "existing")

    def test_backend_failure_never_creates_success_receipt(self):
        with patch("phiserve.phistream_lab.probe_checkpoint", return_value=REVIEW):
            with self.assertRaises(PhiStreamLabError):
                text_lab(self.catalog, "research", tokenizer_path=self.tokenizer,
                         prompt="hello world", tokens=2, device="cpu",
                         capacity_mib=4096, reserve_mib=128,
                         output=self.receipt, approved=True,
                         runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=2))
        self.assertFalse(self.receipt.exists())

    def test_cli_options_require_explicit_consent_and_memory_budget(self):
        p = parser()
        args = p.parse_args([
            "phistream", "text-lab", "research", "--tokenizer", str(self.tokenizer),
            "--prompt", "hello", "--capacity-mib", "4096",
            "--approve-local-execution", "--output", str(self.receipt),
        ])
        self.assertTrue(args.approve_local_execution)
        self.assertEqual(args.capacity_mib, 4096)
        self.assertFalse(args.show_output)
        args = p.parse_args([
            "phistream", "tokenizer-check", "research", "--tokenizer",
            str(self.tokenizer), "--prompt", "hello",
        ])
        self.assertEqual(args.action, "tokenizer-check")


if __name__ == "__main__":
    unittest.main()
