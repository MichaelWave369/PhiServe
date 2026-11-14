"""R21 realistic synthetic end-to-end R18 -> R19 -> R20 runner tests.

The runtime stage functions are the real modules, with only the underlying
R18 benchmark and hardware snapshot replaced by locally generated fixtures.
No CUDA, GGUF inference, network, or real physical performance is exercised.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phiserve.field_campaign import collect
from phiserve.field_study import StudyHeld, layout, main, study
from phiserve.campaign_review import inspect_campaign
from phiserve.campaign_bundle import inspect_bundle
from test_field_campaign import FieldCampaignTests


class FieldStudyTests(unittest.TestCase):
    def setUp(self):
        self.producer=FieldCampaignTests(
            "test_preview_never_writes_anything_or_benchmarks")
        self.producer.setUp()
        self.addCleanup(self.producer.doCleanups)
        self.root=self.producer.root
        self.options={
            "workspace":self.root,
            "site":"main-5070","session":"study-01",
            "alias":"toy","binary":str(self.producer.bin),
            "layers":[0,12],"gpu_probe":"cuda",
            "sample_nvidia":True,"registry":self.producer.registry,
        }

    def run_fake(self, **kwargs):
        def fake_collector(**options):
            return collect(**{
                **options,"benchmark_fn":self.producer.fake_benchmark,
                "snapshot_fn":self.producer.fake_snapshot,
            })
        return study(**{**self.options,
                        "collector":fake_collector, **kwargs})

    def paths(self):
        return layout(self.root,"main-5070","study-01")

    def test_preview_is_read_only_and_has_no_local_paths(self):
        result=self.run_fake()
        campaign, review, archive=self.paths()
        self.assertEqual(result["status"],"READ_ONLY_PREVIEW")
        self.assertFalse(result["inference_started"])
        self.assertFalse(result["files_created"])
        self.assertEqual(result["local_model_and_binary_preflight"],"PASSED")
        self.assertEqual(result["physical_hardware_preflight"],"NOT_PERFORMED")
        self.assertEqual(result["requested_gpu_layers"],[0,12])
        self.assertTrue(result["requires_explicit_execute"])
        self.assertFalse(campaign.exists() or review.exists() or archive.exists())
        self.assertEqual(self.producer.events,[])
        self.assertNotIn(str(self.root),json.dumps(result))

    def test_single_explicit_run_calls_real_three_stage_modules(self):
        result=self.run_fake(execute=True)
        campaign, review_path, archive=self.paths()
        self.assertEqual(result["status"],"LOCAL_OPERATOR_STUDY_FILES_COMPLETE")
        self.assertEqual(result["r15_classification"],
                         "REVIEW_REQUIRED_CAPACITY_NOT_CONFIRMED")
        self.assertEqual(result["r19_status"],
                         "LOCAL_SOURCE_BYTES_MATCH_NOT_MACHINE_AUTHENTICITY")
        self.assertEqual(result["r20_status"],"PRIVATE_LOCAL_BUNDLE_CREATED")
        self.assertTrue(campaign.joinpath("campaign-manifest.json").exists())
        self.assertTrue(review_path.exists())
        self.assertTrue(archive.exists())
        self.assertEqual(self.producer.events,[
            "benchmark:0:sample=False","benchmark:12:sample=True",
            "snapshot:cuda",
        ])
        verified=inspect_campaign(campaign)
        zipped=inspect_bundle(archive)
        self.assertEqual(zipped["manifest_sha256"],verified["manifest_sha256"])
        self.assertTrue(zipped["source_bytes_verified"])
        self.assertFalse(result["automatic_model_routing"])
        self.assertFalse(result["gpu_execution_verified"])
        self.assertFalse(result["hardware_identity_authenticated"])
        self.assertFalse(result["deployment_authorized"])
        self.assertFalse(result["external_upload_performed"])

    def test_same_workspace_second_run_with_new_session_is_allowed(self):
        self.run_fake(execute=True)
        self.run_fake(execute=True,session="study-02")
        self.assertTrue(self.root.joinpath("main-5070-study-02.zip").exists())
        self.assertTrue(self.root.joinpath("main-5070-study-01.zip").exists())

    def test_reusing_session_refuses_before_new_inference(self):
        self.run_fake(execute=True)
        self.producer.events.clear()
        with self.assertRaisesRegex(StudyHeld,"STUDY_OUTPUT_ALREADY_EXISTS"):
            self.run_fake(execute=True)
        self.assertEqual(self.producer.events,[])

    def test_existing_review_file_blocks_all_run_stages(self):
        campaign, review, archive=self.paths()
        review.write_text("existing")
        with self.assertRaisesRegex(StudyHeld,"STUDY_OUTPUT_ALREADY_EXISTS"):
            self.run_fake(execute=True)
        self.assertEqual(review.read_text(),"existing")
        self.assertFalse(campaign.exists())
        self.assertFalse(archive.exists())

    def test_invalid_site_session_and_malicious_labels_rejected(self):
        for field,value in (
            ("site","../escape"),("session","../escape"),
            ("session","with space"),("site",""),("session","X"*65),
            ("site","C:\\Users"),("session",True)
        ):
            with self.subTest(field=field,value=value):
                with self.assertRaises(StudyHeld):
                    self.run_fake(**{field:value},execute=True)
                self.assertEqual(self.producer.events,[])

    def test_requires_existing_private_workspace(self):
        with self.assertRaisesRegex(StudyHeld,"WORKSPACE_MUST_BE_EXISTING_REAL_DIRECTORY"):
            self.run_fake(workspace=self.root/"missing",execute=True)
        git=self.root/"repository"
        git.mkdir()
        (git/".git").mkdir()
        with self.assertRaisesRegex(StudyHeld,"WORKSPACE_INSIDE_GIT_CHECKOUT"):
            self.run_fake(workspace=git,execute=True)

    def test_workspace_symlink_refused(self):
        link=self.root/"alias-folder"
        try:
            link.symlink_to(self.root,target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation unavailable")
        with self.assertRaisesRegex(StudyHeld,"WORKSPACE_MUST_BE_EXISTING_REAL_DIRECTORY"):
            self.run_fake(workspace=link,execute=True)

    def test_wrong_type_execute_gate_fails_before_inference(self):
        with self.assertRaisesRegex(StudyHeld,"INVALID_EXECUTION_GATE"):
            self.run_fake(execute="true")
        self.assertEqual(self.producer.events,[])

    def test_missing_model_or_binary_is_not_silently_installed(self):
        with self.assertRaises(ValueError):
            self.run_fake(alias="missing",execute=True)
        with self.assertRaises(ValueError):
            self.run_fake(binary=str(self.root/"missing-server.exe"),execute=True)
        self.assertFalse(self.root.joinpath("main-5070-study-01").exists())

    def test_backend_failure_creates_no_review_or_zip(self):
        def broken_collector(**options):
            options.update(benchmark_fn=self.producer.fake_benchmark,
                           snapshot_fn=self.producer.fake_snapshot)
            def fail_after_cpu(*args, **kwargs):
                if kwargs["gpu_layers"]>0:
                    raise RuntimeError("SIMULATED_OFFLOAD_FAILURE")
                return self.producer.fake_benchmark(*args,**kwargs)
            options["benchmark_fn"]=fail_after_cpu
            return collect(**options)
        with self.assertRaisesRegex(RuntimeError,"SIMULATED_OFFLOAD_FAILURE"):
            study(**{**self.options,
                     "collector":broken_collector,"execute":True})
        campaign, review, archive=self.paths()
        self.assertTrue((campaign/"benchmark-layers-0.json").exists())
        self.assertFalse((campaign/"campaign-manifest.json").exists())
        self.assertFalse(review.exists())
        self.assertFalse(archive.exists())

    def test_verifier_failure_prevents_pack_and_success_claim(self):
        def rejected_review(*args,**kwargs):
            raise RuntimeError("SIMULATED_REVIEW_REFUSAL")
        with self.assertRaisesRegex(RuntimeError,"SIMULATED_REVIEW_REFUSAL"):
            self.run_fake(execute=True,reviewer=rejected_review)
        campaign, review, archive=self.paths()
        self.assertTrue((campaign/"campaign-manifest.json").exists())
        self.assertFalse(review.exists())
        self.assertFalse(archive.exists())

    def test_zip_failure_never_reports_complete(self):
        def rejected_zip(*args,**kwargs):
            raise RuntimeError("SIMULATED_PACK_FAILURE")
        with self.assertRaisesRegex(RuntimeError,"SIMULATED_PACK_FAILURE"):
            self.run_fake(execute=True,packer=rejected_zip)
        campaign, review, archive=self.paths()
        self.assertTrue(campaign.exists())
        self.assertTrue(review.exists())
        self.assertFalse(archive.exists())

    def test_fail_closed_on_fake_success_result(self):
        def forged_collector(**options):
            return {"status":"SUCCESS","r15_classification":"fake"}
        with self.assertRaisesRegex(StudyHeld,"R18_NOT_COMPLETE"):
            self.run_fake(execute=True,collector=forged_collector)
        _, review, archive=self.paths()
        self.assertFalse(review.exists())
        self.assertFalse(archive.exists())

    def test_cli_run_requires_literal_execute_and_creates_nothing(self):
        argv=[
            "run","--workspace",str(self.root),"--site","main-5070",
            "--session","study-01","--alias","toy",
            "--llama-server",str(self.producer.bin),
            "--gpu-layers","0","--gpu-layers","12",
        ]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as err:
            main(argv)
        self.assertEqual(err.exception.code,2)
        self.assertEqual(self.producer.events,[])
        self.assertFalse(self.paths()[0].exists())

    def test_cli_preview_succeeds_without_any_inference(self):
        argv=[
            "preview","--workspace",str(self.root),"--site","main-5070",
            "--session","study-01","--alias","toy",
            "--llama-server",str(self.producer.bin),
            "--gpu-layers","0","--gpu-layers","12",
        ]
        with patch.dict(os.environ,{"PHISERVE_HOME":str(self.producer.registry.home)}):
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(argv),0)
        report=json.loads(stdout.getvalue())
        self.assertEqual(report["status"],"READ_ONLY_PREVIEW")
        self.assertFalse(self.paths()[0].exists())
        self.assertEqual(self.producer.events,[])


if __name__ == "__main__":
    unittest.main()
