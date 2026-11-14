"""R18 field-campaign contract tests. Synthetic GGUF/HTTP receipts only."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

from phiserve.benchmark import LIMITS, PROMPT, _summary, validate_receipt
from phiserve.field_campaign import CampaignHeld, collect, main, SCHEMA
from phiserve.registry import ModelRegistry


class FieldCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.model=self.root/"tiny.gguf"
        self.model.write_bytes(b"GGUF"+b"dummy synthetic bytes"*30)
        self.bin=self.root/"llama-server"
        self.bin.write_bytes(b"synthetic executable fixture")
        self.registry=ModelRegistry(self.root/"home")
        self.registry.add("toy",self.model)
        self.output=self.root/"campaign"
        self.events=[]
        self.now=datetime.now(timezone.utc)
        self.args={
            "alias":"toy","binary":str(self.bin),"out_dir":self.output,
            "site":"lab-main","layers":[0,12],"gpu_probe":"cuda",
            "registry":self.registry,
        }

    def fake_benchmark(self,alias,binary,path,*,gpu_layers,ctx_size,max_tokens,
                       warmups,trials,backend_port,timeout_seconds,
                       startup_timeout,sample_nvidia,registry):
        self.events.append("benchmark:"+str(gpu_layers)+":sample="+str(sample_nvidia))
        measurements=[]
        for i in range(1,trials+1):
            wall=100.0+i
            measurements.append({
                "index":i,"wall_ms":wall,"first_content_ms":10.0+i,
                "generated_chars":32,"completion_tokens_reported":5,
                "throughput_tps":5000.0/wall,
                "response_sha256":"d"*64,"end_marker_observed":True,
            })
        r={
            "schema":"PHISERVE_BENCHMARK_FIELD_V1",
            "status":"LOCAL_OPERATOR_OBSERVATION_UNATTESTED",
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "experiment":{
                "name":"fixed-public-prompt-v1",
                "prompt_sha256":hashlib.sha256(PROMPT.encode()).hexdigest(),
                "warmups":warmups,"trials":trials
            },
            "model":{
                "alias":alias,
                "gguf_sha256":hashlib.sha256(self.model.read_bytes()).hexdigest(),
                "size_bytes":self.model.stat().st_size,
            },
            "runtime":{
                "binary_name":self.bin.name,
                "binary_sha256":hashlib.sha256(self.bin.read_bytes()).hexdigest(),
                "phiserve_version":"0.10.0",
                "platform_family":platform.system(),
                "startup_ms":150.0,
            },
            "requested":{
                "gpu_layers":gpu_layers,"ctx_size":ctx_size,
                "max_tokens":max_tokens,"seed":42,"temperature":0,
                "backend_port":backend_port,
            },
            "measurements":measurements,
            "summary":_summary(measurements),
            "gpu_process_telemetry":None,
            "unverified_limits":dict(LIMITS),
        }
        validate_receipt(r)
        path.write_text(json.dumps(r),encoding="utf-8")
        return r

    def fake_snapshot(self,*,gpu,drives):
        self.events.append("snapshot:"+gpu)
        self.assertEqual(drives,())
        return {
            "schema":"PHISTREAM_HARDWARE_SNAPSHOT_V1",
            "status":"OBSERVED_READ_ONLY",
            "provenance":"LOCAL_HARDWARE_COUNTERS_NOT_A_BENCHMARK",
            "observed_utc":datetime.now(timezone.utc).isoformat(),
            "platform":platform.platform(),"logical_cpu_count":16,
            "ram":{
                "total_bytes":32*1024**3,
                "available_bytes":24*1024**3,
                "source":("WIN32_GLOBAL_MEMORY_STATUS_EX"
                          if platform.system()=="Windows" else
                          "LINUX_PROC_MEMINFO_MEMAVAILABLE"),
            },
            "gpu":None if gpu=="none" else {
                "device":"cuda","name":"synthetic GPU",
                "device_index":0,"compute_capability":"8.9",
                "total_bytes":12*1024**3,"free_bytes":8*1024**3,
                "source":"TORCH_CUDA_MEM_GET_INFO",
                "torch_version":"synthetic","cuda_build":"synthetic",
            },
            "gpu_requested":gpu,"drives":[],
            "limitations":["Synthetic fixture; no physical CUDA was used."],
        }

    def run_fake(self,**options):
        kwargs={**self.args,"benchmark_fn":self.fake_benchmark,
                "snapshot_fn":self.fake_snapshot,**options}
        return collect(**kwargs)

    def test_preview_never_writes_anything_or_benchmarks(self):
        view=self.run_fake()
        self.assertEqual(view["status"],"PLAN_ONLY_NOT_EXECUTED")
        self.assertTrue(view["requires_explicit_execute"])
        self.assertFalse(view["real_hardware_measurement_claimed"])
        self.assertEqual(view["gpu_layers_requested"],[0,12])
        self.assertEqual(self.events,[])
        self.assertFalse(self.output.exists())
        self.assertNotIn(str(self.model),json.dumps(view))
        self.assertNotIn(str(self.bin),json.dumps(view))

    def test_explicit_run_sequences_cpu_gpu_then_snapshot_then_r15(self):
        result=self.run_fake(execute=True,sample_nvidia=True)
        self.assertEqual(self.events,[
            "benchmark:0:sample=False",
            "benchmark:12:sample=True",
            "snapshot:cuda",
        ])
        self.assertEqual(result["status"],"OPERATOR_EXECUTED_UNATTESTED_REVIEW_ONLY")
        self.assertEqual(result["r15_classification"],
                         "REVIEW_REQUIRED_CAPACITY_NOT_CONFIRMED")
        manifest=json.loads((self.output/"campaign-manifest.json").read_text())
        self.assertEqual(manifest["schema"],SCHEMA)
        self.assertEqual(len(manifest["profiles"]),2)
        self.assertEqual([p["gpu_layers_requested"] for p in manifest["profiles"]],[0,12])
        self.assertFalse(manifest["claims"]["gpu_execution_independently_verified"])
        self.assertFalse(manifest["claims"]["automatic_inference_routing"])
        self.assertTrue((self.output/"hardware-snapshot.json").exists())
        self.assertTrue((self.output/"capacity-advice.json").exists())
        self.assertFalse(result["production_approval"])

    def test_missing_cuda_counters_yields_honest_hold(self):
        result=self.run_fake(execute=True,gpu_probe="none")
        self.assertEqual(result["r15_classification"],"HOLD_CUDA_COUNTERS_MISSING")
        doc=json.loads((self.output/"capacity-advice.json").read_text())
        self.assertIsNone(doc["capacity_floor"]["gpu_budget_bytes"])
        self.assertFalse(doc["assurances"]["execution_authorized"])

    def test_refuse_without_cpu_baseline_or_duplicate_layers(self):
        for layers in ([12],[0,0],[0,12,12],[-1,0],[],[0,1,2,3,4]):
            with self.subTest(layers=layers),self.assertRaises(CampaignHeld):
                self.run_fake(layers=layers,execute=True)
            self.assertFalse(self.output.exists())

    def test_bad_options_held_before_any_write_or_inference(self):
        for key,value in (("site","outside/label"),("ctx_size",8),
                          ("trials",2),("gpu_probe","magic"),
                          ("backend_port",0),("sample_nvidia","yes")):
            with self.subTest(key=key),self.assertRaises(CampaignHeld):
                self.run_fake(execute=True,**{key:value})
            self.assertEqual(self.events,[])
            self.assertFalse(self.output.exists())

    def test_missing_local_binary_or_registry_model_refused(self):
        with self.assertRaises(CampaignHeld):
            self.run_fake(execute=True,binary=str(self.root/"missing.exe"))
        with self.assertRaises(ValueError):
            self.run_fake(execute=True,alias="unknown")
        self.assertFalse(self.output.exists())

    def test_no_overwrite_existing_campaign_directory(self):
        self.output.mkdir()
        marker=self.output/"do-not-delete"
        marker.write_text("unchanged")
        with self.assertRaisesRegex(CampaignHeld,"OUTPUT_ALREADY_EXISTS"):
            self.run_fake(execute=True)
        self.assertEqual(marker.read_text(),"unchanged")

    def test_cannot_publish_private_campaign_into_git_checkout(self):
        code=self.root/"repo"
        code.mkdir()
        (code/".git").mkdir()
        with self.assertRaisesRegex(CampaignHeld,"OUTPUT_WITHIN_GIT_CHECKOUT"):
            self.run_fake(execute=True,out_dir=code/"receipts")
        self.assertFalse((code/"receipts").exists())

    def test_failed_second_run_leaves_partial_sources_but_no_complete_manifest(self):
        def fail_after_first(*args,**kwargs):
            if kwargs["gpu_layers"]>0:
                raise RuntimeError("SYNTHETIC_BACKEND_FAILURE")
            return self.fake_benchmark(*args,**kwargs)
        with self.assertRaisesRegex(RuntimeError,"SYNTHETIC_BACKEND_FAILURE"):
            self.run_fake(execute=True,benchmark_fn=fail_after_first)
        self.assertEqual(self.events,["benchmark:0:sample=False"])
        self.assertTrue((self.output/"benchmark-layers-0.json").exists())
        self.assertFalse((self.output/"campaign-manifest.json").exists())
        self.assertFalse((self.output/"capacity-advice.json").exists())

    def test_snapshot_failure_does_not_emit_completed_campaign(self):
        def broken_snapshot(*args,**kwargs):
            self.events.append("snapshot:fail")
            raise RuntimeError("CUDA_UNAVAILABLE")
        with self.assertRaisesRegex(RuntimeError,"CUDA_UNAVAILABLE"):
            self.run_fake(execute=True,snapshot_fn=broken_snapshot)
        self.assertEqual(self.events,[
            "benchmark:0:sample=False","benchmark:12:sample=False","snapshot:fail"
        ])
        self.assertFalse((self.output/"campaign-manifest.json").exists())
        self.assertFalse((self.output/"capacity-advice.json").exists())

    def test_import_failure_prevents_starting_any_benchmark(self):
        with patch("phiserve.field_campaign.importlib.import_module",
                   side_effect=ImportError("no phi")):
            with self.assertRaisesRegex(CampaignHeld,"PHISTREAM_HARDWARE_SNAPSHOT_UNAVAILABLE"):
                collect(**{**self.args,"benchmark_fn":self.fake_benchmark},execute=True)
        self.assertEqual(self.events,[])
        self.assertFalse(self.output.exists())

    def test_no_status_fabricated_when_advisor_refuses(self):
        def fake_reject(*args,**kwargs):
            raise ValueError("SYNTHETIC_EVIDENCE_REJECTED")
        with self.assertRaisesRegex(ValueError,"SYNTHETIC_EVIDENCE_REJECTED"):
            self.run_fake(execute=True,advisor_fn=fake_reject)
        self.assertFalse((self.output/"campaign-manifest.json").exists())

    def test_cli_requires_separate_execute_flag(self):
        args=[
            "collect","--alias","toy","--llama-server",str(self.bin),
            "--site","lab-main","--out-dir",str(self.output),
            "--gpu-layers","0","--gpu-layers","12"
        ]
        with self.assertRaises(SystemExit) as caught:
            main(args)
        self.assertEqual(caught.exception.code,2)
        self.assertFalse(self.output.exists())

    def test_cli_preview_does_not_start_benchmark(self):
        args=[
            "preview","--alias","toy","--llama-server",str(self.bin),
            "--site","lab-main","--out-dir",str(self.output),
            "--gpu-layers","0",
        ]
        with patch.dict(os.environ,{"PHISERVE_HOME":str(self.registry.home)}):
            self.assertEqual(main(args),0)
        self.assertFalse(self.output.exists())


if __name__=="__main__":
    unittest.main()
