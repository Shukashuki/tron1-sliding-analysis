"""Serial single-wheel diagnostics after the current PPO/evaluation job exits.

Fresh full NVML/PCI inspection per case, no retries or process signals. This
driver creates no training jobs, never claims a GPU reservation, and stops at
the first failed geometry/force check. Runtime sources are manifest-bound.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def run(a):
    source = a.source_dir.resolve()
    manifest = read(a.source_manifest)
    entries = manifest["files"]
    assert len(entries) == 5 and all(sha(source/row["path"]) == row["sha256"] for row in entries)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert sha(a.gpu_audit_module) == a.gpu_audit_sha256
    spec = importlib.util.spec_from_file_location("_wheel_geometry_gpu_audit", a.gpu_audit_module)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    assert helper.GPU_ORDER == (3, 5, 4)
    launch = read(a.predecessor_launch)
    protocol = read(source/"config/physx_wheel_geometry_bench_v1.json")
    assert protocol["kind"] == "PhysX_single_wheel_geometry_force_factorial_v1"
    geometry_levels = [g for g in protocol["geometry_levels"]
                       if a.geometry_filter == "all" or g["approximation"] == "smooth"]
    expected_cases = len(protocol["force_levels"])*len(geometry_levels)
    a.output_dir.mkdir()
    target = a.output_dir/"status.json"
    status = dict(kind="serial_PhysX_single_wheel_geometry_force_factorial", status="waiting_for_predecessor",
        cases=[], predecessor_launch_sha256=sha(a.predecessor_launch),
        source_manifest_sha256=sha(a.source_manifest), driver_sha256=sha(__file__),
        task_success_claimed=False, policy_training=False, automatic_retry=False,
        signals_sent=False, GPU_reservation_claimed=False)
    status.update(geometry_filter=a.geometry_filter, expected_cases=expected_cases)
    write(target, status)
    deadline = time.monotonic()+a.deadline_seconds
    try:
        while True:
            assert time.monotonic() < deadline, "Predecessor has not completed before deadline"
            current = helper.process_snapshot(launch["pid"])
            if current is None or current["state"] == "Z":
                break
            assert all(current[k] == launch["process_identity_at_launch"][k]
                       for k in ("uid", "start_ticks", "argv")), "Predecessor PID identity changed"
            time.sleep(10)
        predecessor = read(a.predecessor_status)
        assert predecessor["status"] == "nominal_dual_engine_validation_completed_requires_task_review"
        assert len(predecessor["children"]) == 2
        assert all(c["status"] == "completed_verified" and c["returncode"] == 0 for c in predecessor["children"])
        status["predecessor_completed_status_sha256"] = sha(a.predecessor_status)
        for law in protocol["force_levels"]:
            for geometry in geometry_levels:
                if len(status["cases"]) >= a.case_limit:
                    break
                assert all(sha(source/r["path"]) == r["sha256"] for r in entries), "Frozen source changed"
                name = geometry["id"]+"__"+law
                folder = a.output_dir/name
                assert not folder.exists()
                while True:
                    assert time.monotonic() < deadline, "No fully idle approved GPU before deadline"
                    gpus = helper.query_gpus()
                    idle = next((g for g in gpus if g["idle"]), None)
                    if idle is not None:
                        break
                    status.update(status="waiting_for_fully_idle_GPU", gpu_snapshot=gpus)
                    write(target, status); time.sleep(10)
                argv = [str(a.python), "-B", "-u", str(source/"scripts/probe_physx_wheel_geometry.py"),
                    "--law", law, "--slices", str(geometry["slices"]),
                    "--approximation", geometry["approximation"],
                    "--dt", str(protocol["physics_dt_s"]), "--duration", str(protocol["duration_s"]),
                    "--speed", str(protocol["speed_m_s"]), "--side-angle-deg", str(protocol["initial_hub_side_angle_deg"]),
                    "--mu", str(protocol["mu"]), "--device", f"cuda:{idle['physical_index']}",
                    "--output-dir", str(folder)]
                with (a.output_dir/(name+".log")).open("xb") as log:
                    child = subprocess.Popen(argv, cwd=a.output_dir, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID"), shell=False)
                identity = helper.process_snapshot(child.pid)
                assert identity and identity["uid"] == os.getuid() and identity["argv"] == argv
                case = dict(id=name, geometry=geometry, law=law, pid=child.pid, argv=argv,
                    process_identity_at_launch=identity, gpu_idle_before_launch=idle, status="running")
                status["cases"].append(case); status["status"] = "running_single_case"; write(target, status)
                while child.poll() is None:
                    assert time.monotonic() < deadline, "Owned child exceeded deadline; preserved, not signalled"
                    time.sleep(1)
                case["returncode"] = child.returncode
                assert child.returncode == 0, "Case failed; preserve evidence without automatic retry"
                assert not (folder/"failure.json").exists(), "Physics preflight failed; inspect failure.json"
                result = read(folder/"status.json")
                assert result["status"] == "completed" and result["physical_steps"] == 800
                assert result["law"] == law and result["compound_geometry"]["slices"] == geometry["slices"]
                assert result["compound_geometry"]["approximation_mode"] == geometry["approximation"]
                assert sha(folder/"native_trace.npz") == result["native_trace_sha256"]
                assert all(sha(p) == digest for p, digest in result["source_sha256"].items())
                case.update(status="completed_verified", report_sha256=sha(folder/"status.json"),
                    native_trace_sha256=result["native_trace_sha256"])
                write(target, status)
        status["status"] = ("selected_single_wheel_cases_completed_require_analysis"
                            if len(status["cases"]) == expected_cases else "limited_single_wheel_preflight_completed")
    except BaseException as error:
        if status["cases"] and status["cases"][-1]["status"] == "running":
            status["cases"][-1]["status"] = "failed_validation"
        status.update(status="failed_no_automatic_retry", error_type=type(error).__name__, error=str(error))
        write(target, status)
        raise
    write(target, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-dir", "source-manifest", "gpu-audit-module", "predecessor-launch",
                 "predecessor-status", "python", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--gpu-audit-sha256", required=True)
    parser.add_argument("--deadline-seconds", type=float, default=1800)
    parser.add_argument("--case-limit", type=int, choices=range(1,10), default=9)
    parser.add_argument("--geometry-filter", choices=("all", "smooth"), default="all")
    run(parser.parse_args())
