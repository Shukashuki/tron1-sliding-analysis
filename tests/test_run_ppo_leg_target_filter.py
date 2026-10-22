"""Injected protocol fixtures—not actual filtered-physics experiments."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"scripts"))
import run_ppo_leg_target_filter as runner
import eval_ppo_leg_target_filter as evaluator
from test_eval_ppo_leg_target_filter import telemetry, make_plan, BASE_REPORT, LEGACY
from run_ppo_smoothness_sequence import write_json


@pytest.fixture
def args(tmp_path, monkeypatch):
    inputs = tmp_path/"inputs"; inputs.mkdir()
    values = {name: inputs/name for name in ("checkpoint", "guard_source", "motion_file", "asset_path", "runner_config", "task_profile")}
    for path in values.values(): path.write_text("CPU protocol fixture")
    bm = inputs/"bm"; config = bm/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    config.parent.mkdir(parents=True); config.write_text("# CPU fixture\n")
    original = runner.base.sha256
    monkeypatch.setattr(runner.base, "sha256", lambda path: runner.base.CHECKPOINT_SHA if Path(path) == values["checkpoint"] else original(path))
    monkeypatch.setattr(runner.base, "freeze", lambda args: {"frozen": 1}); monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return NS(**values, legacy_evaluator=LEGACY, beyondmimic_source=bm, python=Path(sys.executable), output_dir=tmp_path/"fresh")


def make_output(folder, alpha, monkeypatch, physical_gpu_index=1):
    folder.mkdir(); plan = make_plan(alpha, monkeypatch, physical_gpu_index); plan["paths"] = {"output_dir": folder}
    report = json.loads(BASE_REPORT.read_text()); report.pop("contact_stability_diagnostic")
    with np.load(BASE_REPORT.parent/"physics_trajectory.npz", allow_pickle=False) as archive: raw = {key: archive[key] for key in archive.files}
    control, physics, runtime = telemetry(alpha, requested=raw["actions_clipped"][1::4, 0], initial=raw["joint_position_rad"][0, 0, :6])
    runtime["device"] = f"cuda:{physical_gpu_index}"
    device = report["runtime_device_audit"]
    for key in ("requested_device", "environment_device", "physics_sim_view_frontend_device", "robot_native_transform_tensor_device"):
        device[key] = f"cuda:{physical_gpu_index}"
    for key in ("native_physics_backend_device_ordinal", "torch_current_device", "physics_cudaDevice_setting", "device_index"):
        device[key] = physical_gpu_index
    class Handler:
        def control_arrays(self): return control
        def physics_arrays(self): return physics
        def close(self): pass
        def audit(self, *, check_device_health): return runtime
    evaluator.finalize_filter(None, Handler(), plan)
    report["policy_export"].pop("contact_stability_diagnostic"); evaluator.decorate_contract(report["policy_export"], plan)
    report["leg_target_filter_diagnostic"] = plan["audit"]; report["evaluator_source_sha256"] = evaluator.base.sha256(evaluator.__file__)
    (folder/"physics_trajectory.npz").symlink_to(BASE_REPORT.parent/"physics_trajectory.npz"); write_json(folder/"report.json", report)


class Guard:
    KIT_ARGS = "--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false --/physics/cudaDevice=1 --/physics/sceneMultiGPUMode=0"
    write_json = staticmethod(write_json)
    def __init__(self, args, monkeypatch, *, busy_at=None, nonzero=False):
        self.args, self.monkeypatch = args, monkeypatch; self.calls = self.probes = 0; self.busy_at, self.nonzero = busy_at, nonzero
    def query_gpu_one(self, physical_gpu_index=1):
        self.probes += 1
        return {"physical_index": physical_gpu_index, "idle": self.probes != self.busy_at, "uuid": f"GPU-{physical_gpu_index}",
            "pci_bus_id": f"00000000:{physical_gpu_index+1:02x}:00.0", "pci_bus_id_cuda_ordinal_verified": True}
    def run_child(self, command, log, cwd, environment, spawned):
        self.calls += 1; spawned(6000+self.calls); log.write_text("CPU fixture\n")
        assert "CUDA_VISIBLE_DEVICES" not in environment and environment["TRON1_BEYONDMIMIC_ROOT"] == str(self.args.beyondmimic_source)
        assert "--leg-kd" not in command and "--leg-kp" not in command
        index = int(command[command.index("--physical-gpu-index")+1])
        assert command[command.index("--device")+1] == f"cuda:{index}"
        assert command[command.index("--kit_args")+1] == evaluator.kit_args_for(index)
        make_output(Path(command[command.index("--output-dir")+1]), float(command[command.index("--alpha")+1]), self.monkeypatch, index)
        return 7 if self.nonzero else 0


def test_identity_first_two_guards_and_finalized_telemetry(args, monkeypatch):
    guard = Guard(args, monkeypatch); sleeps = []; status = runner.run(args, guard=guard, quiesce=sleeps.append, gpu_probe=guard.query_gpu_one)
    assert status["status"] == "episodes_complete_requires_contact_and_tracking_review" and guard.calls == guard.probes == 2
    assert sleeps == [2., 2.] and status["children"][0]["verification"]["recording_sha256"] == runner.BASELINE_RAW_SHA256
    assert all(row["status"] == "completed_verified" for row in status["children"])
    assert status["contact_success_claimed"] is False


@pytest.mark.parametrize("busy_at", [1, 2])
def test_guard_before_each_child(args, monkeypatch, busy_at):
    guard = Guard(args, monkeypatch, busy_at=busy_at)
    with pytest.raises(ValueError, match="fully idle"): runner.run(args, guard=guard, quiesce=lambda s: None, gpu_probe=guard.query_gpu_one)
    assert guard.calls == busy_at-1


def test_alpha1_original_raw_gate_before_alpha05(args, monkeypatch):
    guard = Guard(args, monkeypatch); monkeypatch.setattr(runner, "BASELINE_RAW_SHA256", "0"*64)
    with pytest.raises(ValueError, match="reproduce original54e01"): runner.run(args, guard=guard, quiesce=lambda s: None, gpu_probe=guard.query_gpu_one)
    assert guard.calls == 1


def test_nonzero_child_not_retried(args, monkeypatch):
    guard = Guard(args, monkeypatch, nonzero=True)
    with pytest.raises(ValueError, match="Owned evaluator failed"): runner.run(args, guard=guard, quiesce=lambda s: None, gpu_probe=guard.query_gpu_one)
    assert guard.calls == 1


def test_separate_runtime_evidence_hash_tamper_rejected(args, monkeypatch):
    folder = args.output_dir.parent/"case"; make_output(folder, 1., monkeypatch); runner.validate_output(folder, 1.)
    (folder/evaluator.RUNTIME_FILE).write_text('{}')
    with pytest.raises(ValueError, match="evidence hash"): runner.validate_output(folder, 1.)


def test_existing_output_precedes_guard(args):
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="Preserve existing"): runner.run(args)


def gpu_xml(*, process_at=None, swap=False, missing_processes=False):
    rows = []
    buses = list(range(1, 7))
    if swap: buses[0], buses[3] = buses[3], buses[0]
    for index, bus in enumerate(buses):
        process = '<process_info><pid>12345</pid><type>G</type><process_name>foreign-graphics</process_name></process_info>' if index == process_at else ''
        processes = '' if missing_processes and index == 3 else f'<processes>{process}</processes>'
        rows.append(f'<gpu><uuid>GPU-{index}</uuid><pci><pci_bus_id>00000000:{bus:02x}:00.0</pci_bus_id></pci><fb_memory_usage><used>6 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization>{processes}</gpu>')
    return '<nvidia_smi_log>'+''.join(rows)+'</nvidia_smi_log>'


def test_gpu3_xml_proves_complete_idle_ordinal_and_graphics_process_rejection():
    good = runner.parse_gpu_snapshot(gpu_xml(), 3)
    assert good['physical_index'] == 3 and good['uuid'] == 'GPU-3' and good['idle'] is True
    assert good['pci_bus_id_cuda_ordinal_verified'] is True
    busy = runner.parse_gpu_snapshot(gpu_xml(process_at=3), 3)
    assert busy['idle'] is False and busy['all_reported_compute_and_graphics_processes'][0]['type'] == 'G'
    with pytest.raises(ValueError, match='CUDA ordinal'): runner.parse_gpu_snapshot(gpu_xml(swap=True), 3)
    with pytest.raises(ValueError, match='enumeration'): runner.parse_gpu_snapshot(gpu_xml(missing_processes=True), 3)
    with pytest.raises(ValueError, match='Unauthorized'): runner.parse_gpu_snapshot(gpu_xml(), 2)


def test_gpu3_fixed_for_both_children_no_fallback_and_runtime_binding(args, monkeypatch):
    args.physical_gpu_index = 3; guard = Guard(args, monkeypatch)
    status = runner.run(args, guard=guard, quiesce=lambda s: None, gpu_probe=guard.query_gpu_one)
    assert status['physical_gpu_index'] == 3 and status['requested_device'] == 'cuda:3'
    assert status['automatic_gpu_fallback'] is False and guard.calls == guard.probes == 2
    assert all(child['gpu_preflight']['physical_index'] == 3 for child in status['children'])
    case = args.output_dir/'alpha0p5'; report_path = case/'report.json'; report = json.loads(report_path.read_text())
    report['runtime_device_audit']['native_physics_backend_device_ordinal'] = 1; write_json(report_path, report)
    with pytest.raises(ValueError, match='inconsistent or masked'): runner.validate_output(case, .5, 3)


def test_gpu3_busy_or_masked_stops_without_fallback(args, monkeypatch):
    args.physical_gpu_index = 3; guard = Guard(args, monkeypatch, busy_at=1)
    with pytest.raises(ValueError, match='no fallback'):
        runner.run(args, guard=guard, quiesce=lambda s: None, gpu_probe=guard.query_gpu_one)
    assert guard.calls == 0 and guard.probes == 1
    fresh = args.output_dir.parent/'masked'; args.output_dir = fresh; monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    with pytest.raises(ValueError, match='unmasked'): runner.run(args, guard=guard, gpu_probe=guard.query_gpu_one)
    assert not fresh.exists()
