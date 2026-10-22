"""Synthetic fixed-actor native-clock/GPU-guard fixtures; never run a job."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"scripts"))
import run_ppo_contact_timestep as runner
import eval_ppo_contact_timestep as evaluator
from test_eval_ppo_contact_timestep import plan, clock_helper, profiles, LEGACY
from run_ppo_smoothness_sequence import write_json


def make_output(folder, dt, index, monkeypatch):
    folder.mkdir(parents=True)
    p = plan(dt, monkeypatch, index); trained, actual = profiles()
    actual.update(leg_kp=500., leg_kd=10., critic_dim=120)
    evaluator.decorate_contract(actual, p)
    contract = evaluator.refine_export_contract(actual, dt, trained_contract=trained,
        clock_helper=clock_helper().apply_physics_clock_to_contract, plan=p)
    decimation = evaluator.DECIMATION[dt]; n = 349*decimation+1; device = f"cuda:{index}"
    raw = {"time_s": np.arange(n)*dt, "physics_dt_s": np.array(dt), "policy_dt_s": np.array(.02),
        "valid_mask": np.ones((n, 1), dtype=bool), "simulation_step_counter": np.arange(n),
        "actions_clipped": np.zeros((n, 1, 8)), "action_was_applied": np.r_[False, np.ones(n-1, dtype=bool)][:, None],
        "reference_frame_for_action": np.r_[0, np.repeat(np.arange(349), decimation)][:, None]}
    for name, width in (("root_position_m", 3), ("root_quaternion_wxyz", 4), ("root_linear_velocity_w_m_s", 3),
        ("root_angular_velocity_w_rad_s", 3), ("joint_position_rad", 8), ("joint_velocity_rad_s", 8), ("applied_joint_torque_nm", 8)):
        raw[name] = np.zeros((n, 1, width))
    raw["root_quaternion_wxyz"][:, 0, 0] = 1.
    np.savez(folder/"physics_trajectory.npz", **raw)
    report = {"status": "evaluated", "simulator": "IsaacLab/PhysX", "checkpoint_sha256": runner.base.CHECKPOINT_SHA,
        "reference_frames": 349, "domain_randomization_enabled": False, "observation_noise_enabled": False,
        "full_horizon_diagnostic": True, "episodes_per_environment": 1, "deterministic_start_frame": 0,
        "reference_state_initialization_during_evaluation": False, "terminal_states_captured_before_auto_reset": True,
        "evaluation_contact_friction_override": {"evaluation_contact_friction": .7},
        "contact_timestep_diagnostic": copy.deepcopy(p["audit"]), "policy_export": contract,
        "evaluator_source_sha256": evaluator.base.sha256(evaluator.__file__),
        "runtime_device_audit": {"status": "verified", "requested_device": device, "environment_device": device,
            "physics_sim_view_frontend_device": device, "robot_native_transform_tensor_device": device,
            "cuda_visible_devices": None, "native_and_tensor_device_consistency_verified": True,
            "native_physics_backend_device_ordinal": index, "torch_current_device": index, "physics_cudaDevice_setting": index},
        "physics_clock_readback": {"status": "verified", "decimation": decimation, "simulator_physics_dt_s": dt,
            "environment_physics_dt_s": dt, "net_contact_sensor_update_period_s": dt, "policy_dt_s": .02},
        "physics_dt_s": dt, "physics_clock_refinement": contract["physics_clock_refinement"],
        "contact_material_audit": {"leg_pd_gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]},
            "robot_runtime_friction_range": [.7, .7], "ground_colliders": [{"static_friction": .7, "dynamic_friction": .7}]},
        "physics_trajectory_sha256": runner.base.sha256(folder/"physics_trajectory.npz")}
    write_json(folder/"report.json", report)
    return report


@pytest.mark.parametrize("dt,native,decimation", [(.005, 1397, 4), (.0025, 2793, 8)])
@pytest.mark.parametrize("index", [1, 3, 4, 5])
def test_actual_decisions_native_states_clock_contract_and_selected_device(tmp_path, monkeypatch, dt, native, decimation, index):
    folder = tmp_path/"case"; report = make_output(folder, dt, index, monkeypatch)
    if dt == .005: monkeypatch.setattr(runner, "BASELINE_RAW_SHA256", report["physics_trajectory_sha256"])
    proof = runner.validate_output(folder, physics_dt_s=dt, physical_gpu_index=index)
    assert proof["native_states"] == native and proof["decimation"] == decimation
    assert proof["policy_decisions"] == 349 and proof["duration_s"] == 6.98
    assert proof["native_clock_and_device_verified"] and proof["physical_gpu_index_requested"] == index


@pytest.mark.parametrize("bad", ["device", "sensor_clock", "actor_dim", "pd", "bridge", "material", "dr"])
def test_wrong_native_or_physical_contract_is_rejected(tmp_path, monkeypatch, bad):
    folder = tmp_path/"case"; report = make_output(folder, .0025, 3, monkeypatch)
    if bad == "device": report["runtime_device_audit"]["native_physics_backend_device_ordinal"] = 1
    if bad == "sensor_clock": report["physics_clock_readback"]["net_contact_sensor_update_period_s"] = .005
    if bad == "actor_dim": report["policy_export"]["actor_dim"] = 52
    if bad == "pd": report["contact_material_audit"]["leg_pd_gain_readback"]["kd_range"] = [5., 5.]
    if bad == "bridge": report["contact_timestep_diagnostic"]["final_export_profile_bridge"]["evaluation_task_profile_canonical_sha256"] = "0"*64
    if bad == "material": report["contact_material_audit"]["ground_colliders"][0]["dynamic_friction"] = .15
    if bad == "dr": report["domain_randomization_enabled"] = True
    write_json(folder/"report.json", report)
    with pytest.raises(ValueError): runner.validate_output(folder, .0025, 3)


@pytest.mark.parametrize("bad", ["reset_splice", "counter_gap", "missing_state", "nonfinite", "wrong_count"])
def test_native_data_failures_are_not_physical_passes(tmp_path, monkeypatch, bad):
    folder = tmp_path/"case"; report = make_output(folder, .0025, 3, monkeypatch)
    path = folder/"physics_trajectory.npz"
    with np.load(path) as archive: raw = {name: archive[name] for name in archive.files}
    if bad == "reset_splice": raw["valid_mask"][500, 0] = False
    if bad == "counter_gap": raw["simulation_step_counter"][500] += 1
    if bad == "missing_state": raw["joint_position_rad"] = raw["joint_position_rad"][:-1]
    if bad == "nonfinite": raw["root_position_m"][500, 0, 0] = np.nan
    if bad == "wrong_count": raw["time_s"] = raw["time_s"][:-8]
    np.savez(path, **raw); report["physics_trajectory_sha256"] = runner.base.sha256(path); write_json(folder/"report.json", report)
    with pytest.raises(ValueError): runner.validate_output(folder, .0025, 3)


@pytest.fixture
def args(tmp_path, monkeypatch):
    inputs = tmp_path/"inputs"; inputs.mkdir()
    values = {name: inputs/name for name in ("checkpoint", "guard_source", "motion_file", "asset_path", "runner_config", "task_profile")}
    for path in values.values(): path.write_text("CPU fixture only")
    bm = inputs/"bm"; config = bm/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    config.parent.mkdir(parents=True); config.write_text("# CPU fixture\n")
    original = runner.base.sha256
    monkeypatch.setattr(runner.base, "sha256", lambda path: runner.base.CHECKPOINT_SHA if Path(path) == values["checkpoint"] else original(path))
    monkeypatch.setattr(runner.base, "freeze", lambda args: {"frozen": True})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return NS(**values, legacy_evaluator=LEGACY, beyondmimic_source=bm, python=Path(sys.executable),
        physical_gpu_index=3, output_dir=tmp_path/"fresh")


class Guard:
    write_json = staticmethod(write_json)
    def __init__(self, args, monkeypatch, *, nonzero=False): self.args, self.monkeypatch, self.calls, self.nonzero = args, monkeypatch, 0, nonzero
    def run_child(self, command, log, cwd, environment, spawned):
        self.calls += 1; spawned(6000+self.calls); log.write_text("CPU fixture; no simulator\n")
        assert "CUDA_VISIBLE_DEVICES" not in environment and environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert command[command.index("--device")+1] == "cuda:3" and "--alpha" not in command and "--leg-kd" not in command
        folder = Path(command[command.index("--output-dir")+1]); dt = float(command[command.index("--physics-dt-s")+1])
        report = make_output(folder, dt, 3, self.monkeypatch)
        if dt == .005: self.monkeypatch.setattr(runner, "BASELINE_RAW_SHA256", report["physics_trajectory_sha256"])
        return 7 if self.nonzero else 0


def idle(index): return {"idle": True, "physical_index": index, "uuid": "GPU-fixture", "pci_bus_id": "0000:03:00.0", "pci_bus_id_cuda_ordinal_verified": True}


def test_two_sequential_full_episodes_no_analysis_or_retry(args, monkeypatch):
    guard = Guard(args, monkeypatch); calls = []; sleeps = []
    def probe(index): calls.append(index); return idle(index)
    status = runner.run(args, guard=guard, gpu_probe=probe, quiesce=sleeps.append)
    assert guard.calls == 2 and calls == [3, 3] and sleeps == [2., 2.]
    assert status["status"] == "episodes_complete_requires_contact_and_tracking_review"
    assert [child["verification"]["native_states"] for child in status["children"]] == [1397, 2793]
    assert not status["automatic_gpu_fallback"] and not status["additional_training_updates"]


@pytest.mark.parametrize("busy_at", [1, 2])
def test_fresh_idle_each_episode_no_fallback(args, monkeypatch, busy_at):
    guard = Guard(args, monkeypatch); probes = []
    def probe(index):
        probes.append(index); proof = idle(index); proof["idle"] = len(probes) != busy_at; return proof
    with pytest.raises(ValueError, match="freshly fully idle"): runner.run(args, guard=guard, gpu_probe=probe, quiesce=lambda t: None)
    assert guard.calls == busy_at-1


def test_baseline_identity_gate_stops_before_2p5ms(args, monkeypatch):
    guard = Guard(args, monkeypatch)
    original = runner.validate_output
    def reject_baseline(folder, dt, index):
        monkeypatch.setattr(runner, "BASELINE_RAW_SHA256", "0"*64)
        return original(folder, dt, index)
    monkeypatch.setattr(runner, "validate_output", reject_baseline)
    with pytest.raises(ValueError, match="original54e01"): runner.run(args, guard=guard, gpu_probe=idle, quiesce=lambda t: None)
    assert guard.calls == 1


def test_nonzero_owned_exit_never_retried(args, monkeypatch):
    guard = Guard(args, monkeypatch, nonzero=True)
    with pytest.raises(ValueError, match="Owned evaluator failed"): runner.run(args, guard=guard, gpu_probe=idle, quiesce=lambda t: None)
    assert guard.calls == 1
