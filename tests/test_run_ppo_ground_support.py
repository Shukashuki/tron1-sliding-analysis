"""Injected CPU orchestration fixtures: no real GPU, training or inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_ppo_ground_support as trial


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


@pytest.fixture
def args(tmp_path, monkeypatch):
    directories = {name: tmp_path/name for name in ("snapshot", "source_root", "validation_source_root", "beyondmimic_source", "baseline_root", "baseline_training_dir")}
    for path in directories.values():
        path.mkdir()
    for name in set(trial.PINS)|{"scripts/run_car_validation.py", "scripts/run_ppo_smoothness_sequence.py", "scripts/analyze_ppo_smoothness.py", "scripts/analyze_ppo_contact_stability.py", "scripts/summarize_ppo_smoothness.py"}:
        path = directories["snapshot"]/name; path.parent.mkdir(exist_ok=True); path.write_bytes((ROOT/name).read_bytes())
    fake_sources = {}
    for name in trial.BASELINE_SOURCES:
        path = directories["source_root"]/name; path.parent.mkdir(exist_ok=True); path.write_text("# injected CPU test source\n")
        fake_sources[name] = trial.common.sha256(path)
    monkeypatch.setattr(trial, "BASELINE_SOURCES", fake_sources)
    fake_metadata = {}
    for name in trial.BASELINE_METADATA:
        path = directories["baseline_training_dir"]/name; write(path, {"fixture": name})
        fake_metadata[name] = trial.common.sha256(path)
    monkeypatch.setattr(trial, "BASELINE_METADATA", fake_metadata)
    bm = directories["beyondmimic_source"]/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    bm.parent.mkdir(parents=True); bm.write_text("# injected official-source fixture\n")
    inputs = {}
    for name, constant in (("motion_file", "BASELINE_MOTION_SHA256"), ("reference_file", "REFERENCE_SHA256"),
                           ("asset_path", "BASELINE_ASSET_SHA256"), ("mujoco_model", "MODEL_SHA256"), ("task_profile", "BASELINE_PROFILE_SHA256")):
        path = tmp_path/(name+".input"); path.write_text("synthetic locked "+name); inputs[name] = path
        monkeypatch.setattr(trial, constant, trial.common.sha256(path))
    protocol = json.loads((ROOT/"config/ppo_ground_support_v1.json").read_text())
    protocol["baseline_task_profile_sha256"] = trial.BASELINE_PROFILE_SHA256
    evaluator = directories["validation_source_root"]/"scripts/eval_tracking_friction_isaac.py"
    evaluator.parent.mkdir(); evaluator.write_text("# injected evaluator fixture\n")
    # Keep the real immutable evaluator pin in the declared protocol; only this
    # fake file's digest is mocked. No production hashing code is modified.
    original_sha = trial.common.sha256
    monkeypatch.setattr(trial.common, "sha256", lambda path: protocol["legacy_physx_evaluator_sha256"] if Path(path) == evaluator else original_sha(path))
    protocol_path = tmp_path/"protocol.json"; write(protocol_path, protocol)
    make_evaluation(directories["baseline_root"]/"mu0p700/isaac", inputs["motion_file"], trial.BASELINE_CHECKPOINT_SHA256)
    monkeypatch.setattr(trial, "ROOT", directories["snapshot"])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return argparse.Namespace(**directories, **inputs, protocol=protocol_path, python=Path(sys.executable),
        output_dir=tmp_path/"new_run", run_formal_after_verified_smoke=False)


def make_evaluation(folder, motion, checkpoint):
    folder.mkdir(parents=True)
    (folder/"physics_trajectory.npz").write_bytes(b"injected analyzer fixture, not physical data")
    (folder/"actor_normalized.pt").write_bytes(b"fixture actor never loaded")
    write(folder/"report.json", {"checkpoint_sha256": checkpoint, "motion_file_sha256": trial.common.sha256(motion),
        "physics_trajectory_sha256": trial.common.sha256(folder/"physics_trajectory.npz"),
        "evaluator_source_sha256": "76d78337d70519693bc7e8d819a6523a800e1e4889de5a0466c2d34356d2ae9d",
        "evaluation_contact_friction_override": {"evaluation_contact_friction": .7},
        "domain_randomization_enabled": False, "full_horizon_diagnostic": True,
        "summary": {"episodes": [{"completed_full_reference": True, "start_reference_frame": 0}]},
        "runtime_device_audit": {"status": "verified", "requested_device": "cuda:1", "cuda_visible_devices": None,
            "native_and_tensor_device_consistency_verified": True},
        "contact_material_audit": {"leg_pd_gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]}}})


def runtime_audit(smoke):
    steps, envs = (24, 2) if smoke else (24000, 512)
    return {"steps_started": steps, "steps_completed": steps, "unique_cost_windows": steps,
        "physics_samples_total": 4*steps, "physics_environment_substeps": 4*steps*envs,
        "steps_failed": 0, "samples_last_window": 4, "num_envs": envs, "closed": True,
        "device_health_checked": True, "device_health_finite": True, "cpu_fixture_only": False, "device": "cuda:1",
        "module_sha256": trial.PINS["scripts/ppo_substep_ground_support.py"], "wrapper_sha256": trial.PINS["scripts/train_ppo_ground_support.py"],
        "clock_helper_sha256": trial.PINS["scripts/ppo_leg_torque_smoothness.py"], "physics_dt_s": .005, "policy_dt_s": .02, "decimation": 4,
        "scale_fraction": .2, "raw_reward_bounds": [0., 1.], "external_reward_weight": .4,
        "sensor_names": ["wheel_L_ground", "wheel_R_ground"], "wheel_body_names": ["wheel_L_Link", "wheel_R_Link"],
        "force_matrix_shape_each_sensor": [envs, 1, 1, 3], "ground_filter": {"path": "/World/ground/terrain/GroundPlane/CollisionPlane", "collision_api_verified": True},
        "gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]}, "mass_gravity_reads": 1,
        "mass_kg_range": [22., 22.], "mg_N_range": [215.82, 215.82], "gravity_m_s2": [0., 0., -9.81],
        "positive_normal_force_samples_left_right": [4*steps, 4*steps], "negative_normal_force_samples": 0,
        "observed_ground_fn_z_min_N": [0., 0.], "observed_ground_fn_z_max_N": [500., 600.], "observed_normal_force_xy_abs_peak_N": .01}


def make_training(output, smoke):
    output.mkdir(); steps, envs, iterations = (24, 2, 1) if smoke else (24000, 512, 1000)
    scope_status = "smoke_passed" if smoke else "training_completed"
    reward = {"term": trial.TERM, "weight": .4, "scale_fraction": .2, "additional_reward_term": False}
    write(output/"ablation_manifest.json", {"kind": trial.KIND, "status": "configured_before_environment_creation", "smoke_only": smoke,
        "prepared_method_runtime_verified": False, "trained_policy_claimed": False,
        "initialization": "from_scratch_no_checkpoint_resume", "num_envs": envs, "iterations": iterations, "seed": 42,
        "expected_policy_steps": steps, "expected_environment_transitions": steps*envs,
        "baseline_action_rate_weight": -.5, "reward_replacement": reward})
    write(output/"run_status.json", {"status": scope_status, "environment_transitions": steps*envs, "policy_steps_completed": steps})
    write(output/"manifest.json", {"num_envs": envs, "iterations": iterations, "seed": 42, "setup_smoke_control_steps": 0,
        "environment_transitions_requested": steps*envs, "training_reward_overrides": {"action_rate_l2": -.5, trial.TERM: reward}, "contact_reward_replacement": reward})
    write(output/"runner_config.json", {"num_steps_per_env": 24}); write(output/trial.RUNTIME_FILE, runtime_audit(smoke))
    (output/("model_smoke.pt" if smoke else "model_final.pt")).write_bytes(b"fixture checkpoint never loaded")


def fake_finalizer(output, *, child_returncode):
    path = output/"ablation_manifest.json"; metadata = json.loads(path.read_text())
    if child_returncode != 0:
        metadata.update(status="failed"); write(path, metadata); return metadata
    smoke = metadata["smoke_only"]
    metadata.update(status="smoke_passed" if smoke else "training_completed", trained_policy_claimed=not smoke,
        prepared_method_runtime_verified=True, actual_environment_transitions=metadata["expected_environment_transitions"],
        checkpoint_sha256=trial.common.sha256(output/("model_smoke.pt" if smoke else "model_final.pt")),
        final_training_manifest_sha256=trial.common.sha256(output/"manifest.json"), runtime_audit_sha256=trial.common.sha256(output/trial.RUNTIME_FILE),
        run_status_sha256=trial.common.sha256(output/"run_status.json"))
    write(path, metadata); return metadata


class Children:
    def __init__(self, args, *, bad_audit=False, nonzero=False, mutate_source=False):
        self.args, self.calls = args, []; self.bad_audit, self.nonzero, self.mutate_source = bad_audit, nonzero, mutate_source
    def __call__(self, command, log, cwd, environment, spawned):
        self.calls.append(list(command)); spawned(9000+len(self.calls)); log.write_text("CPU fixture\n")
        assert "CUDA_VISIBLE_DEVICES" not in environment
        assert environment["TRON1_BEYONDMIMIC_ROOT"] == str(self.args.beyondmimic_source)
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        if any(str(x).endswith("train_ppo_ground_support.py") for x in command):
            assert "--action-rate-weight" not in command and "--resume" not in command
            assert command[command.index("--kit_args")+1] == trial.KIT_ARGS
            output = Path(command[command.index("--output-dir")+1]); make_training(output, "--smoke-only" in command)
            if self.bad_audit:
                runtime = runtime_audit(True); runtime["positive_normal_force_samples_left_right"] = [1, 0]; write(output/trial.RUNTIME_FILE, runtime)
        if self.mutate_source:
            (self.args.snapshot/"scripts/train_ppo_ground_support.py").write_text("# changed synthetic source\n")
        return 7 if self.nonzero else 0


def fake_validation(args, training, validation, launch):
    launch([str(args.python), "fake_isaac.py"], args.snapshot, "validation")
    isaac = validation/"mu0p700/isaac"; make_evaluation(isaac, args.motion_file, trial.common.sha256(training/"model_final.pt"))
    launch([str(args.python), "fake_mujoco.py"], args.snapshot, "validation")
    mujoco = validation/"mu0p700/mujoco"; mujoco.mkdir(); (mujoco/"rollout.npz").write_bytes(b"injected rollout")
    write(mujoco/"report.json", {"status": "completed", "completed_full_reference": True,
        "motion_sha256": trial.common.sha256(args.motion_file), "model_sha256": trial.common.sha256(args.mujoco_model),
        "policy_sha256": trial.common.sha256(isaac/"actor_normalized.pt")})


def run_fake(args, *, children=None, busy_at=None):
    children = children or Children(args); probes, sleeps = [], []
    def probe():
        probes.append(len(children.calls)); return {"physical_index": 1, "idle": len(probes) != busy_at}
    def analyzer(path, *, reference, report, high_frequency_hz):
        data = json.loads(report.read_text()); return {"evaluation_mu": .7, "reference_sha256": trial.common.sha256(reference), "checkpoint_sha256": data["checkpoint_sha256"]}
    code = trial.run_trial(args, gpu_probe=probe, child_runner=children, training_finalizer=fake_finalizer,
        validation_runner=fake_validation, trajectory_analyzer=analyzer, contact_analyzer=lambda folder: {"injected": True}, quiesce=sleeps.append)
    status = json.loads((args.output_dir/"ground_support_status.json").read_text())
    return code, status, children, probes, sleeps


def test_default_only_real_scope_smoke_and_no_policy_claim(args):
    code, status, children, probes, sleeps = run_fake(args)
    assert code == 0 and status["status"] == "smoke_passed_requires_explicit_formal_adoption"
    assert len(children.calls) == len(probes) == len(sleeps) == 1
    assert status["smoke"]["environment_transitions"] == 48
    assert "training" not in status and not (args.output_dir/"training").exists()
    assert status["new_policy_quality_claimed"] is status["continuous_contact_certified"] is False


def test_explicit_full_sequence_only_nominal_dual_engine_and_no_auto_success(args):
    args.run_formal_after_verified_smoke = True
    code, status, children, probes, sleeps = run_fake(args)
    assert code == 0 and status["status"] == "requires_visual_and_contact_review"
    assert len(children.calls) == len(probes) == len(sleeps) == 4
    assert sleeps == [2.]*4 and status["protocol"]["validation_mu_in_order"] == [.7]
    assert status["training"]["environment_transitions"] == 12288000
    assert all(child["pid"] >= 9001 and child["status"] == "completed" for child in status["children"])
    assert status["requires_visual_contact_review"] is True


@pytest.mark.parametrize("busy_at", [1, 2, 3, 4])
def test_fresh_idle_guard_before_every_child(args, busy_at):
    args.run_formal_after_verified_smoke = True
    code, status, children, probes, _ = run_fake(args, busy_at=busy_at)
    assert code == 3 and status["status"] == "stopped_gpu_busy"
    assert len(children.calls) == busy_at-1 and len(probes) == busy_at
    assert status["children"][-1]["status"] == "not_started_gpu_busy"


def test_incomplete_force_proof_stops_before_formal(args):
    args.run_formal_after_verified_smoke = True
    code, status, children, _, _ = run_fake(args, children=Children(args, bad_audit=True))
    assert code == 1 and len(children.calls) == 1
    assert "Both real upward" in status["error"]


def test_nonzero_owned_child_is_not_promoted_or_retried(args):
    code, status, children, _, _ = run_fake(args, children=Children(args, nonzero=True))
    assert code == 1 and len(children.calls) == 1
    assert json.loads((args.output_dir/"smoke/ablation_manifest.json").read_text())["status"] == "failed"


def test_source_mutation_stops_without_next_launch(args):
    args.run_formal_after_verified_smoke = True
    code, status, children, _, _ = run_fake(args, children=Children(args, mutate_source=True))
    assert code == 1 and len(children.calls) == 1 and "SourceChangedError" in status["error"]


@pytest.mark.parametrize("key,value", [("physics_samples_total", 95), ("physics_environment_substeps", 191),
    ("device_health_finite", False), ("cpu_fixture_only", True), ("mg_N_range", [0., 0.]),
    ("positive_normal_force_samples_left_right", [5, 0]), ("negative_normal_force_samples", 1),
    ("observed_ground_fn_z_max_N", [10., 0.]), ("observed_ground_fn_z_min_N", [-1., 0.]),
    ("gravity_m_s2", [0., 0., 9.81]), ("gain_readback", {"kp_range": [250., 250.], "kd_range": [10., 10.]})])
def test_smoke_exit_zero_cannot_replace_physical_runtime_proof(key, value):
    runtime = runtime_audit(True); runtime[key] = value
    with pytest.raises(ValueError):
        trial.verify_physical_audit(runtime, smoke=True)


def test_previous_mu015_and_17point_grids_are_rejected(args):
    protocol = json.loads(args.protocol.read_text()); protocol["validation_mu_in_order"] = [.7, .15]; write(args.protocol, protocol)
    with pytest.raises(ValueError, match="validation_mu_in_order"):
        trial.resolve_args(args)


def test_official_beyondmimic_config_and_unmasked_device_are_required(args, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="unset"):
        trial.resolve_args(args)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    (args.beyondmimic_source/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py").unlink()
    with pytest.raises(ValueError, match="BeyondMimic"):
        trial.resolve_args(args)


def test_source_pins_and_existing_output_fail_before_jobs(args):
    (args.snapshot/"scripts/train_ppo_ground_support.py").write_text("# changed fixture\n")
    with pytest.raises(ValueError, match="implementation SHA"):
        trial.resolve_args(args)
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="Preserve existing"):
        trial.resolve_args(args)
