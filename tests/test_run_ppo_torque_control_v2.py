"""Injected CPU dispatcher lifecycle tests, never GPU/jobs/Isaac."""
import argparse
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_ppo_torque_control_v2 as trial
from test_train_ppo_torque_control_v2 import runtime


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def evaluation(folder, motion, checkpoint):
    folder.mkdir(parents=True); (folder/"physics_trajectory.npz").write_bytes(b"fixture raw, never simulated")
    (folder/"actor_normalized.pt").write_bytes(b"fixture actor, never loaded")
    write(folder/"report.json", {"checkpoint_sha256": checkpoint, "motion_file_sha256": trial.common.sha256(motion),
        "physics_trajectory_sha256": trial.common.sha256(folder/"physics_trajectory.npz"),
        "evaluator_source_sha256": trial.LEGACY_EVALUATOR_SHA256,
        "evaluation_contact_friction_override": {"evaluation_contact_friction": .7},
        "domain_randomization_enabled": False, "full_horizon_diagnostic": True,
        "summary": {"episodes": [{"completed_full_reference": True, "start_reference_frame": 0}]},
        "runtime_device_audit": {"status": "verified", "requested_device": "cuda:1", "cuda_visible_devices": None,
            "native_and_tensor_device_consistency_verified": True},
        "contact_material_audit": {"leg_pd_gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]}}})


@pytest.fixture
def args(tmp_path, monkeypatch):
    dirs = {name: tmp_path/name for name in ("snapshot", "source_root", "validation_source_root", "beyondmimic_source", "baseline_root", "baseline_training_dir")}
    for path in dirs.values(): path.mkdir()
    for name in ("train_ppo_torque_control_v2.py", "ppo_leg_torque_smoothness.py", "train_beyondmimic_style.py", "height_smoothness.py",
        "train_ppo_smoothness.py", "run_car_validation.py", "run_ppo_smoothness_sequence.py", "analyze_ppo_smoothness.py", "analyze_ppo_contact_stability.py"):
        path = dirs["snapshot"]/"scripts"/name; path.parent.mkdir(exist_ok=True); path.write_bytes((ROOT/"scripts"/name).read_bytes())
    sources = {}
    for name in trial.trainer.BASELINE_SOURCES:
        path = dirs["source_root"]/name; path.parent.mkdir(exist_ok=True); path.write_text("# injected historical source\n")
        sources[name] = trial.common.sha256(path)
    monkeypatch.setattr(trial.trainer, "BASELINE_SOURCES", sources)
    metadata = {}
    for name in trial.trainer.BASELINE_METADATA:
        path = dirs["baseline_training_dir"]/name; write(path, {"fixture": name}); metadata[name] = trial.common.sha256(path)
    monkeypatch.setattr(trial.trainer, "BASELINE_METADATA", metadata)
    bm = dirs["beyondmimic_source"]/"source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    bm.parent.mkdir(parents=True); bm.write_text("# injected official source\n")
    inputs = {}
    for name, owner, constant in (("motion_file", trial.trainer, "BASELINE_MOTION_SHA256"),
        ("asset_path", trial.trainer, "BASELINE_ASSET_SHA256"), ("task_profile", trial.trainer, "BASELINE_PROFILE_SHA256"),
        ("reference_file", trial, "REFERENCE_SHA256"), ("mujoco_model", trial, "MODEL_SHA256")):
        path = tmp_path/(name+".input"); path.write_text("synthetic locked input "+name); inputs[name] = path
        monkeypatch.setattr(owner, constant, trial.common.sha256(path))
    protocol = json.loads((ROOT/"config/ppo_torque_control_v2.json").read_text())
    protocol["baseline_task_profile_sha256"] = trial.trainer.BASELINE_PROFILE_SHA256
    protocol_path = tmp_path/"protocol.json"; write(protocol_path, protocol)
    evaluator = dirs["validation_source_root"]/"scripts/eval_tracking_friction_isaac.py"
    evaluator.parent.mkdir(); evaluator.write_text("# injected evaluator fixture\n")
    original_sha = trial.common.sha256
    monkeypatch.setattr(trial.common, "sha256", lambda path: trial.LEGACY_EVALUATOR_SHA256 if Path(path) == evaluator else original_sha(path))
    evaluation(dirs["baseline_root"]/"mu0p700/isaac", inputs["motion_file"], trial.trainer.BASELINE_CHECKPOINT_SHA256)
    monkeypatch.setattr(trial, "ROOT", dirs["snapshot"]); monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return argparse.Namespace(**dirs, **inputs, protocol=protocol_path, python=Path(sys.executable),
        output_dir=tmp_path/"fresh", run_formal_after_verified_smoke=False)


def make_training(output, *, smoke):
    output.mkdir(); steps, envs, iterations = (24, 2, 1) if smoke else (24000, 512, 1000)
    reward = {"term": trial.trainer.TERM_NAME, "weight": -.05, "additional_reward_term": True}
    write(output/"ablation_manifest.json", {"kind": trial.trainer.KIND, "status": "configured_before_environment_creation",
        "smoke_only": smoke, "trained_policy_claimed": False, "prepared_method_runtime_verified": False,
        "baseline_checkpoint_used_for_initialization": False, "additional_reward": reward})
    write(output/"run_status.json", {"status": "smoke_passed" if smoke else "training_completed",
        "environment_transitions": steps*envs, "policy_steps_completed": steps})
    write(output/"manifest.json", {"environment_transitions_requested": steps*envs, "num_envs": envs, "iterations": iterations,
        "seed": 42, "setup_smoke_control_steps": 0, "training_reward_overrides": {"action_rate_l2": -.5},
        "additional_training_reward_terms": {trial.trainer.TERM_NAME: reward}, "original_grounded_reward_preserved": True})
    write(output/"runner_config.json", {"num_steps_per_env": 24}); write(output/trial.trainer.RUNTIME_FILE, runtime(smoke))
    (output/("model_smoke.pt" if smoke else "model_final.pt")).write_bytes(b"fixture model never loaded")


def finalizer(output, *, child_returncode):
    path = output/"ablation_manifest.json"; ablation = json.loads(path.read_text())
    if child_returncode != 0:
        ablation.update(status="failed"); write(path, ablation); return ablation
    smoke = ablation["smoke_only"]
    ablation.update(status="smoke_passed" if smoke else "training_completed", trained_policy_claimed=not smoke,
        prepared_method_runtime_verified=True, actual_environment_transitions=48 if smoke else 12288000,
        checkpoint_sha256=trial.common.sha256(output/("model_smoke.pt" if smoke else "model_final.pt")),
        final_training_manifest_sha256=trial.common.sha256(output/"manifest.json"),
        runtime_audit_sha256=trial.common.sha256(output/trial.trainer.RUNTIME_FILE), run_status_sha256=trial.common.sha256(output/"run_status.json"))
    write(path, ablation); return ablation


class Children:
    def __init__(self, args, *, bad=False, failure=False, mutate=False):
        self.args, self.calls, self.bad, self.failure, self.mutate = args, [], bad, failure, mutate
    def __call__(self, command, log, cwd, environment, spawned):
        self.calls.append(list(command)); spawned(9000+len(self.calls)); log.write_text("CPU fixture only\n")
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and "CUDA_VISIBLE_DEVICES" not in environment
        assert environment["TRON1_BEYONDMIMIC_ROOT"] == str(self.args.beyondmimic_source)
        if any(str(x).endswith("train_ppo_torque_control_v2.py") for x in command):
            assert "--resume" not in command and "--action-rate-weight" not in command
            assert command[command.index("--leg-torque-rate-weight")+1] == "-0.05"
            assert command[command.index("--device")+1] == "cuda:1"
            assert command[command.index("--smoke-steps")+1] == "0"
            output = Path(command[command.index("--output-dir")+1]); make_training(output, smoke="--smoke-only" in command)
            if self.bad:
                audit = runtime(); audit["physics_samples_total"] = 95; write(output/trial.trainer.RUNTIME_FILE, audit)
        if self.mutate: (self.args.snapshot/"scripts/train_ppo_torque_control_v2.py").write_text("# changed source\n")
        return 7 if self.failure else 0


def validation(args, training, output, launch):
    launch([str(args.python), "fake_isaac.py"], args.snapshot, "validation")
    ia = output/"mu0p700/isaac"; evaluation(ia, args.motion_file, trial.common.sha256(training/"model_final.pt"))
    launch([str(args.python), "fake_mujoco.py"], args.snapshot, "validation")
    mj = output/"mu0p700/mujoco"; mj.mkdir(); (mj/"rollout.npz").write_bytes(b"fixture rollout not simulation")
    write(mj/"report.json", {"status": "completed", "completed_full_reference": True, "hidden_resets": 0, "root_state_writes": 1,
        "motion_sha256": trial.common.sha256(args.motion_file), "model_sha256": trial.common.sha256(args.mujoco_model),
        "policy_sha256": trial.common.sha256(ia/"actor_normalized.pt")})


def run(args, *, children=None, busy_at=None):
    children = children or Children(args); probes, sleeps = [], []
    def probe(): probes.append(True); return {"physical_index": 1, "idle": len(probes) != busy_at}
    def analyzer(path, *, reference, report, high_frequency_hz): return {"injected_cpu_analysis": True}
    code = trial.run_trial(args, gpu_probe=probe, child_runner=children, validation_runner=validation,
        training_finalizer=finalizer, trajectory_analyzer=analyzer, contact_analyzer=lambda path: {"injected": True}, quiesce=sleeps.append)
    status = json.loads((args.output_dir/"torque_control_v2_status.json").read_text())
    return code, status, children, probes, sleeps


def test_default_smoke_only_cannot_implicitly_adopt_formal(args):
    code, status, children, probes, sleeps = run(args)
    assert code == 0 and status["status"] == "smoke_passed_requires_explicit_formal_adoption"
    assert len(children.calls) == len(probes) == 1 and sleeps == [2.]
    assert "training" not in status and not (args.output_dir/"training").exists()
    assert not status["new_policy_quality_claimed"] and not status["continuous_contact_certified"]


def test_manual_formal_smoke_then_exact_full_then_nominal_dual_engine(args):
    args.run_formal_after_verified_smoke = True
    code, status, children, probes, sleeps = run(args)
    assert code == 0 and status["status"] == "requires_visual_and_contact_review"
    assert len(children.calls) == len(probes) == 4 and sleeps == [2.]*4
    assert status["training"]["environment_transitions"] == 12288000
    assert status["protocol"]["validation_mu_in_order"] == [.7]
    assert status["training"]["original_grounded_preserved"]


@pytest.mark.parametrize("busy_at", [1, 2, 3, 4])
def test_fresh_gpu_guard_for_every_owned_child_no_substitution_or_retry(args, busy_at):
    args.run_formal_after_verified_smoke = True
    code, status, children, probes, _ = run(args, busy_at=busy_at)
    assert code == 3 and status["status"] == "stopped_gpu_busy"
    assert len(children.calls) == busy_at-1 and len(probes) == busy_at
    assert status["children"][-1]["status"] == "not_started_gpu_busy"


@pytest.mark.parametrize("option", ["bad", "failure", "mutate"])
def test_bad_smoke_never_launches_formal(args, option):
    args.run_formal_after_verified_smoke = True
    children = Children(args, **{option: True})
    code, status, children, _, _ = run(args, children=children)
    assert code == 1 and status["status"] == "failed" and len(children.calls) == 1
    assert not (args.output_dir/"training").exists()


def test_fresh_output_required_no_mutation_existing_results(args):
    args.output_dir.mkdir(); (args.output_dir/"keep.txt").write_text("preserved")
    with pytest.raises(ValueError, match="fresh"): run(args)
    assert (args.output_dir/"keep.txt").read_text() == "preserved"


def test_domain_masks_fail_closed(args, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="unmasked"): run(args)


def test_only_nominal_protocol_no_old_mu0p15_automatic_extension(args):
    protocol = json.loads(args.protocol.read_text()); protocol["validation_mu_in_order"] = [.7, .15]; write(args.protocol, protocol)
    with pytest.raises(ValueError, match="protocol"): run(args)
