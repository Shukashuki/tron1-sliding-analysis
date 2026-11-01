"""Prepared orchestration tests with injected CPU children; no real jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_ppo_torque_smoothness as trial
import train_ppo_torque_smoothness as trainer
from test_run_ppo_smoothness_sequence import make_evaluation, fake_analyzer


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def args(tmp_path, monkeypatch):
    directories = {name: tmp_path/name for name in
        ("snapshot", "source_root", "validation_source_root", "beyondmimic_source", "baseline_root")}
    for path in directories.values():
        path.mkdir()
    for name in ("train_ppo_torque_smoothness.py", "ppo_leg_torque_smoothness.py", "train_beyondmimic_style.py",
                 "height_smoothness.py", "run_car_validation.py", "analyze_ppo_smoothness.py"):
        path = directories["snapshot"]/"scripts"/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((ROOT/"scripts"/name).read_bytes())
    inputs = {}
    for name in ("motion_file", "reference_file", "asset_path", "mujoco_model", "task_profile"):
        inputs[name] = tmp_path/(name+".input")
        inputs[name].write_text("synthetic locked "+name)
    protocol = json.loads((ROOT/"config/ppo_leg_torque_smoothness_v1.json").read_text())
    protocol["baseline_task_profile_sha256"] = trial.common.sha256(inputs["task_profile"])
    protocol_path = tmp_path/"protocol.json"
    write_json(protocol_path, protocol)
    previous = {"status": "needs_next_method", "finished_unix_time": 100.,
        "protocol": {"method": "action_rate_l2_weight_only", "candidate_action_rate_weights_in_order": [-.3, -.5, -1.]},
        "candidates": [{"action_rate_weight": weight, "status": "failed_engineering_screen",
                        "comparisons": {"0.7": {"passes": True}, "0.15": {"passes": False}}}
                       for weight in (-.3, -.5, -1.)],
        "children": [{"status": "completed"}]}
    previous_path = tmp_path/"previous_sequence/sequence_status.json"
    write_json(previous_path, previous)
    for mu in (.7, .15):
        make_evaluation(directories["baseline_root"]/trial.common.case_name(mu)/"isaac",
                        mu=mu, checkpoint=protocol["baseline_checkpoint_sha256"], motion=inputs["motion_file"])
    monkeypatch.setattr(trial, "ROOT", directories["snapshot"])
    return argparse.Namespace(**directories, **inputs, protocol=protocol_path,
        python=Path(sys.executable), previous_sequence_status=previous_path, output_dir=tmp_path/"new_torque_trial")


def make_training(output, *, smoke, partial_audit=False, partial_training=False):
    output.mkdir()
    steps, envs, iterations = (24, 2, 1) if smoke else (24000, 512, 1000)
    transitions = steps*envs
    status = "smoke_passed" if smoke else "training_completed"
    wrapper = trainer.sha256(trainer.__file__)
    sampler = trainer.sha256(trainer.torque.__file__)
    reward = {"term": trial.TERM, "weight": -.01, "delta_torque_scale_nm": 20.,
              "wheel_cost_included": False, "denominator_after_reset": 4, "first_actual_substep_after_reset_cost": 0.}
    ablation = {"kind": "historical_physx_ppo_leg_applied_torque_rate_ablation",
        "status": "configured_before_environment_creation", "prepared_method_runtime_verified": False,
        "trained_policy_claimed": False, "smoke_only": smoke, "initialization": "from_scratch_no_checkpoint_resume",
        "baseline_action_rate_weight": -.1, "additional_reward": reward,
        "expected_policy_steps": steps, "expected_environment_transitions": transitions,
        "num_envs": envs, "iterations": iterations, "seed": 42,
        "wrapper_sha256": wrapper, "sampler_module_sha256": sampler}
    write_json(output/"ablation_manifest.json", ablation)
    write_json(output/"run_status.json", {"status": "training" if partial_training else status,
        "trained_policy_claimed": not smoke, "environment_transitions": 100 if partial_training else transitions})
    write_json(output/"manifest.json", {"num_envs": envs, "iterations": iterations, "seed": 42,
        "setup_smoke_control_steps": 0, "environment_transitions_requested": transitions})
    write_json(output/"runner_config.json", {"num_steps_per_env": 24})
    (output/("model_smoke.pt" if smoke else "model_final.pt")).write_bytes(b"synthetic checkpoint, never inferred")
    write_json(output/"torque_rate_runtime_audit.json", {"module_sha256": sampler, "wrapper_sha256": wrapper,
        "steps_started": steps, "steps_completed": steps, "unique_cost_windows": steps,
        "physics_samples_total": 4*steps-1 if partial_audit else 4*steps,
        "steps_failed": 0, "samples_last_window": 4, "device_health_finite": True, "closed": True,
        "num_envs": envs, "delta_torque_scale_nm": 20., "wheel_cost_included": False})


class FakeChildren:
    def __init__(self, args, *, partial_audit=False, partial_training=False, nonzero=False,
                 change_source=False, change_previous=False):
        self.args, self.calls = args, []
        self.partial_audit, self.partial_training, self.nonzero = partial_audit, partial_training, nonzero
        self.change_source, self.change_previous = change_source, change_previous

    def __call__(self, command, log_path, cwd, environment, spawned):
        assert environment.get("CUDA_VISIBLE_DEVICES") is None
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        self.calls.append(list(command)); spawned(8000+len(self.calls))
        log_path.write_text("CPU fake child log\n")
        if any(str(part).endswith("train_ppo_torque_smoothness.py") for part in command):
            assert "--action-rate-weight" not in command
            assert command[command.index("--leg-torque-rate-weight")+1] == "-0.01"
            assert command[command.index("--device")+1] == "cuda:1"
            assert command[command.index("--smoke-steps")+1] == "0"
            smoke = "--smoke-only" in command
            output = Path(command[command.index("--output-dir")+1])
            make_training(output, smoke=smoke, partial_audit=self.partial_audit,
                          partial_training=self.partial_training and not smoke)
        if self.change_source:
            (self.args.snapshot/"scripts/ppo_leg_torque_smoothness.py").write_text("changed frozen synthetic source\n")
        if self.change_previous:
            previous = json.loads(self.args.previous_sequence_status.read_text())
            previous["finished_unix_time"] += 1.
            write_json(self.args.previous_sequence_status, previous)
        return 7 if self.nonzero else 0


def fake_validation(args, training_output, validation_output, launch_checked):
    for mu in (.7, .15):
        launch_checked([str(args.python), "fake_isaac.py", "--device", "cuda:1"], args.snapshot, "validation")
        make_evaluation(validation_output/trial.common.case_name(mu)/"isaac", mu=mu,
            checkpoint=trial.common.sha256(training_output/"model_final.pt"), motion=args.motion_file, weight=-.01)
        launch_checked([str(args.python), "fake_mujoco.py"], args.snapshot, "validation")


def run_fake(args, *, passes=True, probe=None, children=None):
    children = children or FakeChildren(args)
    probes, sleeps = [], []
    def gpu():
        probes.append(len(children.calls))
        return probe(len(probes)) if probe else {"physical_index": 1, "idle": True}
    def compare(candidate, baseline, **kwargs):
        assert candidate["evaluation_mu"] == baseline["evaluation_mu"]
        assert kwargs == {"reduction_fraction": .5, "xy_max_rmse_m": .15, "xy_extra_rmse_m": .03,
                          "speed_retention_min": .9, "beta_extra_rmse_deg": 5.}
        return {"passes": passes, "status": "passes_unvalidated_screen" if passes else "fails_unvalidated_screen"}
    code = trial.run_trial(args, gpu_probe=gpu, child_runner=children, validation_runner=fake_validation,
        trajectory_analyzer=fake_analyzer, candidate_comparator=compare, quiesce=sleeps.append)
    status = json.loads((args.output_dir/"torque_trial_status.json").read_text())
    return code, status, children, probes, sleeps


def test_prepared_trial_smoke_formal_and_dual_engine_children_are_strictly_sequential(args):
    previous = args.previous_sequence_status.read_bytes()
    code, status, children, probes, sleeps = run_fake(args)
    assert code == 0 and status["status"] == "screen_passed_requires_fullbody_and_translation_review"
    assert len(children.calls) == len(probes) == len(sleeps) == 6
    assert sleeps == [2.]*6
    assert status["smoke"]["environment_transitions"] == 48
    assert status["training"]["environment_transitions"] == 12288000
    assert status["smoke"]["physical_sampling_verified"] and status["training"]["physical_sampling_verified"]
    assert all(item["status"] == "completed" and item["pid"] >= 8001 for item in status["children"])
    assert not status["jitter_resolved_claimed"] and not status["whole_body_jitter_reduction_claimed"]
    assert status["requires_fullbody_visual_review"] and status["requires_translation_and_root_height_hf_diagnostics"]
    assert args.previous_sequence_status.read_bytes() == previous
    frozen = json.loads((args.output_dir/"frozen_source_and_inputs.json").read_text())
    assert frozen["previous_sequence_status"]["sha256"] == trial.common.sha256(args.previous_sequence_status)
    assert "action_rate_" not in str(args.output_dir)


def test_failed_screen_is_needs_next_method_not_jitter_complete(args):
    code, status, children, _, _ = run_fake(args, passes=False)
    assert code == 4 and status["status"] == "needs_next_method"
    assert len(children.calls) == 6 and not status["jitter_resolved_claimed"]


@pytest.mark.parametrize("status", ["running_training", "running_validation", "screen_passed_requires_visual_review", "failed"])
def test_previous_not_ready_stops_before_output_creation_or_gpu_probe(args, status):
    previous = json.loads(args.previous_sequence_status.read_text()); previous["status"] = status
    write_json(args.previous_sequence_status, previous)
    with pytest.raises(ValueError, match="terminal needs_next_method"):
        trial.run_trial(args, gpu_probe=lambda: pytest.fail("No GPU check before previous method completes"))
    assert not args.output_dir.exists()


def test_missing_previous_trial_evidence_cannot_be_promoted_by_terminal_string(args):
    previous = json.loads(args.previous_sequence_status.read_text()); previous["candidates"].pop()
    write_json(args.previous_sequence_status, previous)
    with pytest.raises(ValueError, match="all three declared trials"):
        trial.run_trial(args)
    assert not args.output_dir.exists()


def test_busy_gpu_stops_before_any_child_without_retry(args):
    code, status, children, probes, sleeps = run_fake(args, probe=lambda _: {"physical_index": 1, "idle": False})
    assert code == 3 and status["status"] == "stopped_gpu_busy"
    assert children.calls == sleeps == [] and len(probes) == 1


def test_busy_gpu_before_validation_preserves_completed_training(args):
    code, status, children, probes, _ = run_fake(args,
        probe=lambda count: {"physical_index": 1, "idle": count <= 2})
    assert code == 3 and len(children.calls) == 2 and len(probes) == 3
    assert status["training"]["status"] == "training_completed"
    assert status["comparisons"] == {}


def test_partial_smoke_physical_audit_blocks_formal_training(args):
    children = FakeChildren(args, partial_audit=True)
    code, status, _, _, _ = run_fake(args, children=children)
    assert code == 1 and status["status"] == "failed" and len(children.calls) == 1
    assert not (args.output_dir/"training").exists()
    metadata = json.loads((args.output_dir/"smoke/ablation_manifest.json").read_text())
    assert metadata["trained_policy_claimed"] is False
    assert metadata["prepared_method_runtime_verified"] is False


def test_partial_formal_training_is_not_a_completed_new_method(args):
    children = FakeChildren(args, partial_training=True)
    code, status, _, _, _ = run_fake(args, children=children)
    assert code == 1 and status["status"] == "failed" and len(children.calls) == 2
    assert "training" not in status and not (args.output_dir/"validation").exists()


def test_nonzero_owned_child_is_finalized_as_failure_even_with_complete_fake_artifacts(args):
    children = FakeChildren(args, nonzero=True)
    code, status, _, _, _ = run_fake(args, children=children)
    assert code == 1 and len(children.calls) == 1 and status["children"][0]["returncode"] == 7
    ablation = json.loads((args.output_dir/"smoke/ablation_manifest.json").read_text())
    assert ablation["status"] == "failed" and ablation["trained_policy_claimed"] is False


@pytest.mark.parametrize("change", ["source", "previous"])
def test_frozen_source_or_previous_terminal_evidence_change_stops_next_job(args, change):
    children = FakeChildren(args, change_source=change == "source", change_previous=change == "previous")
    code, status, _, _, _ = run_fake(args, children=children)
    assert code == 1 and len(children.calls) == 1
    assert "SourceChangedError" in status["error"]
    assert not (args.output_dir/"training").exists()


def test_existing_output_and_changed_screen_are_rejected(args):
    args.output_dir.mkdir()
    with pytest.raises(ValueError, match="fresh torque output"):
        trial.resolve_args(args)
    args.output_dir.rmdir()
    protocol = json.loads(args.protocol.read_text()); protocol["screening"]["maximum_core_xy_rmse_m"] = 1.
    write_json(args.protocol, protocol)
    with pytest.raises(ValueError, match="Predeclared screen changed"):
        trial.resolve_args(args)


def test_cli_requires_previous_sequence_status():
    with pytest.raises(SystemExit):
        trial.parse_args([])

