"""Fake-GPU/child tests: no simulator, training, SSH or remote access occurs."""
import argparse
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_ppo_smoothness_sequence as sequence


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_evaluation(directory, *, mu, checkpoint, motion, weight=None):
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory/"physics_trajectory.npz"
    raw.write_bytes(b"fake raw measured trajectory for an injected analyzer")
    report = {"checkpoint_sha256": checkpoint, "motion_file_sha256": sequence.sha256(motion),
              "physics_trajectory_sha256": sequence.sha256(raw),
              "evaluation_contact_friction_override": {"evaluation_contact_friction": mu},
              "domain_randomization_enabled": False, "full_horizon_diagnostic": True,
              "summary": {"episodes": [{"completed_full_reference": True, "start_reference_frame": 0}]},
              "runtime_device_audit": {"status": "verified", "requested_device": "cuda:1",
                                        "cuda_visible_devices": None,
                                        "native_and_tensor_device_consistency_verified": True},
              "test_weight": weight}
    write_json(directory/"report.json", report)


@pytest.fixture
def args(tmp_path, monkeypatch):
    directories = {name: tmp_path/name for name in
                   ("snapshot", "source_root", "validation_source_root", "beyondmimic_source", "baseline_root")}
    for path in directories.values():
        path.mkdir()
    for name in ("train_ppo_smoothness.py", "train_beyondmimic_style.py",
                 "run_car_validation.py", "analyze_ppo_smoothness.py", "height_smoothness.py"):
        path = directories["snapshot"]/"scripts"/name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# fake child source, never imported\n")
    for name in ("scripts/train_tracking.py", "training/tron1_tracking.py", "training/tron1_drift.py",
                 "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py"):
        path = directories["source_root"]/name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# fake historical source\n")
    inputs = {}
    for name in ("motion_file", "reference_file", "asset_path", "mujoco_model", "task_profile"):
        inputs[name] = tmp_path/(name+".input")
        inputs[name].write_text("locked "+name)
    protocol = json.loads((ROOT/"config/ppo_smoothness_sequence_v1.json").read_text())
    protocol["baseline_task_profile_sha256"] = sequence.sha256(inputs["task_profile"])
    protocol_path = tmp_path/"protocol.json"
    write_json(protocol_path, protocol)
    monkeypatch.setattr(sequence, "ROOT", directories["snapshot"])
    for mu in (.7, .15):
        make_evaluation(directories["baseline_root"]/sequence.case_name(mu)/"isaac",
                        mu=mu, checkpoint=protocol["baseline_checkpoint_sha256"], motion=inputs["motion_file"])
    return argparse.Namespace(**directories, **inputs, protocol=protocol_path,
                              python=Path(sys.executable), output_dir=tmp_path/"fresh_output")


def make_training(output, *, weight, smoke, protocol):
    output.mkdir()
    transitions = 48 if smoke else 12288000
    status = "smoke_passed" if smoke else "training_completed"
    manifest = {"num_envs": 2 if smoke else 512, "iterations": 1 if smoke else 1000,
                "environment_transitions_requested": transitions, "setup_smoke_control_steps": 0,
                "seed": 42, "smoke_only": smoke,
                "task_profile_sha256": protocol["baseline_task_profile_sha256"],
                "training_reward_overrides": {"action_rate_l2": weight}}
    write_json(output/"manifest.json", manifest)
    write_json(output/"run_status.json", {"status": status, "environment_transitions": transitions,
                                         "trained_policy_claimed": not smoke})
    write_json(output/"runner_config.json", {"num_steps_per_env": 24})
    checkpoint = output/("model_smoke.pt" if smoke else "model_final.pt")
    checkpoint.write_bytes(("new from scratch "+str(weight)).encode())
    ablation = {"status": status, "initialization": "from_scratch_no_checkpoint_resume",
                "smoke_only": smoke, "trained_policy_claimed": not smoke,
                "expected_environment_transitions": transitions, "actual_environment_transitions": transitions,
                "reward_override": {"term": "action_rate_l2", "baseline_weight": -.1, "weight": weight,
                                    "applied_before_environment_creation": True},
                "checkpoint_sha256": sequence.sha256(checkpoint),
                "final_training_manifest_sha256": sequence.sha256(output/"manifest.json")}
    write_json(output/"ablation_manifest.json", ablation)


class FakeChildren:
    def __init__(self, args, *, partial_training=False, mutate_source=False,
                 unfinalized=False, nonzero=False, bad_function=False):
        self.args = args
        self.calls = []
        self.partial_training = partial_training
        self.mutate_source = mutate_source
        self.unfinalized = unfinalized
        self.nonzero = nonzero
        self.bad_function = bad_function

    def __call__(self, command, log_path, cwd, environment, on_spawn):
        assert environment.get("CUDA_VISIBLE_DEVICES") is None
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert environment["TRON1_BEYONDMIMIC_ROOT"] == str(self.args.beyondmimic_source)
        self.calls.append(command)
        log_path.write_text("fake child preserved log\n")
        on_spawn(7000+len(self.calls))
        if any(str(part).endswith("train_ppo_smoothness.py") for part in command):
            output = Path(command[command.index("--output-dir")+1])
            weight = float(command[command.index("--action-rate-weight")+1])
            smoke = "--smoke-only" in command
            assert command[command.index("--device")+1] == "cuda:1"
            assert command[command.index("--smoke-steps")+1] == "0"
            make_training(output, weight=weight, smoke=smoke, protocol=self.args.protocol_data)
            if self.unfinalized:
                ablation = json.loads((output/"ablation_manifest.json").read_text())
                ablation.update(status="configured_before_environment_creation", trained_policy_claimed=False,
                                kind="historical_physx_ppo_action_rate_ablation", algorithm="PPO",
                                training_engine="PhysX", seed=42, num_envs=2 if smoke else 512,
                                iterations=1 if smoke else 1000,
                                historical_trainer_sha256=sequence.sha256(self.args.snapshot/"scripts/train_beyondmimic_style.py"),
                                height_adapter_sha256=sequence.sha256(self.args.snapshot/"scripts/height_smoothness.py"),
                                task_profile_sha256=sequence.sha256(self.args.task_profile),
                                motion_sha256=sequence.sha256(self.args.motion_file),
                                asset_sha256=sequence.sha256(self.args.asset_path),
                                wrapper_sha256=sequence.sha256(self.args.snapshot/"scripts/train_ppo_smoothness.py"),
                                DR_recipe_modified=False, base_task_profile_file_modified=False,
                                baseline_checkpoint_modified=False)
                ablation["reward_override"].update(
                    function="wrong.function" if self.bad_function else "isaaclab.envs.mdp.rewards.action_rate_l2",
                    divided_by_control_dt=False)
                for name in ("checkpoint_sha256", "actual_environment_transitions", "final_training_manifest_sha256"):
                    ablation.pop(name, None)
                source_names = ("scripts/train_tracking.py", "training/tron1_tracking.py", "training/tron1_drift.py",
                                "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py")
                sources = {name: sequence.sha256(self.args.source_root/name) for name in source_names}
                ablation["baseline_source_sha256"] = sources
                manifest = json.loads((output/"manifest.json").read_text())
                manifest.pop("training_reward_overrides")
                manifest["source_training_modules_sha256"] = sources
                write_json(output/"manifest.json", manifest)
                write_json(output/"ablation_manifest.json", ablation)
            if self.partial_training and not smoke:
                write_json(output/"run_status.json", {"status": "training", "environment_transitions": 100})
        if self.mutate_source:
            (self.args.snapshot/"scripts/train_ppo_smoothness.py").write_text("changed after first child\n")
        return 1 if self.nonzero else 0


def fake_validation(args, training_output, validation_output, launch_checked):
    weight = json.loads((training_output/"ablation_manifest.json").read_text())["reward_override"]["weight"]
    for mu in args.protocol_data["validation_mu_in_order"]:
        directory = validation_output/sequence.case_name(mu)/"isaac"
        # Both physical-engine children get their own fresh idle observation.
        launch_checked([str(args.python), "fake_isaac.py", "--device", "cuda:1"], args.snapshot, "validation")
        make_evaluation(directory, mu=mu, checkpoint=sequence.sha256(training_output/"model_final.pt"),
                        motion=args.motion_file, weight=weight)
        launch_checked([str(args.python), "fake_mujoco.py"], args.snapshot, "validation")


def fake_analyzer(path, *, reference, report, high_frequency_hz):
    recorded = json.loads(Path(report).read_text())
    return {"evaluation_mu": recorded["evaluation_contact_friction_override"]["evaluation_contact_friction"],
            "checkpoint_sha256": recorded["checkpoint_sha256"],
            "reference_sha256": sequence.sha256(reference), "test_weight": recorded["test_weight"]}


def run_fake(args, *, passes=(), gpu_probe=None, children=None):
    children = children or FakeChildren(args)
    probes, sleeps = [], []

    def probe():
        probes.append(len(children.calls))
        return gpu_probe(len(probes)) if gpu_probe else {"physical_index": 1, "idle": True}

    def compare(candidate, baseline, **kwargs):
        assert candidate["evaluation_mu"] == baseline["evaluation_mu"]
        assert kwargs["reduction_fraction"] == .5
        assert kwargs["xy_max_rmse_m"] == .15
        assert kwargs["beta_extra_rmse_deg"] == 5.
        return {"passes": candidate["test_weight"] in passes}

    code = sequence.run_sequence(args, gpu_probe=probe, child_runner=children,
                                 validation_runner=fake_validation,
                                 trajectory_analyzer=fake_analyzer,
                                 candidate_comparator=compare, quiesce=sleeps.append)
    status = json.loads((args.output_dir/"sequence_status.json").read_text())
    return code, status, children, probes, sleeps


def test_stops_after_first_passing_weight_and_preserves_earlier_failure(args):
    code, status, children, probes, sleeps = run_fake(args, passes=(-.5,))
    assert code == 0
    assert status["status"] == "screen_passed_requires_visual_review"
    assert status["selected_action_rate_weight"] == -.5
    assert [case["status"] for case in status["candidates"]] == ["failed_engineering_screen", "passed_engineering_screen"]
    assert len(children.calls) == len(probes) == len(sleeps) == 12
    assert sleeps == [2.]*12
    assert all(case["pid"] >= 7001 for case in status["children"])
    assert all(case["status"] == "completed" for case in status["children"])
    assert not status["jitter_resolved_claimed"]
    assert not (args.output_dir/sequence.candidate_name(-1.)).exists()


def test_no_weight_pass_is_needs_next_method_not_completed(args):
    code, status, children, probes, _ = run_fake(args)
    assert code == 4
    assert status["status"] == "needs_next_method"
    assert len(children.calls) == len(probes) == 18
    assert len(status["candidates"]) == 3


def test_gpu_busy_before_first_child_is_fail_closed(args):
    code, status, children, _, sleeps = run_fake(
        args, gpu_probe=lambda _: {"physical_index": 1, "idle": False})
    assert code == 3
    assert status["status"] == "stopped_gpu_busy"
    assert children.calls == sleeps == []
    assert status["children"][0]["status"] == "not_started_gpu_busy"


def test_fresh_probe_blocks_validation_if_other_gpu_work_appears(args):
    code, status, children, probes, _ = run_fake(
        args, gpu_probe=lambda count: {"physical_index": 1, "idle": count <= 2})
    assert code == 3
    assert len(children.calls) == 2  # Only owned smoke and full train ran.
    assert len(probes) == 3
    assert status["children"][-1]["phase"] == "validation"


def test_existing_output_is_not_overwritten(args):
    args.output_dir.mkdir()
    original = args.output_dir/"keep.txt"
    original.write_text("existing user evidence")
    with pytest.raises(FileExistsError):
        run_fake(args)
    assert original.read_text() == "existing user evidence"


def test_source_mutation_after_child_stops_without_next_launch(args):
    children = FakeChildren(args, mutate_source=True)
    code, status, children, _, _ = run_fake(args, children=children)
    assert code == 1
    assert len(children.calls) == 1
    assert "SourceChangedError" in status["error"]


def test_partial_training_cannot_be_validated_or_accepted(args):
    children = FakeChildren(args, partial_training=True)
    code, status, children, _, _ = run_fake(args, children=children)
    assert code == 1
    assert len(children.calls) == 2
    assert "collection budget" in status["error"]
    assert status["candidates"][0]["comparisons"] == {}


def test_parent_finalizes_successful_isaac_exit_without_promoting_smoke_to_policy(args):
    children = FakeChildren(args, unfinalized=True)
    code, status, _, _, _ = run_fake(args, children=children, passes=(-.3,))
    assert code == 0
    train_children = [child for child in status["children"] if child["phase"] in ("smoke", "training")]
    assert len(train_children) == 2
    assert all(child["parent_metadata_finalization"]["performed"] for child in train_children)
    assert all(child["parent_metadata_finalization"]["metadata_finalize_location"] == "parent_after_isaac_exit"
               for child in train_children)
    candidate = args.output_dir/"action_rate_0p3"
    smoke = json.loads((candidate/"smoke/ablation_manifest.json").read_text())
    formal = json.loads((candidate/"training/ablation_manifest.json").read_text())
    assert smoke["status"] == "smoke_passed" and smoke["trained_policy_claimed"] is False
    assert smoke["actual_environment_transitions"] == 48
    assert formal["status"] == "training_completed" and formal["trained_policy_claimed"] is True
    assert formal["actual_environment_transitions"] == 12288000
    for directory in (candidate/"smoke", candidate/"training"):
        ablation = json.loads((directory/"ablation_manifest.json").read_text())
        manifest = json.loads((directory/"manifest.json").read_text())
        assert manifest["training_reward_overrides"] == {"action_rate_l2": -.3}
        assert ablation["final_training_manifest_sha256"] == sequence.sha256(directory/"manifest.json")


def test_failed_child_cannot_be_parent_finalized_even_if_status_claims_complete(args):
    children = FakeChildren(args, unfinalized=True, nonzero=True)
    code, status, children, _, _ = run_fake(args, children=children)
    assert code == 1 and len(children.calls) == 1
    assert status["children"][0]["returncode"] == 1
    ablation = json.loads((args.output_dir/"action_rate_0p3/smoke/ablation_manifest.json").read_text())
    assert ablation["status"] == "configured_before_environment_creation"
    assert "checkpoint_sha256" not in ablation


def test_parent_refuses_wrong_pre_environment_reward_function_without_mutation(args):
    children = FakeChildren(args, unfinalized=True, bad_function=True)
    code, status, children, _, _ = run_fake(args, children=children)
    assert code == 1 and len(children.calls) == 1
    assert "weight/function" in status["error"]
    directory = args.output_dir/"action_rate_0p3/smoke"
    ablation = json.loads((directory/"ablation_manifest.json").read_text())
    manifest = json.loads((directory/"manifest.json").read_text())
    assert ablation["status"] == "configured_before_environment_creation"
    assert "training_reward_overrides" not in manifest


def test_wrong_baseline_checkpoint_stops_before_any_gpu_child(args):
    report = args.baseline_root/"mu0p700/isaac/report.json"
    data = json.loads(report.read_text())
    data["checkpoint_sha256"] = "0"*64
    write_json(report, data)
    code, status, children, _, _ = run_fake(args)
    assert code == 1 and children.calls == []
    assert "matching checkpoint" in status["error"]


def gpu_xml(*, processes="", utilization="0 %", memory="7 MiB", reverse=False):
    devices = [f'''<gpu><uuid>GPU-{index}</uuid><pci><pci_bus_id>00000000:0{index+1}:00.0</pci_bus_id></pci>
        <fb_memory_usage><used>{memory}</used></fb_memory_usage><utilization><gpu_util>{utilization}</gpu_util></utilization>
        <processes>{processes if index == 1 else ""}</processes></gpu>''' for index in range(2)]
    return "<nvidia_smi_log>"+"".join(reversed(devices) if reverse else devices)+"</nvidia_smi_log>"


def test_gpu_probe_includes_graphics_processes_and_does_not_claim_reservation():
    clear = sequence.parse_gpu_one_snapshot(gpu_xml())
    assert clear["idle"] and clear["physical_index"] == 1
    assert clear["reservation_claimed"] is False
    active = sequence.parse_gpu_one_snapshot(gpu_xml(
        processes="<process_info><pid>123</pid><type>G</type></process_info>"))
    assert not active["idle"]
    assert active["all_reported_compute_and_graphics_processes"][0]["type"] == "G"


def test_unknown_readback_or_wrong_pci_order_is_fail_closed():
    with pytest.raises(RuntimeError, match="memory/utilization"):
        sequence.parse_gpu_one_snapshot(gpu_xml(utilization="N/A"))
    with pytest.raises(RuntimeError, match="PCI_BUS_ID"):
        sequence.parse_gpu_one_snapshot(gpu_xml(reverse=True))
    assert not sequence.parse_gpu_one_snapshot(gpu_xml(utilization="1 %"))["idle"]
    assert not sequence.parse_gpu_one_snapshot(gpu_xml(memory="64 MiB"))["idle"]
