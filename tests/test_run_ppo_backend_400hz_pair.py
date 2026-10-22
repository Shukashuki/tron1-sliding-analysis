"""Offline dispatcher tests with injected NVML snapshots and child processes."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ppo_backend_400hz_pair as driver


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


class WrapperFixture:
    def validate_protocol(self, value):
        return value

    def prepare_declared_inputs(self, protocol, output, **kwargs):
        output.mkdir()
        for name in ("contract.json", "task_profile.json"):
            write_json(output / name, {"deterministic_fixture": True})
        return {"contract": output / "contract.json", "task_profile": output / "task_profile.json",
                "derived_input_sha256": {"contract": driver.sha256(output / "contract.json"),
                                         "task_profile": driver.sha256(output / "task_profile.json")}}


def gpu(index=3, *, idle=True, uuid="GPU-fixture"):
    return {"physical_index": index, "uuid": uuid, "pci_bus_id": "00000000:65:04.0",
            "idle": idle, "pci_bus_id_cuda_ordinal_verified": True,
            "all_reported_compute_and_graphics_processes": [] if idle else [{"pid": "999", "type": "G"}],
            "reservation_claimed": False}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    snapshot = tmp_path / "snapshot"
    for directory in (snapshot / "scripts", snapshot / "config", snapshot / "native",
                      tmp_path / "tron1/scripts", tmp_path / "tron1/training",
                      tmp_path / "bm/source", tmp_path / "mjlab/src"):
        directory.mkdir(parents=True)
    for relative in ("scripts/train_ppo_backend_400hz.py", "scripts/benchmark_mjlab_gpu_task.py",
                     "scripts/guard.py", "scripts/probe.py"):
        (snapshot / relative).write_text("# synthetic source fixture only\n")
    for name in ("contract.json", "task.json", "model.xml", "motion.npz", "robot.usda", "device.py"):
        (tmp_path / name).write_bytes(b"synthetic fixture, not a real artifact")
    protocol = {"cohort_id": "fixture", "shared_dr_recipe": "fixture",
                "algorithm": {"num_envs": 512, "iterations": 1000, "transitions": 12288000},
                "smoke": {"num_envs": 32, "iterations": 2, "transitions": 1536}}
    write_json(tmp_path / "protocol.json", protocol)
    args = argparse.Namespace(snapshot=snapshot, protocol=tmp_path / "protocol.json",
        contract=tmp_path / "contract.json", task_profile=tmp_path / "task.json",
        model=tmp_path / "model.xml", motion=tmp_path / "motion.npz", asset_path=tmp_path / "robot.usda",
        tron1_source=tmp_path / "tron1", beyondmimic_source=tmp_path / "bm", mjlab_source=tmp_path / "mjlab",
        guard_source=snapshot / "scripts/guard.py", gpu_probe_source=snapshot / "scripts/probe.py",
        isaac_device_audit_source=tmp_path / "device.py", output_dir=tmp_path / "preflight",
        python=Path(sys.executable), physical_gpu_index=3, stage="preflight", preflight_dir=None)
    monkeypatch.setattr(driver, "ROOT", snapshot)
    monkeypatch.setattr(driver, "validate_parity", lambda folder, args: {"status": "verified", "kind": "synthetic_parity"})
    monkeypatch.setattr(driver, "validate_training", lambda folder, backend, mode, args: {
        "status": "verified", "backend": backend, "run_kind": mode,
        "initial_policy_state_sha256": "a" * 64, "kind": "synthetic_training"})
    commands, sleeps = [], []

    def child(command, log, cwd, environment, callback):
        commands.append(command)
        callback(1234 + len(commands))
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert "CUDA_VISIBLE_DEVICES" not in environment
        assert environment["TRON1_BEYONDMIMIC_ROOT"] == str(args.beyondmimic_source)
        Path(log).write_text("synthetic child log\n")
        return 0

    options = {"guard": SimpleNamespace(write_json=write_json), "gpu_probe": lambda index: gpu(index),
               "wrapper": WrapperFixture(), "quiesce": sleeps.append, "child_runner": child}
    return SimpleNamespace(args=args, options=options, commands=commands, sleeps=sleeps, root=tmp_path)


def test_default_preflight_only_three_sequential_children_and_explicit_budgets(harness):
    result = driver.run(harness.args, **harness.options)
    assert result["status"] == "preflight_completed_requires_formal_authorization"
    assert [child["stage"] for child in result["children"]] == ["parity", "physx_smoke", "mjlab_smoke"]
    assert len(harness.commands) == 3 and harness.sleeps == [2., 2., 2.]
    parity, physx, mjlab = harness.commands
    assert parity[parity.index("--parity-num-envs") + 1] == "4"
    assert parity[parity.index("--parity-steps") + 1] == "8"
    for command in (physx, mjlab):
        assert command[command.index("--num-envs") + 1] == "32"
        assert command[command.index("--iterations") + 1] == "2"
        assert command[command.index("--smoke-steps") + 1] == "8"
        assert command[command.index("--device") + 1] == "cuda:3"
        assert command[command.index("--mjlab-task-implementation") + 1] == "gpu"
    assert result["validation_matrix_status"] == "pending_not_evaluated"
    assert result["exclusive_gpu_reservation_claimed"] is False


def test_formal_requires_prior_verified_unchanged_preflight_then_exact_two_arms(harness):
    driver.run(harness.args, **harness.options)
    old = harness.args.output_dir
    harness.args.output_dir = harness.root / "formal"
    harness.args.stage, harness.args.preflight_dir = "formal", old
    harness.commands.clear()
    result = driver.run(harness.args, **harness.options)
    assert result["status"] == "training_pair_completed_validation_pending"
    assert [child["stage"] for child in result["children"]] == ["physx_formal", "mjlab_formal"]
    assert len(harness.commands) == 2
    for command in harness.commands:
        assert command[command.index("--num-envs") + 1] == "512"
        assert command[command.index("--iterations") + 1] == "1000"
    assert result["preflight_verification"]["status"] == "verified"


@pytest.mark.parametrize("busy_at", [0, 1, 2])
def test_complete_gpu_busy_snapshot_prevents_that_and_all_later_children(harness, busy_at):
    probes = []

    def query(index):
        probes.append(index)
        return gpu(index, idle=len(probes) - 1 != busy_at)

    harness.options["gpu_probe"] = query
    with pytest.raises(ValueError, match="freshly fully idle"):
        driver.run(harness.args, **harness.options)
    assert len(harness.commands) == busy_at
    status = driver.read_json(harness.args.output_dir / "backend_400hz_pair_status.json")
    assert status["status"] == "stopped_without_retry"
    assert all(child["status"] == "completed_verified" for child in status["children"])


def test_failed_child_is_preserved_and_never_retried(harness):
    original = harness.options["child_runner"]

    def child(*args):
        original(*args)
        return 17

    harness.options["child_runner"] = child
    with pytest.raises(ValueError, match="Owned child failed"):
        driver.run(harness.args, **harness.options)
    assert len(harness.commands) == 1
    status = driver.read_json(harness.args.output_dir / "backend_400hz_pair_status.json")
    assert status["children"][0]["returncode"] == 17
    assert status["children"][0]["status"] == "failed_or_unverified"
    assert (harness.args.output_dir / "parity.log").is_file()


def test_child_success_without_actual_verification_does_not_advance(harness, monkeypatch):
    def reject(*args):
        raise ValueError("missing physical parity")

    monkeypatch.setattr(driver, "validate_parity", reject)
    with pytest.raises(ValueError, match="missing physical parity"):
        driver.run(harness.args, **harness.options)
    assert len(harness.commands) == 1


def test_source_changed_by_child_stops_before_its_proof_or_next_launch(harness):
    original = harness.options["child_runner"]

    def child(*args):
        original(*args)
        (harness.args.snapshot / "scripts/train_ppo_backend_400hz.py").write_text("# changed fixture\n")
        return 0

    harness.options["child_runner"] = child
    with pytest.raises(ValueError, match="Frozen source or input bytes changed"):
        driver.run(harness.args, **harness.options)
    assert len(harness.commands) == 1


def test_embedded_upstream_source_is_frozen_and_symlinks_are_rejected(harness):
    embedded = harness.args.snapshot / "upstream/tron1-learning-from-human/scripts"
    embedded.mkdir(parents=True)
    source = embedded / "eval_tracking_mujoco.py"
    source.write_text("# embedded startup dependency fixture\n")
    harness.args.parity_contract = harness.args.contract
    harness.args.parity_task_profile = harness.args.task_profile
    frozen = driver.source_snapshot(harness.args)
    assert str(source) in frozen["files"]
    source.write_text("# changed embedded startup code\n")
    with pytest.raises(ValueError, match="Frozen source or input bytes changed"):
        driver.verify_snapshot(harness.args, frozen)
    (embedded / "linked.py").symlink_to(source)
    with pytest.raises(ValueError, match="symlinked source"):
        driver.source_snapshot(harness.args)


def test_gpu_identity_changed_after_parity_prevents_first_training(harness):
    calls = []

    def query(index):
        calls.append(index)
        return gpu(index, uuid="GPU-one" if len(calls) == 1 else "GPU-other")

    harness.options["gpu_probe"] = query
    with pytest.raises(ValueError, match="UUID/PCI identity changed"):
        driver.run(harness.args, **harness.options)
    assert len(harness.commands) == 1


def test_no_overwrite_and_masked_or_unauthorized_device_fail_before_creating_output(harness, monkeypatch):
    harness.args.output_dir.mkdir()
    sentinel = harness.args.output_dir / "keep.txt"
    sentinel.write_text("owned old evidence")
    with pytest.raises(ValueError, match="fresh directory"):
        driver.run(harness.args, **harness.options)
    assert sentinel.read_text() == "owned old evidence"
    harness.args.output_dir = harness.root / "masked"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(ValueError, match="unmasked"):
        driver.run(harness.args, **harness.options)
    assert not harness.args.output_dir.exists()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    harness.args.physical_gpu_index = 1
    with pytest.raises(ValueError, match="GPU3/4/5"):
        driver.run(harness.args, **harness.options)
    assert not harness.commands


def test_formal_without_explicit_preflight_has_no_side_effects(harness):
    harness.args.stage = "formal"
    with pytest.raises(ValueError, match="--preflight-dir"):
        driver.run(harness.args, **harness.options)
    assert not harness.args.output_dir.exists() and not harness.commands


def test_changed_source_between_preflight_and_formal_is_rejected_without_gpu_job(harness):
    driver.run(harness.args, **harness.options)
    old = harness.args.output_dir
    (harness.args.snapshot / "scripts/train_ppo_backend_400hz.py").write_text("# changed after preflight\n")
    harness.args.stage, harness.args.preflight_dir = "formal", old
    harness.args.output_dir = harness.root / "formal"
    harness.commands.clear()
    with pytest.raises(ValueError, match="Formal sources/input bytes differ"):
        driver.run(harness.args, **harness.options)
    assert not harness.commands


def test_incomplete_preflight_cannot_authorize_formal(harness):
    driver.run(harness.args, **harness.options)
    old = harness.args.output_dir
    status_path = old / "backend_400hz_pair_status.json"
    status = driver.read_json(status_path)
    status["children"].pop()
    write_json(status_path, status)
    harness.args.stage, harness.args.preflight_dir = "formal", old
    harness.args.output_dir = harness.root / "formal"
    harness.commands.clear()
    with pytest.raises(ValueError, match="Preflight is incomplete"):
        driver.run(harness.args, **harness.options)
    assert not harness.commands


def test_network_initialization_mismatch_stops_before_next_training(harness, monkeypatch):
    calls = []

    def validate(folder, backend, mode, args):
        calls.append(backend)
        return {"status": "verified", "initial_policy_state_sha256": ("a" if len(calls) == 1 else "b") * 64}

    monkeypatch.setattr(driver, "validate_training", validate)
    with pytest.raises(ValueError, match="Initial policy network differs"):
        driver.run(harness.args, **harness.options)
    assert len(harness.commands) == 3


def test_source_pin_mismatch_does_not_import_or_execute_changed_helper(tmp_path):
    source = tmp_path / "changed.py"
    source.write_text("raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match="Pinned helper source differs"):
        driver._load(source, "fixture", "0" * 64)


def training_artifact_fixture(harness, backend, mode):
    """Synthetic bytes test verification plumbing, never a trained-policy claim."""
    args = harness.args
    args.protocol_data = driver.read_json(args.protocol)
    args.wrapper = driver._load(ROOT / "scripts/train_ppo_backend_400hz.py", "_fixture_actual_field_validator")
    declared = WrapperFixture().prepare_declared_inputs(args.protocol, harness.root / (backend + mode + "_declared_inputs"))
    args.derived_input_sha256 = declared["derived_input_sha256"]
    folder = harness.root / (backend + mode)
    folder.mkdir()
    (folder / "model_final.pt").write_bytes(b"synthetic non-policy checkpoint fixture")
    (folder / "policy.pt").write_bytes(b"synthetic non-actor fixture")
    n, iterations, transitions = (32, 2, 1536) if mode == "smoke" else (512, 1000, 12288000)
    flags = {"smoke_only": mode == "smoke", "trained_policy_claimed": mode == "formal"}
    fields = {"physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02, "actor_dim": 51,
              "critic_dim": 120, "action_dim": 8, "action_rate_weight": -.5, "grounded_reward_weight": .4,
              "reward_term_count": 15, "leg_kp": 500., "leg_kd": 10.,
              "absolute_height_termination": False, "absolute_xy_termination": False}
    device = {"requested_device": "cuda:3", "torch_current_device": 3, "cuda_visible_devices": None,
              "status": "verified", "native_and_tensor_device_consistency_verified": True,
              "environment_device": "cuda:3", "physics_sim_view_frontend_device": "cuda:3",
              "robot_native_transform_tensor_device": "cuda:3", "native_physics_backend_device_ordinal": 3,
              "physics_cudaDevice_setting": 3, "task_arrays_device": "cuda:3",
              "native_backend": {"device": "cuda:3", "physics_dt_s": .0025, "decimation": 8, "host_task_adapter": False}}
    cohort = {"id": "fixture", "run_kind": mode, "protocol_sha256": driver.sha256(args.protocol),
              "wrapper_sha256": driver.sha256(args.snapshot / "scripts/train_ppo_backend_400hz.py"),
              "from_scratch": True, "historical_checkpoint_used": False,
              "network_seed_after_environment_setup_and_smoke": True, "normalized_actor_export_verified": True,
              "initial_policy_state_sha256": "a" * 64,
              "source_files_sha256": {"scripts/train_ppo_backend_400hz.py": driver.sha256(args.snapshot / "scripts/train_ppo_backend_400hz.py")},
              "derived_input_sha256": args.derived_input_sha256,
              "runtime": {"status": "verified", "no_extra_environment_steps": True,
                          "shared_DR_recipe": "fixture", "fields": fields, "device": device}, **flags}
    checkpoint, actor = driver.sha256(folder / "model_final.pt"), driver.sha256(folder / "policy.pt")
    task = {"action_rate_weight": -.5}
    contract = {"contract_status": "smoke_completed_normalized_actor_not_formal_policy" if mode == "smoke" else "trained_completed_normalized_actor",
                "policy_training_backend": backend, "policy_algorithm": "PPO", "training_environment_transitions": transitions,
                "physics_dt": .0025, "decimation": 8, "task_profile": task, "trained_task_profile": task,
                "actor_includes_observation_normalizer": True, "runtime_clock_and_interface_verified": True,
                "checkpoint_sha256": checkpoint, "exported_policy_sha256": actor, **flags}
    manifest = {"backend": backend, "num_envs": n, "iterations": iterations, "cohort": cohort,
                "initial_policy_state_sha256": "a" * 64, "task_profile": task, "contract": contract, **flags}
    report = {"status": "training_completed", "backend": backend, "training_environment_steps": transitions,
              "initial_policy_state_sha256": "a" * 64, "checkpoint_sha256": checkpoint,
              "exported_policy_sha256": actor, **flags}
    for name, value in (("manifest.json", manifest), ("policy_contract.json", contract),
                        ("final_policy_contract.json", contract), ("cohort_startup_audit.json", cohort),
                        ("run_status.json", report)):
        write_json(folder / name, value)
    completion = {"status": "completed_verified", "backend": backend, "run_kind": mode, "cohort_id": "fixture",
                  "training_environment_steps": transitions, "protocol_sha256": driver.sha256(args.protocol),
                  "source_files_unchanged": True, "historical_checkpoint_used": False, "additional_reward_terms": False,
                  "physics_dt_s": .0025, "decimation": 8, "policy_dt_s": .02, "action_rate_weight": -.5,
                  "grounded_reward_weight": .4, "checkpoint_sha256": checkpoint, "exported_policy_sha256": actor,
                  "initial_policy_state_sha256": "a" * 64, **flags}
    for name, key in (("manifest.json", "manifest_sha256"), ("policy_contract.json", "policy_contract_sha256"),
                      ("final_policy_contract.json", "final_policy_contract_sha256"),
                      ("cohort_startup_audit.json", "startup_audit_sha256")):
        completion[key] = driver.sha256(folder / name)
    write_json(folder / "cohort_completion_audit.json", completion)
    actual = transitions + 8 * n
    write_json(folder / "environment_final.json", {"physical_counters": {
        "environment_transitions": actual, "physical_environment_substeps": actual * 8}})
    return folder, args


@pytest.mark.parametrize("backend,mode", [("physx", "smoke"), ("mjlab", "smoke"), ("physx", "formal"), ("mjlab", "formal")])
def test_real_completion_verifier_binds_mode_budget_runtime_and_actual_bytes(harness, monkeypatch, backend, mode):
    # harness normally mocks verification only for workflow tests.
    verifier = REAL_VALIDATE_TRAINING
    folder, args = training_artifact_fixture(harness, backend, mode)
    assert verifier(folder, backend, mode, args)["status"] == "verified"
    (folder / "policy.pt").write_bytes(b"changed after completion")
    with pytest.raises(ValueError, match="checkpoint/export bytes differ"):
        verifier(folder, backend, mode, args)


def test_partial_training_never_passes_completed_artifact_verifier(harness):
    folder, args = training_artifact_fixture(harness, "physx", "formal")
    report = driver.read_json(folder / "run_status.json")
    report["training_environment_steps"] -= 12288
    write_json(folder / "run_status.json", report)
    with pytest.raises(ValueError, match="exact declared budget"):
        REAL_VALIDATE_TRAINING(folder, "physx", "formal", args)


REAL_VALIDATE_TRAINING = driver.validate_training
