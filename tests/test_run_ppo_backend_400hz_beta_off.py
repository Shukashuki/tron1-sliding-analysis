"""Offline queue tests only: never launch a simulator or access a GPU."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ppo_backend_400hz_beta_off as driver


def write(path, value):
    Path(path).write_text(json.dumps(value))


def gpu(index=3, idle=True):
    return {"physical_index": index, "idle": idle, "uuid": "fixture-gpu",
        "pci_bus_id": "0000:65:00.0", "pci_bus_id_cuda_ordinal_verified": True,
        "all_reported_compute_and_graphics_processes": [] if idle else [{"pid": 999}]}


class Trainer:
    def validate_protocol(self, value):
        return value

    def prepare_declared_inputs(self, protocol, folder, **kwargs):
        folder.mkdir()
        for name in ("contract.json", "task_profile.json"):
            write(folder / name, {"fixture": True})
        return {"contract": folder / "contract.json", "task_profile": folder / "task_profile.json",
                "derived_input_sha256": {"contract": driver.sha256(folder / "contract.json"),
                    "task_profile": driver.sha256(folder / "task_profile.json")}}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    base = driver.load(ROOT / "scripts/run_ppo_backend_400hz_pair.py", "_beta_off_test_base", driver.BASE_DRIVER_SHA256)
    snapshot = tmp_path / "snapshot"
    for folder in (snapshot / "scripts", snapshot / "config", snapshot / "native",
                   tmp_path / "tron1/scripts", tmp_path / "tron1/training",
                   tmp_path / "bm/source", tmp_path / "mjlab/src", tmp_path / "baseline", tmp_path / "legacy/scripts"):
        folder.mkdir(parents=True)
    for filename in ("train_ppo_backend_400hz.py", "train_ppo_backend_400hz_beta_off.py",
                     "benchmark_mjlab_gpu_task.py", "guard.py", "probe.py"):
        (snapshot / "scripts" / filename).write_text("# fixture only\n")
    for filename in ("contract.json", "profile.json", "motion.npz", "model.xml", "robot.usda", "device.py"):
        (tmp_path / filename).write_bytes(b"fixture, not an experiment artifact")
    protocol = {"cohort_id": "synthetic", "shared_dr_recipe": "fixture",
        "algorithm": {"num_envs": 512, "iterations": 1000, "transitions": 12288000},
        "smoke": {"num_envs": 32, "iterations": 2, "transitions": 1536}}
    write(tmp_path / "protocol.json", protocol)
    write(tmp_path / "baseline_protocol.json", protocol)
    args = argparse.Namespace(snapshot=snapshot, protocol=tmp_path / "protocol.json",
        contract=tmp_path / "contract.json", task_profile=tmp_path / "profile.json",
        model=tmp_path / "model.xml", motion=tmp_path / "motion.npz", asset_path=tmp_path / "robot.usda",
        tron1_source=tmp_path / "tron1", beyondmimic_source=tmp_path / "bm", mjlab_source=tmp_path / "mjlab",
        guard_source=snapshot / "scripts/guard.py", gpu_probe_source=snapshot / "scripts/probe.py",
        isaac_device_audit_source=tmp_path / "device.py", output_dir=tmp_path / "preflight",
        python=Path(sys.executable), physical_gpu_index=3, stage="preflight", preflight_dir=None,
        beta_on_pair_dir=tmp_path / "baseline", beta_on_protocol=tmp_path / "baseline_protocol.json",
        legacy_source=tmp_path / "legacy")
    monkeypatch.setattr(base, "ROOT", snapshot)
    monkeypatch.setattr(base, "validate_parity", lambda folder, args: {"status": "verified", "kind": "synthetic_parity"})
    proof = {"status": "verified", "initial_policy_state_sha256": driver.EXPECTED_INITIAL_SHA256}
    monkeypatch.setattr(driver, "validate_training", lambda folder, backend, mode, *a: {
        **proof, "backend": backend, "run_kind": mode})
    baseline_path = tmp_path / "baseline/artifact.json"
    write(baseline_path, {"fixture_baseline": True})
    def cohort(folder, protocol=None, **kwargs):
        backend = "mjlab" if "mjlab" in Path(folder).name else "physx"
        return {"training_backend": backend, "smoke_only": False, "directory": Path(folder),
            "manifest": {"initial_policy_state_sha256": driver.EXPECTED_INITIAL_SHA256},
            "contract": {"trained_task_profile": {"reward_weights": {"sideslip": 2.}}},
            "paths": {"manifest.json": baseline_path},
            "input_sha256": {"model_final.pt": "a" * 64, "policy.pt": "b" * 64,
                             "manifest.json": driver.sha256(baseline_path)}}
    def validate_output(folder, cohort_dir, protocol, *, validation_backend):
        return {"status": "verified", "native_states": 2793, "policy_decisions": 349,
            "duration_s": 6.98, "physics_dt_s": .0025, "policy_dt_s": .02,
            "training_backend": "mjlab" if "mjlab" in str(cohort_dir) else "physx",
            "validation_backend": validation_backend, "evaluation_contact_friction": .7, "smoke_only": False}
    shared = SimpleNamespace(verify_cohort=cohort, validate_output=validate_output)
    commands, envs, sleeps = [], [], []
    def child(command, log, cwd, env, spawned):
        commands.append(command)
        envs.append(env)
        spawned(1234 + len(commands))
        Path(log).write_text("synthetic child\n")
        return 0
    opts = dict(base=base, shared=shared, on_shared=shared, wrapper=Trainer(),
        guard=SimpleNamespace(write_json=write, run_child=child), gpu_probe=lambda index: gpu(index),
        child_runner=child, sleep=sleeps.append)
    return SimpleNamespace(args=args, options=opts, commands=commands, environments=envs, sleeps=sleeps, root=tmp_path)


def test_preflight_runs_only_parity_and_two_beta_off_smokes(harness):
    status = driver.run(harness.args, **harness.options)
    assert status["status"] == "preflight_completed_requires_formal_authorization"
    assert len(harness.commands) == 3
    for command in harness.commands[1:]:
        assert command[2].endswith("/train_ppo_backend_400hz_beta_off.py")
        assert command[command.index("--num-envs") + 1] == "32"
    assert (harness.args.output_dir / driver.STATUS_FILE).is_file()
    assert not status["beta_on_matrix_resumed"]
    assert not status["beta_on_checkpoint_initialization"]


def test_explicit_gpu6_is_authorized_and_reported_without_fallback(harness):
    harness.args.physical_gpu_index = 6
    status = driver.run(harness.args, **harness.options)
    assert status["physical_gpu_index"] == 6
    assert status["requested_device"] == "cuda:6"
    assert status["authorized_physical_gpu_indices"] == [3, 4, 5, 6]
    assert status["automatic_gpu_fallback"] is False
    assert status["gpu_selection"] == "explicit_requested_ordinal_no_automatic_fallback"
    for command in harness.commands:
        assert command[command.index("--device") + 1] == "cuda:6"


def test_parser_authorizes_only_explicit_gpu3_4_5_6():
    parser = driver.parser()
    assert tuple(parser._option_string_actions["--physical-gpu-index"].choices) == (3, 4, 5, 6)


def gpu_xml(*, process=None, swap=False, missing_processes=False, memory=7, utilization=0):
    rows = []
    buses = list(range(1, 9))
    if swap:
        buses[1], buses[6] = buses[6], buses[1]
    for index, bus in enumerate(buses):
        entry = (f"<process_info><pid>999</pid><type>{process}</type>"
                 "<process_name>unrelated</process_name></process_info>" if index == 6 and process else "")
        processes = "" if missing_processes and index == 6 else f"<processes>{entry}</processes>"
        rows.append(f"<gpu><uuid>GPU-{index}</uuid><pci><pci_bus_id>00000000:{bus:02x}:00.0</pci_bus_id></pci>"
            f"<fb_memory_usage><used>{memory} MiB</used></fb_memory_usage>"
            f"<utilization><gpu_util>{utilization} %</gpu_util></utilization>{processes}</gpu>")
    return "<nvidia_smi_log>" + "".join(rows) + "</nvidia_smi_log>"


def test_real_private_probe_gpu6_reuses_complete_nvml_checks_without_source_mutation():
    base = driver.load(ROOT / "scripts/run_ppo_backend_400hz_pair.py", "_probe_test_base", driver.BASE_DRIVER_SHA256)
    path = ROOT / "scripts/run_ppo_leg_target_filter.py"
    before = driver.sha256(path)
    probe = driver.gpu_probe_module(base, SimpleNamespace(gpu_probe_source=path))
    result = probe.parse_gpu_snapshot(gpu_xml(), 6)
    assert result["physical_index"] == 6 and result["uuid"] == "GPU-6" and result["idle"] is True
    assert result["pci_bus_id_cuda_ordinal_verified"] is True
    assert driver.sha256(path) == before == base.GPU_PROBE_SHA256
    original = __import__("eval_ppo_leg_target_filter")
    assert probe.evaluator is not original
    assert tuple(original.AUTHORIZED_GPU_INDICES) == (1, 3, 4, 5)
    for kind in ("G", "C", "C+G"):
        busy = probe.parse_gpu_snapshot(gpu_xml(process=kind), 6)
        assert busy["idle"] is False
        assert busy["all_reported_compute_and_graphics_processes"][0]["type"] == kind
    assert not probe.parse_gpu_snapshot(gpu_xml(memory=64), 6)["idle"]
    assert not probe.parse_gpu_snapshot(gpu_xml(utilization=1), 6)["idle"]
    with pytest.raises(ValueError, match="CUDA ordinal"):
        probe.parse_gpu_snapshot(gpu_xml(swap=True), 6)
    with pytest.raises(ValueError, match="enumeration"):
        probe.parse_gpu_snapshot(gpu_xml(missing_processes=True), 6)
    with pytest.raises(ValueError, match="Unauthorized"):
        probe.parse_gpu_snapshot(gpu_xml(), 0)


@pytest.mark.parametrize("index", [0, 1, 2, 7])
def test_unauthorized_gpu_rejected_before_any_output_or_child(harness, index):
    harness.args.physical_gpu_index = index
    with pytest.raises(ValueError, match="authorized physical GPU3/4/5/6"):
        driver.run(harness.args, **harness.options)
    assert not harness.args.output_dir.exists()
    assert not harness.commands


def test_formal_then_four_nominal_cells_sequential_and_cpu_has_no_gpu(harness):
    driver.run(harness.args, **harness.options)
    harness.args.stage, harness.args.preflight_dir = "formal", harness.args.output_dir
    harness.args.output_dir = harness.root / "formal"
    harness.commands.clear()
    harness.environments.clear()
    status = driver.run(harness.args, **harness.options)
    assert status["status"] == "training_and_nominal_validation_completed"
    assert len(harness.commands) == 6
    assert status["nominal_counts"] == {"expected": 4, "completed_verified": 4, "runtime_failed": 0}
    for command in harness.commands[:2]:
        assert command[command.index("--num-envs") + 1] == "512"
        assert command[command.index("--iterations") + 1] == "1000"
    for command, env in zip(harness.commands[2:], harness.environments[2:]):
        assert command[command.index("--evaluation-contact-friction") + 1] == ".7"
        if command[2].endswith("_mujoco.py"):
            assert env["CUDA_VISIBLE_DEVICES"] == ""
        else:
            assert "CUDA_VISIBLE_DEVICES" not in env
    assert "remaining_friction_not_launched" in status["validation_matrix_status"]


@pytest.mark.parametrize("at", [0, 1, 2])
def test_busy_gpu_stops_before_next_child_without_retry(harness, at):
    calls = []
    def query(index):
        calls.append(index)
        return gpu(index, idle=len(calls) - 1 != at)
    harness.options["gpu_probe"] = query
    with pytest.raises(ValueError, match="freshly fully idle"):
        driver.run(harness.args, **harness.options)
    assert len(harness.commands) == at
    state = json.loads((harness.args.output_dir / driver.STATUS_FILE).read_text())
    assert state["status"] == "stopped_without_retry"


def test_failed_child_does_not_launch_following_child(harness):
    child = harness.options["child_runner"]
    def fail(*args):
        child(*args)
        return 23
    harness.options["child_runner"] = fail
    with pytest.raises(ValueError, match="Owned child failed"):
        driver.run(harness.args, **harness.options)
    assert len(harness.commands) == 1


def test_reject_baseline_nested_output(harness):
    harness.args.output_dir = harness.args.beta_on_pair_dir / "do_not_write"
    with pytest.raises(ValueError, match="Never write beta-off"):
        driver.run(harness.args, **harness.options)
    assert not harness.args.output_dir.exists()


def test_gpu_becomes_busy_before_nominal_preserves_trained_pair(harness):
    driver.run(harness.args, **harness.options)
    harness.args.stage, harness.args.preflight_dir = "formal", harness.args.output_dir
    harness.args.output_dir = harness.root / "formal"
    harness.commands.clear()
    probes = []
    def query(index):
        probes.append(index)
        return gpu(index, idle=len(probes) < 3)
    harness.options["gpu_probe"] = query
    with pytest.raises(ValueError, match="freshly completely idle"):
        driver.run(harness.args, **harness.options)
    assert len(harness.commands) == 2
    status = json.loads((harness.args.output_dir / driver.STATUS_FILE).read_text())
    assert status["status"] == "stopped_without_retry_training_preserved"
    assert all(child["status"] == "completed_verified" for child in status["children"])
    assert status["nominal_counts"] == {"expected": 4, "completed_verified": 0, "runtime_failed": 0}


def test_failed_nominal_is_not_retried_and_following_cells_not_started(harness):
    driver.run(harness.args, **harness.options)
    harness.args.stage, harness.args.preflight_dir = "formal", harness.args.output_dir
    harness.args.output_dir = harness.root / "formal"
    harness.commands.clear()
    child = harness.options["child_runner"]
    def fail_nominal(*parts):
        code = child(*parts)
        return 14 if "eval_ppo_backend_400hz_beta_off_" in parts[0][2] else code
    harness.options["child_runner"] = fail_nominal
    with pytest.raises(ValueError, match="Nominal evaluation child failed"):
        driver.run(harness.args, **harness.options)
    assert len(harness.commands) == 3
    status = json.loads((harness.args.output_dir / driver.STATUS_FILE).read_text())
    assert status["nominal_counts"] == {"expected": 4, "completed_verified": 0, "runtime_failed": 1}


def test_reject_artifact_changed_between_verification_and_freeze(tmp_path):
    path = tmp_path / "manifest.json"
    write(path, {"old": True})
    verified = {"paths": {"manifest.json": path}, "input_sha256": {"manifest.json": driver.sha256(path)}}
    write(path, {"changed": True})
    with pytest.raises(ValueError, match="before freezing"):
        driver.cohort_files({"physx": verified})


def test_fixed_initializer_and_only_reward_change_checked():
    profile = {"reward_weights": {"sideslip": 0., "grounded": .4}, "action_rate_weight": -.5}
    baseline = {"contract": {"trained_task_profile": copy.deepcopy(profile)},
                "manifest": {"initial_policy_state_sha256": driver.EXPECTED_INITIAL_SHA256},
                "input_sha256": {"model_final.pt": "a" * 64, "policy.pt": "b" * 64}}
    baseline["contract"]["trained_task_profile"]["reward_weights"]["sideslip"] = 2.
    device = {"requested_device": "cuda:3", "torch_current_device": 3, "cuda_visible_devices": None,
        "status": "verified", "native_and_tensor_device_consistency_verified": True,
        "environment_device": "cuda:3", "physics_sim_view_frontend_device": "cuda:3",
        "robot_native_transform_tensor_device": "cuda:3", "native_physics_backend_device_ordinal": 3,
        "physics_cudaDevice_setting": 3}
    cohort = {"contract": {"trained_task_profile": profile}, "training_backend": "physx",
        "manifest": {"initial_policy_state_sha256": driver.EXPECTED_INITIAL_SHA256,
                     "cohort": {"runtime": {"device": device}}}}
    assert driver.compare_training(cohort, baseline, SimpleNamespace(physical_gpu_index=3))["initial_network_identical_to_beta_on"]
    cohort["contract"]["trained_task_profile"]["action_rate_weight"] = -.1
    with pytest.raises(ValueError, match="More than"):
        driver.compare_training(cohort, baseline, SimpleNamespace(physical_gpu_index=3))
