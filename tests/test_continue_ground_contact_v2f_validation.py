"""Offline continuation tests; no SSH, simulator, GPU or real child process."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import continue_ground_contact_v2f_validation as module


def put(path, value):
    Path(path).write_text(json.dumps(value))


def gpus(idle_index=3):
    return [{"physical_index": index, "idle": index == idle_index,
             "pci_bus_id_cuda_ordinal_verified": True, "uuid": f"GPU-{index}", "pci_bus_id": f"00000000:65:0{index}.0",
             "all_reported_compute_and_graphics_processes": [] if index == idle_index else [{"pid": "foreign", "type": "C+G"}]}
            for index in module.GPU_ORDER]


@pytest.fixture
def harness(tmp_path):
    workspace, dependencies, legacy, bm = [tmp_path / name for name in ("workspace", "dependencies", "legacy", "bm")]
    for path in (workspace / "physx_formal_v1", dependencies / "scripts", legacy / "scripts", bm / "source"):
        path.mkdir(parents=True)
    for name in module.DEPENDENCIES:
        (dependencies / "scripts" / name).write_text("# synthetic dependency fixture\n")
    for name in ("native.py", "physx.py"):
        (legacy / "scripts" / name).write_text("# frozen legacy fixture\n")
    (bm / "source/config.py").write_text("# frozen BeyondMimic fixture\n")
    for name in ("motion.npz", "model.xml", "asset.usda"):
        (tmp_path / name).write_bytes(b"synthetic input fixture")
    (tmp_path / "protocol.json").write_bytes((ROOT/"config/ppo_ground_contact_v2f_400hz_v1.json").read_bytes())
    args = argparse.Namespace(workspace=workspace, evaluation_dependencies=dependencies,
        protocol=tmp_path / "protocol.json", legacy_native=legacy / "scripts/native.py",
        legacy_physx=legacy / "scripts/physx.py", motion=tmp_path / "motion.npz",
        model=tmp_path / "model.xml", asset=tmp_path / "asset.usda", beyondmimic_source=bm,
        python=Path(sys.executable), launch_record=None, expected_training_pid=3554880,
        poll_seconds=.01, deadline_seconds=100.)
    argv = [str(args.python), "-u", str(tmp_path / "source/scripts/train_ppo_ground_contact_v2f.py"),
        "--output-dir", str(workspace / "physx_formal_v1"), "--cohort-protocol", str(args.protocol),
        "--cohort-run-kind", "formal", "--motion", str(args.motion), "--model", str(args.model),
        "--asset-path", str(args.asset), "--beyondmimic-source", str(args.beyondmimic_source)]
    put(workspace / "physx_formal_v1_launch.json", {"pid": 3554880, "argv": argv,
        "run_kind": "formal_from_scratch_12288000_transitions", "started_at_utc": "2026-10-23T17:24:59+00:00"})
    put(workspace / "physx_formal_v1/run_status.json", {"status": "training"})
    commands, environments, sleeps, gpu_calls, checks = [], [], [], [], []

    def verify_training(directory, protocol, allow_smoke):
        checks.append("training")
        assert allow_smoke is False
        return {"smoke_only": False, "input_sha256": {"model_final.pt": "c" * 64, "policy.pt": "a" * 64,
            "ground_contact_v2f_training_audit.json": "d" * 64}}

    def validate_output(folder, directory, protocol, backend, allow_smoke):
        checks.append(backend)
        assert allow_smoke is False
        return {"status": "verified", "checkpoint_sha256": "c" * 64, "recording_sha256": "b" * 64,
                "native_states": 2793, "policy_decisions": 349, "evaluation_mu": .7}

    def child(command, log, cwd, environment, on_spawn):
        commands.append(command); environments.append(environment)
        on_spawn(100 + len(commands))
        Path(log).write_text("synthetic child log\n")
        Path(module.option(command, "--output-dir")).mkdir()
        return 0

    def query():
        gpu_calls.append(True)
        assert checks == ["training", "mujoco"]
        return gpus()

    options = {"evaluator": SimpleNamespace(verify_training=verify_training, validate_output=validate_output),
               "process_probe": lambda pid: None, "gpu_probe": query, "child_runner": child,
               "sleep": sleeps.append, "monotonic": lambda: 0.}
    return SimpleNamespace(args=args, options=options, argv=argv, commands=commands, environments=environments,
                           sleeps=sleeps, gpu_calls=gpu_calls, checks=checks)


def running(harness, *, ticks=10, command=None, state="R"):
    return {"pid": 3554880, "uid": os.getuid(), "state": state, "start_ticks": ticks,
            "argv": harness.argv if command is None else command}


def test_already_completed_then_exact_native_before_physx_same_actor(harness):
    status = module.run(harness.args, **harness.options)
    assert status["status"] == "nominal_dual_engine_validation_completed_requires_task_review"
    assert status["training_terminal_observation"]["observed_alive_before"] is False
    assert harness.checks == ["training", "mujoco", "physx"]
    assert len(harness.commands) == 2 and len(harness.gpu_calls) == 1
    for command in harness.commands:
        assert module.option(command, "--mu") == "0.7" and "--allow-smoke-policy" not in command
        assert Path(command[2]).name == "eval_ppo_ground_contact_v2f.py"
    assert harness.environments[0]["CUDA_VISIBLE_DEVICES"] == ""
    assert harness.environments[0]["MUJOCO_GL"] == "disable"
    assert "CUDA_VISIBLE_DEVICES" not in harness.environments[1]
    assert harness.environments[1]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert module.option(harness.commands[1], "--physical-gpu-index") == "3"
    for backend in ("mujoco", "physx"):
        proof = harness.args.workspace / (backend + "_nominal_eval_v1/remote_verification.json")
        assert module.read_json(proof)["checkpoint_sha256"] == "c" * 64
    assert status["process_signals_sent"] is False and status["future_gpu_reservation_claimed"] is False


def test_exact_training_identity_pinned_until_it_naturally_exits(harness):
    snapshots = iter([running(harness), running(harness), None])
    harness.options["process_probe"] = lambda _: next(snapshots)
    status = module.run(harness.args, **harness.options)
    assert status["training_process_identity"]["start_ticks"] == 10
    assert status["training_terminal_observation"]["observed_alive_before"] is True
    assert harness.sleeps[:2] == [.01, .01]


@pytest.mark.parametrize("change", ["command", "start_ticks", "owner"])
def test_reused_or_wrong_owned_process_never_launches_validation(harness, change):
    first, second = running(harness), running(harness)
    if change == "command": second["argv"] = ["other-process"]
    if change == "start_ticks": second["start_ticks"] = 11
    if change == "owner": second["uid"] += 1
    snapshots = iter([first, second])
    harness.options["process_probe"] = lambda _: next(snapshots)
    with pytest.raises(ValueError):
        module.run(harness.args, **harness.options)
    assert not harness.commands and not harness.gpu_calls


def test_busy_gpus_wait_without_reservation_then_only_authorized_idle5(harness):
    snapshots = iter([gpus(None), gpus(None), gpus(5)])
    harness.options["gpu_probe"] = lambda: next(snapshots)
    status = module.run(harness.args, **harness.options)
    assert module.option(harness.commands[-1], "--physical-gpu-index") == "5"
    assert harness.sleeps == [2., .01, .01, 2.]
    assert status["future_gpu_reservation_claimed"] is False


def test_gpu4_is_allowed_only_when_its_full_snapshot_is_idle(harness):
    harness.options["gpu_probe"] = lambda: gpus(4)
    module.run(harness.args, **harness.options)
    assert module.option(harness.commands[-1], "--physical-gpu-index") == "4"


def test_failed_training_proof_or_failed_child_stops_without_retry(harness):
    def reject(*args, **kwargs):
        raise ValueError("incomplete training")
    harness.options["evaluator"].verify_training = reject
    with pytest.raises(ValueError, match="incomplete training"):
        module.run(harness.args, **harness.options)
    assert not harness.commands


def test_native_child_failure_preserves_evidence_and_no_gpu_validation(harness):
    original = harness.options["child_runner"]

    def failed(*args):
        original(*args)
        return 4

    harness.options["child_runner"] = failed
    with pytest.raises(ValueError, match="Owned nominal evaluation failed"):
        module.run(harness.args, **harness.options)
    assert len(harness.commands) == 1 and not harness.gpu_calls
    state = module.read_json(harness.args.workspace / "nominal_validation_continuation_v1/continuation_status.json")
    assert state["status"] == "stopped_without_retry" and state["children"][0]["returncode"] == 4
    assert (harness.args.workspace / "nominal_validation_continuation_v1/mujoco.log").exists()


def test_source_changed_while_waiting_aborts_before_any_child(harness):
    harness.options["process_probe"] = lambda _: running(harness)

    def sleep(_):
        harness.args.protocol.write_text("changed protocol")

    harness.options["sleep"] = sleep
    with pytest.raises(ValueError, match="Frozen evaluation source"):
        module.run(harness.args, **harness.options)
    assert not harness.commands


def test_bounded_deadline_does_not_signal_training(harness):
    ticks = iter([0., 101.])
    harness.options["monotonic"] = lambda: next(ticks)
    with pytest.raises(ValueError, match="deadline exceeded"):
        module.run(harness.args, **harness.options)
    assert not harness.commands


def test_no_overwrite_of_previous_nominal_or_state_dirs(harness):
    target = harness.args.workspace / "mujoco_nominal_eval_v1"
    target.mkdir(); (target / "keep").write_text("existing failed evidence")
    with pytest.raises(ValueError, match="Preserve existing"):
        module.run(harness.args, **harness.options)
    assert (target / "keep").read_text() == "existing failed evidence"
    assert not harness.commands


def test_wrong_launch_pid_is_not_a_generic_training_queue(harness):
    harness.args.expected_training_pid = 999
    with pytest.raises(ValueError, match="exact authorized"):
        module.run(harness.args, **harness.options)
    assert not harness.commands


def test_new_explicit_positive_pid_is_bound_without_a_hardcoded_historical_pid(harness):
    path = harness.args.workspace/"physx_formal_v1_launch.json"
    launch = module.read_json(path)
    launch["pid"] = harness.args.expected_training_pid = 4567891
    put(path, launch)
    snapshots = iter([{**running(harness), "pid": 4567891}, None])
    harness.options["process_probe"] = lambda pid: next(snapshots) if pid == 4567891 else pytest.fail("Wrong process queried")
    result = module.run(harness.args, **harness.options)
    assert result["training_pid"] == 4567891
    assert result["training_process_identity"]["pid"] == 4567891


@pytest.mark.parametrize("value", [0, -1, True])
def test_nonpositive_or_boolean_training_pid_is_rejected(harness, value):
    path = harness.args.workspace/"physx_formal_v1_launch.json"
    launch = module.read_json(path)
    launch["pid"] = harness.args.expected_training_pid = value
    put(path, launch)
    with pytest.raises(ValueError, match="exact authorized"):
        module.run(harness.args, **harness.options)
    assert not harness.commands


def test_pinned_d_protocol_is_checked_before_wait_or_child(harness):
    protocol = module.read_json(harness.args.protocol)
    protocol["ground_contact_reward_v2"]["minimum_weight"] = .25
    put(harness.args.protocol, protocol)
    with pytest.raises(ValueError, match="V2f protocol"):
        module.run(harness.args, **harness.options)
    assert not harness.commands


def test_d_trainer_must_be_actual_python_script_not_a_flag_or_old_a_script(harness):
    path = harness.args.workspace/"physx_formal_v1_launch.json"
    launch = module.read_json(path)
    launch["argv"][2] = launch["argv"][2].replace("_v2f.py", "_v2.py")
    launch["argv"].append("/unrelated/scripts/train_ppo_ground_contact_v2f.py")
    put(path, launch)
    with pytest.raises(ValueError, match="trainer differs"):
        module.run(harness.args, **harness.options)
    assert not harness.commands


def test_packet_pins_c_evaluator_all_abc_definitions_and_protocol():
    assert len(module.DEPENDENCIES) == 11
    assert {"ground_contact_reward_v2.py", "ground_contact_reward_v2b.py", "ground_contact_reward_v2c.py", "ground_contact_reward_v2d.py", "ground_contact_reward_v2f.py"} <= module.DEPENDENCIES.keys()
    for name, digest in module.DEPENDENCIES.items():
        assert module.sha256(ROOT/"scripts"/name) == digest
    assert module.sha256(ROOT/"config/ppo_ground_contact_v2f_400hz_v1.json") == module.PROTOCOL_SHA256


def test_pinned_dependency_hash_is_checked_before_import(tmp_path):
    path = tmp_path / "scripts"
    path.mkdir()
    (path / next(iter(module.DEPENDENCIES))).write_text("raise Exception('must not execute')")
    with pytest.raises(ValueError, match="Pinned evaluation dependency differs"):
        module.load_evaluator(tmp_path)


def xml_fixture(*, graphics=False, unavailable=False):
    rows = []
    for index in range(6):
        processes = "<process_info><pid>999</pid><type>G</type></process_info>" if graphics and index == 3 else ""
        process_node = "" if unavailable and index == 3 else "<processes>" + processes + "</processes>"
        rows.append(f"<gpu><uuid>GPU-{index}</uuid><pci><pci_bus_id>00000000:65:0{index}.0</pci_bus_id></pci>"
            "<fb_memory_usage><used>6 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization>"
            + process_node + "</gpu>")
    return "<nvidia_smi_log>" + "".join(rows) + "</nvidia_smi_log>"


def test_full_nvml_graphics_process_counts_as_busy_and_missing_enumeration_fails():
    snapshots = module.parse_gpu_snapshot(xml_fixture(graphics=True))
    assert snapshots[0]["physical_index"] == 3 and snapshots[0]["idle"] is False
    assert snapshots[1]["physical_index"] == 5 and snapshots[1]["idle"] is True
    with pytest.raises(ValueError, match="enumeration"):
        module.parse_gpu_snapshot(xml_fixture(unavailable=True))
