"""Dispatcher tests use synthetic child files, never training or GPU jobs."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_ppo_backend_pair", ROOT / "scripts/run_ppo_backend_pair.py")
PAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIR)


@pytest.fixture
def arguments(tmp_path, monkeypatch):
    snapshot, workspace, upstream = tmp_path / "snapshot", tmp_path / "workspace", tmp_path / "upstream"
    (snapshot / "scripts").mkdir(parents=True)
    (snapshot / "config").mkdir()
    (snapshot / "scripts/train_ppo_backend.py").write_text("# synthetic test source; never executed\n")
    for filename in ("nominal_actuator_contract.json", "car_drift_height_smooth_task.json"):
        (snapshot / "config" / filename).write_text("{}\n")
    upstream.mkdir()
    for filename in ("model.xml", "motion.npz", "robot.usda"):
        (tmp_path / filename).write_text("synthetic test fixture\n")
    args = PAIR.parse_args([
        "--snapshot", str(snapshot), "--workspace", str(workspace), "--python", sys.executable,
        "--mjlab-source", str(upstream), "--model", str(tmp_path / "model.xml"),
        "--motion", str(tmp_path / "motion.npz"), "--asset-path", str(tmp_path / "robot.usda"),
        "--tron1-source", str(snapshot), "--beyondmimic-source", str(upstream)])
    monkeypatch.setattr(PAIR.subprocess, "check_output", lambda *args, **kwargs: PAIR.MJLAB_PIN + "\n")
    return args


def idle():
    return {"physical_index": 0, "uuid": "GPU-synthetic-test-only", "memory_used_mib": 7,
            "utilization_percent": 0, "compute_processes": [], "idle": True}


def make_child(command, log_path, cwd, environment, *, status="training_completed", initial="a" * 64,
               count=None, returncode=0):
    backend = command[command.index("--backend") + 1]
    output = Path(command[command.index("--output-dir") + 1])
    num_envs = int(command[command.index("--num-envs") + 1])
    iterations = int(command[command.index("--iterations") + 1])
    transitions = num_envs * 24 * iterations
    output.mkdir(parents=True, exist_ok=False)
    Path(log_path).write_text("synthetic child fixture; no training performed\n")
    for filename in ("model_final.pt", "policy.pt"):
        (output / filename).write_bytes(b"synthetic unit-test bytes, not a learned policy")
    PAIR.write_json(output / "run_status.json", {
        "status": status, "training_environment_steps": transitions if count is None else count,
        "checkpoint_sha256": PAIR.sha256(output / "model_final.pt"),
        "exported_policy_sha256": PAIR.sha256(output / "policy.pt")})
    PAIR.write_json(output / "manifest.json", {
        "backend": backend, "num_envs": num_envs, "iterations": iterations,
        "transitions_requested": transitions, "initial_policy_state_sha256": initial})
    return returncode


def read_status(args):
    return json.loads((args.output_dir / "pair_status.json").read_text())


def test_defaults_and_children_are_sequential_with_unmasked_gpu_zero(arguments, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    events = []

    def probe():
        events.append("probe")
        return idle()

    def child(command, log, cwd, environment):
        backend = command[command.index("--backend") + 1]
        events.append(backend)
        assert "CUDA_VISIBLE_DEVICES" not in environment
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert command[command.index("--device") + 1] == "cuda:0"
        assert cwd == arguments.snapshot
        return make_child(command, log, cwd, environment)

    assert PAIR.run_pair(arguments, gpu_probe=probe, child_runner=child) == 0
    assert events == ["probe", "mjlab", "probe", "physx"]
    status = read_status(arguments)
    assert status["status"] == "training_completed"
    assert status["matched_initialization_verified"] is True
    assert status["transitions_per_backend"] == 12_288_000
    assert status["num_envs"] == 512 and status["iterations"] == 1000
    assert status["automatic_retries"] is False
    assert (arguments.output_dir / "source_snapshot_hashes.json").is_file()


def test_busy_gpu_stops_without_starting_any_child(arguments):
    busy = dict(idle(), idle=False, compute_processes=[{"pid": "other-user-test-fixture"}])
    assert PAIR.run_pair(arguments, gpu_probe=lambda: busy,
                         child_runner=lambda *args: pytest.fail("must not start")) == 3
    status = read_status(arguments)
    assert status["status"] == "stopped_gpu_busy"
    assert status["children"][0]["status"] == "not_started_gpu_busy"
    assert not (arguments.output_dir / "mjlab").exists()


def test_busy_second_preflight_preserves_completed_first_child(arguments):
    probes = iter([idle(), dict(idle(), idle=False, memory_used_mib=120)])
    assert PAIR.run_pair(arguments, gpu_probe=lambda: next(probes), child_runner=make_child) == 3
    status = read_status(arguments)
    assert [child["status"] for child in status["children"]] == [
        "training_completed_verified", "not_started_gpu_busy"]
    assert not (arguments.output_dir / "physx").exists()


@pytest.mark.parametrize("status,count,returncode", [
    ("failed", None, 0), ("training_completed", 12, 0), ("training_completed", None, 7)])
def test_zero_exit_is_not_sufficient_for_completion(arguments, status, count, returncode):
    calls = []

    def child(*args):
        calls.append(args[0])
        return make_child(*args, status=status, count=count, returncode=returncode)

    assert PAIR.run_pair(arguments, gpu_probe=idle, child_runner=child) == 1
    assert len(calls) == 1
    assert read_status(arguments)["status"] == "failed"


def test_both_initial_policy_hashes_must_match(arguments):
    counter = iter(["a" * 64, "b" * 64])
    assert PAIR.run_pair(arguments, gpu_probe=idle,
                         child_runner=lambda *args: make_child(*args, initial=next(counter))) == 1
    status = read_status(arguments)
    assert status["status"] == "failed"
    assert "different initial policy states" in status["error"]
    assert len(status["children"]) == 2


def test_source_change_stops_before_second_child(arguments):
    def child(*args):
        result = make_child(*args)
        (arguments.snapshot / "scripts/train_ppo_backend.py").write_text("# changed test fixture\n")
        return result

    assert PAIR.run_pair(arguments, gpu_probe=idle, child_runner=child) == 1
    status = read_status(arguments)
    assert "SourceChangedError" in status["error"]
    assert len(status["children"]) == 1


def test_artifact_hash_is_verified_not_trusted(arguments):
    def child(*args):
        result = make_child(*args)
        output = Path(args[0][args[0].index("--output-dir") + 1])
        (output / "policy.pt").write_bytes(b"changed synthetic test bytes")
        return result

    assert PAIR.run_pair(arguments, gpu_probe=idle, child_runner=child) == 1
    assert "mismatched completed artifact" in read_status(arguments)["error"]


def test_existing_pair_is_never_overwritten(arguments):
    assert PAIR.run_pair(arguments, gpu_probe=idle, child_runner=make_child) == 0
    original = (arguments.output_dir / "pair_status.json").read_bytes()
    with pytest.raises(FileExistsError):
        PAIR.run_pair(arguments, gpu_probe=idle, child_runner=make_child)
    assert (arguments.output_dir / "pair_status.json").read_bytes() == original


@pytest.mark.parametrize("memory,utilization,processes,expected", [
    (95, 1, [], True), (96, 0, [], False), (7, 2, [], False),
    (7, 0, [{"gpu_uuid": "GPU-0", "pid": "123"}], False),
    (7, 0, [{"gpu_uuid": "GPU-6", "pid": "789"}], True)])
def test_gpu_thresholds_and_foreign_process_guard(monkeypatch, memory, utilization, processes, expected):
    gpu = [{"index": "0", "uuid": "GPU-0", "pci.bus_id": "00000000:18:00.0",
            "memory.used": str(memory), "utilization.gpu": str(utilization)}]
    monkeypatch.setattr(PAIR, "_query_csv", lambda fields, kind: gpu if kind == "gpu" else processes)
    assert PAIR.query_gpu_zero()["idle"] is expected


def test_source_hash_ignores_only_caches_and_detects_new_code(arguments):
    before = PAIR.snapshot_hashes(arguments.snapshot)
    cache = arguments.snapshot / "scripts/__pycache__"
    cache.mkdir()
    (cache / "test.pyc").write_bytes(b"generated test cache")
    assert PAIR.snapshot_hashes(arguments.snapshot) == before
    (arguments.snapshot / "scripts/new_source.py").write_text("# new source test fixture\n")
    with pytest.raises(PAIR.SourceChangedError):
        PAIR.verify_snapshot(arguments.snapshot, before)


def test_wrong_mjlab_pin_refused_before_output_creation(arguments, monkeypatch):
    monkeypatch.setattr(PAIR.subprocess, "check_output", lambda *args, **kwargs: "0" * 40 + "\n")
    with pytest.raises(ValueError, match="required"):
        PAIR.run_pair(arguments, gpu_probe=idle, child_runner=make_child)
    assert not arguments.output_dir.exists()


def test_python_symlink_is_preserved_for_virtual_environment_selection(arguments, tmp_path):
    interpreter = tmp_path / "isolated-env/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    arguments.python = interpreter
    PAIR._resolve_and_validate(arguments)
    assert arguments.python == interpreter
    assert arguments.python.is_symlink()


def test_gpu_pci_order_mismatch_is_refused(monkeypatch):
    devices = [
        {"index": "0", "uuid": "GPU-0", "pci.bus_id": "00000000:88:00.0",
         "memory.used": "7", "utilization.gpu": "0"},
        {"index": "1", "uuid": "GPU-1", "pci.bus_id": "00000000:18:00.0",
         "memory.used": "7", "utilization.gpu": "0"}]
    monkeypatch.setattr(PAIR, "_query_csv", lambda *args: devices)
    with pytest.raises(RuntimeError, match="not CUDA ordinal 0"):
        PAIR.query_gpu_zero()
