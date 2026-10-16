"""Orchestration tests only: no simulator, SSH, compiler or GPU job."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_controller_pair as pair
import run_nmpc_validation as driver


def arguments(tmp_path, action="plan", reference_id="drift_drl_traj4"):
    return driver.parser().parse_args([action, "--source-root", str(tmp_path / "source"),
        "--motion", str(tmp_path / "motion.npz"), "--reference-file", str(tmp_path / "reference.npz"),
        "--asset-path", str(tmp_path / "robot.usda"), "--model", str(tmp_path / "model.xml"),
        "--contract", str(tmp_path / "contract.json"), "--output-dir", str(tmp_path / "grid"),
        "--native-library", str(tmp_path / "lib_native.so"), "--reference-id", reference_id,
        "--device", "cuda:2"])


def option(command, name):
    return command[command.index(name) + 1]


def normalized(command):
    value = list(command)
    for name in ("--plant-mu", "--output-dir"):
        value[value.index(name) + 1] = "<allowed-case-change>"
    return value


@pytest.mark.parametrize("reference_id,frames", [("drift_drl_traj4", 349), ("doa_sa29_scale2", 258)])
def test_plan_only_has_34_commands_no_runtime_checks_or_side_effects(tmp_path, monkeypatch, reference_id, frames):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not launch, check GPUs, freeze or contact a host")
    for name in ("run", "Popen", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    monkeypatch.setattr(driver, "qualification", forbidden)
    monkeypatch.setattr(driver, "require_idle_gpu", forbidden)
    args = arguments(tmp_path, reference_id=reference_id)
    result = driver.build_plan(args)
    assert result["engine_episodes_planned"] == 34 and len(result["cases"]) == 17
    assert result["reference_spec"]["frames"] == frames
    assert not result["execution_performed"] and not result["freeze_verified"]
    assert not result["remote_permission_granted"] and not result["pooled_success_statistics"]
    assert not result["height_or_flight_failure_gate"] and not result["heading_offset_is_mismatch"]
    assert not list(tmp_path.iterdir())  # Inputs may be absent: this is explicitly an unverified plan.
    assert result["settings_contract"]["effective_cem_config"]["parameterization"] == "warm_start_residual_blocks"
    assert result["settings_contract"]["seed"] == 42
    for engine in ("isaac", "mujoco"):
        commands = [item[engine + "_command"] for item in result["cases"]]
        assert all(normalized(command) == normalized(commands[0]) for command in commands)
        for command in commands:
            assert option(command, "--settings") == str(args.settings.resolve())
            assert option(command, "--prediction-mu-scenarios") == "0.7"
            assert "--max-policy-steps" not in command and "--integration-only" not in command
            assert not any("ssh" in Path(part).name or part == "scp" for part in command[:2])
            assert not any(name in command for name in ("--checkpoint", "--policy", "--runner-config"))
            assert option(command, "--reference-id") == reference_id
        if engine == "isaac":
            assert option(commands[0], "--num-envs") == "1" and option(commands[0], "--seed") == "42"
            assert "--source-root" in commands[0] and "--headless" in commands[0]
        else:
            assert "--source-root" not in commands[0] and "--seed" not in commands[0]


@pytest.mark.parametrize("values", [[.7], [], [.7]*17, [float("nan")]*17,
                                  [.01, .03, .05, .1, .15, .2, .3, .45, .6, .7, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.1]])
def test_grid_does_not_silently_change_or_subset_the_frozen_17_cases(values):
    with pytest.raises(ValueError):
        driver.grid_values(values)


def test_explicit_per_engine_native_builds_and_python_optin(tmp_path):
    args = arguments(tmp_path)
    args.isaac_native_library = tmp_path / "isaac.so"
    args.mujoco_native_library = tmp_path / "mj.so"
    plan = driver.build_plan(args)
    assert option(plan["cases"][0]["isaac_command"], "--native-library") == str(args.isaac_native_library)
    assert option(plan["cases"][0]["mujoco_command"], "--native-library") == str(args.mujoco_native_library)
    args.prediction_backend = "python"
    with pytest.raises(ValueError, match="Native libraries"):
        driver.build_plan(args)
    args.native_library = args.isaac_native_library = args.mujoco_native_library = None
    plan = driver.build_plan(args)
    assert "--native-library" not in plan["cases"][0]["isaac_command"]


def test_missing_freeze_never_launches_even_with_execute_selected(tmp_path, monkeypatch):
    args = arguments(tmp_path, "execute")
    monkeypatch.setattr(driver, "launch", lambda *a: pytest.fail("No child may run"))
    monkeypatch.setattr(driver, "require_idle_gpu", lambda *a: pytest.fail("No GPU query before qualification"))
    with pytest.raises(ValueError, match="freeze-manifest"):
        driver.execute(args)
    assert not args.output_dir.exists()


def test_missing_both_nominal_directories_cannot_create_freeze(tmp_path, monkeypatch):
    args = arguments(tmp_path, "freeze"); args.freeze_output = tmp_path / "freeze.json"
    monkeypatch.setattr(driver, "runtime_paths", lambda *a: (_ for _ in ()).throw(ValueError("Missing Isaac nominal")))
    monkeypatch.setattr(driver, "verify_reference_pair", lambda *a: driver.get_reference(args.reference_id))
    with pytest.raises(ValueError, match="Missing Isaac nominal"):
        driver.freeze_controller(args)
    assert not args.freeze_output.exists() and not args.output_dir.exists()


def test_freeze_output_cannot_precreate_or_overwrite_grid_directory(tmp_path):
    args = arguments(tmp_path, "freeze")
    args.freeze_output = args.output_dir / "freeze.json"
    with pytest.raises(ValueError, match="outside"):
        driver.freeze_controller(args)
    args.freeze_output = tmp_path / "existing.json"; args.freeze_output.write_text("user owned")
    with pytest.raises(ValueError, match="existing"):
        driver.freeze_controller(args)
    assert args.freeze_output.read_text() == "user owned"


def test_gpu_guard_is_read_only_strict_unmasked_and_never_kills(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="6, 0\n")
    monkeypatch.setattr(subprocess, "run", fake)
    driver.require_idle_gpu("cuda:2")
    assert calls == [["nvidia-smi", "-i", "2", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]]
    for masked in ("", "2", "0,1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", masked)
        with pytest.raises(ValueError, match="Unset"):
            driver.require_idle_gpu("cuda:2")
    assert len(calls) == 1
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="100, 0"))
    with pytest.raises(ValueError, match="not idle"):
        driver.require_idle_gpu("cuda:2")


def mocked_execution(args, monkeypatch):
    args.freeze_manifest = args.output_dir.parent / "frozen.json"
    args.freeze_manifest.write_text("fixture only; not a real controller freeze")
    frozen = {"nominal": {engine: {"engine": engine, "frames": 349, "controller_contract": {"fixed": True}}
                           for engine in ("isaac", "mujoco")}}
    checks, launches = [], []
    monkeypatch.setattr(driver, "verified_freeze", lambda *a: checks.append(True) or copy.deepcopy(frozen))
    monkeypatch.setattr(driver, "require_beyondmimic_source", lambda *a: None)
    monkeypatch.setattr(driver, "require_idle_gpu", lambda *a: None)
    monkeypatch.setattr(driver, "launch", lambda command, log: launches.append(command) or 0)
    return checks, launches, frozen


def test_mocked_full_grid_varies_only_plant_mu_and_keeps_measured_failures(tmp_path, monkeypatch):
    args = arguments(tmp_path, "execute")
    checks, launches, frozen = mocked_execution(args, monkeypatch)
    def episode(directory, engine, spec, config, mu):
        return {**copy.deepcopy(frozen["nominal"][engine]), "plant_mu": mu, "report": {"plant_seed": 42}, "_directory": str(directory),
                "solver_diagnostics": {"fallback_used": 349}, "hashes": {"report": "a"*64}}
    def analyze(config):
        config.output_dir.mkdir()
        result = {"windows": {"core": {"isaac": {"large_xy_exploratory_screen": True},
                                       "mujoco": {"large_xy_exploratory_screen": True}}}, "posture_diagnostics": {}}
        (config.output_dir / "summary.json").write_text(json.dumps(result))
        return result
    monkeypatch.setattr(driver, "episode_evidence", episode)
    monkeypatch.setattr(driver, "validate_case_against_nominal", lambda actual, nominal:
                        driver.require(actual["controller_contract"] == nominal["controller_contract"], "changed"))
    monkeypatch.setattr(pair, "run", analyze)
    result = driver.execute(args)
    assert result["status"] == "completed" and result["completed_engine_episodes"] == 34
    assert len(launches) == 34 and len(checks) == 69
    assert all(row["executions"]["isaac"] == {"exit_code": 0, "log_file": "isaac.log", "artifacts_validated": True}
               for row in result["cases"])
    assert all(row["windows"]["core"]["isaac"]["large_xy_exploratory_screen"] for row in result["cases"])
    assert all(row["engines"]["mujoco"]["solver_diagnostics"]["fallback_used"] == 349 for row in result["cases"])
    assert not result["real_time_verified"] and not result["actuation_delay_injected"]
    assert (args.output_dir / "summary.json").is_file()
    assert all(normalized(command) == normalized(launches[0]) for command in launches[::2])
    assert all(normalized(command) == normalized(launches[1]) for command in launches[1::2])


def test_exit_zero_without_artifacts_is_runtime_incomplete_not_physical_success(tmp_path, monkeypatch):
    args = arguments(tmp_path, "execute")
    _, launches, _ = mocked_execution(args, monkeypatch)
    with pytest.raises(FileNotFoundError):
        driver.execute(args)
    manifest = json.loads((args.output_dir / "grid_manifest.json").read_text())
    assert manifest["status"] == "runtime_failure_incomplete"
    assert len(launches) == 1 and not (args.output_dir / "summary.json").exists()
    assert manifest["cases"][0]["engines"] == {}
    assert manifest["cases"][0]["executions"]["isaac"]["exit_code"] == 0
    assert not manifest["cases"][0]["executions"]["isaac"]["artifacts_validated"]


def test_cross_case_mutation_is_detected_before_another_child(tmp_path, monkeypatch):
    args = arguments(tmp_path, "execute")
    checks, launches, _ = mocked_execution(args, monkeypatch)
    def changed(*unused):
        checks.append(True)
        if len(checks) >= 3:
            raise ValueError("Frozen settings changed during child execution")
        return {"nominal": {}}
    monkeypatch.setattr(driver, "verified_freeze", changed)
    with pytest.raises(ValueError, match="changed during"):
        driver.execute(args)
    assert len(launches) == 1
    assert json.loads((args.output_dir / "grid_manifest.json").read_text())["status"] == "runtime_failure_incomplete"


def test_nonzero_child_exit_is_explicit_and_not_a_physical_xy_verdict(tmp_path, monkeypatch):
    args = arguments(tmp_path, "execute")
    mocked_execution(args, monkeypatch)
    monkeypatch.setattr(driver, "launch", lambda *unused: 13)
    with pytest.raises(ValueError, match="Child exit 13"):
        driver.execute(args)
    manifest = json.loads((args.output_dir / "grid_manifest.json").read_text())
    assert manifest["status"] == "runtime_failure_incomplete"
    row = manifest["cases"][0]
    assert row["executions"]["isaac"]["exit_code"] == 13 and row["engines"] == {}
    assert "windows" not in row and not (args.output_dir / "summary.json").exists()


def test_modified_freeze_evidence_rejected_even_when_structural_checks_are_mocked(tmp_path, monkeypatch):
    args = arguments(tmp_path, "execute"); args.freeze_manifest = tmp_path / "frozen.json"
    current = {"reference_spec": {}, "settings_contract": {}, "bindings": {},
               "evaluation_mu": driver.grid_values(), "validation_driver_sha256": driver.file_sha(driver.__file__),
               "nominal": {"isaac": {"immutable_value": 1}}}
    modified = copy.deepcopy(current); modified["nominal"]["isaac"]["immutable_value"] = 2
    args.freeze_manifest.write_text(json.dumps(modified))
    monkeypatch.setattr(driver, "qualification", lambda *unused: (current, {}))
    monkeypatch.setattr(driver, "validate_freeze", lambda *unused: None)
    with pytest.raises(ValueError, match="evidence.*modified"):
        driver.verified_freeze(args)


def test_different_reward_profile_cannot_qualify_for_the_matched_grid(tmp_path):
    args = arguments(tmp_path, "freeze")
    args.task_profile = tmp_path / "different_task.json"; args.task_profile.write_text("{}")
    with pytest.raises(ValueError, match="shared height-smooth"):
        driver.qualification(args)


def test_cem_seed_is_not_silently_decoupled_from_matched_seed42(tmp_path):
    args = arguments(tmp_path)
    config = json.loads(args.settings.read_text()); config["nmpc"]["seed"] = 43
    args.settings = tmp_path / "changed_settings.json"; args.settings.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="CEM seed 42"):
        driver.build_plan(args)
