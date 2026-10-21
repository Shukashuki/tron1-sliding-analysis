"""Training configuration/guard checks without Torch or simulator imports."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_ppo_backend.py"
SPEC = importlib.util.spec_from_file_location("train_ppo_backend", SCRIPT)
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def load_config_function(path):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "ppo_config")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ppo_config"]


@pytest.mark.parametrize("seed,save_interval", [(42, 100), (7, 2), (123, 5)])
def test_algorithm_config_matches_original_training_exactly(seed, save_interval):
    original = ROOT / "upstream/tron1-learning-from-human/scripts/train_tracking.py"
    if not original.is_file():
        pytest.skip("Initialize the TRON1 source submodule for original-config parity")
    expected = load_config_function(original)(seed, save_interval)
    actual = TRAIN.ppo_config(seed, save_interval)
    expected.pop("experiment_name")
    actual.pop("experiment_name")
    assert actual == expected


def test_config_calls_do_not_share_mutable_nested_state():
    first = TRAIN.ppo_config()
    first["policy"].pop("class_name")
    first["obs_groups"]["policy"].append("unexpected")
    second = TRAIN.ppo_config()
    assert second["policy"]["class_name"] == "ActorCritic"
    assert second["obs_groups"]["policy"] == ["policy"]


def test_existing_experiment_is_refused_before_importing_gpu_packages(tmp_path):
    evidence = tmp_path / "keep.json"
    evidence.write_text('{"untouched": true}\n')
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "mjlab", "--model", "unused.xml",
         "--motion", "unused.npz", "--output-dir", str(tmp_path)],
        capture_output=True, text=True)
    assert completed.returncode == 2
    assert "never overwrite an experiment" in completed.stderr
    assert json.loads(evidence.read_text()) == {"untouched": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.json"]


def test_nonfinite_manifest_is_refused_without_writing(tmp_path):
    target = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        TRAIN.write_json(target, {"invalid": float("nan")})
    assert not target.exists()


def test_initial_network_is_seeded_after_environment_setup_before_runner():
    tree = ast.parse(SCRIPT.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    runner_line = next(node.lineno for node in calls if isinstance(node.func, ast.Name)
                       and node.func.id == "OnPolicyRunner")
    seed_lines = [node.lineno for node in calls if isinstance(node.func, ast.Attribute)
                  and node.func.attr in ("manual_seed", "manual_seed_all")]
    assert len(seed_lines) == 2 and all(line < runner_line for line in seed_lines)
    assert min(seed_lines) > max(node.lineno for node in calls
                                if isinstance(node.func, ast.Attribute)
                                and node.func.attr == "reset")


def test_physx_height_adapter_is_installed_before_environment_construction():
    tree = ast.parse(SCRIPT.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    install = next(n.lineno for n in calls if n.func.id == "install_height_smoothness")
    construct = next(n.lineno for n in calls if n.func.id == "ManagerBasedRLEnv")
    assert install < construct
    assert "PhysX must resolve the same 15-term height-smooth objective" in SCRIPT.read_text()


def test_locked_protocol_separates_training_dr_and_wide_validation_grid():
    config = json.loads((ROOT/"config/ppo_mjlab_backend_comparison_v1.json").read_text())
    assert config["training_randomization"]["sliding_friction"]["range"] == [.3, 1.2]
    assert len(config["validation"]["friction_grid"]) == 17
    assert config["validation"]["friction_grid"][0] == .01
    assert config["validation"]["friction_grid"][-1] == 2.
    assert config["task_contract"]["absolute_height_gate"] is False
    assert config["task_contract"]["airborne_failure"] is False
    assert config["algorithm"]["transitions"] == 512*24*1000
