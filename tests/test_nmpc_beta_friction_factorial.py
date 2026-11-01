"""Asset-free checks for the locked exact-friction MPC comparison grid."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_nmpc_beta_friction_factorial import case_name, directory
from dispatch_nmpc_oracle_isaac import case_key, GRID
from dispatch_nmpc_beta_friction_17_isaac import GRID as ISAAC_17_GRID, BRANCHES as ISAAC_17_BRANCHES
from dispatch_nmpc_beta_friction_17_mujoco import GRID as MUJOCO_17_GRID, BRANCHES as MUJOCO_17_BRANCHES


def test_common_grid_and_case_names():
    config = json.loads((ROOT / "config/nmpc_beta_friction_factorial.json").read_text())
    assert config["ground_mu_equal_static_dynamic"] == [.05, .1, .2, .3, .7, 1.6]
    assert GRID == (.05, .1, .2, .3, 1.6)
    assert [case_name(mu) for mu in config["ground_mu_equal_static_dynamic"]] == [
        "mu0p050", "mu0p100", "mu0p200", "mu0p300", "mu0p700", "mu1p600"]
    assert [case_key(mu) for mu in GRID] == ["mu0p050", "mu0p100", "mu0p200", "mu0p300", "mu1p600"]


def test_profiles_differ_only_in_sideslip_weight_and_description():
    side = json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())
    no_side = json.loads((ROOT / "config/car_drift_height_smooth_no_sideslip_task.json").read_text())
    assert side["reward_weights"]["sideslip"] == 2.
    assert no_side["reward_weights"]["sideslip"] == 0.
    side["reward_weights"]["sideslip"] = 0.
    side.pop("description")
    no_side.pop("description")
    assert side == no_side


def test_nominal_reuse_and_prior_oracle_cases_are_explicit():
    assert directory("side", "oracle", "MuJoCo", .7) == directory("side", "fixed", "MuJoCo", .7)
    assert directory("no_side", "oracle", "Isaac", .7) == directory("no_side", "fixed", "Isaac", .7)
    assert "no_sideslip_cem_oracle_friction_pilot_20261016_v1" in str(
        directory("no_side", "oracle", "MuJoCo", .1))
    assert "sideslip_cem_oracle_friction_grid_20261017_v1" in str(
        directory("side", "oracle", "MuJoCo", .1))
    assert "nmpc_oracle_isaac_grid_20261017_v1" in str(
        directory("side", "oracle", "Isaac", .1))


def test_legacy_seventeen_point_extension_keeps_old_cases():
    config = json.loads((ROOT / "config/nmpc_beta_friction_factorial_17.json").read_text())
    old = json.loads((ROOT / "config/nmpc_beta_friction_factorial.json").read_text())
    assert len(config["ground_mu_equal_static_dynamic"]) == 17
    assert config["reused_six_point_mu"] == old["ground_mu_equal_static_dynamic"]
    assert sorted(config["new_mu"] + config["reused_six_point_mu"]) == config["ground_mu_equal_static_dynamic"]
    assert tuple(config["new_mu"]) == ISAAC_17_GRID == MUJOCO_17_GRID
    assert tuple(ISAAC_17_BRANCHES) == tuple(MUJOCO_17_BRANCHES) == (
        "side_oracle", "no_side_fixed", "no_side_oracle")
    assert "nmpc_primary_isaac_friction_development" in str(directory("side", "fixed", "Isaac", .45))
    assert "nmpc_beta_friction_17_isaac" in str(directory("side", "oracle", "Isaac", .45))
    assert "nmpc_beta_friction_17_mujoco" in str(directory("no_side", "fixed", "MuJoCo", .45))
    assert directory("no_side", "oracle", "Isaac", .7) == directory("no_side", "fixed", "Isaac", .7)
