"""Asset-free synthetic effort/action clocks; never import or run simulators."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_nominal_control_burden as audit


def raw_fixture(kind, action_value=.5):
    n = 1397
    fields = {"position": np.zeros((n, 3)), "quaternion": np.tile([1., 0., 0., 0.], (n, 1)),
        "com_velocity": np.zeros((n, 3)), "angular_velocity": np.zeros((n, 3)), "joint_pos": np.zeros((n, 8)),
        "joint_vel": np.zeros((n, 8)), "wheel_pos": np.zeros((n, 2, 3)), "contact_count": np.zeros((n, 2)),
        "contact_force": np.zeros((n, 2, 3)), "contact_slip": np.zeros((n, 2))}
    mapping = {
        "policy_isaac": ("root_position_m", "root_quaternion_wxyz", "root_linear_velocity_w_m_s", "root_angular_velocity_w_rad_s",
                          "joint_position_rad", "joint_velocity_rad_s", "wheel_position_m"),
        "policy_mujoco": ("root_pos", "root_quat_wxyz", "root_com_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel", "wheel_pos"),
        "nmpc": ("root_pos", "root_quaternion_wxyz", "base_com_velocity_w", "base_angular_velocity_w", "joint_pos", "joint_vel", "wheel_pos")}
    names = (*mapping[kind], "wheel_ground_contact_count", "wheel_ground_contact_force_w_n", "wheel_lateral_contact_slip_m_s")
    raw = {source: value[:, None] if kind == "policy_isaac" else value.copy()
           for source, value in zip(names, fields.values())}
    raw.update(time_s=np.arange(n) * .005, physics_dt_s=np.array(.005), policy_dt_s=np.array(.02))
    raw["action_joint_names" if kind == "nmpc" else "joint_names"] = np.array(audit.JOINT_NAMES)
    actions = np.full((349, 8), action_value, dtype=np.float32)
    held = np.vstack((np.zeros((1, 8)), np.repeat(actions, 4, axis=0)))
    phase = np.r_[0, np.repeat(np.arange(349), 4)]
    applied = np.r_[False, np.ones(1396, dtype=bool)]
    torque = np.tile([3.] * 6 + [4., 4.], (n, 1)); torque[0] = 1e6
    if kind == "policy_isaac":
        raw.update(actions_clipped=held[:, None], applied_joint_torque_nm=torque[:, None],
            reference_frame_for_action=phase[:, None], action_was_applied=applied[:, None],
            valid_mask=np.ones((n, 1), dtype=bool), simulation_step_counter=np.arange(n) + 124)
    elif kind == "policy_mujoco":
        raw.update(actions_clipped=actions.astype(float), joint_torque_nm=torque,
            reference_frame_for_action=np.arange(349), reference_frame=phase, policy_time_s=np.arange(349) * .02)
    else:
        raw.update(actions=actions.astype(float), held_action=held, applied_joint_torque_nm=torque,
            reference_frame_for_action=phase, action_was_applied=applied)
    return raw


def contract_fixture():
    return {"physics_dt": .005, "decimation": 4, "raw_action_clip": [-1., 1.], "leg_position_scale": 1.,
        "leg_position_offset": "reference_frame_0_joint_pos", "wheel_torque_scale_nm": 12., "leg_kp": 500., "leg_kd": 10.,
        "leg_torque_limit_nm": 80., "wheel_torque_limit_nm": 12., "actuator_model": "dc_motor",
        "leg_saturation_effort_nm": 80., "wheel_saturation_effort_nm": 12.,
        "leg_motor_velocity_limit_rad_s": 15., "wheel_motor_velocity_limit_rad_s": 100.,
        "observed_joint_velocity_order": list(audit.JOINT_NAMES)}


def test_statistics_distinguish_component_pool_from_group_norm():
    result = audit.statistics([[3., 4.], [0., 0.]], ["a", "b"])
    assert result["pooled_component"]["rms"] == 2.5
    assert result["pooled_component"]["absolute_p95"] == pytest.approx(3.85)
    assert result["group_l2"]["rms"] == pytest.approx(np.sqrt(12.5))
    assert result["group_l2"]["peak"] == 5.
    assert result["per_joint"]["b"]["rms"] == pytest.approx(np.sqrt(8.))
    with pytest.raises(ValueError): audit.statistics([[float("nan")]], ["a"])
    with pytest.raises(ValueError): audit.statistics(np.empty((0, 2)), ["a", "b"])


@pytest.mark.parametrize("kind", ["policy_isaac", "policy_mujoco", "nmpc"])
def test_full_and_core_clocks_exclude_initial_cached_torque_and_invented_rate(kind):
    result = audit.burden_windows(audit.native_control(raw_fixture(kind), kind))
    assert {k: result["full"][k] for k in ("command_intervals", "poststep_torque_samples", "successive_command_rate_samples")} == {
        "command_intervals": 349, "poststep_torque_samples": 1396, "successive_command_rate_samples": 348}
    assert {k: result["core"][k] for k in ("command_intervals", "poststep_torque_samples", "successive_command_rate_samples")} == {
        "command_intervals": 128, "poststep_torque_samples": 512, "successive_command_rate_samples": 128}
    for window in result.values():
        assert window["actual_joint_torque_nm"]["legs"]["pooled_component"]["rms"] == 3.
        assert window["actual_joint_torque_nm"]["wheels"]["pooled_component"]["absolute_peak"] == 4.
        assert window["actual_joint_torque_nm"]["legs"]["group_l2"]["rms"] == pytest.approx(3 * np.sqrt(6))
        assert window["leg_target_rate_rad_s"]["pooled_component"]["rms"] == 0.
        assert window["wheel_requested_torque_nm"]["pooled_component"]["rms"] == 6.
        assert window["wheel_action_rate_normalized_s"]["pooled_component"]["absolute_peak"] == 0.


def test_core_boundary_uses_actual_prior_command_but_not_previous_interval_torque():
    control = audit.native_control(raw_fixture("nmpc", 0.), "nmpc")
    control["actions"][72:, :6] = 1.
    control["torque"][288, :6] = 80.  # t=1.44 belongs to the previous interval.
    result = audit.burden_windows(control)
    assert result["core"]["leg_target_rate_rad_s"]["pooled_component"]["rms"] == pytest.approx(50 / np.sqrt(128))
    assert result["core"]["actual_joint_torque_nm"]["legs"]["pooled_component"]["absolute_peak"] == 3.


@pytest.mark.parametrize("problem", ["short", "clock", "dt", "order", "phase", "counter", "held", "mask", "floatprecision", "nonfinite", "bounds", "torque"])
def test_bad_native_trace_or_action_contract_is_rejected(problem):
    raw = raw_fixture("policy_isaac")
    if problem == "short": raw["time_s"] = raw["time_s"][:-1]
    elif problem == "clock": raw["time_s"][100:] += .005
    elif problem == "dt": raw["policy_dt_s"] = np.array(.04)
    elif problem == "order": raw["joint_names"] = raw["joint_names"][::-1]
    elif problem == "phase": raw["reference_frame_for_action"][4, 0] = 1
    elif problem == "counter": raw["simulation_step_counter"][100] -= 1
    elif problem == "held": raw["actions_clipped"][2, 0, 0] = .1
    elif problem == "mask": raw["action_was_applied"] = raw["action_was_applied"].astype(int)
    elif problem == "floatprecision": raw["actions_clipped"] = np.full((1397, 1, 8), .123456789)
    elif problem == "nonfinite": raw["applied_joint_torque_nm"][10, 0, 0] = np.nan
    elif problem == "bounds": raw["actions_clipped"][1:5] = 1.01
    elif problem == "torque": raw["applied_joint_torque_nm"][10, 0, 7] = 13.
    with pytest.raises(ValueError): audit.native_control(raw, "policy_isaac")


def test_censored_or_reset_spliced_episode_is_not_full_coverage():
    raw = raw_fixture("policy_isaac"); raw["valid_mask"][-1] = False
    with pytest.raises(ValueError): audit.native_control(raw, "policy_isaac")
    raw["valid_mask"][100] = False
    with pytest.raises(ValueError): audit.native_control(raw, "policy_isaac")


def pair_fixture(tmp_path, label="PPO"):
    algorithm, blocks = audit.STUDY[label]
    folder = tmp_path / label; folder.mkdir()
    contract = contract_fixture(); contract_path = tmp_path / (label + "_contract.json")
    contract_path.write_text(json.dumps(contract))
    pair = {"algorithm": algorithm, "reference_spec": audit.get_reference("drift_drl_traj4"), "fixed_mu": .7,
        "core_coverage_complete": True, "integration_only": False, "height_or_flight_failure_gate": False,
        "heading_offset_is_mismatch": False, "audit": {"height_free_objective_and_evaluation": True, "trace_bindings": {}},
        "analysis_source_sha256": {"synthetic_fixture": "a" * 64}, "reported_plant_source_provenance": {},
        "input_sha256": {}, "windows": {"core": {}}, "posture_diagnostics": {}}
    compact = {}; paths = []
    for engine in ("isaac", "mujoco"):
        kind = "nmpc" if algorithm == "NMPC" else "policy_" + engine
        raw = raw_fixture(kind); rawpath = folder / (engine + ".npz"); np.savez_compressed(rawpath, **raw)
        report = {"policy_export" if engine == "isaac" else "contract": {**contract, "policy_algorithm": algorithm}}
        if algorithm == "NMPC":
            report.update(status="completed", solver_failures=0, plant_mu=.7, prediction_mu_scenarios=[.7],
                source_sha256={"private_contract": audit.sha(contract_path)}, rollout_sha256=audit.sha(rawpath),
                effective_cem_config={"horizon_steps": 30, "control_blocks": blocks, "candidates": 64, "iterations": 3,
                                      "elite_count": 8, "seed": 42})
        elif engine == "isaac": report["physics_trajectory_sha256"] = audit.sha(rawpath)
        reportpath = folder / (engine + ".json"); reportpath.write_text(json.dumps(report))
        pair["input_sha256"].update({engine + "_trace": audit.sha(rawpath), engine + "_report": audit.sha(reportpath)})
        trace = audit.native_control(raw, kind)["trace"]
        for name, values in trace.items():
            if name not in ("physics_dt_s", "joint_names"): compact[engine + "_" + name] = values
        pair["audit"]["trace_bindings"][engine] = {"sha256": audit.sha(rawpath), "scope": "synthetic fixture"}
        pair["reported_plant_source_provenance"][engine] = {"reported_source_hashes": []}
        pair["windows"]["core"][engine] = {key: None for key in audit.CORE_FIELDS}
        pair["windows"]["core"][engine].update(samples_in_window=513, coverage_complete=True,
            reference_xy_rmse_m=.1, reference_xy_peak_m=.2, mean_speed_retention_ratio=1., window_s=[1.44, 4.])
        pair["posture_diagnostics"][engine] = {"max_base_tilt_deg": 0.}
        paths.extend([rawpath, reportpath])
    np.savez_compressed(folder / "pair_trace.npz", **compact); (folder / "topdown.png").write_bytes(b"synthetic existing PNG identity only")
    pair["artifacts"] = {"compact_trace_sha256": audit.sha(folder / "pair_trace.npz"), "topdown_sha256": audit.sha(folder / "topdown.png")}
    (folder / "summary.json").write_text(json.dumps(pair))
    return [label, folder, *paths], contract_path


def test_hashbound_native_pairs_copy_canonical_metrics_without_new_fk(tmp_path):
    case, path = pair_fixture(tmp_path)
    result = audit.analyze_case(*case, common_contract=audit.actuator_contract(path), contract_sha256=audit.sha(path))
    assert result["engines"]["isaac"]["canonical_core_metrics"]["reference_xy_rmse_m"] == .1
    assert result["engines"]["mujoco"]["burden"]["core"]["poststep_torque_samples"] == 512
    assert str(tmp_path) not in json.dumps(result)
    case[2].write_bytes(b"changed trace")
    with pytest.raises(ValueError, match="byte binding"):
        audit.analyze_case(*case, common_contract=audit.actuator_contract(path), contract_sha256=audit.sha(path))


@pytest.mark.parametrize("problem", ["compact", "png", "mapping", "metrics", "fallback", "iterations"])
def test_fail_closed_pair_artifact_and_nominal_configuration(problem, tmp_path):
    case, path = pair_fixture(tmp_path, "NMPC_H30_B5")
    folder = case[1]; pair = json.loads((folder / "summary.json").read_text())
    if problem in ("compact", "png"): (folder / ("pair_trace.npz" if problem == "compact" else "topdown.png")).write_bytes(b"changed")
    elif problem == "mapping": path.write_text(json.dumps({**contract_fixture(), "wheel_torque_scale_nm": 10.}))
    elif problem == "metrics": del pair["windows"]["core"]["isaac"]["heading_geometry"]
    else:
        report = json.loads(case[3].read_text())
        if problem == "fallback": report["solver_failures"] = 1
        else: report["effective_cem_config"]["iterations"] = 6
        case[3].write_text(json.dumps(report)); pair["input_sha256"]["isaac_report"] = audit.sha(case[3])
    (folder / "summary.json").write_text(json.dumps(pair))
    with pytest.raises(ValueError): audit.analyze_case(*case, common_contract=audit.actuator_contract(path), contract_sha256=audit.sha(path))


def test_portable_allowlist_and_no_simulator_imports():
    with pytest.raises(ValueError): audit.portable({"path": "/home/private/model.xml"})
    code = """
import sys
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mujoco','torch','isaaclab'}: raise RuntimeError('Forbidden '+fullname)
sys.meta_path.insert(0,Block())
import analyze_nominal_control_burden
assert len(analyze_nominal_control_burden.STUDY)==4
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "scripts", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_complete_four_pair_study_has_eight_episodes_no_rank_or_new_physics(tmp_path):
    fixtures = [pair_fixture(tmp_path, label) for label in audit.STUDY]
    result = audit.analyze([case for case, _ in fixtures], fixtures[0][1])
    assert result["engine_episodes"] == 8 and result["controller_pairs"] == 4
    assert {row["label"] for row in result["cases"]} == set(audit.STUDY)
    assert not result["new_physics_or_inference"] and not result["algorithm_ranking_claimed"]
    assert not result["controller_freeze_claimed"] and not result["friction_grid_claimed"]
    assert "512" in result["definitions"]["burden_window"]
    assert "513" in result["definitions"]["canonical_metrics"]
    assert "energy" in " ".join(result["not_computed"])
    assert str(tmp_path) not in json.dumps(result)
    with pytest.raises(ValueError): audit.analyze([fixtures[0][0]] * 4, fixtures[0][1])


def test_public_aggregate_source_hashes_counts_and_canonical_metrics_preserved():
    folder = ROOT / "results/nominal_control_burden_20261016_v1"
    assert {path.name for path in folder.iterdir()} == {"summary.json"}
    assert audit.sha(folder / "summary.json") == "99ac180db404e55e71f8d0a416ca8c1a7857b30b436be70665bc6327a8c1cf4f"
    result = audit.read_json(folder / "summary.json")
    audit.portable(result)
    assert result["source_sha256"]["analyze_nominal_control_burden.py"] == audit.sha(ROOT / "scripts/analyze_nominal_control_burden.py")
    assert result["common_actuator_contract_sha256"] == "6e105cbd6efe7d35117a895fa097e8ec64b6ac822a9de07c00acfaef4d09642f"
    for case in result["cases"]:
        pair_dir = ROOT / "results" / case["canonical_pair"]
        pair = audit.read_json(pair_dir / "summary.json")
        assert case["canonical_pair_sha256"] == {name: audit.sha(pair_dir / name) for name in ("summary.json", "pair_trace.npz", "topdown.png")}
        assert case["reported_plant_source_provenance"] == pair["reported_plant_source_provenance"]
        for engine, item in case["engines"].items():
            assert item["canonical_core_metrics"] == pair["windows"]["core"][engine]
            assert item["canonical_posture_diagnostic"] == pair["posture_diagnostics"][engine]
            assert item["input_sha256"]["raw_trace"] == pair["input_sha256"][engine + "_trace"]
            assert item["policy_decisions"] == 349 and item["native_samples"] == 1397
            for label, commands, physical, rates in (("core", 128, 512, 128), ("full", 349, 1396, 348)):
                window = item["burden"][label]
                assert (window["command_intervals"], window["poststep_torque_samples"], window["successive_command_rate_samples"]) == (commands, physical, rates)
                for group, count in (("legs", 6), ("wheels", 2)):
                    values = window["actual_joint_torque_nm"][group]
                    assert values["group_l2"]["rms"] == pytest.approx(np.sqrt(count) * values["pooled_component"]["rms"])
    note = (ROOT / "results/2026-10-16-nominal-control-burden.md").read_text()
    for private in ("/home/", "/data/", "zengxiong", "14.103.233.29"):
        assert private not in note
    assert "not a controller ranking" in note
    assert "not measured joint velocity" in note
    assert "no power or energy" in note
