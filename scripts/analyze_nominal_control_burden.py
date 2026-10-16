"""Offline control-burden audit of four existing nominal controller pairs.

Only NumPy/stdlib and the pure canonical trace adapter are used. No model,
policy, FK, dynamics, simulator process, network or new friction case is used.
Raw reports/traces must match their existing public canonical pair byte hashes.
The output contains portable aggregate statistics, never raw licensed states.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from controller_references import get_reference
from controller_trace import native_trace
from nmpc_tracking_objective import JOINT_NAMES


STUDY = {"PPO": ("PPO", None), "SAC": ("SAC", None),
         "NMPC_H30_B5": ("NMPC", 5), "NMPC_H30_B10": ("NMPC", 10)}
ACTUATOR_FIELDS = ("physics_dt", "decimation", "raw_action_clip", "leg_position_scale", "leg_position_offset",
    "wheel_torque_scale_nm", "leg_kp", "leg_kd", "leg_torque_limit_nm", "wheel_torque_limit_nm",
    "actuator_model", "leg_saturation_effort_nm", "wheel_saturation_effort_nm",
    "leg_motor_velocity_limit_rad_s", "wheel_motor_velocity_limit_rad_s", "observed_joint_velocity_order")
CORE_FIELDS = {"coverage_complete", "fraction_below_80pct_reference_speed", "ground_contact_heading_geometry",
    "ground_contact_lateral_slip_m_s", "heading_geometry", "heading_offset_is_mismatch", "height_or_flight_failure_gate",
    "large_xy_exploratory_screen", "mean_speed_m_s", "mean_speed_retention_ratio", "net_course_change_deg",
    "reference_mean_speed_m_s", "reference_xy_peak_m", "reference_xy_rmse_m", "samples_in_window", "window_s", "xy_screen_limits_m"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    json.dumps(value, allow_nan=False)
    return value


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def portable(value):
    """Fail rather than publish arbitrary raw metadata or private host paths."""
    json.dumps(value, allow_nan=False)
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), "Non-string metadata key")
        for key, item in value.items(): portable(key); portable(item)
    elif isinstance(value, list):
        for item in value: portable(item)
    elif isinstance(value, str):
        require(not value.startswith(("/", "~", "\\"))
                and not any(token in value for token in ("/home/", "/data/", "zengxiong", "14.103.233.29", "117.50.226.213")),
                "Private/absolute path in publication")
    return value


def statistics(values, names):
    """RMS and absolute component quantiles; group norm uses all named axes."""
    values = np.asarray(values, dtype=float)
    require(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] == len(names)
            and np.isfinite(values).all(), "Require a nonempty finite named sample matrix")
    absolute = np.abs(values)
    norm = np.linalg.norm(values, axis=1)
    return {"samples": len(values), "components": len(names),
        "pooled_component": {"rms": float(np.sqrt(np.mean(values ** 2))),
                             "absolute_p95": float(np.quantile(absolute, .95)), "absolute_peak": float(absolute.max())},
        "group_l2": {"rms": float(np.sqrt(np.mean(norm ** 2))), "p95": float(np.quantile(norm, .95)), "peak": float(norm.max())},
        "per_joint": {name: {"rms": float(np.sqrt(np.mean(values[:, k] ** 2))),
                             "absolute_p95": float(np.quantile(absolute[:, k], .95)), "absolute_peak": float(absolute[:, k].max())}
                      for k, name in enumerate(names)}}


def native_control(raw, kind):
    """Validate applied actions/torques and decimation, excluding reset caches."""
    require(kind in ("policy_isaac", "policy_mujoco", "nmpc"), "Unsupported trace kind")
    trace = native_trace(raw, kind)
    n = 349; samples = 4 * n + 1
    require(len(trace["time_s"]) == samples and np.allclose(trace["time_s"], np.arange(samples) * .005, rtol=0, atol=1e-9)
            and trace["physics_dt_s"] == .005 and float(raw["policy_dt_s"]) == .02, "Require full349/1397 original5ms/20ms clocks")
    names_key = "action_joint_names" if kind == "nmpc" else "joint_names"
    require(tuple(raw[names_key].tolist()) == JOINT_NAMES, "Recorded joint/torque axes differ from canonical action order")
    if kind == "policy_isaac":
        require(raw["valid_mask"].shape == (samples, 1) and raw["valid_mask"].dtype.kind == "b"
                and raw["valid_mask"].all(), "Require one complete first episode; no reset splicing")
        held = np.asarray(raw["actions_clipped"])
        require(held.shape == (samples, 1, 8), "Invalid Isaac held actions")
        held = held[:, 0]
        actions = held[1::4]
        torque = np.asarray(raw["applied_joint_torque_nm"])
        require(torque.shape == (samples, 1, 8), "Invalid Isaac applied torques")
        torque = torque[:, 0]
        phase = raw["reference_frame_for_action"][:, 0]
        applied = raw["action_was_applied"][:, 0]
    elif kind == "policy_mujoco":
        actions = np.asarray(raw["actions_clipped"])
        torque = np.asarray(raw["joint_torque_nm"])
        require(np.array_equal(raw["reference_frame_for_action"], np.arange(n))
                and np.allclose(raw["policy_time_s"], np.arange(n) * .02, atol=1e-9, rtol=0), "MuJoCo policy issue-clock mismatch")
        phase = raw["reference_frame"]
        held = np.vstack((np.zeros((1, 8)), np.repeat(actions, 4, axis=0)))
        applied = np.r_[False, np.ones(samples - 1, dtype=bool)]
    else:
        actions = np.asarray(raw["actions"])
        torque = np.asarray(raw["applied_joint_torque_nm"])
        held = np.asarray(raw["held_action"])
        phase = raw["reference_frame_for_action"]
        applied = raw["action_was_applied"]
    require(actions.shape == (n, 8) and actions.dtype.kind == "f" and np.isfinite(actions).all()
            and np.max(np.abs(actions)) <= 1. and np.array_equal(actions, actions.astype(np.float32).astype(actions.dtype)),
            "Require finite normalized applied float32 actions, not raw policy/CEM proposals")
    require(held.shape == (samples, 8) and np.array_equal(held, np.vstack((np.zeros((1, 8)), np.repeat(actions, 4, axis=0)))),
            "Applied actions were not held for four consecutive physical steps")
    require(np.array_equal(phase, np.r_[0, np.repeat(np.arange(n), 4)])
            and applied.dtype.kind == "b" and np.array_equal(applied, np.r_[False, np.ones(samples - 1, dtype=bool)]),
            "Action application/reference phase mismatch")
    require(torque.shape == (samples, 8) and torque.dtype.kind in "if" and np.isfinite(torque).all(), "Invalid actual torque samples")
    require(np.all(np.abs(torque[1:, :6]) <= 80.001) and np.all(np.abs(torque[1:, 6:]) <= 12.001),
            "Actual torque violates the shared bounded actuator contract")
    if "simulation_step_counter" in raw:
        counter = raw["simulation_step_counter"]
        require(counter.shape == (samples,) and np.all(np.diff(counter) == 1), "Physical counter skipped/reset")
    return {"trace": trace, "actions": actions.astype(float), "torque": torque.astype(float)}


def burden_windows(control, core=(1.44, 4.00)):
    """Align issued controls [start,end) with post-step torque (start,end].

    No invented initial previous action: rates use differences between actual
    successive issued commands, so full-episode rates have348 rather than349
    rows. A core boundary change includes its actual preceding command.
    """
    actions, torque = control["actions"], control["torque"]
    issue_time = np.arange(349) * .02
    physical_time = control["trace"]["time_s"]
    rate = np.diff(actions, axis=0) / .02
    results = {}
    for label, (start, end) in (("full", (0., 6.98)), ("core", tuple(core))):
        commands = (issue_time >= start - 1e-9) & (issue_time < end - 1e-9)
        physical = (physical_time > start + 1e-9) & (physical_time <= end + 1e-9)
        rates = commands[1:]
        require(physical.sum() == 4 * commands.sum(), "Control and torque window clocks are not matched")
        results[label] = {"interval_s": [start, end], "command_intervals": int(commands.sum()),
            "poststep_torque_samples": int(physical.sum()), "successive_command_rate_samples": int(rates.sum()),
            "actual_joint_torque_nm": {"legs": statistics(torque[physical, :6], JOINT_NAMES[:6]),
                                       "wheels": statistics(torque[physical, 6:], JOINT_NAMES[6:])},
            "leg_target_rate_rad_s": statistics(rate[rates, :6], JOINT_NAMES[:6]),
            "wheel_action_rate_normalized_s": statistics(rate[rates, 6:], JOINT_NAMES[6:]),
            "wheel_requested_torque_nm": statistics(12. * actions[commands, 6:], JOINT_NAMES[6:])}
    return results


def actuator_contract(path):
    raw = read_json(path)
    contract = {name: raw.get(name) for name in ACTUATOR_FIELDS}
    require(contract["physics_dt"] == .005 and contract["decimation"] == 4
            and contract["raw_action_clip"] == [-1., 1.] and contract["leg_position_scale"] == 1.
            and contract["leg_position_offset"] == "reference_frame_0_joint_pos" and contract["wheel_torque_scale_nm"] == 12.
            and contract["leg_kp"] == 500. and contract["leg_kd"] == 10.
            and contract["actuator_model"] == "dc_motor" and contract["observed_joint_velocity_order"] == list(JOINT_NAMES),
            "Unsupported common actuator mapping")
    return contract


def audit_pair(folder, algorithm):
    folder = Path(folder)
    pair = read_json(folder / "summary.json")
    require(pair.get("algorithm") == algorithm and pair.get("reference_spec") == get_reference("drift_drl_traj4")
            and pair.get("fixed_mu") == .7 and pair.get("core_coverage_complete") is True
            and pair.get("integration_only") is False and pair.get("height_or_flight_failure_gate") is False
            and pair.get("heading_offset_is_mismatch") is False, "Require canonical primary full nominal pair")
    require(pair.get("audit", {}).get("height_free_objective_and_evaluation") is True,
            "Canonical pair is not the height-free task")
    for name, key in (("pair_trace.npz", "compact_trace_sha256"), ("topdown.png", "topdown_sha256")):
        require(sha(folder / name) == pair["artifacts"][key], "Canonical pair artifact SHA mismatch")
    return pair, load_arrays(folder / "pair_trace.npz")


def analyze_case(label, pair_dir, isaac_trace, isaac_report, mujoco_trace, mujoco_report, *, common_contract, contract_sha256):
    require(label in STUDY, "Only the four predeclared nominal cases are supported; no failed I6 substitution")
    algorithm, blocks = STUDY[label]
    pair_dir = Path(pair_dir)
    require(re.fullmatch(r"[A-Za-z0-9_-]+", pair_dir.name) is not None, "Nonportable public pair folder")
    pair, compact = audit_pair(pair_dir, algorithm)
    result = {"label": label, "algorithm": algorithm, "canonical_pair": pair_dir.name,
        "canonical_pair_sha256": {name: sha(pair_dir / name) for name in ("summary.json", "pair_trace.npz", "topdown.png")},
        "canonical_analysis_source_sha256": copy.deepcopy(pair["analysis_source_sha256"]),
        "reported_plant_source_provenance": copy.deepcopy(pair["reported_plant_source_provenance"]), "engines": {}}
    for engine, trace_path, report_path in (("isaac", isaac_trace, isaac_report), ("mujoco", mujoco_trace, mujoco_report)):
        trace_sha, report_sha = sha(trace_path), sha(report_path)
        require(pair["input_sha256"][engine + "_trace"] == trace_sha
                and pair["input_sha256"][engine + "_report"] == report_sha, "Raw trace/report differs from canonical pair byte binding")
        report = read_json(report_path)
        raw = load_arrays(trace_path)
        if algorithm == "NMPC":
            require(report.get("status") == "completed" and report.get("solver_failures") == 0
                    and report.get("plant_mu") == .7 and report.get("prediction_mu_scenarios") == [.7],
                    "Require completed zero-failure nominal NMPC, not a fallback diagnostic")
            config = report.get("effective_cem_config", {})
            require({key: config.get(key) for key in ("horizon_steps", "control_blocks", "candidates", "iterations", "elite_count", "seed")} ==
                    {"horizon_steps": 30, "control_blocks": blocks, "candidates": 64, "iterations": 3, "elite_count": 8, "seed": 42},
                    "NMPC case configuration differs from its predeclared label")
            require(report.get("rollout_sha256") == trace_sha and contract_sha256 in report.get("source_sha256", {}).values(),
                    "NMPC report lacks raw trace/shared actuator file byte binding")
            kind = "nmpc"
        else:
            contract = report.get("policy_export" if engine == "isaac" else "contract", {})
            require({name: contract.get(name) for name in ACTUATOR_FIELDS} == common_contract
                    and contract.get("policy_algorithm") == algorithm, "RL algorithm/action mapping mismatch")
            if engine == "isaac":
                require(report.get("physics_trajectory_sha256") == trace_sha, "Isaac report native trace SHA mismatch")
            elif "rollout_sha256" in report:
                require(report["rollout_sha256"] == trace_sha, "MuJoCo report native trace SHA mismatch")
            kind = "policy_" + engine
        control = native_control(raw, kind)
        # Canonical metrics are copied only after all underlying native states,
        # measured contacts and physical clocks match the public derivative.
        for name, values in control["trace"].items():
            key = engine + "_" + name
            if name in ("physics_dt_s", "joint_names"): continue
            require(key in compact and np.array_equal(values, compact[key]), "Canonical/native mismatch: " + key)
        core = pair["windows"]["core"][engine]
        require(set(core) == CORE_FIELDS and core["samples_in_window"] == 513, "Changed canonical core metric schema/coverage")
        result["engines"][engine] = {"input_sha256": {"raw_report": report_sha, "raw_trace": trace_sha},
            "native_kind": kind, "policy_decisions": 349, "native_samples": 1397,
            "canonical_trace_binding": copy.deepcopy(pair["audit"]["trace_bindings"][engine]),
            "canonical_core_metrics": copy.deepcopy(core),
            "canonical_posture_diagnostic": copy.deepcopy(pair["posture_diagnostics"][engine]),
            "burden": burden_windows(control),
            "effective_cem_config": copy.deepcopy(report.get("effective_cem_config")),
            "training_provenance": copy.deepcopy(pair.get("training_provenance")),
            "reported_replanning_ms_median_p95_max": report.get("compute_ms_median_p95_max") if algorithm == "NMPC" else None}
    return portable(result)


def analyze(cases, contract_path):
    require(len(cases) == 4 and {case[0] for case in cases} == set(STUDY), "Require each predeclared pair exactly once")
    common = actuator_contract(contract_path)
    rows = [analyze_case(*case, common_contract=common, contract_sha256=sha(contract_path)) for case in cases]
    result = {"schema_version": 1, "kind": "offline_nominal_control_burden_aggregate_audit",
        "reference_spec": get_reference("drift_drl_traj4"), "plant_mu": .7,
        "engine_episodes": 8, "controller_pairs": 4, "cases": rows,
        "source_sha256": {"analyze_nominal_control_burden.py": sha(Path(__file__)),
            "controller_trace.py": sha(Path(__file__).with_name("controller_trace.py")),
            "controller_references.py": sha(Path(__file__).with_name("controller_references.py"))},
        "common_actuator_contract_sha256": sha(contract_path), "actuator_fields_only": common,
        "definitions": {"actual_torque": "Recorded post-step applied effort for the preceding 5 ms substep; initial t0 cached/reset effort excluded",
            "leg_target_rate": "diff(actual clipped leg action)/.02 in rad/s because q_target=q0+action and q0 is constant; not measured joint velocity",
            "wheel_action_rate": "diff(actual clipped wheel action)/.02 in normalized units/s; not wheel angular acceleration",
            "wheel_requested_torque": "12*actual clipped wheel action at 50 Hz, before motor speed-dependent clipping; recorded actual torque is authoritative",
            "burden_window": "Commands issued in [start,end), their post-step torques in (start,end]; core 128 commands/512 torques, full 349/1396",
            "rate_boundary": "Differences between actual successive issued commands; no invented t0 delta; full 348 rates, core 128 including its preceding-command boundary change",
            "canonical_metrics": "Copied unchanged from hash-bound public pairs; closed 1.44–4.00 s window has 513 physical samples, distinct from aligned burden window",
            "group_l2": "Per-time norm over six leg or two wheel components, then RMS/p95/peak; pooled-component quantiles flatten all samples/components"},
        "not_computed": ["DC-envelope occupancy or command-clipping rate", "power, work, energy or battery consumption", "new FK or policy inference", "new physics or friction cases"],
        "limitations": ["One seed and one nominal episode per engine/controller; not an algorithm ranking or causal explanation.",
            "SAC MuJoCo includes fallen/posture-failure states; effort is not conditional on successful tracking.",
            "PPO/SAC actor sees 51 inputs; NMPC has full physical/contact state and future preview; matched stage terms do not equalize discount/termination/normalization or compute.",
            "NMPC source versions and host-specific native builds differ across development cases; canonical provenance is preserved, not relabeled.",
            "Lower effort/action variation alone does not establish better control or successful speed-retaining turning; no numerical speed/posture/height/beta gate is added."],
        "algorithm_ranking_claimed": False, "controller_freeze_claimed": False, "friction_grid_claimed": False,
        "new_physics_or_inference": False, "hardware_ready": False}
    return portable(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", nargs=6, required=True,
        metavar=("LABEL", "PUBLIC_PAIR_DIR", "ISAAC_TRACE", "ISAAC_REPORT", "MUJOCO_TRACE", "MUJOCO_REPORT"))
    parser.add_argument("--actuator-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.case, args.actuator_contract)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary_sha256": sha(output), "engine_episodes": 8, "new_physics_or_inference": False}))


if __name__ == "__main__":
    main()
