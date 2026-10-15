"""Audit physical-clock retiming of frozen GMR paths, not just GIF playback.

Same qpos / geometry; t'=t/alpha, dq'=alpha*dq, ddq'=alpha^2*ddq.
CoM momentum and translation energy are reference-state estimates, not actual
simulated motion. Motor torque availability is not required/realized torque.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from hdm05_gmr import ROOT, axial, heading_features, sha, wrap


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_motor_envelope(settings):
    repository = ROOT / "upstream/tron1-learning-from-human"
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if commit != settings["motor_source_commit"]:
        raise ValueError("Motor source submodule commit mismatch")
    for kind in ("contract", "envelope"):
        if sha(ROOT / settings["motor_" + kind + "_source"]) != settings["motor_" + kind + "_sha256"]:
            raise ValueError("Motor source hash mismatch")
    tree = ast.parse((ROOT / settings["motor_contract_source"]).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "observation_action_contract")
    returned = next(node.value for node in function.body if isinstance(node, ast.Return))
    pairs = {ast.literal_eval(key): value for key, value in zip(returned.keys, returned.values)}
    policy_fps = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "POLICY_FPS")
    for key, expected in settings["motor_contract"].items():
        value = policy_fps if isinstance(pairs[key], ast.Name) and pairs[key].id == "POLICY_FPS" else ast.literal_eval(pairs[key])
        if value != expected:
            raise ValueError("Configured motor values differ from pinned task contract")
    # Load only the exact hash-locked pure NumPy helper, without importing
    # deployment/Isaac/Torch modules or executing their top-level code.
    tree = ast.parse((ROOT / settings["motor_envelope_source"]).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "dc_motor_torque_bounds")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), settings["motor_envelope_source"], "exec"), namespace)
    return namespace["dc_motor_torque_bounds"]


def intervals(time, mask):
    """Observed contiguous windows only; a lone sample has zero duration."""
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return [{"start_s": float(time[start]), "end_s": float(time[stop - 1]),
             "duration_s": float(time[stop - 1] - time[start]),
             "samples": int(stop - start)} for start, stop in edges.reshape(-1, 2)]


def max_window(time, mask):
    windows = intervals(time, mask)
    return max(windows, key=lambda value: value["duration_s"], default=None)


def distribution(values):
    values = np.asarray(values)
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": float(np.max(values))}


def fixed_phase_core(time, velocity, settings):
    # The same source-phase core is used for every alpha; never choose entry
    # or exit from a speed threshold that would make retention tautological.
    phase = (time - time[0]) / (time[-1] - time[0])
    lower, upper = settings["core_phase_fraction"]
    width = settings["entry_exit_phase_window"]
    core = (phase >= lower) & (phase <= upper)
    entry = (phase >= lower) & (phase <= lower + width)
    exit_ = (phase >= upper - width) & (phase <= upper)
    if not all(mask.any() for mask in (core, entry, exit_)):
        raise ValueError("Fixed-phase core requires sampled entry and exit windows")
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    entry_speed, exit_speed = float(np.median(speed[entry])), float(np.median(speed[exit_]))
    first, last = velocity[entry, :2].mean(axis=0), velocity[exit_, :2].mean(axis=0)
    course_change = (float(wrap(np.degrees(np.arctan2(last[1], last[0]) - np.arctan2(first[1], first[0]))))
                     if min(np.linalg.norm(first), np.linalg.norm(last)) > 1e-6 else None)
    return {"source_phase_fraction": [lower, upper], "entry_exit_window_fraction": width,
            "entry_native_frame_indices": np.flatnonzero(entry).tolist(),
            "exit_native_frame_indices": np.flatnonzero(exit_).tolist(),
            "entry_speed_median_m_s": entry_speed, "exit_speed_median_m_s": exit_speed,
            "exit_over_entry_speed": exit_speed / entry_speed if entry_speed > 1e-8 else None,
            "core_speed_p10_over_entry": float(np.quantile(speed[core], .1)) / entry_speed if entry_speed > 1e-8 else None,
            "net_course_change_deg": course_change}


def retime_case(source, geometry, factor, gmr_settings, settings, motor_bounds):
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Speed factor must be finite and positive")
    native_time = source["time_s"]
    if len(native_time) < 4 or np.any(np.diff(native_time) <= 0):
        raise ValueError("Expected at least four strictly increasing reference samples")
    time = native_time / factor
    names = list(source["human_joint_names"])
    features = heading_features(time, source["root_target_m"], source["qpos"][:, :3],
        source["human_positions_robot_scale_m"][:, [names.index(n) for n in gmr_settings["bvh_ankle_nodes"]]],
        source["human_positions_robot_scale_m"][:, [names.index(n) for n in gmr_settings["bvh_toe_base_nodes"]]],
        source["robot_wheel_heading_deg"], source["robot_wheel_bottom_m"], gmr_settings,
        wheel_centers=source["robot_site_positions_m"][:, 5:7])
    np.testing.assert_allclose(wrap(features["robot_course_wheel_offset_deg"] - source["robot_course_wheel_offset_deg"]), 0, atol=1e-8)
    qvel = source["qvel_intervals"] * factor
    mid_time = .5 * (time[1:] + time[:-1])
    qacc = np.diff(qvel, axis=0) / np.diff(mid_time)[:, None]
    com = geometry["com_position_m"]
    com_velocity = np.gradient(com, time, axis=0)
    com_acceleration = np.gradient(com_velocity, time, axis=0)
    mass = float(geometry["mass_kg"])
    speed = np.linalg.norm(com_velocity[:, :2], axis=1)
    momentum = mass * com_velocity[:, :2]
    energy = .5 * mass * speed ** 2
    course = np.degrees(np.arctan2(com_velocity[:, 1], com_velocity[:, 0]))
    gamma = wrap(course[:, None] - source["robot_wheel_heading_deg"])
    gamma_axis = axial(gamma)
    contact = source["robot_near_ground_proxy"]
    leg_vids, wheel_vids = geometry["leg_velocity_indices"], geometry["wheel_velocity_indices"]
    motor = settings["motor_contract"]
    leg_velocity = qvel[:, leg_vids]
    lower, upper = motor_bounds(leg_velocity, motor["leg_saturation_effort_nm"],
                                motor["leg_torque_limit_nm"], motor["leg_motor_velocity_limit_rad_s"])
    motoring = np.maximum(0., np.where(leg_velocity >= 0, upper, -lower))
    speed_ratio = np.abs(leg_velocity) / motor["leg_motor_velocity_limit_rad_s"]
    policy_t = np.arange(time[0], time[-1] + 1e-10, 1 / motor["policy_fps"])
    qids = geometry["leg_position_indices"]
    policy_joints = np.column_stack([np.interp(policy_t, time, source["qpos"][:, qid]) for qid in qids])
    per_speed = {}
    high_angle = np.abs(gamma_axis) >= settings["high_axial_offset_deg"]
    for threshold in settings["speed_thresholds_m_s"]:
        fast = speed >= threshold
        high = fast[:, None] & high_angle
        per_speed[f"{threshold:g}"] = {
            "fast_high_angle_ungrounded_max_window": max_window(time, np.any(high, axis=1)),
            "fast_high_angle_any_near_ground_max_window": max_window(time, np.any(high & contact, axis=1)),
            "fast_high_angle_near_ground_max_window_L_R": [max_window(time, high[:, side] & contact[:, side]) for side in range(2)],
            "fast_high_angle_near_ground_samples_L_R": (high & contact).sum(axis=0).tolist()}
    # Select windows on the original clock to avoid floating-point boundary
    # changes when t/alpha is normalized again at a different alpha.
    core = fixed_phase_core(native_time, com_velocity, settings)
    goal_window = per_speed[f"{settings['goal_speed_m_s']:g}"]["fast_high_angle_any_near_ground_max_window"]
    goal_duration = 0. if goal_window is None else goal_window["duration_s"]
    screens = {"fast_high_angle_duration": goal_duration >= settings["goal_duration_s"],
               "fixed_core_course_change": core["net_course_change_deg"] is not None and abs(core["net_course_change_deg"]) >= settings["exploratory_min_core_course_change_deg"],
               "fixed_core_speed_retention": core["core_speed_p10_over_entry"] is not None and core["core_speed_p10_over_entry"] >= settings["exploratory_min_core_speed_retention"]}
    result = {"factor": factor, "duration_s": float(time[-1] - time[0]), "frames": len(time),
              "mass_kg": mass, "com_xy_speed_m_s": distribution(speed),
              "com_xy_momentum_magnitude_kg_m_s": distribution(np.linalg.norm(momentum, axis=1)),
              "com_xy_translation_energy_J": distribution(energy),
              "com_xy_acceleration_m_s2": distribution(np.linalg.norm(com_acceleration[:, :2], axis=1)),
              "peak_leg_speed_rad_s": float(np.abs(leg_velocity).max()),
              "peak_leg_acceleration_rad_s2": float(np.abs(qacc[:, leg_vids]).max()),
              "peak_leg_no_load_speed_ratio": float(speed_ratio.max()),
              "leg_interval_joint_fraction_at_or_above_no_load": float((speed_ratio >= 1).mean()),
              "min_directional_motoring_torque_available_nm": float(motoring.min()),
              "max_policy_50Hz_leg_reference_step_rad": float(np.abs(np.diff(policy_joints, axis=0)).max()),
              "peak_reference_wheel_spin_rate_rad_s": float(np.abs(qvel[:, wheel_vids]).max()),
              "motor_torque_demand_assessed": False, "physical_feasibility_assessed": False,
              "heading_speed_windows": per_speed, "fixed_phase_core": core,
              "exploratory_screens": screens, "all_exploratory_screens": all(screens.values())}
    delta = {"time_s": time, "source_time_s": native_time, "speed_factor": np.array(factor),
             "qvel_intervals": qvel, "qacc_interior": qacc, "qacc_time_s": time[1:-1],
             "com_position_m": com, "mass_kg": np.array(mass), "com_velocity_m_s": com_velocity,
             "com_acceleration_m_s2": com_acceleration, "com_xy_speed_m_s": speed,
             "com_xy_momentum_kg_m_s": momentum, "com_xy_translation_energy_J": energy,
             "com_course_deg": course, "com_course_wheel_offset_deg": gamma,
             "com_course_wheel_axial_deg": gamma_axis,
             "leg_velocity_rad_s": leg_velocity, "leg_acceleration_rad_s2": qacc[:, leg_vids],
             "nominal_leg_torque_lower_nm": lower, "nominal_leg_torque_upper_nm": upper,
             "directional_motoring_torque_available_nm": motoring, **features}
    return result, delta


def geometry_from_model(model, source):
    data = mujoco.MjData(model)
    com = []
    for pose in source["qpos"]:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        com.append(data.subtree_com[model.body("base_Link").id].copy())
    intervals_ = np.empty_like(source["qvel_intervals"])
    for i, dt in enumerate(np.diff(source["time_s"])):
        mujoco.mj_differentiatePos(model, intervals_[i], dt, source["qpos"][i], source["qpos"][i + 1])
    np.testing.assert_allclose(intervals_, source["qvel_intervals"], atol=1e-10)
    legs = [model.joint(i) for i in range(1, model.njnt) if not model.joint(i).name.startswith("wheel_")]
    wheels = [model.joint(i) for i in range(1, model.njnt) if model.joint(i).name.startswith("wheel_")]
    return {"com_position_m": np.asarray(com), "mass_kg": np.array(model.body_mass.sum()),
            "leg_joint_names": np.array([joint.name for joint in legs]),
            "leg_velocity_indices": np.array([joint.dofadr[0] for joint in legs]),
            "leg_position_indices": np.array([joint.qposadr[0] for joint in legs]),
            "wheel_velocity_indices": np.array([joint.dofadr[0] for joint in wheels])}


def make_plots(rows, output, settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clips = list(dict.fromkeys(row["clip"] for row in rows))
    fields = (("CoM XY speed P95 (m/s)", lambda row: row["com_xy_speed_m_s"]["p95"], "viridis"),
              ("CoM XY translation energy P95 (J)", lambda row: row["com_xy_translation_energy_J"]["p95"], "magma"),
              ("Peak leg speed / nominal no-load speed", lambda row: row["peak_leg_no_load_speed_ratio"], "YlOrRd"),
              ("Longest >=2 m/s, >=20 deg near-ground window (s)", lambda row: (row["heading_speed_windows"]["2"]["fast_high_angle_any_near_ground_max_window"] or {"duration_s": 0})["duration_s"], "Blues"))
    fig, axes = plt.subplots(1, 4, figsize=(17, 7))
    for axis, (title, value, cmap) in zip(axes, fields):
        values = np.array([[value(next(row for row in rows if row["clip"] == clip and row["factor"] == factor))
                            for factor in settings["rates"]] for clip in clips])
        image = axis.imshow(values, aspect="auto", cmap=cmap)
        axis.set_title(title, fontsize=9)
        axis.set_xticks(range(len(settings["rates"])), [f"{factor:g}x" for factor in settings["rates"]])
        axis.set_yticks(range(len(clips)), [clip.split("_", 2)[2].rsplit("_", 2)[0] for clip in clips] if axis is axes[0] else [])
        for y, x in np.ndindex(values.shape):
            rgba = image.cmap(image.norm(values[y, x]))
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            axis.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=9,
                      color="black" if luminance > .5 else "white")
        fig.colorbar(image, ax=axis, shrink=.6)
    fig.suptitle("Frozen GMR geometry, physical reference clocks 1x / 2x / 3x\nKinematic estimates and motor availability only; no policy or friction rollout", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(output / "rate_sweep.png", dpi=135)
    plt.close(fig)


def replay(summary_path, base_dir):
    from audit_hdm05_gmr import compare
    overview = json.loads(summary_path.read_text())
    base = json.loads((base_dir / "experiment.json").read_text())
    if sha(base_dir / "experiment.json") != overview["base_experiment_sha256"]:
        raise ValueError("Base GMR experiment mismatch")
    motor_bounds = load_motor_envelope(overview["settings"])
    for row in overview["cases"]:
        stem = Path(row["clip"]).stem
        trace_path = base_dir / stem / "retarget_trace.npz"
        if sha(trace_path) != row["base_trace_sha256"]:
            raise ValueError("Base GMR trace mismatch")
        geometry_path = summary_path.parent / stem / "geometry.npz"
        if sha(geometry_path) != row["geometry_sha256"]:
            raise ValueError("Saved CoM geometry hash mismatch")
        report, delta = retime_case(read_npz(trace_path), read_npz(geometry_path), row["factor"],
                                   base["settings"], overview["settings"], motor_bounds)
        compare(row["metrics"], report)
        delta_path = summary_path.parent / row["delta_file"]
        if sha(delta_path) != row["delta_sha256"]:
            raise ValueError("Retiming delta hash mismatch")
        saved = read_npz(delta_path)
        for key in delta:
            np.testing.assert_allclose(saved[key], delta[key], atol=1e-9, rtol=1e-9, err_msg=key)
    print(json.dumps({"status": "all_retimed_metrics_reproduced", "cases": len(overview["cases"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-experiment", type=Path, default=ROOT / "results/hdm05_gmr_bd")
    parser.add_argument("--config", type=Path, default=ROOT / "config/hdm05_rate_sweep.json")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay", type=Path, help="Model-free verification of a published summary.json")
    args = parser.parse_args()
    if args.replay:
        replay(args.replay, args.base_experiment)
        return
    if args.model is None or args.output_dir is None:
        parser.error("Creation requires --model and --output-dir")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Preserving existing rate-sweep output")
    settings = json.loads(args.config.read_text())
    base = json.loads((args.base_experiment / "experiment.json").read_text())
    if sha(args.model) != base["model_sha256"]:
        raise ValueError("Model differs from GMR source")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    motor_bounds = load_motor_envelope(settings)
    output.mkdir(parents=True, exist_ok=True)
    rows, plot_rows = [], []
    for report in base["clips"]:
        stem = Path(report["clip"]).stem
        source_path = args.base_experiment / stem / "retarget_trace.npz"
        if sha(source_path) != report["trace_sha256"]:
            raise ValueError("Base trace mismatch")
        source = read_npz(source_path)
        geometry = geometry_from_model(model, source)
        folder = output / stem
        folder.mkdir()
        np.savez_compressed(folder / "geometry.npz", **geometry)
        for factor in settings["rates"]:
            result, delta = retime_case(source, geometry, factor, base["settings"], settings, motor_bounds)
            filename = stem + f"/rate_{factor:g}.npz"
            np.savez_compressed(output / filename, **delta)
            rows.append({"clip": report["clip"], "factor": factor,
                         "base_trace_sha256": report["trace_sha256"], "geometry_sha256": sha(folder / "geometry.npz"),
                         "delta_file": filename, "delta_sha256": sha(output / filename), "metrics": result})
            plot_rows.append({"clip": report["clip"], **result})
    overview = {"kind": "HDM05_frozen_GMR_reference_physical_clock_rate_sweep", "settings": settings,
                "settings_sha256": sha(args.config), "analysis_source_sha256": sha(Path(__file__)),
                "base_experiment_sha256": sha(args.base_experiment / "experiment.json"),
                "model_sha256": sha(args.model), "mass_kg": float(model.body_mass.sum()), "cases": rows,
                "policy_trained": False, "dynamics_validated": False,
                "limitations": ["CoM momentum/energy are imposed reference-state estimates, not achieved motion.",
                                "Motor-envelope availability is nominal; demanded torque and contact forces are not solved.",
                                "No-load speed is not a hard joint-speed limit or an automatic feasibility verdict.",
                                "Geometry, source human pre-processing, wheel-spin freeze and contact-proxy limitations are inherited.",
                                "Friction DR and friction validation have not been changed or performed.",
                                "Fixed-phase turn and sustained-speed thresholds are exploratory screens, not physical success criteria."]}
    (output / "summary.json").write_text(json.dumps(overview, indent=2, allow_nan=False) + "\n")
    make_plots(plot_rows, output, settings)
    print(json.dumps({"cases": len(rows), "output": str(output), "kinematic_screen_passes": sum(row["metrics"]["all_exploratory_screens"] for row in rows)}))


if __name__ == "__main__":
    main()
