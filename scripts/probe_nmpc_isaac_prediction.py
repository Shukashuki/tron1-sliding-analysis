"""Sixteen observed Isaac actions, each predicted for ONE isolated 20 ms interval.

This is not a new controller evaluation or a contact-cause experiment. The
exact recorded 111D controller inputs, nonzero reset sensor history and applied
float32 actions are preserved. Recorded root/joint/twist/torque/force channels
are direct evidence; other body poses/velocities reconstructed by MuJoCo FK
are explicitly NOT additional measured Isaac channels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXED_FRAMES = (0, 19, 43, 72, 90, 100, 125, 145, 150, 175, 195, 225)
CONTACT_STRATA_S = ((.02, 1.44), (1.44, 2.3), (2.3, 3.2), (3.2, 4.))
EXPECTED_SELECTED_FRAMES = (0, 14, 19, 43, 72, 73, 90, 100, 117, 125, 145, 150, 161, 175, 195, 225)
EXPECTED_ARCHIVE = {
    "report.json": "954b8449a654219de0bd473bae330cea2ea97fa9399bde3fd5db2f2211f01980",
    "rollout.npz": "46b4422669c3d2f915ae4492d825bbe248eef8cfc11776a6b8a583c13f15b9bc",
    "solver_log.json": "71856b9de0f1e70986bd5265a7acfa37394feb733aaa26b5586f58d6bae1a1d4",
}
COURSE_SPEED_THRESHOLD_M_S = .25


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite {shape}")
    return value


def validate_trace(trace, solver):
    """Require the complete original physical clock and actual action interface."""
    from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES
    shapes = {"time_s": (1397,), "simulation_step_counter": (1397,),
              "root_pos": (1397, 3), "root_quaternion_wxyz": (1397, 4),
              "base_com_velocity_w": (1397, 3), "base_angular_velocity_w": (1397, 3),
              "joint_pos": (1397, 8), "joint_vel": (1397, 8), "wheel_pos": (1397, 2, 3),
              "body_net_forces_w": (1397, 9, 3), "applied_joint_torque_nm": (1397, 8),
              "wheel_ground_contact_count": (1397, 2), "actions": (349, 8),
              "packed_controller_states": (349, 111), "held_action": (1397, 8),
              "reference_frame_for_action": (1397,), "action_was_applied": (1397,)}
    for name, shape in shapes.items():
        if name not in trace:
            raise ValueError("Missing actual measured channel: " + name)
        _array(trace[name], shape, name)
    if tuple(trace["body_names"].tolist()) != BODY_NAMES or tuple(trace["action_joint_names"].tolist()) != JOINT_NAMES:
        raise ValueError("Recorded body/joint ordering differs")
    if float(trace["physics_dt_s"]) != .005 or float(trace["policy_dt_s"]) != .02:
        raise ValueError("Recorded control/physics dt differs")
    if len(solver) != 349:
        raise ValueError("Incomplete solver log")
    quaternion_norm = np.linalg.norm(trace["root_quaternion_wxyz"], axis=1)
    if not np.allclose(quaternion_norm, 1., atol=1e-5, rtol=0):
        raise ValueError("Actual root quaternions must be unit wxyz")
    if not np.allclose(trace["time_s"], np.arange(1397)*.005, atol=1e-12, rtol=0):
        raise ValueError("Recorded clock differs from consecutive 5 ms samples")
    if not np.array_equal(np.diff(trace["simulation_step_counter"]), np.ones(1396)):
        raise ValueError("Recorded native physics counter skipped/reset")
    actions = np.asarray(trace["actions"])
    if np.max(np.abs(actions)) > 1. or not np.array_equal(actions, actions.astype(np.float32).astype(float)):
        raise ValueError("Use exact float32-rounded applied actions, not raw CEM outputs")
    if (np.asarray(trace["action_was_applied"]).dtype.kind != "b"
            or trace["action_was_applied"][0] or not np.all(trace["action_was_applied"][1:])):
        raise ValueError("Incorrect initial/action physical sample flags")
    if not np.array_equal(trace["reference_frame_for_action"][1:], np.repeat(np.arange(349), 4)):
        raise ValueError("Recorded action phase does not hold four physical samples")
    if not np.array_equal(trace["held_action"][1:], np.repeat(actions, 4, axis=0)):
        raise ValueError("Recorded substeps do not use actual logged actions")
    for k, row in enumerate(solver):
        if row["frame"] != k or row["reference_frame_for_reward"] != k:
            raise ValueError("Actual reward must use held pre-action reference[k]")
        if not np.array_equal(row["action"], actions[k]):
            raise ValueError("Actual solver/trace actions differ")
    packed = np.asarray(trace["packed_controller_states"])
    if not np.array_equal(packed[:, -1], np.r_[1., np.zeros(348)]):
        raise ValueError("Only the first recorded reward suppresses reset smoothness")
    current = packed[:, 29:56].reshape(349, 9, 3)
    if not np.array_equal(current, np.asarray(trace["body_net_forces_w"])[4*np.arange(349)]):
        raise ValueError("Actual newest-first sensor-history row differs from directly logged net forces")
    # PhysX ContactSensor stores newest first. This is proven against recorded
    # samples, not assumed from MuJoCo's opposite chronological append order.
    for k in range(1, 349):
        expected = np.asarray(trace["body_net_forces_w"])[4*k-np.arange(3)]
        if not np.array_equal(packed[k, 29:110].reshape(3, 9, 3), expected):
            raise ValueError("Recorded packed contact history is not the actual newest-first sensor history")


def select_intervals(trace):
    """Time strata plus earliest unused measured contact changes; no error input."""
    time = _array(trace["time_s"], (1397,), "time_s")
    counts = _array(trace["wheel_ground_contact_count"], (1397, 2), "wheel_ground_contact_count")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("Contact counts must be nonnegative integers")
    active = counts > 0
    transitions = np.flatnonzero(np.any(active[1:] != active[:-1], axis=1))+1
    selection = {k: {"kind": "fixed_clock_stratum", "policy_frame": k} for k in FIXED_FRAMES}
    for lo, hi in CONTACT_STRATA_S:
        available = [int(s) for s in transitions
                     if lo <= time[s] < hi and int((s-1)//4) not in selection]
        if not available:
            raise ValueError("Missing unused measured wheel-contact transition in declared stratum")
        sample = available[0]
        frame = (sample-1)//4  # a boundary sample belongs to the preceding action
        selection[frame] = {"kind": "first_unused_measured_wheel_contact_transition",
                            "policy_frame": frame, "stratum_s": [lo, hi],
                            "event_sample": sample, "event_time_s": float(time[sample]),
                            "active_before_L_R": active[sample-1].tolist(),
                            "active_after_L_R": active[sample].tolist()}
    if len(selection) != 16:
        raise ValueError("Expected exactly sixteen distinct predeclared intervals")
    return [selection[k] for k in sorted(selection)]


def actual_sensor_history(trace, sample):
    """Raw newest-first actual history, retaining the recorded nonzero reset row."""
    if isinstance(sample, bool) or not isinstance(sample, (int, np.integer)) or not 0 <= sample < 1397:
        raise ValueError("Invalid actual physical sample")
    packed = _array(trace["packed_controller_states"], (349, 111), "packed_controller_states")
    forces = _array(trace["body_net_forces_w"], (1397, 9, 3), "body_net_forces_w")
    boundary = min(sample//4, 348)
    history = packed[boundary, 29:110].reshape(3, 9, 3).copy()
    for index in range(4*boundary+1, sample+1):
        history = np.concatenate((forces[index:index+1], history[:2]), axis=0)
    return history


def measurement_at(trace, sample):
    """Root is already environment-origin-subtracted LINK position, not COM."""
    if isinstance(sample, bool) or not isinstance(sample, (int, np.integer)) or not 0 <= sample < 1397:
        raise ValueError("Invalid measured sample")
    fields = {"root_pos": ("root_pos", 3), "root_quat_wxyz": ("root_quaternion_wxyz", 4),
              "root_com_lin_vel_w": ("base_com_velocity_w", 3),
              "root_ang_vel_w": ("base_angular_velocity_w", 3),
              "joint_pos": ("joint_pos", 8), "joint_vel": ("joint_vel", 8)}
    return {name: _array(trace[key][sample], (width,), key).copy()
            for name, (key, width) in fields.items()}


def course_error(predicted_velocity, actual_velocity):
    predicted = _array(predicted_velocity, (4, 3), "predicted COM velocity")
    actual = _array(actual_velocity, (4, 3), "actual COM velocity")
    valid = (np.linalg.norm(predicted[:, :2], axis=1) >= COURSE_SPEED_THRESHOLD_M_S) & (
        np.linalg.norm(actual[:, :2], axis=1) >= COURSE_SPEED_THRESHOLD_M_S)
    difference = np.arctan2(predicted[:, 1], predicted[:, 0])-np.arctan2(actual[:, 1], actual[:, 0])
    difference = np.arctan2(np.sin(difference), np.cos(difference))
    return {"valid": valid.tolist(), "predicted_minus_measured_rad": [float(x) if v else None for x, v in zip(difference, valid)]}


def archived_source_hash(report, suffix):
    found = [digest for path, digest in report["source_sha256"].items() if path.endswith("/"+suffix)]
    if len(found) != 1:
        raise ValueError("Missing/ambiguous archived source role: " + suffix)
    return found[0]


def max_abs(value):
    return float(np.max(np.abs(value)))


def analyze(args):
    # Importing these CPU helpers does not import Torch/Isaac or build anything.
    import mujoco
    from nmpc_native_batch import NativeBatchTransition
    from nmpc_tracking_objective import HeightSmoothTrackingObjective, MujocoObjectiveState
    from nmpc_shooting import CEMConfig
    from run_nmpc_tracking_isaac import SourceModelProvider, IsaacToMujocoState
    from run_nmpc_tracking import MujocoTrackingPredictor
    from controller_references import DEFAULT_CATALOG, get_reference

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserving existing audit: " + str(output))
    episode = Path(args.episode)
    for name, digest in EXPECTED_ARCHIVE.items():
        if sha(episode/name) != digest:
            raise ValueError("Audit is bound to the exact archived residual Isaac nominal bytes: " + name)
    report = json.loads((episode/"report.json").read_text())
    solver = json.loads((episode/"solver_log.json").read_text())
    with np.load(episode/"rollout.npz", allow_pickle=False) as z:
        trace = {name: z[name].copy() for name in z.files}
    if (report["status"] != "completed" or report["engine"] != "Isaac/PhysX"
            or report["reference_id"] != "drift_drl_traj4" or report["plant_mu"] != .7
            or report["prediction_mu_scenarios"] != [.7] or report["prediction_receives_plant_mu"]
            or report["prediction_backend"] != "native" or report["learned_policy_used"]
            or report["action_parameterization"]["mode"] != "warm_start_residual_blocks"):
        raise ValueError("Only the observed completed PRIMARY residual nominal Isaac episode is in scope")
    if sha(episode/"rollout.npz") != report["rollout_sha256"] or sha(episode/"solver_log.json") != report["solver_log_sha256"]:
        raise ValueError("Archived report/trace/solver integrity mismatch")
    spec = get_reference("drift_drl_traj4")
    if (sha(args.model) != report["mujoco_model_sha256"] or sha(args.motion) != report["motion_sha256"]
            or sha(args.motion) != spec["motion_npz_sha256"]
            or sha(DEFAULT_CATALOG) != report["reference_catalog_sha256"] or spec != report["reference_spec"]):
        raise ValueError("Actual model/motion/reference differs from locked experiment")
    sources = {}
    for relative in ("scripts/run_nmpc_tracking.py", "scripts/run_nmpc_tracking_isaac.py", "scripts/nmpc_shooting.py",
                     "scripts/nmpc_tracking_objective.py", "scripts/nmpc_batch_objective.py", "scripts/nmpc_native_batch.py",
                     "scripts/classical_tracking.py", "scripts/controller_references.py", "scripts/height_smoothness.py", "native/nmpc_batch.cpp"):
        actual = sha(ROOT/relative)
        if actual != archived_source_hash(report, relative):
            raise ValueError("Current frozen controller/physical/objective source differs: " + relative)
        sources[relative] = actual
    source_root = Path(args.source_root)
    for relative in ("scripts/eval_tracking_mujoco.py", "scripts/eval_tracking.py", "training/tron1_domain_randomization.py",
                     "training/tron1_contact_slip.py", "training/tron1_tracking.py", "training/tron1_drift.py",
                     "training/tron1_runtime_device.py", "training/tron1_physics_clock.py"):
        actual = sha(source_root/relative)
        if actual != archived_source_hash(report, relative):
            raise ValueError("Actual source-root helper differs: " + relative)
        sources["source-root/"+relative] = actual
    for path, suffix in ((args.settings, "config/nmpc_height_smooth_residual.json"),
                         (args.task_profile, "config/car_drift_height_smooth_task.json"),
                         (args.contract, "config/nominal_actuator_contract.json")):
        if sha(path) != archived_source_hash(report, suffix):
            raise ValueError("Frozen settings/profile/actuator contract differs: " + suffix)
    settings = json.loads(Path(args.settings).read_text())
    task = json.loads(Path(args.task_profile).read_text())
    contract = json.loads(Path(args.contract).read_text())
    if settings != report["settings"] or asdict(CEMConfig(**settings["nmpc"])) != report["effective_cem_config"]:
        raise ValueError("Frozen settings/effective CEM configuration differs")
    validate_trace(trace, solver)
    selection = select_intervals(trace)
    selected = np.array([row["policy_frame"] for row in selection])
    if tuple(selected) != EXPECTED_SELECTED_FRAMES:
        raise ValueError("Recorded contact-strata selection differs from declared pre-prediction scope")
    provider = SourceModelProvider(source_root, args.model, args.motion, contract, settings, (.7,))
    predictor = MujocoTrackingPredictor(provider, args.model, task, (.7,))
    limits = _array(report["runtime_soft_joint_pos_limits"], (8, 2), "actual Isaac runtime soft joint limits")
    predictor.objective = HeightSmoothTrackingObjective(task, dt=.02, soft_joint_limits=limits)
    converter = IsaacToMujocoState(provider)
    extractor = MujocoObjectiveState(provider.model, soft_joint_limits=limits)
    state_errors = []
    fk_errors = []
    measured_states = {}
    fk_states = {}
    # Kinematics only. No integration, contact solve calibration, CEM or search.
    for k in selected:
        for sample in range(4*int(k), 4*int(k)+5):
            if sample in measured_states:
                continue
            history = actual_sensor_history(trace, sample)
            converted, velocity_error = converter.convert(measurement_at(trace, sample), history,
                                                           first_reward_after_reset=sample == 0)
            measured_states[sample] = converted
            data = converter.data
            state = extractor.extract(data, contact_force_history=history,
                                      body_net_forces_w=trace["body_net_forces_w"][sample])
            fk_states[sample] = state
            root_error = max_abs(state["body_pos_w"][0]-trace["root_pos"][sample])
            wheel_error = max_abs(state["body_pos_w"][-2:]-trace["wheel_pos"][sample])
            actual_quat = np.asarray(trace["root_quaternion_wxyz"][sample], dtype=float)
            actual_quat = actual_quat/np.linalg.norm(actual_quat)
            orientation_error = float(2*np.arccos(np.clip(abs(np.dot(state["body_quat_w"][0], actual_quat)), 0., 1.)))
            fk_errors.append({"sample": sample, "root_origin_max_abs_error_m": root_error,
                              "wheel_origins_max_abs_error_m": wheel_error,
                              "root_quaternion_error_rad": orientation_error,
                              "base_COM_twist_reconstruction_max_abs_error": velocity_error})
            if sample % 4 == 0 and sample//4 < 349:
                stored = trace["packed_controller_states"][sample//4]
                error = max_abs(converted-stored)
                if error > 2e-6:
                    raise ValueError("Independent actual-state conversion differs from originally recorded controller input")
                state_errors.append(error)
    initial = trace["packed_controller_states"][selected]
    qpos = initial[:, :15].copy();qvel = initial[:, 15:29].copy()
    history = initial[:, 29:110].reshape(16, 3, 9, 3).copy()
    actions = trace["actions"][selected, None].copy()
    result = {"schema_version": 1, "kind": "bounded_existing_isaac_action_one_interval_prediction_audit",
              "status": "selected_before_prediction", "selection": selection, "selected_policy_frames": selected.tolist(),
              "predicted_control_intervals": 16, "predicted_physics_substeps": 64,
              "new_closed_loop_episode": False, "new_cem_search": False, "new_native_build": False,
              "new_isaac_app": False, "learned_policy_data_used": False,
              "selection_rule": {"fixed_clock_frames": list(FIXED_FRAMES), "contact_event_strata_s": CONTACT_STRATA_S,
                                 "event": "Earliest measured binary wheel-ground contact-mode change in each half-open stratum whose interval is not already selected; boundary sample belongs to preceding action.",
                                 "prediction_errors_used_for_selection": False},
              "archive_sha256": dict(EXPECTED_ARCHIVE), "model_sha256": sha(args.model), "motion_sha256": sha(args.motion),
              "settings_sha256": sha(args.settings), "task_profile_sha256": sha(args.task_profile), "actuator_contract_sha256": sha(args.contract),
              "matched_frozen_source_sha256": sources, "probe_source_sha256": sha(__file__),
              "runtime_soft_joint_pos_limits": limits.tolist(), "independent_converter_vs_recorded_packed_max_abs_error": max(state_errors),
              "independent_FK_vs_direct_measured_channel_checks": fk_errors,
              "prediction_input_history": "EXACT raw recorded PhysX newest-first history, including nonzero reset history; no fabricated zero forces or solver warm-start.",
              "force_history_comparison": "Reorder native chronological output to newest-first ONLY for measured-array comparison; actual/predicted objective history uses the respective original order, whose max-norm statistic is order-invariant.",
              "actual_objective_authority": "Logged actual_isaac_objective 15 raw/weighted terms and actual Isaac reward; no missing nine-body state is invented.",
              "kinematic_reconstruction_scope": "Root LINK pose/COM twist and eight joint q/dq are measured. Other seven body poses/COM velocities and held relative-body targets use MuJoCo FK; differences from actual logged reward quantify this reconstruction, not an exact same-state Isaac parity claim.",
              "root_frame": "Recorded LINK origin world positions already subtract environment origin; converter origin=zero. Quaternions wxyz; COM and angular velocities world; 8 joints named in action order.",
              "course_speed_threshold_m_s": COURSE_SPEED_THRESHOLD_M_S,
              "caveats": ["One-step observed-action differences do not establish which engine/contact parameter caused whole-trajectory drift, or how counterfactual CEM plans would evolve.",
                          "Native prediction uses a cold qacc/time reset per interval exactly as the original controller. No unknown PhysX solver state is reconstructed.",
                          "Native body net contact forces and PhysX sensor forces are engine-specific solver outputs, not calibrated equivalent tire forces.",
                          "Wheel-ground contact counts are actual measurements; no ground labels are inferred from FK or net-body forces. Flight, height and nonzero beta are not failure gates.",
                          "The local prebuilt native library differs from the original remote build; exact source, MuJoCo/header/library/model hashes and both binary identities are retained."],
              "started_utc": datetime.now(timezone.utc).isoformat()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    try:
        # THE ONLY PHYSICAL PREDICTION CALL: 16 independent H=1 observed actions.
        with NativeBatchTransition(predictor.models[.7], provider.q0, contract, args.native_library) as native:
            original_worker = report["prediction_backend_metadata"]["workers"]["0.7"]
            for field in ("source_sha256", "mujoco_version", "mujoco_header_sha256",
                          "mujoco_library_sha256", "model_snapshot_sha256"):
                if native.metadata[field] != original_worker[field]:
                    raise ValueError("Local predictor differs from original remote worker: " + field)
            prediction = native.rollout(qpos, qvel, actions, initial_history=history)
            native_metadata = native.metadata.copy()
        model = predictor.models[.7]
        data = mujoco.MjData(model)
        predicted_extractor = MujocoObjectiveState(model, soft_joint_limits=limits)
        rows = []
        for row, k in enumerate(selected):
            k = int(k);begin = 4*k;end = begin+4
            actual_qpos = np.stack([measured_states[i][:15] for i in range(begin+1, end+1)])
            actual_qvel = np.stack([measured_states[i][15:29] for i in range(begin+1, end+1)])
            tangent = np.empty((4, 14))
            for sub in range(4):
                mujoco.mj_differentiatePos(model, tangent[sub], 1., actual_qpos[sub], prediction["qpos"][row, 0, sub])
            qvel_error = prediction["qvel"][row, 0]-actual_qvel
            torque_error = prediction["torques"][row, 0]-trace["applied_joint_torque_nm"][begin+1:end+1]
            same_state_torque = np.stack([provider.torque(trace["actions"][k], measured_states[i][:15], measured_states[i][15:29])
                                         for i in range(begin, end)])
            actuator_error = same_state_torque-trace["applied_joint_torque_nm"][begin+1:end+1]
            actual_COM_velocity = trace["base_com_velocity_w"][begin+1:end+1]
            predicted_COM_velocity = prediction["body_lin_vel_w"][row, 0, :, 0]
            force_error = prediction["body_net_forces_w"][row, 0]-trace["body_net_forces_w"][begin+1:end+1]
            data.qpos[:], data.qvel[:] = prediction["qpos"][row, 0, -1], prediction["qvel"][row, 0, -1]
            mujoco.mj_fwdPosition(model, data)
            predicted_state = predicted_extractor.extract(data, contact_force_history=prediction["contact_force_history_w"][row, 0],
                                                          body_net_forces_w=prediction["body_net_forces_w"][row, 0, -1])
            held = predictor.objective.hold_reference(predictor.reference_frame(k), fk_states[begin])
            previous = np.zeros(8) if k == 0 else trace["actions"][k-1]
            previous_vz = float(trace["base_com_velocity_w"][begin, 2])
            predicted_objective = predictor.objective.evaluate(predicted_state, held, trace["actions"][k], previous, previous_vz, reset_sample=k == 0)
            reconstructed_objective = predictor.objective.evaluate(fk_states[end], held, trace["actions"][k], previous, previous_vz, reset_sample=k == 0)
            logged = solver[k]["actual_isaac_objective"]
            raw_error = {term: float(predicted_objective["raw_terms"][term]-logged["raw_terms"][term]) for term in logged["raw_terms"]}
            weighted_error = {term: float(predicted_objective["weighted_terms"][term]-logged["weighted_terms"][term]) for term in logged["weighted_terms"]}
            reconstruction_error = {term: float(reconstructed_objective["weighted_terms"][term]-logged["weighted_terms"][term]) for term in logged["weighted_terms"]}
            rows.append({"policy_frame": k, "interval_s": [float(trace["time_s"][begin]), float(trace["time_s"][end])],
                         "selection": selection[row], "actual_applied_float32_action": trace["actions"][k].tolist(),
                         "direct_measured_comparisons": {
                             "root_xy_prediction_error_each_substep_m": np.linalg.norm(prediction["qpos"][row, 0, :, :2]-trace["root_pos"][begin+1:end+1, :2], axis=1).tolist(),
                             "root_rotation_tangent_prediction_error_each_substep_rad": np.linalg.norm(tangent[:, 3:6], axis=1).tolist(),
                             "joint_position_error_each_substep_rad": (prediction["qpos"][row, 0][:, provider.qids]-trace["joint_pos"][begin+1:end+1]).tolist(),
                             "qvel_error_max_each_substep_mixed_units": np.max(np.abs(qvel_error), axis=1).tolist(),
                             "root_COM_velocity_error_each_substep_m_s": (predicted_COM_velocity-actual_COM_velocity).tolist(),
                             "root_angular_velocity_error_each_substep_rad_s": (prediction["body_ang_vel_w"][row, 0, :, 0]-trace["base_angular_velocity_w"][begin+1:end+1]).tolist(),
                             "root_COM_course_error": course_error(predicted_COM_velocity, actual_COM_velocity),
                             "predicted_minus_measured_torque_each_substep_nm": torque_error.tolist(),
                             "pure_DC_actuator_model_on_actual_prestates_minus_measured_torque_nm": actuator_error.tolist(),
                             "net_body_contact_force_error_each_substep_max_n": np.max(np.abs(force_error), axis=(1, 2)).tolist(),
                             "net_body_contact_force_error_each_substep_L2_n": np.linalg.norm(force_error, axis=-1).tolist(),
                             "net_force_history_error_max_n_after_explicit_order_alignment": max_abs(prediction["contact_force_history_w"][row, 0, ::-1]-actual_sensor_history(trace, end)),
                             "actual_wheel_ground_counts_each_substep": trace["wheel_ground_contact_count"][begin+1:end+1].tolist(),
                             "native_total_contact_count_each_substep_not_wheel_ground_counts": prediction["contact_count"][row, 0].tolist()},
                         "objective_prediction_vs_authoritative_logged_actual": {
                             "logged_actual_stage_cost": logged["stage_cost"], "predicted_stage_cost": predicted_objective["stage_cost"],
                             "predicted_minus_actual_stage_cost": float(predicted_objective["stage_cost"]-logged["stage_cost"]),
                             "predicted_minus_actual_raw_terms": raw_error, "predicted_minus_actual_weighted_terms": weighted_error,
                             "logged_actual_isaac_dt_reward": solver[k]["actual_dt_scaled_reward"],
                             "predicted_minus_actual_isaac_dt_reward": float(.02*predicted_objective["total_reward"]-solver[k]["actual_dt_scaled_reward"])},
                         "FK_reconstruction_not_an_additional_measured_9body_state": {
                             "reconstructed_minus_logged_stage_cost": float(reconstructed_objective["stage_cost"]-logged["stage_cost"]),
                             "reconstructed_minus_logged_weighted_terms": reconstruction_error}})
        direct = [row["direct_measured_comparisons"] for row in rows]
        aggregate = {
            "maximum_root_xy_prediction_error_m": max(max(x["root_xy_prediction_error_each_substep_m"]) for x in direct),
            "maximum_root_rotation_prediction_error_rad": max(max(x["root_rotation_tangent_prediction_error_each_substep_rad"]) for x in direct),
            "maximum_root_COM_velocity_component_error_m_s": max(max_abs(x["root_COM_velocity_error_each_substep_m_s"]) for x in direct),
            "maximum_root_angular_velocity_component_error_rad_s": max(max_abs(x["root_angular_velocity_error_each_substep_rad_s"]) for x in direct),
            "maximum_torque_prediction_error_nm": max(max_abs(x["predicted_minus_measured_torque_each_substep_nm"]) for x in direct),
            "maximum_first_substep_torque_prediction_error_nm": max(max_abs(x["predicted_minus_measured_torque_each_substep_nm"][0]) for x in direct),
            "maximum_pure_actuator_model_error_on_actual_prestates_nm": max(max_abs(x["pure_DC_actuator_model_on_actual_prestates_minus_measured_torque_nm"]) for x in direct),
            "maximum_net_body_contact_force_prediction_error_n": max(max(x["net_body_contact_force_error_each_substep_max_n"]) for x in direct),
            "maximum_absolute_stage_cost_prediction_error": max(abs(x["objective_prediction_vs_authoritative_logged_actual"]["predicted_minus_actual_stage_cost"]) for x in rows),
            "maximum_FK_reconstruction_vs_logged_stage_cost_error": max(abs(x["FK_reconstruction_not_an_additional_measured_9body_state"]["reconstructed_minus_logged_stage_cost"]) for x in rows),
            "maximum_FK_reconstruction_vs_logged_weighted_term_error": max(max_abs(list(x["FK_reconstruction_not_an_additional_measured_9body_state"]["reconstructed_minus_logged_weighted_terms"].values())) for x in rows)}
        result.update(status="audited", aggregate=aggregate, rows=rows, native_backend=native_metadata,
                      original_remote_native_backend=report["prediction_backend_metadata"],
                      local_native_library_sha256=sha(args.native_library),
                      local_native_build_manifest_sha256=sha(Path(args.native_library).with_suffix(".json")),
                      original_remote_native_library_sha256=report["native_library_sha256"],
                      native_binary_matches_original_remote=sha(args.native_library) == report["native_library_sha256"],
                      probe_mujoco_version=mujoco.__version__, probe_numpy_version=np.__version__,
                      completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        result.update(status="failed_preserving_selected_scope", error=f"{type(error).__name__}: {error}")
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        raise
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "selected_policy_frames": selected.tolist(), **aggregate}, indent=2))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "model", "motion", "contract", "source-root", "native-library", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--settings", type=Path, default=ROOT/"config/nmpc_height_smooth_residual.json")
    p.add_argument("--task-profile", type=Path, default=ROOT/"config/car_drift_height_smooth_task.json")
    return p


if __name__ == "__main__":
    analyze(parser().parse_args())
