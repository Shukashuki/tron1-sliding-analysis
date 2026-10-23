"""Offline native any-wheel load/gap severity, not a controller counterfactual.

Optional V2a/V2b comparison changes ONLY an ungated force-window statistic on
the same recorded trajectory. It is NOT an executed/retrained V2b controller,
PPO's actual pre-command tracking-gated reward, or a counterfactual trajectory.
No physics, inference, FK, height/flight classification or policy update occurs.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import analyze_ppo_ground_contact_v2 as ppo
from analyze_cem_ground_contact_v2 import support_summary
from analyze_ppo_ground_support_failure import persistent_events
from controller_trace import native_trace
from ppo_backend_400hz_contract import INPUT_PINS, read_json, require, sha256, write_json


KINDS = ("nmpc", "policy_isaac", "policy_mujoco")
INTERVALS = {"full_native": (0., 6.98), "core": (1.44, 4.), "contact": (.02, 6.96)}


def _statistics(value):
    value = np.asarray(value, dtype=float)
    if not value.size:
        return {"samples": 0, "minimum": None, "p01": None, "p05": None, "p25": None,
                "median": None, "p75": None, "p95": None, "p99": None, "maximum": None, "mean": None}
    require(np.isfinite(value).all(), "Nonfinite recorded diagnostic signal")
    keys = ("minimum", "p01", "p05", "p25", "median", "p75", "p95", "p99", "maximum")
    result = dict(zip(keys, map(float, np.quantile(value, (0., .01, .05, .25, .5, .75, .95, .99, 1.)))))
    return {"samples": int(value.size), **result, "mean": float(value.mean())}


def _mask(time, interval):
    selected = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
    require(selected.sum() >= 2 and abs(time[selected][0]-interval[0]) < 1e-8
            and abs(time[selected][-1]-interval[1]) < 1e-8, "Native diagnostic window is incomplete")
    return selected


def _duty(time, value):
    value = np.asarray(value, dtype=bool)
    return {"sample_fraction": float(value.mean()),
            "time_fraction_zoh": float(np.sum(np.diff(time)*value[:-1])/(time[-1]-time[0])),
            "true_samples": int(value.sum()), "native_samples": int(len(time))}


def _event_indices(time, event):
    start, end = event["onset_s"], event["end_s"]
    return ((time >= start-1e-9) & (time <= end+1e-9 if event.get("right_window_censored")
                                 else time < end-1e-9))


def _context(time, position, velocity, tilt, force_z, mg, reference, selected, nonwheel):
    query = time[selected]
    require(len(query) > 0, "Empty event context")
    speed = np.linalg.norm(velocity[selected, :2], axis=-1)
    available = ((query >= reference["time_s"][0]-1e-10)
                 & (query <= reference["time_s"][-1]+1e-10))
    xy = {"reference_overlap_samples": int(available.sum()), "native_samples": int(len(query)),
          "reference_coverage_complete": bool(available.all()), "reference_extrapolation": False,
          "rmse_on_available_overlap_m": None, "peak_on_available_overlap_m": None,
          "mean_reference_speed_on_available_overlap_m_s": None,
          "speed_ratio_on_available_overlap": None}
    if available.any():
        target = np.column_stack([np.interp(query[available], reference["time_s"], reference["position"][:, i]) for i in (0, 1)])
        error = np.linalg.norm(position[selected][available, :2]-target, axis=-1)
        desired_speed = np.interp(query[available], reference["time_s"],
                                  np.linalg.norm(reference["velocity"][:, :2], axis=-1))
        denominator = float(desired_speed.mean())
        xy.update(rmse_on_available_overlap_m=float(np.sqrt(np.mean(error**2))),
                  peak_on_available_overlap_m=float(error.max()),
                  mean_reference_speed_on_available_overlap_m_s=denominator,
                  speed_ratio_on_available_overlap=float(speed[available].mean()/denominator) if denominator > 1e-12 else None)
    measured = {"available": False}
    if nonwheel is not None:
        measured = {"available": True, "aggregate_ground_normal_force_norm_n": _statistics(nonwheel[selected]),
                    "sample_fraction_strict_gt1N": float(np.mean(nonwheel[selected] > 1.)),
                    "used_as_automatic_failure_gate": False}
    return {"native_interval_s": [float(query[0]), float(query[-1])], "xy": xy,
            "COM_speed_xy_m_s": _statistics(speed), "COM_vertical_velocity_world_m_s": _statistics(velocity[selected, 2]),
            "body_tilt_deg_diagnostic_only": _statistics(tilt[selected]),
            "total_upward_wheel_load_over_mg": _statistics(force_z[selected].clip(min=0).sum(-1)/mg),
            "nonwheel_ground_contact": measured, "height_or_flight_classification": False}


def _gap_category(event):
    duration = event["observed_duration_s"]
    censored = event["left_window_censored"] or event["right_window_censored"]
    if censored:
        return "boundary_censored_ge100ms_lower_bound" if duration >= .1-1e-10 else "boundary_censored_lt100ms_observed"
    if event["native_samples"] == 1: return "isolated_one_sample_2p5ms"
    if duration < .02-1e-10: return "multi_sample_lt20ms"
    if duration < .1-1e-10: return "ge20ms_lt100ms"
    return "ge100ms"


def analyze_contact(time, position, velocity, quaternion, force_z, mg, reference, *, nonwheel=None,
                    include_sensitivity=False):
    """Pure arithmetic on one full continuous native episode and measured loads."""
    time, position, velocity, quaternion, force_z = [np.asarray(x, dtype=float)
        for x in (time, position, velocity, quaternion, force_z)]
    require(time.shape == (2793,) and np.allclose(time, np.arange(2793)*.0025, atol=1e-10, rtol=0),
            "Require all 2793 consecutive native400Hz states, no partial or reset-spliced episode")
    require(position.shape == velocity.shape == (2793, 3) and quaternion.shape == (2793, 4)
            and force_z.shape == (2793, 2) and all(np.isfinite(x).all() for x in (position, velocity, quaternion, force_z)),
            "Invalid finite measured root/normal-force arrays")
    require(not isinstance(mg, (bool, np.bool_)) and np.ndim(mg) == 0 and np.isfinite(mg)
            and mg > 0 and (force_z >= -1e-4).all(), "Require positive documented mg and upward floor-normal force")
    require(type(include_sensitivity) is bool, "Sensitivity selection must be boolean")
    norm = np.linalg.norm(quaternion, axis=-1)
    require(np.allclose(norm, 1., atol=1e-5, rtol=0), "Root orientation must be measured unit wxyz")
    unit = quaternion/norm[:, None]
    tilt = np.degrees(np.arccos(np.clip(1.-2.*np.sum(unit[:, 1:3]**2, axis=-1), -1., 1.)))
    if nonwheel is not None:
        nonwheel = np.asarray(nonwheel, dtype=float)
        require(nonwheel.shape == (2793,) and np.isfinite(nonwheel).all() and (nonwheel >= 0).all(),
                "Invalid measured nonwheel-ground normal-force norm")
    rt = np.asarray(reference["time_s"], dtype=float)
    require(rt.shape == (349,) and np.allclose(rt, np.arange(349)*.02, atol=1e-10, rtol=0)
            and np.asarray(reference["position"]).shape == np.asarray(reference["velocity"]).shape == (349, 3)
            and np.isfinite(reference["position"]).all() and np.isfinite(reference["velocity"]).all(), "Invalid real reference clock/states")
    positive = force_z.clip(min=0)
    windows = {}
    for name, interval in INTERVALS.items():
        selected = _mask(time, interval); t, f = time[selected], positive[selected]
        thresholds = {}
        for label, threshold in (("strict_gt0N", 0.), ("strict_gt5N", 5.), ("strict_gt0p1mg", .1*mg), ("strict_gt0p5mg", .5*mg)):
            wheel = f > threshold
            thresholds[label] = {"threshold_n_strict": float(threshold),
                "per_wheel": {side: _duty(t, wheel[:, i]) for i, side in enumerate(("left", "right"))},
                "any_wheel": _duty(t, wheel.any(-1)), "both_wheels_diagnostic_not_required": _duty(t, wheel.all(-1)),
                "total_wheel_force_strict_gt_threshold": _duty(t, f.sum(-1) > threshold)}
        support = support_summary(time, force_z, interval, threshold=5.)
        histogram = {}; events = []
        for event in support["no_support_gaps"]:
            category = _gap_category(event)
            slot = histogram.setdefault(category, {"events": 0, "observed_duration_sum_s": 0., "native_samples": 0})
            slot["events"] += 1; slot["observed_duration_sum_s"] += event["observed_duration_s"]
            slot["native_samples"] += event["native_samples"]
            events.append({**event, "severity_bin": category,
                "context": _context(time, position, velocity, tilt, force_z, mg, reference, _event_indices(time, event), nonwheel)})
        longest = sorted(events, key=lambda item: (-item["observed_duration_s"], -item["native_samples"], item["onset_s"]))[:10]
        persistent = [event for event in events if event["observed_duration_s"] >= .1-1e-10]
        measured = {"available": False}
        if nonwheel is not None:
            measured = {"available": True, "normal_force_norm_n": _statistics(nonwheel[selected]),
                        "aggregate_norm_strict_gt1N": _duty(t, nonwheel[selected] > 1.),
                        "persistent_events": persistent_events(t, nonwheel[selected] > 1., .1),
                        "used_as_automatic_failure_gate": False}
        windows[name] = {"interval_s": list(interval), "native_samples": len(t), "threshold_load_fractions": thresholds,
            "total_upward_Fz_over_mg_quantiles_native_samples": _statistics(f.sum(-1)/mg),
            "per_wheel_Fz_over_mg_quantiles_native_samples": {side: _statistics(f[:, i]/mg) for i, side in enumerate(("left", "right"))},
            "any_wheel_5N": support, "gap_duration_histogram": histogram,
            "isolated_uncensored_2p5ms_gap_count": histogram.get("isolated_one_sample_2p5ms", {}).get("events", 0),
            "observed_ge100ms_gap_count_including_censored_lower_bounds": len(persistent),
            "longest_gap_events_with_native_context": longest,
            "gap_intervals_are_observed_not_flight_or_failure_labels": True,
            "nonwheel_ground_contact_diagnostic": measured}
    # Sample0 is pre-action. Each decision holds through states8h+1 ... 8h+8.
    force_windows = positive[1:].reshape(349, 8, 2)
    q = np.clip(force_windows.sum(-1)/(.5*mg), 0., 1.)
    mean, minimum = q.mean(-1), q.min(-1)
    v2a, v2b = .75*mean+.25*minimum, minimum.copy()
    excess = np.maximum(force_windows/(3.*mg)-1., 0.)
    impact_gate = np.exp(-np.mean(excess**2, axis=(1, 2)))
    impact = (force_windows > 3.*mg).any(axis=(1, 2))
    impact_events = []
    for frame in np.flatnonzero(impact):
        start, end = int(frame)*8, (int(frame)+1)*8
        selected = np.zeros(2793, dtype=bool); selected[start+1:end+1] = True
        impact_events.append({"decision_index": int(frame), "pre_action_time_s": float(time[start]),
            "physical_samples_interval_s": [float(time[start+1]), float(time[end])],
            "native_physical_indices_inclusive": [start+1, end],
            "per_wheel_peak_Fz_over_mg": (force_windows[frame].max(axis=0)/mg).tolist(),
            "samples_over_3mg_per_wheel": (force_windows[frame] > 3.*mg).sum(axis=0).tolist(),
            "impact_gate_force_factor_only": float(impact_gate[frame]),
            "wholly_within_contact_interval": bool(time[start+1] >= .02-1e-9 and time[end] <= 6.96+1e-9),
            "context": _context(time, position, velocity, tilt, force_z, mg, reference, selected, nonwheel)})
    sensitivity = {"included": include_sensitivity, "not_actual_PPO_reward_or_return": True,
        "not_V2b_controller_performance_or_counterfactual_trajectory": True,
        "tracking_gate_not_reconstructed": True, "no_action_policy_or_simulation_changed": True}
    if include_sensitivity:
        sensitivity.update(formula="same recorded q_j=clip(total_positive_Fz/(.5mg),0,1); V2a=.75mean8(q)+.25min8(q); V2b=min8(q)",
            V2a_ungated_support_factor=_statistics(v2a), V2b_ungated_support_factor=_statistics(v2b),
            V2a_minus_V2b_ungated_support_factor=_statistics(v2a-v2b),
            windows_with_zero_minimum_support=int(np.count_nonzero(minimum == 0.)),
            windows_with_positive_mean_but_zero_minimum=int(np.count_nonzero((mean > 0.) & (minimum == 0.))))
    return {"kind": "recorded_native_ground_load_and_gap_severity", "physics_dt_s": .0025, "policy_dt_s": .02,
        "normalization_weight_n": float(mg), "windows": windows,
        "eight_substep_windows": {"decisions": 349, "samples_per_decision": 8,
            "pre_action_snapshot_excluded": True, "native_array_slice": "[1:].reshape(349,8,2)",
            "per_wheel_3mg_threshold_n_strict": float(3.*mg), "impact_window_count": int(impact.sum()),
            "impact_windows": impact_events, "impact_gate_force_factor": _statistics(impact_gate)},
        "offline_contact_factor_sensitivity": sensitivity,
        "native_signal_context": _context(time, position, velocity, tilt, force_z, mg, reference, np.ones(2793, dtype=bool), nonwheel),
        "N": 1, "new_simulation_training_policy_inference_or_FK": False,
        "new_controller_performance_claimed": False, "height_or_flight_inference": False,
        "bilateral_requirement_added": False, "overall_task_success_claimed": False}


def _load_inputs(args):
    require(args.kind in KINDS, "Select an explicit recording schema")
    paths = {"raw": Path(args.raw), "report": Path(args.report)}
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(), "Missing/direct input file required: " + name)
        expected = ppo._digest(getattr(args, name+"_sha256"), name)
        require(sha256(path) == expected, "Explicit input SHA differs: " + name)
    report = read_json(paths["report"])
    ppo.load_protocol(args.protocol)
    proof = None
    if args.kind.startswith("policy_"):
        engine = "physx" if args.kind == "policy_isaac" else "mujoco"
        expected_raw = "physics_trajectory.npz" if engine == "physx" else "rollout.npz"
        require(paths["raw"].resolve() == paths["report"].resolve().parent/expected_raw
                and paths["report"].name == "report.json", "PPO requires its canonical parent-verified evaluation folder")
        raw, evidence = ppo.load_completed(paths["report"].parent, engine, args.protocol)
        proof = evidence["evaluation_proof"]
    else:
        require(report.get("status") == "completed" and report.get("smoke_only") is False
                and report.get("kind") == "400hz_CEM_ground_contact_reward_v2a_anywheel"
                and report.get("completed_policy_steps") == 349 and report.get("native_samples") == 2793
                and report.get("completed_full_reference") is True
                and report.get("termination_or_reset_after_initialization") is False
                and report.get("floor_filter_verified") is True
                and report.get("rollout_sha256") == args.raw_sha256, "Require completed source-bound CEM V2 native episode")
        with np.load(paths["raw"], allow_pickle=False) as archive: raw = {name: archive[name].copy() for name in archive.files}
        actions = np.asarray(raw.get("actions"))
        require(actions.shape == (349, 8) and np.isfinite(actions).all(), "Missing full CEM held-action decisions")
        require(np.asarray(raw["ground_normal_force_window_w"]).shape == (349, 8, 2, 3)
                and np.array_equal(raw["ground_normal_force_w"][1:].reshape(349, 8, 2, 3), raw["ground_normal_force_window_w"]),
                "CEM stored eight-substep forces differ from native records")
    trace = native_trace(raw, args.kind)
    ppo._clock(trace)
    field = "ground_normal_force_w" if "ground_normal_force_w" in raw else "wheel_ground_contact_force_w_n"
    force = np.asarray(raw[field], dtype=float)
    if args.kind == "policy_isaac": force = force[:, 0]
    require(force.shape == (2793, 2, 3), "Invalid native ground NORMAL force shape")
    nonwheel = raw.get("nonwheel_ground_contact_force_norm_n")
    if nonwheel is not None:
        nonwheel = np.asarray(nonwheel, dtype=float)
        if args.kind == "policy_isaac": nonwheel = nonwheel[:, 0]
    mg = getattr(args, "mg_n", None)
    if mg is None:
        mg = report.get("model_weight_n")
        source = "Actual validation model_weight_n in the SHA-bound CEM report; not training mass or force inference"
    else:
        source = getattr(args, "mg_source", None)
        require(isinstance(source, str) and source.strip(), "Explicit mg requires a documented evaluation-weight source")
        if "model_weight_n" in report:
            require(math.isclose(mg, report["model_weight_n"], abs_tol=1e-8, rel_tol=0), "Explicit mg disagrees with validation report")
    require(isinstance(mg, (int, float)) and not isinstance(mg, bool) and math.isfinite(mg) and mg > 0,
            "Evaluation mg is unavailable: supply --mg-n and --mg-source from actual evaluation readback; never use training mass")
    require(all(sha256(path) == getattr(args, name+"_sha256") for name, path in paths.items()), "Inputs changed during loading")
    return trace, force[..., 2], nonwheel, float(mg), source, proof, field


def recompute_compact(arrays, *, include_sensitivity=False):
    reference = {name: arrays["reference_"+name] for name in ("time_s", "position", "velocity")}
    return analyze_contact(arrays["time_s"], arrays["position"], arrays["velocity"], arrays["quaternion"],
        arrays["ground_normal_Fz_n"], float(arrays["evaluation_weight_n"]), reference,
        nonwheel=arrays.get("nonwheel_ground_force_norm_n"), include_sensitivity=include_sensitivity)


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve diagnostics; use a fresh output directory")
    trace, force, nonwheel, mg, source, proof, field = _load_inputs(args)
    reference = ppo.metrics.reference_data(args.reference)
    report = read_json(args.report)
    require(report.get("motion_sha256", report.get("motion_file_sha256")) == INPUT_PINS["motion"], "Report/reference identity differs")
    result = analyze_contact(trace["time_s"], trace["position"], trace["com_velocity"], trace["quaternion"],
        force, mg, reference, nonwheel=nonwheel, include_sensitivity=args.include_v2b_sensitivity)
    result.update(input_schema=args.kind, input_sha256={"raw": args.raw_sha256, "report": args.report_sha256},
        input_files={"raw": str(Path(args.raw).resolve()), "report": str(Path(args.report).resolve())},
        protocol_sha256=sha256(args.protocol), reference_sha256=sha256(args.reference),
        analyzer_sha256=sha256(__file__), normalization_weight_source=source,
        actual_ground_normal_force_source=field, parent_PPO_verification_proof=proof,
        limitations=["N=1 recorded episode; no new controller experiment or statistical qualification.",
            "Force thresholds are descriptive. Zero/weak load does not establish geometric separation or flight.",
            "Both-wheel fractions are diagnostics only; any-wheel support is the user's requirement.",
            "Normalization uses documented evaluation weight, never inferred from force or copied from training.",
            "Quantiles are native-sample quantiles; separate duty fractions use zero-order-held native time.",
            "Boundary-censored gap duration is an observed lower bound; the last state has no invented future duration.",
            "XY context uses only the real reference overlap through6.96s; no endpoint extrapolation to6.98s.",
            "Optional V2b factors are same-record force sensitivity, NOT actual PPO reward, return or counterfactual performance."])
    arrays = {"time_s": trace["time_s"], "position": trace["position"], "velocity": trace["com_velocity"],
              "quaternion": trace["quaternion"], "ground_normal_Fz_n": force, "evaluation_weight_n": np.array(mg)}
    arrays.update({"reference_"+name: reference[name] for name in ("time_s", "position", "velocity")})
    if nonwheel is not None: arrays["nonwheel_ground_force_norm_n"] = nonwheel
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"native_contact_context.npz", **arrays)
    with (output/"gap_events.csv").open("x", newline="") as stream:
        fields = ("window", "onset_s", "end_s", "observed_duration_s", "native_samples", "severity_bin",
                  "left_censored", "right_censored")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for name, value in result["windows"].items():
            for event in value["any_wheel_5N"]["no_support_gaps"]:
                writer.writerow(dict(zip(fields, (name, event["onset_s"], event["end_s"], event["observed_duration_s"],
                    event["native_samples"], _gap_category(event), event["left_window_censored"], event["right_window_censored"]))))
    require(sha256(args.raw) == args.raw_sha256 and sha256(args.report) == args.report_sha256, "Completed source changed during diagnostics")
    result["artifact_sha256"] = {name: sha256(output/name) for name in ("native_contact_context.npz", "gap_events.csv")}
    write_json(output/"summary.json", result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--kind", choices=KINDS, required=True)
    for name in ("raw", "report", "output-dir"): result.add_argument("--"+name, type=Path, required=True)
    for name in ("raw-sha256", "report-sha256"): result.add_argument("--"+name, required=True)
    result.add_argument("--protocol", type=Path, default=ppo.DEFAULT_PROTOCOL)
    result.add_argument("--reference", type=Path, default=ppo.DEFAULT_REFERENCE)
    result.add_argument("--mg-n", type=float)
    result.add_argument("--mg-source")
    result.add_argument("--include-v2b-sensitivity", action="store_true")
    return result


if __name__ == "__main__": run(parser().parse_args())
