"""Offline native-MuJoCo PPO wheel-resultant Coulomb-reference diagnostics.

Requires archived floor TOTAL and matching NORMAL forces. This is a lower-
bound wheel proxy, NOT a per-contact/actual-pyramidal-edge measurement, reward,
drift success rule, or task acceptance change. No simulator or policy is loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import friction_cone_diagnostics as cone

ROOT = Path(__file__).resolve().parents[1]
MOTION_SHA256 = "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
MODEL_SHA256 = "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"
LEGACY_EVALUATOR_SHA256 = "2251d740d7568874a47a232556241cc8d1ee5f5b7be1351d1c657aea8fa85f87"
NORMAL_SLIP_HELPER_SHA256 = "52ef38de9cece19a8ee02c120cb85d472ffd831e4587e5f0d1f340974ab699f4"
NATIVE_WRAPPER_SHA256 = "9887aa0b4b7ef74845f47f8623356490a9990e4449e0b42f4a77b2b35a49c1e4"
DIAGNOSTIC_HELPER_SHA256 = "79ce452f24869545b859b0fd27af8e0ff393b2c140225dfe0b040606138c54fe"
NORMAL_HELPER = ROOT / "upstream/tron1-learning-from-human/training/tron1_contact_slip.py"
GEOM_NAMES = ("floor", "base_collision", "abad_L_collision", "hip_L_collision", "knee_L_collision",
    "wheel_L_collision", "abad_R_collision", "hip_R_collision", "knee_R_collision", "wheel_R_collision")
WINDOWS = {"core": (1.44, 4.), "full_contact": (.02, 6.96)}
NORMAL_THRESHOLD_N = 5.
NEAR_REFERENCE_THRESHOLD = .9
LATERAL_RMS_THRESHOLD_M_S = .05
FORCE_ATOL_N = 1e-6
FORCE_RTOL = 1e-9
RHO_NUMERICAL_EXCESS_TOL = 1e-8


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            name + " requires an explicit lowercase SHA256")
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_collector_sources(report, evidence, cohort_key):
    wrapper_name = "eval_ppo_" + cohort_key.removesuffix("_evaluation") + ".py"
    wrapper = ROOT / "scripts" / wrapper_name
    sources = {str(NORMAL_HELPER.relative_to(ROOT)): NORMAL_SLIP_HELPER_SHA256,
        "scripts/eval_ppo_backend_400hz_mujoco.py": NATIVE_WRAPPER_SHA256}
    require(wrapper.is_file() and not wrapper.is_symlink(), "Unknown/direct cohort wrapper source required")
    require(report.get("evaluator_source_sha256") == evidence.get("wrapper_sha256") == sha256(wrapper),
            "Archived evaluator wrapper source identity differs")
    sources[str(wrapper.relative_to(ROOT))] = evidence["wrapper_sha256"]
    for name, expected in sources.items():
        require((ROOT / name).is_file() and sha256(ROOT / name) == expected,
                "Frozen normal/total collector source changed: " + name)
    return sources


def validate_report(report, raw_sha256, legacy_source=None):
    require(report.get("engine") == "MuJoCo" and report.get("status") == "completed"
            and report.get("full_horizon_diagnostic") is True and report.get("completed_full_reference") is True
            and report.get("termination") == "motion_end" and report.get("hidden_resets") == 0
            and report.get("root_state_writes") == 1 and report.get("start_reference_frame") == 0
            and report.get("final_reference_frame") == 348
            and (report.get("reference_frames"), report.get("policy_steps"), report.get("physics_steps")) == (349, 349, 2792),
            "Require completed full native MuJoCo PPO episode without hidden resets; not PhysX/CEM/smoke")
    require(report.get("expected_duration_s") == 6.98
            and np.isclose(report.get("recorded_duration_s", np.nan), 6.98, rtol=0, atol=1e-9),
            "Incomplete or inconsistent reported native duration")
    require(report.get("motion_sha256") == MOTION_SHA256 and report.get("model_sha256") == MODEL_SHA256,
            "Unknown locked reference/model provenance")
    require(report.get("terrain") is None and report.get("terrain_runtime_audit") is None,
            "Only the registered unaugmented flat-floor model is supported")
    require(report.get("policy_kind") == "normalized_torchscript_actor"
            and report.get("contract", {}).get("policy_algorithm") == "PPO"
            and (report["contract"].get("physics_dt"), report["contract"].get("decimation")) == (.0025, 8),
            "Require archived PPO/native400Hz contract, not generic full-state CEM")
    keys = [key for key in report if re.fullmatch(r"ground_contact_v2(?:[bcde])?_evaluation", key)]
    require(len(keys) == 1, "Require one explicit supported formal contact-PPO evaluation proof")
    evidence = report[keys[0]]
    require(evidence.get("status") == "verified" and evidence.get("smoke_only") is False
            and evidence.get("trained_policy_claimed") is True
            and evidence.get("validation_backend") == "native CPU MuJoCo"
            and evidence.get("additional_training_updates") == 0
            and evidence.get("legacy_evaluator_sha256") == LEGACY_EVALUATOR_SHA256
            and evidence.get("recording_sha256") == raw_sha256
            and (evidence.get("native_states"), evidence.get("policy_decisions")) == (2793, 349)
            and (evidence.get("physics_dt_s"), evidence.get("policy_dt_s"), evidence.get("decimation")) == (.0025, .02, 8),
            "Archived full PPO evaluation/raw/source binding differs")
    require(evidence.get("duration_s") == 6.98
            and digest(evidence.get("normalized_actor_sha256"), "actor proof") == report.get("policy_sha256"),
            "PPO actor/duration proof differs")
    require(report.get("contact_slip_helper_sha256") == NORMAL_SLIP_HELPER_SHA256
            and report.get("wheel_lateral_contact_slip_evidence_kind") == "actual_ground_contact_points",
            "Normal-only/actual-point patch-slip collector provenance is unavailable")
    slip = report.get("contact_slip_measurement", {})
    require(slip.get("active_point_normal_force_threshold_n") == 5.
            and slip.get("active_point_comparison") == ">" and slip.get("ground_is_stationary") is True
            and slip.get("sample_dt_s") == .0025
            and slip.get("aggregation") == "normal-force-weighted RMS over active contact points per wheel"
            and slip.get("ground_force_kind") == "sum of normal contact forces, matching the filtered PhysX contact API",
            "Unknown patch-RMS eligibility, units, clock, or normal-only force semantics")
    source_pins = verify_collector_sources(report, evidence, keys[0])
    material = report.get("contact_material_audit", {})
    require(material.get("native_mapping_verified") is True
            and material.get("explicit_contact_pair_count") == 0
            and material.get("compiled_native_timestep_s") == .0025
            and material.get("colliding_geom_names") == list(GEOM_NAMES)
            and material.get("colliding_geom_ids") == list(range(10))
            and material.get("wheel_geom_ids") == [5, 9], "Unknown flat-floor material/pair mapping")
    friction = np.asarray(material.get("geom_friction"), dtype=float)
    mu = material.get("pair_coefficient")
    require(isinstance(mu, (int, float)) and not isinstance(mu, bool) and np.isfinite(mu) and mu > 0
            and friction.shape == (10, 3) and np.isfinite(friction).all()
            and np.allclose(friction[:, 0], mu, rtol=0, atol=1e-12)
            and np.array_equal(friction[:, 1:], np.zeros((10, 2)))
            and material.get("geom_priority") == [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
            and material.get("geom_condim") == [3] * 10,
            "Require positive uniform isotropic readback mu with declared priorities/condim3; never guess mu")
    solver = material.get("native_plant_readback", {}).get("solver_configuration", {})
    require(solver.get("timestep") == .0025 and solver.get("cone") in (0., 1.),
            "Unknown native timestep/cone provenance")
    legacy_evidence = None
    if legacy_source is not None:
        path = Path(legacy_source)
        require(path.is_file() and not path.is_symlink() and sha256(path) == LEGACY_EVALUATOR_SHA256,
                "Optional frozen legacy source SHA differs")
        legacy_evidence = {"path": str(path.resolve()), "sha256": sha256(path)}
    return {"cohort_key": keys[0], "cohort_id": evidence.get("cohort_id"), "effective_pair_mu": float(mu),
        "mu_source": "SHA-bound native compiled geom_friction readback and pair_coefficient; uniform two tangent coefficients",
        "flat_floor_evidence": "Registered original model SHA, no terrain augmentation, stationary ground, and saved +Z normal vectors; no FK/geometry inference",
        "native_cone_readback": solver["cone"], "native_cone_name": "pyramidal" if solver["cone"] == 0 else "elliptic",
        "actual_engine_boundary_computable": False, "collector_source_sha256": source_pins,
        "legacy_evaluator_declared_sha256": LEGACY_EVALUATOR_SHA256,
        "legacy_whole_file_locally_rehashed": legacy_evidence is not None,
        "optional_legacy_source_evidence": legacy_evidence,
        "legacy_source_note": "Declared frozen legacy SHA is checked; only an explicit separately SHA-verified legacy source is marked locally rehashed. Current upstream bytes are never relabeled."}


def validate_arrays(raw):
    require("wheel_ground_total_contact_force_w_n" in raw,
            "Missing explicit floor TOTAL force; normal-only PhysX or CEM body-net forces are not substitutes")
    required = {"time_s": (2793,), "wheel_ground_total_contact_force_w_n": (2793, 2, 3),
        "wheel_ground_contact_force_w_n": (2793, 2, 3), "wheel_ground_contact_count": (2793, 2),
        "wheel_lateral_contact_slip_m_s": (2793, 2)}
    for name, shape in required.items():
        require(name in raw and np.asarray(raw[name]).shape == shape and np.asarray(raw[name]).dtype.kind in "fiu"
                and np.isfinite(raw[name]).all(), "Missing/nonfinite/wrong-shape native array: " + name)
    time = np.asarray(raw["time_s"], dtype=float)
    require(np.allclose(time, np.arange(2793) * .0025, rtol=0, atol=1e-9)
            and float(raw.get("physics_dt_s", np.nan)) == .0025
            and float(raw.get("policy_dt_s", np.nan)) == .02, "Non-native/incomplete/reset/extrapolated clock")
    total = np.asarray(raw["wheel_ground_total_contact_force_w_n"], dtype=float)
    normal = np.asarray(raw["wheel_ground_contact_force_w_n"], dtype=float)
    counts = np.asarray(raw["wheel_ground_contact_count"])
    slip = np.asarray(raw["wheel_lateral_contact_slip_m_s"], dtype=float)
    require(np.max(np.abs(normal[..., :2])) <= FORCE_ATOL_N and np.min(normal[..., 2]) >= -FORCE_ATOL_N,
            "Saved normal forces are not nonnegative world+Z flat-floor normals")
    require(np.allclose(total[..., 2], normal[..., 2], rtol=FORCE_RTOL, atol=FORCE_ATOL_N),
            "TOTAL-Z and matching NORMAL-Z disagree")
    tangent = total - normal
    require(np.allclose(tangent[..., 2], 0., rtol=0, atol=FORCE_ATOL_N),
            "Derived tangential force has a normal component")
    require(np.all(counts >= 0) and np.array_equal(counts, np.rint(counts)) and np.min(slip) >= 0,
            "Invalid point-count or nonnegative patch-RMS slip")
    require(np.all(slip[counts == 0] == 0.), "Count-zero legacy slip must be its documented zero placeholder")
    lateral = np.where(counts > 0, slip, np.nan)
    return time, total, normal, tangent, counts.astype(np.int32), lateral


def observed_positive_spans(time, mask):
    """Observed positive sample spans, not interval-filled physical durations."""
    time, mask = np.asarray(time, dtype=float), np.asarray(mask)
    require(time.ndim == 1 and mask.shape == time.shape and mask.dtype.kind == "b"
            and np.isfinite(time).all() and (len(time) < 2 or np.all(np.diff(time) > 0)),
            "Observed spans require finite ordered native time and a matching boolean mask")
    ids = np.flatnonzero(mask)
    if not len(ids):
        return {"event_count": 0, "longest_observed_sample_span_lower_bound_s": None,
            "longest_positive_samples": 0, "single_positive_sample_span_s": 0.}
    starts = np.r_[0, np.flatnonzero(np.diff(ids) != 1) + 1]
    ends = np.r_[starts[1:] - 1, len(ids) - 1]
    spans = time[ids[ends]] - time[ids[starts]]
    return {"event_count": int(len(starts)),
        "longest_observed_sample_span_lower_bound_s": float(spans.max()),
        "longest_positive_samples": int(np.max(ends - starts + 1)),
        "single_positive_sample_span_s": 0.}


def analyze(raw, provenance):
    time, total, normal, tangent, counts, lateral = validate_arrays(raw)
    result = cone.contact_diagnostics(np.maximum(normal[..., 2], 0.), tangent[..., :2],
        np.full((2, 2), provenance["effective_pair_mu"]), force_units="N",
        tangent_basis="world_xy_flat_floor", data_scope="wheel_resultant_lower_bound",
        engine_boundary=None, wheel_lateral_slip_m_s=lateral,
        normal_threshold_n=NORMAL_THRESHOLD_N, saturation_threshold=NEAR_REFERENCE_THRESHOLD,
        slip_threshold_m_s=LATERAL_RMS_THRESHOLD_M_S)
    rows = []
    for window, (start, end) in WINDOWS.items():
        clock = (time >= start - 1e-10) & (time <= end + 1e-10)
        for wheel, index in (("L", 0), ("R", 1)):
            loaded = clock & result["cone_valid_loaded"][:, index]
            eligible = loaded & result["wheel_lateral_slip_valid"][:, index]
            near = result["coulomb_reference_near_boundary"][:, index]
            sliding = result["observed_wheel_lateral_slip"][:, index]
            rho = result["coulomb_reference_rho_l2"][loaded, index]
            q = np.quantile(rho, [.5, .95]).tolist() if rho.size else [None, None]
            def fraction(mask, denominator):
                return float(np.count_nonzero(mask) / denominator) if denominator else None
            n, ns = int(loaded.sum()), int(eligible.sum())
            n_near_eligible = int((eligible & near).sum())
            near_span = observed_positive_spans(time, loaded & near)
            joint_span = observed_positive_spans(time, eligible & near & sliding)
            rows.append({"window": window, "wheel": wheel, "native_samples": int(clock.sum()),
                "loaded_valid_resultant_samples": n, "loaded_fraction": fraction(loaded, int(clock.sum())),
                "rho_l2_reference_p50": q[0], "rho_l2_reference_p95": q[1],
                "near_reference_0p9_fraction_of_loaded": fraction(loaded & near, n),
                "raw_strict_float_rho_gt1_fraction_of_loaded_NOT_physical_violation": fraction(
                    loaded & (result["coulomb_reference_rho_l2"][:, index] > 1.), n),
                "rho_gt1plus1e_8_fraction_of_loaded_exploratory": fraction(
                    loaded & (result["coulomb_reference_rho_l2"][:, index] > 1. + RHO_NUMERICAL_EXCESS_TOL), n),
                "rho_above1_max_excess_unclipped": float(max(0., rho.max() - 1.)) if rho.size else None,
                "patch_lateral_RMS_eligible_samples": ns, "patch_lateral_RMS_unavailable_loaded_samples": n - ns,
                "patch_lateral_RMS_gt0p05_fraction_of_eligible": fraction(eligible & sliding, ns),
                "observed_near_reference_AND_patch_lateral_RMS_gt0p05_fraction_of_loaded_lower_bound": fraction(eligible & near & sliding, n),
                "near_reference_AND_patch_lateral_RMS_gt0p05_fraction_of_eligible": fraction(eligible & near & sliding, ns),
                "near_reference_AND_patch_RMS_eligible_samples": n_near_eligible,
                "patch_RMS_gt0p05_fraction_given_near_reference_AND_patch_RMS_eligible": fraction(
                    eligible & near & sliding, n_near_eligible),
                "near_reference_with_patch_RMS_below_cutoff_samples_NOT_static": int((eligible & near & ~sliding).sum()),
                "near_reference_observed_event_count": near_span["event_count"],
                "near_reference_longest_observed_sample_span_lower_bound_s": near_span["longest_observed_sample_span_lower_bound_s"],
                "near_reference_longest_positive_samples": near_span["longest_positive_samples"],
                "near_reference_AND_eligible_patch_RMS_slip_observed_event_count": joint_span["event_count"],
                "near_reference_AND_eligible_patch_RMS_slip_longest_observed_sample_span_lower_bound_s": joint_span["longest_observed_sample_span_lower_bound_s"],
                "near_reference_AND_eligible_patch_RMS_slip_longest_positive_samples": joint_span["longest_positive_samples"],
                "actual_per_contact_engine_boundary_available": False})
    arrays = {"time_s": time, "physics_dt_s": np.array(.0025), "policy_dt_s": np.array(.02),
        "ground_total_force_w_n": total, "ground_normal_force_w_n": normal,
        "ground_tangent_force_w_n": tangent, "active_point_count_gt5N": counts,
        "measured_patch_lateral_RMS_m_s": np.asarray(raw["wheel_lateral_contact_slip_m_s"]),
        "patch_lateral_RMS_with_count_eligibility_m_s": lateral}
    arrays.update({key: value for key, value in result.items() if isinstance(value, np.ndarray)})
    return rows, arrays, result


def recompute_compact(arrays, provenance):
    """Recompute all four scientific metric rows without raw/model/reference files.

    Provenance remains the copied SHA-bound report evidence; this does not
    re-authenticate omitted original raw bytes or recompute policy rewards.
    """
    mu = np.asarray(arrays["effective_mu"], dtype=float)
    require(mu.shape == (2793, 2, 2) and np.isfinite(mu).all()
            and np.all(mu == provenance["effective_pair_mu"]), "Compact effective-mu provenance differs")
    raw = {"time_s": arrays["time_s"], "physics_dt_s": arrays["physics_dt_s"],
        "policy_dt_s": arrays["policy_dt_s"],
        "wheel_ground_total_contact_force_w_n": arrays["ground_total_force_w_n"],
        "wheel_ground_contact_force_w_n": arrays["ground_normal_force_w_n"],
        "wheel_ground_contact_count": arrays["active_point_count_gt5N"],
        "wheel_lateral_contact_slip_m_s": arrays["measured_patch_lateral_RMS_m_s"]}
    rows, rebuilt, result = analyze(raw, provenance)
    require(set(rebuilt) == set(arrays), "Compact array allowlist differs")
    for key, value in rebuilt.items():
        require(np.array_equal(value, arrays[key], equal_nan=True), "Compact derived signal differs: " + key)
    return {"rows": rows, "thresholds": result["thresholds"],
        "actual_per_contact_engine_boundary_computed": False,
        "full_tangent_relative_slip_available": False,
        "original_raw_assets_weights_or_physics_required": False}


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve diagnostics; fresh output directory required")
    paths = {name: Path(getattr(args, name)) for name in ("raw", "report")}
    hashes = {name: digest(getattr(args, name + "_sha256"), name) for name in paths}
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink() and sha256(path) == hashes[name],
                "Missing/direct input or explicit input SHA differs: " + name)
    require(sha256(cone.__file__) == DIAGNOSTIC_HELPER_SHA256, "Frozen shared force-diagnostic helper changed")
    report = read_json(paths["report"])
    provenance = validate_report(report, hashes["raw"], getattr(args, "legacy_source", None))
    with np.load(paths["raw"], allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    rows, arrays, result = analyze(raw, provenance)
    require(recompute_compact(arrays, provenance)["rows"] == rows, "Compact scientific rows do not exactly replay")
    require(all(sha256(path) == hashes[name] for name, path in paths.items()), "Input changed during archived analysis")
    summary = {"kind": "archived_native_PPO_wheel_resultant_Coulomb_reference_diagnostic",
        "status": "completed_offline_diagnostic", "data_scope": "wheel_resultant_lower_bound",
        "input_sha256": hashes, "input_files": {key: str(path.resolve()) for key, path in paths.items()},
        "analyzer_sha256": sha256(__file__), "shared_diagnostic_sha256": sha256(cone.__file__),
        "provenance": provenance, "force_units": "N", "tangent_basis": "world_xy_flat_floor",
        "physical_dt_s": .0025, "native_samples": 2793, "N": 1, "rows": rows,
        "exploratory_thresholds_NOT_acceptance": result["thresholds"],
        "actual_per_contact_engine_boundary_computed": False, "full_tangent_relative_slip_available": False,
        "static_friction_or_rolling_inferred": False, "reward_or_acceptance_changed": False,
        "new_physics_training_policy_inference_FK_or_controller_claim": False, "overall_goal_complete_claimed": False,
        "limitations": ["rho=norm(sum Ft)/(mu*sum Fn) is a wheel-resultant lower-bound Coulomb-reference proxy, not actual MuJoCo pyramidal-edge saturation.",
            "Opposing contact-point tangential forces can cancel. A small resultant rho cannot rule out saturated point contacts.",
            "Uniform isotropic mu is read from verified native materials. Original point tangent frames, per-point forces and effective per-point friction are absent.",
            "Stored patch lateral slip is normal-force-weighted RMS over individual points Fn>5N, not signed slip or full two-tangent velocity.",
            "Count-zero slip0 is masked unavailable, including aggregate Fn>5N from multiple weak points. Low lateral RMS does not establish nonslip or rolling.",
            "Joint observed-slip fractions over all loaded samples are lower bounds when RMS is unavailable; the conditional slip-given-near rate has its own explicit eligible denominator.",
            "Raw rho is never clipped. Strict rho>1 can be float roundoff; the additional1e-8 excess description is numerical/exploratory, not physical failure or actual engine-cone violation.",
            "Near .9 and patch-RMS>.05 are exploratory descriptions only, not a new drift success or task failure criterion.",
            "Persistence is t[last]-t[first] over consecutive observed-positive native samples only: a singleton spans0s, missing/ineligible samples break runs, and no future interval, epsilon filling or physical duration is invented.",
            "Native force samples can include impacts; near-reference force ratio alone does not prove active drift or loss of tracking.",
            "No PhysX normal-only channels, generic CEM body-net forces, guessed reference forces, height gates, or missing-slip imputation are used."]}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "friction_signals.npz", **arrays)
    with (output / "metric_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary["artifact_sha256"] = {name: sha256(output / name) for name in ("friction_signals.npz", "metric_table.csv")}
    write_json(output / "summary.json", summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("raw", "report", "output-dir"):
        result.add_argument("--" + name, type=Path, required=True)
    for name in ("raw-sha256", "report-sha256"):
        result.add_argument("--" + name, required=True)
    result.add_argument("--legacy-source", type=Path,
        help="Optional exact frozen legacy2251 source; current upstream bytes are never silently substituted")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "rows": result["rows"]}, indent=2))
