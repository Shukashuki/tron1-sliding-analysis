"""Add six-DoF base-body diagnostics to a frozen PPO smoothing publication.

The predeclared engineering screen is copied unchanged. These additional
signals are diagnostics, not new reward terms, height limits, or success gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ppo_smoothness import _world_to_body, rms, sha256, spectral_metrics, validate_clock


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT/"scripts/analyze_ppo_smoothness.py"
CORE_INTERVAL_S = (1.44, 4.)
SIGNALS = {
    "angular_world_xy": {"units": "rad/s", "components": ["x", "y"], "frame": "world",
                         "point": "base_Link rigid-body angular velocity; independent of choice of body point"},
    "angular_world_z": {"units": "rad/s", "components": ["z"], "frame": "world",
                        "point": "base_Link rigid-body angular velocity; not an instruction to suppress intended yaw"},
    "com_linear_velocity_world_xy": {"units": "m/s", "components": ["x", "y"], "frame": "world",
                        "point": "Original unmerged base_Link center of mass; not root-link-origin velocity"},
    "com_linear_velocity_world_z": {"units": "m/s", "components": ["z"], "frame": "world",
                        "point": "Original unmerged base_Link center of mass; not a derivative of root height"},
    "root_link_origin_position_world_xy": {"units": "m", "components": ["x", "y"], "frame": "world",
                        "point": "base_Link link origin, not center-of-mass position"},
    "root_link_origin_position_world_z": {"units": "m", "components": ["z"], "frame": "world",
                        "point": "base_Link link-origin height; no absolute-height target or bound"},
    "angular_body_xy": {"units": "rad/s", "components": ["x", "y"], "frame": "base_Link body",
                        "point": "Recorded angular velocity rotated into instantaneous base_Link axes; not quaternion differentiation"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def case_name(mu):
    return "mu"+f"{mu:.3f}".replace(".", "p")


def _state(arrays, prefix, field, count, width):
    name = prefix+"_"+field
    require(name in arrays, "Missing published signal: "+name)
    value = np.asarray(arrays[name], dtype=float)
    require(value.shape == (count, width) and np.isfinite(value).all(),
            "Expected finite native-state shape for "+name)
    return value


def load_publication(publication_dir):
    """Verify all published artifacts before consuming the derived recording."""
    folder = Path(publication_dir)
    summary_path = folder/"summary.json"
    summary = json.loads(summary_path.read_text())
    require(summary.get("kind") == "recorded_physx_ppo_smoothness_publication"
            and summary.get("training_engine") == summary.get("evaluation_engine") == "PhysX",
            "Require the frozen PhysX-to-PhysX smoothing publication")
    expected_helper = summary.get("analysis_source_sha256", {}).get("analyze_ppo_smoothness.py")
    require(expected_helper == sha256(HELPER), "Generic spectral helper differs from the published source hash")
    artifacts = summary.get("artifacts", {})
    require("derived_traces.npz" in artifacts, "Publication has no hash-bound derived recording")
    verified = {}
    for name, artifact in artifacts.items():
        require(isinstance(name, str) and Path(name).name == name and "\\" not in name
                and artifact.get("file") == name, "Artifact must be a publication-local basename")
        path = folder/name
        require(path.is_file() and not path.is_symlink(), "Published artifact is missing or not a regular local file: "+name)
        actual = sha256(path)
        require(actual == artifact.get("sha256"), "Published artifact SHA mismatch: "+name)
        verified[name] = actual
    with np.load(folder/"derived_traces.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return arrays, summary, {"summary.json": sha256(summary_path), **verified}


def _relative_change(candidate, baseline):
    if baseline <= 1e-12:
        return {"ratio": None, "change_percent": None, "reduction_percent": None,
                "note": "Baseline HF RMS is zero/negligible; no percentage or ratio is inferred"}
    return {"ratio": candidate/baseline, "change_percent": 100.*(candidate/baseline-1.),
            "reduction_percent": 100.*(1.-candidate/baseline), "note": "Positive change_percent means more absolute HF motion"}


def analyze_publication(publication_dir, *, high_frequency_hz=10., include_body_frame=True):
    arrays, parent, hashes = load_publication(publication_dir)
    coefficients = parent.get("coefficients")
    require(isinstance(coefficients, list) and len(coefficients) == 2
            and coefficients == [.7, .15], "Require the two predeclared friction cases in their published order")
    rows, index = [], {}
    shared_clock = None
    for mu in coefficients:
        case = case_name(mu)
        require(case in parent["cases"], "Published core-screen case is missing")
        for cohort in ("baseline", "candidate"):
            prefix = case+"_"+cohort
            require(prefix+"_time_s" in arrays, "Missing published native clock: "+prefix)
            time, dt = validate_clock(arrays[prefix+"_time_s"])
            require(abs(time[0]) < 1e-9 and np.isclose(dt, .005, rtol=1e-7, atol=1e-10),
                    "Require the original 200-Hz physical clock, not 50-Hz reference/playback samples")
            require(time[-1] >= CORE_INTERVAL_S[-1]-1e-9, "Physical recording does not cover the predeclared core")
            require(shared_clock is None or np.array_equal(time, shared_clock),
                    "Matched recordings must keep the same native clock; no alignment/resampling")
            shared_clock = time
            count = len(time)
            angular = _state(arrays, prefix, "angular_velocity_w_rad_s", count, 3)
            com_velocity = _state(arrays, prefix, "com_velocity_w_m_s", count, 3)
            root_position = _state(arrays, prefix, "position_m", count, 3)
            signals = {"angular_world_xy": angular[:, :2], "angular_world_z": angular[:, 2],
                "com_linear_velocity_world_xy": com_velocity[:, :2], "com_linear_velocity_world_z": com_velocity[:, 2],
                "root_link_origin_position_world_xy": root_position[:, :2], "root_link_origin_position_world_z": root_position[:, 2]}
            if include_body_frame:
                quaternion = _state(arrays, prefix, "quaternion_wxyz", count, 4)
                require(np.allclose(np.linalg.norm(quaternion, axis=1), 1., atol=1e-5, rtol=0),
                        "Published root quaternion must remain unit wxyz")
                signals["angular_body_xy"] = _world_to_body(quaternion, angular)[:, :2]
            for window, interval in (("full", (float(time[0]), float(time[-1]))), ("core", CORE_INTERVAL_S)):
                mask = (time >= interval[0]-1e-9) & (time <= interval[1]+1e-9)
                for signal_name, values in signals.items():
                    metrics = spectral_metrics(values[mask], dt, high_frequency_hz)
                    row = {"mu": mu, "cohort": cohort, "body": "base_Link", "window": window,
                        "interval_s": list(interval), "signal": signal_name, **SIGNALS[signal_name],
                        "raw_signal_rms": rms(values[mask]), "spectral_metrics": metrics,
                        "mean_square_units": "("+SIGNALS[signal_name]["units"]+")^2",
                        "role": "Additional diagnostic only; not a reward term, constraint or screen criterion"}
                    rows.append(row)
                    index[(mu, cohort, window, signal_name)] = row
    changes = []
    for row in rows:
        if row["cohort"] != "candidate":
            continue
        baseline = index[(row["mu"], "baseline", row["window"], row["signal"])]
        before = baseline["spectral_metrics"]["high_frequency_rms"]
        after = row["spectral_metrics"]["high_frequency_rms"]
        changes.append({"mu": row["mu"], "window": row["window"], "signal": row["signal"],
            "units": row["units"], "baseline_hf_rms": before, "candidate_hf_rms": after,
            "hf_rms_difference": after-before, "absolute_hf_rms_increased": bool(after > before),
            **_relative_change(after, before)})
    return {"schema_version": 1, "kind": "additional_recorded_ppo_base_body_six_dof_jitter_diagnostics",
        "source_publication_sha256": hashes["summary.json"], "verified_source_artifact_sha256": hashes,
        "analysis_source_sha256": {"analyze_ppo_body_jitter.py": sha256(__file__),
                                   "analyze_ppo_smoothness.py": sha256(HELPER)},
        "body": "base_Link", "scope": "One base rigid body only; full robot link states are not present in this archive",
        "coefficient_order": coefficients, "cutoff_hz_exclusive": float(high_frequency_hz),
        "native_sample_hz": 200., "reference_or_video_resampled": False,
        "rows": rows, "relative_changes": changes,
        "predeclared_engineering_screen": {"unchanged": True, "original_status": parent["status"],
            "original_passes_both_conditions": parent["engineering_screen_passes_both_conditions"],
            "original_case_comparisons": {name: value["comparison"] for name, value in parent["cases"].items()}},
        "new_reward_or_constraint": False, "absolute_height_target_added": False,
        "intended_yaw_or_sideslip_targeted_to_zero": False, "jitter_resolved_claimed": False,
        "whole_body_jitter_reduction_claimed": False, "requires_fullbody_visual_review": True,
        "statistical_success_claimed": False, "causal_improvement_claimed": False,
        "new_training": False, "new_simulation": False, "new_policy_inference": False,
        "diagnostic_notes": [
            "A reduction in angular XY jitter does not establish a reduction in every translational/rotational component.",
            "COM velocity is the recorded original unmerged base_Link COM twist. Root position is the recorded link origin, not COM position.",
            "No velocity is obtained by differentiating root position, and no acceleration is obtained by differentiating quaternions.",
            "The six DoF are reported in component groups with their own units; no mixed-unit aggregate score is constructed.",
            "Each band RMS uses the frozen generic helper: linear detrend, Hann window, channel-mean Parseval normalization, frequencies strictly above cutoff.",
            "Raw position RMS includes its world offset; use the detrended band RMS for jitter, not raw absolute height/position.",
            "Intended turning/yaw, height freedom and nonzero sideslip remain unchanged. HF diagnostics are not new failure gates.",
            "No reference subtraction, time warp, coordinate alignment, filtering or resampling is applied.",
            "One seed/episode is descriptive evidence only; inspect full-horizon physical states and full-body video."]}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--publication-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True, help="Fresh JSON file; existing evidence is never overwritten")
    result.add_argument("--high-frequency-hz", type=float, default=10.)
    result.add_argument("--no-body-frame", action="store_true", help="Omit optional instantaneous body-frame angular XY diagnostic")
    return result


def run(args):
    output = Path(args.output)
    require(not output.exists(), "Preserve existing diagnostics: choose a fresh output file")
    result = analyze_publication(args.publication_dir, high_frequency_hz=args.high_frequency_hz,
                                 include_body_frame=not args.no_body_frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    print(json.dumps({"diagnostic_rows": len(outcome["rows"]), "predeclared_screen_unchanged": True,
                      "whole_body_jitter_reduction_claimed": False}))
