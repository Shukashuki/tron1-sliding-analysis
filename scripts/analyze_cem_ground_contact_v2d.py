"""Strict offline audit of a completed ACTUAL V2d CEM nominal episode.

Shares the frozen A metrics and unchanged nominal screen, not its reward or
input-identity gate. No simulation, policy inference, FK, tuning or native build.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import analyze_cem_ground_contact_v2 as base
import diagnose_ground_contact_windows_v2 as contact
import ground_contact_reward_v2d as definition
import analyze_cem_ground_contact_v2b as prior_b
import analyze_cem_ground_contact_v2c as prior_c
import run_cem_ground_contact_v2d as runner
from controller_trace import native_trace
from ppo_backend_400hz_contract import require, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "config/ppo_ground_contact_v2d_400hz_v1.json"
PROTOCOL_SHA256 = "e89a5a3b2336da37097cf7dad163d560b05f38ae15ad6578742b9d567940804a"
METRICS_SOURCE_SHA256 = "540f85cb960669d30ef212a4d1ab68c0c1c6abe8508505ac8103dedbe3b1a0ed"
LIBRARY_SHA256 = "1c4c0a66da7770a9579e6a19d0aded25a6656426f76e068c686a0ad977435095"
BASELINE_PINS = {
    "summary": "4cfa8cb34d2213908c2e90183275974ae5ae6bb2de49ac449849ab0cea605a7f",
    "report.json": "343f6270e8be135e2afd4f06df69d8d4f1273cca2cc651188a1198c81ab901da",
    "rollout.npz": "eff2b735b07b481e6ead32b76a716066f71815401f91231345dde84d451804be"}
V2A_PINS = {
    "report.json": "a1573ea6a2c97d9c1a6f132f14d1b0a3aab9abdd650606b5f73602c980061081",
    "rollout.npz": "b4a6dc3ee5e22977140c669f4a9041fe207abaa1ecd129814bc6d3343e01a8ae"}
V2B_ANALYZER_SHA256 = "ae6cb319af8e3e23e20532f0a172f2c61097a090ea12e2e030355912ced8c2ff"
V2B_PINS = {
    "report.json": "37f8df44d21223be2069027ed4b43288052a5704ea1ba34179ed0ab3ca09e32a",
    "rollout.npz": "fe67b59bfa238d81d0b2528f159ebfcc2f3c415f0dc70da45b2b403caaace319"}
V2C_ANALYZER_SHA256 = "5b8bb74b45816e16d65250039490c75ed1cce4d5b2bcacaee88fd0b8d5be2416"
V2C_PINS = {
    "report.json": "0a44a1478c442b101c561d3a260417f458bb87d862257d66e8e4a55e100d6be1",
    "rollout.npz": "99a4ce2666ecfcf2fe19505dfd5e9527d61e699a45a031636a80ce4c14c7c3ba"}
SOURCE_PINS = {
    "scripts/run_cem_ground_contact_v2d.py": "24dcbe41050c49d80105222fc508c6f07b0b6091dda02d236bfb79d283c32f16",
    "config/ppo_cem_matched_reward_400hz_v1.json": "410180756a7d489a6f8181157d5993813b17b60f7e103449b9bfa81a25c0984d",
    "config/ppo_ground_contact_v2d_400hz_v1.json": PROTOCOL_SHA256,
    "scripts/ground_contact_reward_v2.py": "5794ceb435c6ddc9cc374fa5ebe7718edc99d4c3181ebeac111cafe128461bf5",
    "scripts/ground_contact_reward_v2b.py": "359997a5e4e1734f14ffd8c64a61bbaf0f8fb1d0dd6b274001a77689bb62ba6c",
    "scripts/ground_contact_reward_v2c.py": "758eba1c7234cd43b211262541fce0cab6ad91be8ae0f10a44714d037322ebfb",
    "scripts/ground_contact_reward_v2d.py": "1f6245ca3ca08483704d4f42a044ba38b8d09f01d81dc6e71bbf423d206898a7",
    "scripts/nmpc_batch_ground_contact_v2d_400hz.py": "e4fb8b22d02556579248b8419ffa2f47580be6351c62f46178fcab82b9ac55f1",
    "scripts/nmpc_native_ground_contact_400hz.py": "dccee6518aae5c4e20e513e27ccb27641174cd07836c1f951617c16398b5aa3c",
    "scripts/run_ppo_cem_matched_reward_400hz.py": "f083a62813e8e5c7bf7099d90e603eaa3eea7909e148cfb636e613b200177bea",
    "scripts/nmpc_shooting.py": "7c667ef0cbd32c58ce53882302e75f5d33487bcfea51c2f6ec3435d454a480ce",
    "scripts/nmpc_tracking_objective.py": "ff22eebb2b22b9af7dba114c64fcd10f6b0a90822559af8f32cbe6884015b860",
    "scripts/nmpc_batch_objective.py": "1c9da90ae66778a71ceb2e6bb9d0dcf7f0a2259fafee39b37cbe958b79178fae",
    "scripts/nmpc_batch_objective_400hz.py": "3292c7753b2bbd336cb4944f6d64d417c738dfba8f994e878024dd52f5e11246",
    "scripts/nmpc_native_batch_400hz.py": "9c07f5e3d88f123ca067f73ead43679eeb378c35ed0afe51b6375870f654f335",
    "scripts/run_nmpc_tracking.py": "bce84b92fba8d3ea2f44c6cd5e5d32abd65bed6e20c4f1beb8473bae9cfb1977",
    "scripts/classical_tracking.py": "3fcafdb6a16058280aa1e5a0ade6113f6963ddf3085c25ebe2ed6fbb30661c57",
    "scripts/eval_ppo_backend_400hz_mujoco.py": "9887aa0b4b7ef74845f47f8623356490a9990e4449e0b42f4a77b2b35a49c1e4",
    "scripts/render_ppo_beta_physx_to_mujoco_400hz.py": "9268618566845d8f0fba02e806ce33c9168512466b57f495cfe294ed28acdbee",
    "native/nmpc_batch_ground_contact_400hz.cpp": "e80e76ad68352da00aeb4cf7215fc68f127b6a953d1afee4ad65e3ba0178aef1"}


def load_protocol(path=DEFAULT_PROTOCOL):
    require(sha256(path) == PROTOCOL_SHA256, "Frozen D protocol SHA differs")
    protocol = read_json(path)
    require(protocol["cohort_id"] == "tron1_ground_contact_v2d_any_wheel_400hz_v1"
            and protocol["ground_contact_reward_v2"] == definition.validate_settings(), "Wrong D formula contract")
    require(protocol["nominal_acceptance_before_training"] == read_json(base.DEFAULT_PROTOCOL)["nominal_acceptance_before_training"],
            "C must not relax A nominal acceptance")
    require(sha256(base.__file__) == METRICS_SOURCE_SHA256, "Frozen shared metrics/screen source differs")
    return protocol


def validate_metadata(report, experiment, protocol_path=DEFAULT_PROTOCOL):
    load_protocol(protocol_path)
    require(report.get("status") == experiment.get("status") == "completed"
            and report.get("smoke_only") is False and report.get("truncated_smoke") is False
            and report.get("completed_policy_steps") == 349 and report.get("native_samples") == 2793
            and report.get("completed_full_reference") is True, "Only terminal complete349-step D episodes may be scored")
    require(report.get("kind") == "400hz_CEM_ground_contact_reward_v2d_anywheel"
            and report.get("reward_definition_version") == "ground_contact_v2d_any_wheel"
            and report.get("protocol_sha256") == PROTOCOL_SHA256
            and experiment.get("kind") == "ground_contact_reward_v2d_CEM_nominal_only", "Wrong D identity/version/protocol")
    require(report.get("source_sha256") == experiment.get("source_sha256") == SOURCE_PINS,
            "Require exact complete frozen D source ledger")
    for name, digest in SOURCE_PINS.items():
        require(sha256(ROOT / name) == digest, "Executed frozen D source changed: " + name)
    require(report.get("condition") == "beta_on" and report.get("plant_mu") == .7
            and report.get("prediction_mu_scenarios") == [.7] and report.get("prediction_friction_known") is True
            and report.get("floor_filter_verified") is True
            and report.get("termination_or_reset_after_initialization") is False
            and report.get("height_contact_tilt_failure_gate") is False
            and (report.get("physics_dt_s"), report.get("policy_dt_s"), report.get("decimation")) == (.0025, .02, 8),
            "Wrong D material/floor/reset/native clock")
    require(report.get("ground_contact_reward_v2") == definition.formula_metadata()
            and report.get("ground_reward_external_weight") == .4
            and report.get("reward_term_count") == 15 and report.get("other_reward_terms_unchanged") == 14
            and report.get("actual_reward_uses_ground_contact_v2") is True, "D actual-stage reward contract differs")
    require(experiment.get("native_abi") == 3 and report.get("native_library_sha256") == LIBRARY_SHA256
            and report.get("native_backend", {}).get("native_abi") == 3, "Wrong frozen ABI3 worker")
    require(experiment.get("sample_size") == 1 and experiment.get("new_policy_training_or_inference") is False
            and experiment.get("automatic_retry") is False, "Wrong declared episode scope")
    require(report.get("CEM_config") == read_json(runner.nominal.DEFAULT_SETTINGS)["nmpc"]
            and report.get("discount_factor") == .99, "Frozen D optimizer/discount differs")


def load_completed(folder, protocol_path=DEFAULT_PROTOCOL):
    folder = Path(folder).resolve()
    names = ("report.json", "rollout.npz", "solver_log.json", "experiment.json")
    require(all((folder/name).is_file() and not (folder/name).is_symlink() for name in names),
            "Missing direct completed D files; prefixes/smokes are not analyzable")
    hashes = {name: sha256(folder/name) for name in names}
    report, experiment = read_json(folder/"report.json"), read_json(folder/"experiment.json")
    validate_metadata(report, experiment, protocol_path)
    require(experiment["report_sha256"] == hashes["report.json"]
            and report["rollout_sha256"] == hashes["rollout.npz"]
            and report["solver_log_sha256"] == hashes["solver_log.json"], "D terminal file SHA bindings differ")
    with np.load(folder/"rollout.npz", allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    decisions = json.loads((folder/"solver_log.json").read_text())
    require(isinstance(decisions, list), "D solver log must be a decision list")
    require(len(decisions) == 349 and raw["qpos"].shape == (2793, 15)
            and raw["actions"].shape == (349, 8), "Incomplete D states/actions/decisions")
    failures = sum(bool(row["solver"]["optimization_failed"]) for row in decisions)
    require(failures == report["solver_failures"], "Reported D solver failure count differs")
    reward_audit = runner.recompute_actual_rewards(raw, report, decisions)
    require(reward_audit == report["actual_reward_audit"], "Actual D 15-term reward recomputation differs")
    require(all(sha256(folder/name) == digest for name, digest in hashes.items()), "D completed files changed during loading")
    return raw, report, hashes, reward_audit


def load_original(reference):
    require(sha256(base.BASELINE_SUMMARY) == BASELINE_PINS["summary"], "Published original baseline summary changed")
    hashes = {name: sha256(base.BASELINE_FOLDER/name) for name in ("report.json", "rollout.npz")}
    require(all(hashes[name] == BASELINE_PINS[name] for name in hashes), "Original CEM baseline SHA changed")
    row = next(value for value in read_json(base.BASELINE_SUMMARY)["rows"] if value["id"] == "cem_beta_on")
    require(row["report_sha256"] == hashes["report.json"] and row["recording_sha256"] == hashes["rollout.npz"],
            "Original CEM public row binding differs")
    with np.load(base.BASELINE_FOLDER/"rollout.npz", allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    result = base.recorded_metrics(raw, reference)
    for name, published in (("core", "core"), ("whole_reference_overlap", "reference_overlap")):
        require(result["windows"][name]["tracking"] == row["windows"][published]["tracking"],
                "Original baseline metric recomputation differs")
    return result, hashes


def load_v2a(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(all(sha256(folder/name) == digest for name, digest in V2A_PINS.items()), "Optional A baseline SHA differs")
    raw, report, hashes, audit = base.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "A/D observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "A/D fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2a CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_D_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def load_v2b(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(sha256(prior_b.__file__) == V2B_ANALYZER_SHA256, "Frozen prior B analyzer changed")
    require(all(sha256(folder/name) == digest for name, digest in V2B_PINS.items()), "Optional B baseline SHA differs")
    raw, report, hashes, audit = prior_b.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "B/D observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "B/D fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2b CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_D_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def load_v2c(folder, reference, candidate_raw, candidate_report):
    folder = Path(folder)
    require(sha256(prior_c.__file__) == V2C_ANALYZER_SHA256, "Frozen prior C analyzer changed")
    require(all(sha256(folder/name) == digest for name, digest in V2C_PINS.items()), "Optional C baseline SHA differs")
    raw, report, hashes, audit = prior_c.load_completed(folder)
    require(np.array_equal(raw["qpos"][0], candidate_raw["qpos"][0])
            and np.array_equal(raw["qvel"][0], candidate_raw["qvel"][0]), "C/D observable initial state differs")
    for key in ("CEM_config", "plant_contact_material_audit", "prediction_contact_material_audit", "model_weight_n"):
        require(report[key] == candidate_report[key], "C/D fixed setting differs: " + key)
    return {"label": "Prior ACTUAL V2c CEM closed-loop episode, distinct reward definition; N=1",
        "input_sha256": hashes, "reward_audit": audit, "metrics": base.recorded_metrics(raw, reference),
        "episode_returns_comparable_claimed": False, "counterfactual_D_result_claimed": False,
        "same_observable_initial_state": True, "statistical_or_general_robustness_claimed": False}


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve analyses; fresh output directory required")
    protocol = load_protocol(args.protocol)
    raw, report, hashes, reward_audit = load_completed(args.input_dir, args.protocol)
    require(report["motion_sha256"] == sha256(runner.paired.DEFAULT_REFERENCE), "Locked reference SHA differs")
    reference = base.metrics.reference_data(runner.paired.DEFAULT_REFERENCE)
    old, old_hashes = load_original(reference)
    actual = base.recorded_metrics(raw, reference)
    trace = native_trace(raw, "nmpc")
    diagnosis = contact.analyze_contact(trace["time_s"], trace["position"], trace["com_velocity"], trace["quaternion"],
        raw["ground_normal_force_w"][..., 2], report["model_weight_n"], reference,
        nonwheel=raw.get("nonwheel_ground_contact_force_norm_n"), include_sensitivity=False)
    diagnosis.update(actual_recorded_reward_candidate="ground_contact_v2d",
        normalization_weight_source="SHA-bound D validation model_weight_n; actual native MuJoCo model weight",
        input_sha256=hashes, no_counterfactual_controller_or_PPO_performance=True)
    prior = None
    if getattr(args, "v2a_input_dir", None) is not None:
        prior = load_v2a(args.v2a_input_dir, reference, raw, report)
    previous_b = None
    if getattr(args, "v2b_input_dir", None) is not None:
        previous_b = load_v2b(args.v2b_input_dir, reference, raw, report)
    previous_c = None
    if getattr(args, "v2c_input_dir", None) is not None:
        previous_c = load_v2c(args.v2c_input_dir, reference, raw, report)
    screen = base.nominal_screen(actual, old, protocol)
    summary = {"kind": "CEM_ground_contact_V2d_ACTUAL_nominal_readonly_audit", "status": "completed_analysis",
        "input_directory": str(Path(args.input_dir).resolve()), "input_sha256": hashes,
        "protocol_sha256": sha256(args.protocol), "analyzer_sha256": sha256(__file__),
        "shared_metric_source_sha256": METRICS_SOURCE_SHA256,
        "contact_diagnostic_source_sha256": sha256(contact.__file__), "executed_source_sha256": copy.deepcopy(SOURCE_PINS),
        "sample_size_per_case": 1, "actual_D_reward_audit": reward_audit, "actual_V2d": actual,
        "original_grounded_baseline": old, "baseline_input_sha256": old_hashes,
        "baseline_summary_sha256": BASELINE_PINS["summary"], "optional_prior_actual_V2a": prior,
        "optional_prior_actual_V2b": previous_b, "optional_prior_actual_V2c": previous_c,
        "nominal_screen": screen,
        "runtime": {key: report[key] for key in ("solver_failures", "plant_warning_counts", "compute_ms_median_p95_max")},
        "no_new_dynamics_policy_inference_training_build_or_tuning": True,
        "overall_task_goal_complete_claimed": False,
        "limitations": ["ACTUAL closed-loop D episode only; not offline A-force sensitivity or a PPO result.",
            "N=1 per controller; no causal population effect or universal friction threshold.",
            "Reward definitions differ across original/A/B/C/D: episode reward sums are not a fair performance ranking.",
            "Any-wheel continuity requires actual Fz>5N; the other wheel may lift; zero force is not a geometric flight label.",
            "Acceptance criteria are frozen unchanged; no height/tilt/beta gate replaces independent XY tracking.",
            "Numerical CEM acceptance does not certify qualitative review, PPO/dual-engine results or overall task completion."]}
    arrays = {"time_s": trace["time_s"], "position": trace["position"], "velocity": trace["com_velocity"],
        "quaternion": trace["quaternion"], "ground_normal_Fz_n": raw["ground_normal_force_w"][..., 2],
        "evaluation_weight_n": np.array(report["model_weight_n"])}
    arrays.update({"reference_"+name: reference[name] for name in ("time_s", "position", "velocity")})
    if "nonwheel_ground_contact_force_norm_n" in raw:
        arrays["nonwheel_ground_force_norm_n"] = raw["nonwheel_ground_contact_force_norm_n"]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"native_contact_context.npz", **arrays)
    diagnosis["compact_sha256"] = sha256(output/"native_contact_context.npz")
    write_json(output/"contact_diagnostics.json", diagnosis)
    summary["contact_diagnostics_sha256"] = sha256(output/"contact_diagnostics.json")
    require(all(sha256(Path(args.input_dir)/name) == digest for name, digest in hashes.items()), "D inputs changed during analysis")
    write_json(output/"summary.json", summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--input-dir", type=Path, required=True)
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--v2a-input-dir", type=Path, help="Optional SHA-pinned prior ACTUAL A episode, not reward-sum ranking")
    result.add_argument("--v2b-input-dir", type=Path, help="Optional SHA-pinned prior ACTUAL B episode, not reward-sum ranking")
    result.add_argument("--v2c-input-dir", type=Path, help="Optional SHA-pinned prior ACTUAL C episode, not reward-sum ranking")
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
