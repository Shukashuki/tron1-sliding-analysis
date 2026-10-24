"""Publish completed H CEM evidence and asset-free nine-case native metric replay.

This is an offline packager, not a simulation launcher. It requires terminal
349-decision evidence and the frozen all15-term reward recomputation. Public
compact replay verifies recorded metrics/contact diagnostics, not omitted reward
histories, physics feasibility, PPO performance or overall task completion.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

import analyze_cem_ground_contact_v2h as audit
from controller_metrics import heading_signals
from controller_trace import native_trace
from package_ground_contact_v2_partial import copy_bound, source_archive
from ppo_backend_400hz_contract import read_json, require, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/ground_contact_v2h_20261024_v1"
ANALYZER_SHA256 = "843c546a2545e7ed66b07bd57fa6326903037fd57fe77d026922def62ae573db"
REFERENCE_SHA256 = "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
KIND = "ACTUAL_CEM_H_COMPLETED_NINE_CASE_NOMINAL_PROOF"
CASE_ORDER = ("original_grounded_baseline", "prior_actual_V2a", "prior_actual_V2b",
              "prior_actual_V2c", "prior_actual_V2d", "prior_actual_V2e", "prior_actual_V2f", "prior_actual_V2g", "actual_V2h")
TRACE_FIELDS = ("time_s", "position", "quaternion", "com_velocity", "angular_velocity",
                "contact_count", "contact_force", "wheel_quaternion_wxyz", "ground_normal_Fz_n")


def verify_sources():
    require(sha256(audit.__file__) == ANALYZER_SHA256, "Frozen H CEM analyzer changed")
    protocol = audit.load_protocol()
    require(len(audit.SOURCE_PINS) == 23, "Require complete A-H23-source executed ledger")
    for name, digest in audit.SOURCE_PINS.items():
        require(sha256(ROOT/name) == digest, "Frozen H executed source changed: " + name)
    return protocol


def expected_cases(summary):
    result = {"original_grounded_baseline": summary["original_grounded_baseline"]}
    for suffix in "abcdefg":
        prior = summary.get("optional_prior_actual_V2"+suffix)
        require(isinstance(prior, dict) and isinstance(prior.get("metrics"), dict),
                "Require actual, SHA-bound prior " + suffix.upper() + " episode")
        require(prior.get("same_observable_initial_state") is True
                and prior.get("episode_returns_comparable_claimed") is False
                and prior.get("counterfactual_H_result_claimed") is False,
                "Wrong prior comparison scope")
        result["prior_actual_V2"+suffix] = prior["metrics"]
    result["actual_V2h"] = summary["actual_V2h"]
    return result


def validate_summary(summary, report, experiment, protocol=None):
    protocol = verify_sources() if protocol is None else protocol
    audit.validate_metadata(report, experiment)
    require(summary.get("kind") == "CEM_ground_contact_V2h_ACTUAL_nominal_readonly_audit"
            and summary.get("status") == "completed_analysis"
            and summary.get("analyzer_sha256") == ANALYZER_SHA256
            and summary.get("protocol_sha256") == audit.PROTOCOL_SHA256
            and summary.get("shared_metric_source_sha256") == audit.METRICS_SOURCE_SHA256
            and summary.get("contact_diagnostic_source_sha256") == sha256(audit.contact.__file__)
            and summary.get("executed_source_sha256") == audit.SOURCE_PINS
            and summary.get("sample_size_per_case") == 1
            and summary.get("overall_task_goal_complete_claimed") is False
            and summary.get("no_new_dynamics_policy_inference_training_build_or_tuning") is True,
            "Require actual completed frozen H analysis, not prior/smoke/pending results")
    require(report.get("motion_sha256") == REFERENCE_SHA256, "Locked reference SHA differs")
    reward = summary.get("actual_H_reward_audit")
    require(reward == report.get("actual_reward_audit") and isinstance(reward, dict)
            and reward.get("status") == "verified_exact" and reward.get("stages") == 349
            and reward.get("reward_terms") == 15
            and reward.get("native_substeps_per_stage") == 8,
            "Require preserved terminal349x15 actual reward audit")
    require(summary.get("baseline_summary_sha256") == audit.BASELINE_PINS["summary"]
            and summary.get("baseline_input_sha256") == {name: audit.BASELINE_PINS[name]
                for name in ("report.json", "rollout.npz")}, "Original baseline provenance differs")
    for suffix, pins in (("a", audit.V2A_PINS), ("b", audit.V2B_PINS),
                         ("c", audit.V2C_PINS), ("d", audit.V2D_PINS), ("e", audit.V2E_PINS), ("f", audit.V2F_PINS), ("g", audit.V2G_PINS)):
        prior = summary.get("optional_prior_actual_V2"+suffix)
        require(isinstance(prior, dict) and all(prior.get("input_sha256", {}).get(name) == digest
                    for name, digest in pins.items()), "Wrong prior " + suffix.upper() + " SHA binding")
    cases = expected_cases(summary)
    require(summary.get("nominal_screen") == audit.base.nominal_screen(
                cases["actual_V2h"], cases["original_grounded_baseline"], protocol),
            "H nominal screen differs from unchanged locked criteria")
    require(summary.get("runtime") == {name: report[name] for name in
                ("solver_failures", "plant_warning_counts", "compute_ms_median_p95_max")},
            "H runtime diagnostics differ")
    return cases


def recorded_metrics(trace, reference):
    require(np.asarray(trace["time_s"]).shape == (2793,)
            and np.allclose(trace["time_s"], np.arange(2793)*.0025, rtol=0, atol=1e-9),
            "Require complete consecutive2793-state native clock")
    signals = heading_signals(trace["quaternion"], trace["com_velocity"],
        trace["wheel_quaternion_wxyz"], np.zeros((2793, 2, 3)))
    windows = {name: audit.base.metrics.window_metrics(trace, reference, signals, interval)
        for name, interval in (("core", (1.44, 4.)), ("whole_reference_overlap", (0., 6.96)),
                               ("full_native", (0., 6.98)))}
    support = {"core": audit.base.support_summary(trace["time_s"], trace["ground_normal_Fz_n"], (1.44, 4.)),
        "full_contact": audit.base.support_summary(trace["time_s"], trace["ground_normal_Fz_n"], (.02, 6.96))}
    return {"windows": windows, "any_wheel_ground_support": support,
        "nonzero_beta_is_intended": True, "height_contact_tilt_or_beta_used_as_XY_failure_gate": False,
        "whole_tracking_reference_end_s": 6.96, "physical_recording_end_s": 6.98,
        "trajectory_aligned_smoothed_or_extrapolated": False}


def comparison_arrays(raw_cases, reference):
    require(tuple(raw_cases) == CASE_ORDER, "Wrong nine-case identity/order")
    arrays = {"reference__"+name: np.asarray(reference[name]).copy()
              for name in ("time_s", "position", "velocity", "quaternion")}
    for case, raw in raw_cases.items():
        trace = native_trace(raw, "nmpc")
        trace["wheel_quaternion_wxyz"] = np.asarray(raw["body_quat_w"])[:, -2:]
        trace["ground_normal_Fz_n"] = np.asarray(raw.get(
            "ground_normal_force_w", raw["wheel_ground_contact_force_w_n"]))[..., 2]
        for name in TRACE_FIELDS:
            arrays[case+"__"+name] = np.asarray(trace[name]).copy()
    return arrays


def recompute_comparison(arrays):
    reference = {key.removeprefix("reference__"): value for key, value in arrays.items()
                 if key.startswith("reference__")}
    expected_keys = {"reference__"+name for name in ("time_s", "position", "velocity", "quaternion")}
    expected_keys |= {case+"__"+name for case in CASE_ORDER for name in TRACE_FIELDS}
    require(set(arrays) == expected_keys, "Compact nine-case schema differs")
    return {case: recorded_metrics({name: arrays[case+"__"+name] for name in TRACE_FIELDS}, reference)
            for case in CASE_ORDER}


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def verify_contact(arrays, diagnosis, summary):
    require(diagnosis.get("actual_recorded_reward_candidate") == "ground_contact_v2h"
            and diagnosis.get("input_sha256") == summary["input_sha256"]
            and diagnosis.get("no_counterfactual_controller_or_PPO_performance") is True,
            "Contact diagnostics are not actual H recorded evidence")
    actual = audit.contact.recompute_compact(arrays, include_sensitivity=False)
    for name, value in actual.items():
        require(value == diagnosis.get(name), "Contact diagnostic differs: " + name)
    return actual


def source_ledger():
    """Bind repository helpers used by replay, without models or binary libraries."""
    ledger = dict(audit.SOURCE_PINS)
    todo = [Path(__file__).resolve(), Path(audit.__file__).resolve()]
    seen = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([item.name for item in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                candidate = ROOT/"scripts"/((name or "").split(".")[0]+".py")
                if candidate.is_file() and candidate not in seen:
                    todo.append(candidate)
    for path in seen:
        ledger[str(path.relative_to(ROOT))] = sha256(path)
    for name in ("ppo_ground_contact_v2_400hz_v1.json", "ppo_ground_contact_v2b_400hz_v1.json",
                 "ppo_ground_contact_v2c_400hz_v1.json", "ppo_ground_contact_v2d_400hz_v1.json",
                 "ppo_ground_contact_v2e_400hz_v1.json", "ppo_ground_contact_v2f_400hz_v1.json",
                 "ppo_ground_contact_v2g_400hz_v1.json"):
        ledger["config/"+name] = sha256(ROOT/"config"/name)
    return dict(sorted(ledger.items()))


def verify_archive(folder, manifest):
    folder = Path(folder)
    source = read_json(folder/"source_snapshot.manifest.json")
    require(sha256(folder/"executed_source.tar.gz") == source["archive"]["sha256"]
            == manifest["source_archive_sha256"], "Pure-source archive SHA differs")
    require(source["executed_controller_source_sha256"] == audit.SOURCE_PINS
            and source["models_weights_credentials_native_libraries_included"] is False,
            "Wrong executed H source archive identity/scope")
    entries = {entry["path"]: entry for entry in source["files"]}
    require(len(entries) == len(source["files"]), "Duplicate archive source path")
    require({name: item["sha256"] for name, item in entries.items()} == manifest["replay_helper_sha256"],
            "Source archive and helper ledger differ")
    with tarfile.open(folder/"executed_source.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        require(len(members) == len(entries) and {m.name for m in members} == set(entries),
                "Archive member ledger differs")
        for member in members:
            path = Path(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts
                    and path.suffix in (".py", ".json", ".cpp"), "Unsafe/non-source archive member")
            entry = entries[member.name]
            require(member.size == entry["bytes"]
                    and hashlib.sha256(archive.extractfile(member).read()).hexdigest() == entry["sha256"],
                    "Archive member bytes/hash differs")
    return source


def status_from_summary(summary):
    support = summary["actual_V2h"]["any_wheel_ground_support"]["full_contact"]
    screen = summary["nominal_screen"]
    return {"status": "ACTUAL_CEM_H_COMPLETED_NUMERICAL_SCREENS_PASSED_REVIEW_PENDING"
            if screen["numeric_nominal_episode_pass"] else "ACTUAL_CEM_H_COMPLETED_NUMERICAL_CRITERIA_FAILED",
        "snapshot_utc": datetime.now(timezone.utc).isoformat(), "sample_size_per_case": 1,
        "conditions": {"mu": .7, "friction_known_to_CEM": True, "physics_hz": 400, "policy_hz": 50,
            "beta_reward_on": True, "support_gap_weight": 1., "support_force_threshold_n": 5., "positive_mean_weight": .75, "positive_minimum_weight": .25, "external_grounded_weight": .4,
            "single_wheel_support_allowed": True, "height_flight_or_zero_beta_gate": False,
            "contact_reward_formula": audit.definition.formula_metadata(),
            "beta_gate_source": "existing unweighted reference-beta tracking quality",
            "beta_gate_low_reference_speed_m_s": .3, "support_gap_penalty_ungated": True},
        "CEM_H": {"full_policy_decisions": 349, "native_states": 2793,
            "numeric_nominal_episode_pass": screen["numeric_nominal_episode_pass"],
            "failed_checks": screen["failed_checks"],
            "whole_contact_interval_s": [.02, 6.96],
            "whole_anywheel_support_fraction": support["any_wheel_support_sample_fraction"],
            "whole_longest_no_support_s": support["longest_no_support_observed_s"],
            "runtime": copy.deepcopy(summary["runtime"]), "offline_solver_not_real_time": True},
        "PPO_H": {"status": "OUT_OF_SCOPE_NO_PPO_PERFORMANCE_IN_THIS_CEM_BUNDLE",
            "new_formal_policy_results_claimed": False},
        "strict_continuous_anywheel_contact_achieved": bool(support["every_physical_sample_supported"]
            and support["longest_no_support_observed_s"] == 0.),
        "goal_complete": False, "episode_reward_ranking_across_definitions_claimed": False,
        "limitations": ["N=1 each; no seed statistics, friction-range or general causal robustness claim.",
            "Whole-reference and core tracking/contact/HF screens remain separate and unchanged.",
            "Lower HF, core-only tracking or numerical screens do not establish overall task completion.",
            "Original/A/B/C/D/E/F/H reward definitions differ; their summed rewards are not ranked."]}


def verify_public(folder):
    """Read-only replay: no assets, raw outputs, extraction, physics, FK or inference."""
    folder = Path(folder).resolve()
    manifest = read_json(folder/"artifact_manifest.json")
    require(manifest.get("kind") == KIND and manifest.get("goal_completion_claimed") is False,
            "Wrong actual nine-case H bundle identity")
    entries = manifest["files"]
    names = {item["path"] for item in entries}
    require(len(names) == len(entries) and {p.name for p in folder.iterdir()} == names | {"artifact_manifest.json"},
            "Public bundle file allowlist differs")
    for item in entries:
        relative = Path(item["path"])
        require(not relative.is_absolute() and len(relative.parts) == 1 and ".." not in relative.parts,
                "Unsafe public artifact path")
        path = folder/relative
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"]
                and sha256(path) == item["sha256"], "Public artifact changed: " + str(relative))
    for name, digest in manifest["replay_helper_sha256"].items():
        require(sha256(ROOT/name) == digest, "Frozen repository replay helper changed: " + name)
    verify_archive(folder, manifest)
    summary, report, experiment = (read_json(folder/name) for name in
                                  ("cem_analysis.json", "cem_report.json", "cem_experiment.json"))
    bindings = summary["input_sha256"]
    require(sha256(folder/"cem_report.json") == bindings["report.json"]
            and sha256(folder/"cem_experiment.json") == bindings["experiment.json"]
            and report["rollout_sha256"] == bindings["rollout.npz"]
            and report["solver_log_sha256"] == bindings["solver_log.json"]
            and experiment["report_sha256"] == bindings["report.json"]
            and manifest["raw_input_sha256"] == bindings, "Copied terminal/raw SHA bindings differ")
    require(sha256(folder/"protocol.json") == audit.PROTOCOL_SHA256, "Copied H protocol changed")
    expected = validate_summary(summary, report, experiment, read_json(folder/"protocol.json"))
    require(manifest["executed_H_source_sha256"] == audit.SOURCE_PINS
            and manifest["actual_reward_audit"] == summary["actual_H_reward_audit"],
            "Manifest source/reward evidence differs")
    actual = recompute_comparison(read_arrays(folder/"native_comparison_signals.npz"))
    require(actual == expected, "One or more nine-case metrics do not exactly replay")
    diagnosis = read_json(folder/"contact_diagnostics.json")
    require(diagnosis["compact_sha256"] == sha256(folder/"native_contact_context.npz")
            and sha256(folder/"contact_diagnostics.json") == summary["contact_diagnostics_sha256"],
            "Contact evidence SHA differs")
    verify_contact(read_arrays(folder/"native_contact_context.npz"), diagnosis, summary)
    sidecar = read_json(folder/"completed_CEM_H_provenance.json")
    require(sidecar["input_sha256"] == summary["input_sha256"]
            and sidecar["diagnostic_sha256"] == sha256(folder/"contact_diagnostics.json")
            and sidecar["standalone_all_contact_metric_keys_exact"] is True
            and sidecar["full349x15_reward_audit_recomputed_from_public_compact"] is False,
            "H completion/contact provenance differs")
    status = read_json(folder/"status.json")
    require(status["CEM_H"]["failed_checks"] == summary["nominal_screen"]["failed_checks"]
            and status["CEM_H"]["numeric_nominal_episode_pass"] == summary["nominal_screen"]["numeric_nominal_episode_pass"]
            and status["PPO_H"]["new_formal_policy_results_claimed"] is False
            and status["goal_complete"] is False, "Public status misstates actual H results")
    return {"status": "ALL_NINE_CASE_METRICS_AND_ACTUAL_H_CONTACT_DIAGNOSTICS_EXACT",
        "conditions": list(actual), "actual_H_failed_checks": summary["nominal_screen"]["failed_checks"],
        "full349x15_stage_reward_audit_preserved_not_recomputed_from_compact": True,
        "robot_assets_or_weights_required": False, "new_physics_FK_inference_training_file_writes": False,
        "PPO_H_result_claimed": False, "overall_goal_complete_claimed": False}


def _replay_source():
    return '''"""Read-only nine-case metric/contact replay; no assets, weights or physics."""
import hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/"artifact_manifest.json").read_text())
name="scripts/package_cem_ground_contact_v2h_nominal.py"
assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest["replay_helper_sha256"][name]
from package_cem_ground_contact_v2h_nominal import verify_public
if __name__=="__main__":print(json.dumps(verify_public(HERE),indent=2))
'''


def run(args):
    output, root = Path(args.output_dir), Path(args.input_root)
    require(not output.exists(), "Preserve all prior bundles; fresh output directory required")
    protocol = verify_sources()
    analysis_dir = root/"cem_actual_analysis_v1"
    summary = read_json(analysis_dir/"summary.json")
    raw, report, hashes, reward_audit = audit.load_completed(root/"cem_mu0p7")
    experiment = read_json(root/"cem_mu0p7/experiment.json")
    expected = validate_summary(summary, report, experiment, protocol)
    require(summary["input_sha256"] == hashes and summary["actual_H_reward_audit"] == reward_audit,
            "Terminal H raw/reward evidence differs from analysis")
    reference = audit.base.metrics.reference_data(audit.runner.paired.DEFAULT_REFERENCE)
    raw_cases = {}
    source_folders = {"original_grounded_baseline": audit.base.BASELINE_FOLDER,
        **{"prior_actual_V2"+suffix: ROOT/("outputs/ground_contact_v2"+suffix+"_20261023_v1/cem_mu0p7")
           for suffix in "bcdefg"}}
    source_folders["prior_actual_V2a"] = ROOT/"outputs/ground_contact_v2_20261023_v1/cem_mu0p7"
    for case in CASE_ORDER[:-1]:
        folder = source_folders[case]
        binding = summary["baseline_input_sha256"] if case == CASE_ORDER[0] else summary[
            "optional_"+case]["input_sha256"]
        require(all(sha256(folder/name) == digest for name, digest in binding.items()),
                "Prior actual raw source changed: " + case)
        raw_cases[case] = read_arrays(folder/"rollout.npz")
    raw_cases["actual_V2h"] = raw
    compact = comparison_arrays(raw_cases, reference)
    require(recompute_comparison(compact) == expected, "Nine-case raw/compact metric parity differs")
    diagnosis = read_json(analysis_dir/"contact_diagnostics.json")
    require(sha256(analysis_dir/"contact_diagnostics.json") == summary["contact_diagnostics_sha256"]
            and sha256(analysis_dir/"native_contact_context.npz") == diagnosis["compact_sha256"],
            "Analyzed H contact bindings differ")
    verify_contact(read_arrays(analysis_dir/"native_contact_context.npz"), diagnosis, summary)
    require((root/"cem_launch.json").is_file(), "Missing actual H launch provenance")
    ledger = source_ledger()
    output.mkdir(parents=True, exist_ok=False)
    for source, name, digest in (
        (analysis_dir/"summary.json", "cem_analysis.json", None),
        (root/"cem_mu0p7/report.json", "cem_report.json", hashes["report.json"]),
        (root/"cem_mu0p7/experiment.json", "cem_experiment.json", hashes["experiment.json"]),
        (root/"cem_launch.json", "cem_launch.json", None),
        (analysis_dir/"contact_diagnostics.json", "contact_diagnostics.json", summary["contact_diagnostics_sha256"]),
        (analysis_dir/"native_contact_context.npz", "native_contact_context.npz", diagnosis["compact_sha256"]),
        (audit.DEFAULT_PROTOCOL, "protocol.json", audit.PROTOCOL_SHA256)):
        copy_bound(source, output/name, digest)
    np.savez_compressed(output/"native_comparison_signals.npz", **compact)
    archive = source_archive(output, ledger)
    write_json(output/"source_snapshot.manifest.json", {"kind": "CEM_H_A_B_C_D_E_F_G_EXECUTED_AND_OFFLINE_PURE_SOURCE",
        "archive": archive, "executed_controller_source_sha256": audit.SOURCE_PINS,
        "offline_replay_helper_sha256": ledger,
        "files": [{"path": name, "sha256": digest, "bytes": (ROOT/name).stat().st_size}
                  for name, digest in ledger.items()],
        "models_weights_credentials_native_libraries_included": False,
        "compact_metric_replay_requires_no_assets_or_weights": True})
    write_json(output/"completed_CEM_H_provenance.json", {"kind": "actual_terminal_F_CEM_readonly_completion_proof",
        "input_sha256": hashes, "diagnostic_sha256": summary["contact_diagnostics_sha256"],
        "standalone_all_contact_metric_keys_exact": True, "all_nine_case_metrics_exact": True,
        "actual349x15_reward_audit_verified_against_raw_and_solver_log": True,
        "full349x15_reward_audit_recomputed_from_public_compact": False,
        "PPO_H_result_claimed": False, "solver_failures": report["solver_failures"],
        "plant_warning_counts": report["plant_warning_counts"], "overall_goal_complete_claimed": False})
    write_json(output/"status.json", status_from_summary(summary))
    write_json(output/"reproduction.json", {"kind": "actual_H_CEM_nine_case_readonly_compact_replay",
        "command": "python "+str(output/"reproduce.py"),
        "requires": ["Pinned repository Python helpers (SHA-bound and included as pure source).",
            "NumPy/SciPy and import-time dependencies; no robot model, mesh, checkpoint or raw output archive."],
        "source_archive_read_in_memory_never_extracted_by_replay": True,
        "exact_state_contact_metrics_for_all_nine_cases": True,
        "all_actual_H_contact_diagnostic_keys_recomputed": True,
        "full_stage_reward_audit": "349x15 stages verified from full raw+solver log before packaging; preserved, NOT recomputed from compact.",
        "no_return_ranking_or_PPO_performance_claim": True,
        "no_new_physics_FK_inference_training_or_native_compilation": True,
        "overall_goal_complete_claimed": False})
    with (output/"reproduce.py").open("x") as handle:
        handle.write(_replay_source())
    with (output/"metric_table.csv").open("x", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["case", "core_xy_rmse_m", "whole_xy_rmse_m", "whole_xy_peak_m",
            "core_speed_ratio", "core_beta_rmse_deg", "full_anywheel_support_fraction", "full_max_gap_s"])
        for case, value in expected.items():
            core, whole = (value["windows"][name]["tracking"] for name in ("core", "whole_reference_overlap"))
            support = value["any_wheel_ground_support"]["full_contact"]
            writer.writerow([case, core["xy_rmse_m"], whole["xy_rmse_m"], whole["xy_peak_m"],
                core["speed_retention"], core["beta_rmse_deg"], support["any_wheel_support_sample_fraction"],
                support["longest_no_support_observed_s"]])
    manifest = {"kind": KIND, "files": [{"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir()) if path.is_file()],
        "replay_helper_sha256": ledger, "executed_H_source_sha256": audit.SOURCE_PINS,
        "actual_reward_audit": reward_audit, "raw_input_sha256": hashes,
        "protocol_sha256": audit.PROTOCOL_SHA256, "source_archive_sha256": archive["sha256"],
        "models_weights_credentials_native_libraries_included": False, "images_or_video_included": False,
        "PPO_H_result_claimed": False, "goal_completion_claimed": False}
    write_json(output/"artifact_manifest.json", manifest)
    verify_public(output)
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--verify-public", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    require((args.verify_public is not None) != (args.output_dir is not None),
            "Choose --output-dir OR --verify-public")
    print(json.dumps(verify_public(args.verify_public) if args.verify_public else run(args), indent=2))


