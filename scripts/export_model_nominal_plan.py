"""Offline export of the audited, completed nominal MuJoCo H30/B5 donor.

Never executes physics, inference, optimization, subprocesses or downloads.
The donor's APPLIED float32-roundtrip actions (not raw CEM parameters) are the
only exported numerical data. Historical reported source identities are
allowlisted, not silently replaced by today's modified controller source.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from model_nominal_plan import (
    COUNT_FIELDS, DECLARATIONS, JOINT_NAMES, KIND, PROVENANCE_SCOPE, TAIL_RULE, USE_SCOPE,
    actions_sha, file_sha, generation_budget, json_sha, load_model_nominal_plan, validate_actions,
    validate_metadata,
)


AUDITED_REPORTED_SOURCES = {
    "scripts/nmpc_shooting.py": "8d2ac598d9e0e2765f9566954ba3fd0f443d939d2a5c6566125cca18668b35bd",
    "scripts/run_nmpc_tracking.py": "d121350b553c8a5622e6759a9e69b33e946e9193e01f442ebe3d333541743293",
    "scripts/nmpc_tracking_objective.py": "ff22eebb2b22b9af7dba114c64fcd10f6b0a90822559af8f32cbe6884015b860",
    "native/nmpc_batch.cpp": "deb3afe3367e11c28295e00e4a081e84968818070dcfb7044f06466761875ed2",
}
DONOR_CONFIG = {
    "horizon_steps": 30, "control_blocks": 5, "candidates": 64, "iterations": 3, "elite_count": 8,
    "initial_std": [.04] * 6 + [.35, .35], "min_std": [.005] * 6 + [.03, .03],
    "smoothing": 0., "seed": 42, "scenario_reduction": "worst", "fallback_action": "previous",
    "reference_convention": "held_pre_action", "parameterization": "warm_start_residual_blocks",
}
INITIAL_PLAN = "Bounded bias-force joint feedforward from kinematic reference; not a feasible trajectory certificate."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_source_episode(directory, *, reference_spec, input_sha256, objective_contract):
    """Read/audit an existing episode; return actions and sanitized provenance.

    The canonical pure evidence gate checks native 5 ms clock/contact/action
    holding, complete horizon, original raw hashes and explicit no-root-force
    declarations. No simulator library is imported or called by that gate.
    """
    from nmpc_validation_contract import episode_evidence

    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    require(reference_spec.get("reference_id") == "drift_drl_traj4" and reference_spec.get("frames") == 349,
            "Exporter is scoped to the completed primary 349-action model donor")
    require(report.get("effective_cem_config") == DONOR_CONFIG, "Donor must use audited H30/B5 residual CEM settings")
    evidence = episode_evidence(directory, "mujoco", reference_spec,
        {"effective_cem_config": DONOR_CONFIG, "prediction_mu_scenarios": [.7]}, .7)
    require(report == evidence["report"], "Donor report changed during audit")
    require(report.get("objective_contract") == objective_contract, "Donor objective differs from the supplied task")
    require(report.get("model_sha256") == input_sha256["model"]
            and report.get("motion_sha256") == input_sha256["motion"] == reference_spec["motion_npz_sha256"]
            and input_sha256["robot_reference"] == reference_spec["robot_reference_npz_sha256"]
            and report.get("reference_catalog_sha256") == input_sha256["reference_catalog"], "Donor/reference/model bytes mismatch")
    sources = report["source_sha256"]
    require(all(input_sha256[name] in sources.values() for name in
                ("actuator_contract", "task_profile", "settings", "reference_catalog")),
            "Donor did not report the supplied actuator/settings/task-profile bytes")
    for suffix, digest in AUDITED_REPORTED_SOURCES.items():
        matches = [value for name, value in sources.items() if name == suffix or name.endswith("/" + suffix)]
        require(matches == [digest], "Donor source is not the audited model-only implementation: " + suffix)
    require(report.get("initial_plan") == INITIAL_PLAN, "Unsupported donor initialization")
    previous_plan = report.get("model_generated_nominal_plan", {"enabled": False})
    require(previous_plan == {"enabled": False}, "Do not recursively hide a prior nominal-plan generation budget")
    require(report.get("prediction_backend") == "native", "Donor native execution identity is required")
    backend = report["native_prediction_backend"]
    require(backend.get("learned_initialization") is False and backend.get("plant_mu_used_for_prediction") is False
            and backend.get("terminal_cost") is None, "Unsupported donor prediction initialization/objective")
    workers = backend.get("workers", {})
    require(set(workers) == {"0.7"}, "Unexpected donor prediction worker/scenario")
    worker_fields = ("mujoco_version", "header_version", "abi_version", "source_sha256", "mujoco_header_sha256",
                     "mujoco_library_sha256", "native_library_sha256", "model_snapshot_sha256")
    native = {name: workers["0.7"].get(name) for name in worker_fields}
    with np.load(directory / "rollout.npz", allow_pickle=False) as archive:
        actions = np.asarray(archive["actions"])
        require(tuple(archive["action_joint_names"].tolist()) == JOINT_NAMES, "Donor applied action order mismatch")
        # Explicit conversion is lossless: source records applied float32 controls as float64.
        require(actions.dtype.kind == "f" and np.array_equal(actions, actions.astype("<f4").astype(actions.dtype)),
                "Donor actions are not exact applied float32 controls")
        actions = validate_actions(actions.astype("<f4"), 349)
    solver = json.loads((directory / "solver_log.json").read_text())
    totals = {name: 0 for name in COUNT_FIELDS}
    initial_sources = Counter()
    for k, decision in enumerate(solver):
        info = decision["solver"]
        require(np.isclose(decision.get("time_s", np.nan), (k + 1) * .02, rtol=0, atol=1e-9),
                "Donor decision end-time clock mismatch")
        require(info.get("optimizer") == "bounded_cem_shooting" and info.get("horizon_steps") == 30
                and info.get("control_blocks") == 5 and info.get("prediction_mu_scenarios") == [.7]
                and info.get("reference_convention") == "held_pre_action" and info.get("shared_full_control_sequence") is True
                and info.get("action_parameterization") == "warm_start_residual_blocks"
                and info.get("residual_baseline_fixed_per_solve") is True and info.get("learned_policy_fallback") is False,
                "Missing original model-only solver provenance")
        for name in COUNT_FIELDS:
            value = info.get(name)
            if name in ("optimization_failed", "fallback_used", "initial_candidate_generation_failed"):
                require(type(value) is bool and value is False, "Donor solver failure: " + name)
                value = int(value)
            require(type(value) is int and value >= 0, "Missing/invalid donor solver count: " + name)
            totals[name] += value
        expected = 194 if k == 0 else 195
        require(info["candidate_evaluations"] == info["finite_candidates"] == info["scenario_evaluations"] == expected
                and info["batch_prediction_calls"] == (5 if k == 0 else 6), "Donor search-budget accounting mismatch")
        initial_sources[info.get("initial_candidate_source")] += 1
    require(totals["scenario_evaluations"] == 68054, "Unexpected donor offline prediction budget")
    source = {
        "kind": "completed_closed_loop_model_generated_NMPC", "engine": "MuJoCo", "status": "completed",
        "nominal_plant_mu": .7, "prediction_mu_scenarios": [.7], "effective_cem_config": copy.deepcopy(DONOR_CONFIG),
        "action_parameterization": copy.deepcopy(report["action_parameterization"]),
        "source_artifact_sha256": evidence["hashes"], "reported_source_sha256": sorted(sources.values()),
        "input_sha256": dict(input_sha256), "initial_candidate_source_counts": dict(sorted(initial_sources.items())),
        "initial_plan": INITIAL_PLAN, "declarations": {name: False for name in DECLARATIONS},
        "clock_audit": {"policy_steps": 349, "native_samples": 1397, "start_s": 0., "end_s": 6.98,
                        "four_substep_action_hold_verified": True, "held_reference_phase_verified": True},
        "solver_counts": totals, "native_prediction_identity": native, "provenance_scope": PROVENANCE_SCOPE,
    }
    # Bind all source files again after reading to catch accidental concurrent mutation.
    require(evidence["hashes"] == {name: file_sha(directory / filename) for name, filename in
            (("report", "report.json"), ("trace", "rollout.npz"), ("solver", "solver_log.json"))},
            "Donor artifacts changed during export audit")
    return actions, source


def write_artifact(output_dir, actions, *, identity, source_generation):
    """Create a fresh action-only NPZ plus portable JSON, never overwrite."""
    actions = validate_actions(actions)
    count = len(actions)
    metadata = {
        "schema_version": 1, "kind": KIND, "actions_file": "actions.npz", "actions_file_sha256": "0" * 64,
        "actions_content_sha256": actions_sha(actions),
        "action_contract": {"action_joint_names": list(JOINT_NAMES), "action_count": count, "action_dimension": 8,
                            "dtype": "float32", "normalized_range": [-1., 1.], "physics_dt_s": .005,
                            "control_dt_s": .02, "first_action_start_s": 0., "last_action_start_s": (count - 1) * .02,
                            "applied_interval_end_s": count * .02, "tail_rule": TAIL_RULE},
        "identity": copy.deepcopy(identity), "source_generation": copy.deepcopy(source_generation),
        "offline_generation_budget": generation_budget(source_generation), "use_scope": USE_SCOPE,
    }
    validate_metadata(metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "actions.npz", actions=actions)
    metadata["actions_file_sha256"] = file_sha(output_dir / "actions.npz")
    validate_metadata(metadata)
    metadata_path = output_dir / "plan.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return metadata_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--task-profile", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--reference-catalog", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    # Export-only imports deliberately stay out of the deployed runtime module.
    from controller_references import DEFAULT_CATALOG, get_reference
    from nmpc_tracking_objective import HeightSmoothTrackingObjective

    catalog = args.reference_catalog or DEFAULT_CATALOG
    spec = get_reference("drift_drl_traj4", catalog_path=catalog)
    paths = {"motion": args.motion_file, "robot_reference": args.reference_file, "model": args.model,
             "actuator_contract": args.contract, "task_profile": args.task_profile, "settings": args.settings,
             "reference_catalog": catalog}
    hashes = {name: file_sha(path) for name, path in paths.items()}
    objective = HeightSmoothTrackingObjective(json.loads(args.task_profile.read_text())).contract()
    actions, source = audit_source_episode(args.source_run, reference_spec=spec, input_sha256=hashes,
                                            objective_contract=objective)
    identity = {"reference_id": spec["reference_id"], "reference_spec": spec,
                **{name + "_sha256": hashes[name] for name in
                   ("motion", "robot_reference", "model", "actuator_contract", "task_profile")},
                "objective_contract": objective, "objective_contract_sha256": json_sha(objective)}
    metadata_path = write_artifact(args.output_dir, actions, identity=identity, source_generation=source)
    plan = load_model_nominal_plan(metadata_path, expected_reference_id=spec["reference_id"],
        expected_motion_sha256=hashes["motion"], expected_model_sha256=hashes["model"],
        expected_robot_reference_sha256=hashes["robot_reference"], expected_actuator_contract_sha256=hashes["actuator_contract"],
        expected_objective_contract=objective)
    print(json.dumps({"metadata_sha256": plan.metadata_sha256, "actions_sha256": file_sha(metadata_path.parent / "actions.npz"),
                      "offline_generation_budget": plan.generation_budget}, sort_keys=True))


if __name__ == "__main__":
    main()
