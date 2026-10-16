"""Hash-bound model-generated NMPC initialization, not a policy or action replay.

The sole numerical payload is a float32 array of normalized actuator actions.
No state, root trajectory, simulator, pickle, current plant friction or learned
policy is loaded. A consumer must re-score each candidate from its measured
state and apply only its newly optimized first action. Last-action padding is
explicitly unverified: the donor closed-loop result does not certify replay,
counterfactual states, or this padded tail.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


KIND = "model_generated_nominal_action_plan"
JOINT_NAMES = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
               "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint")
TAIL_RULE = "Clamp action index to last donor action; repeated-last tail is unverified, not certified stabilization"
USE_SCOPE = "Additional candidate re-scored from measured state; never prescribe root states or replay directly"
PROVENANCE_SCOPE = "Hashes bind donor files and reported source identity, not independent proof of original execution"
BUDGET_SCOPE = "This donor episode only; antecedent development/tuning and subsequent online CEM are additional"
DECLARATIONS = ("learned_policy_used", "tvlqr_used", "root_state_prescribed_after_reset",
                "external_root_forces_used", "prediction_receives_plant_mu", "pose_height_failure_gate",
                "action_replay", "old_qr_used", "model_generated_nominal_plan_used")
COUNT_FIELDS = ("candidate_evaluations", "scenario_evaluations", "finite_candidates", "failed_candidates",
                "failed_scenarios", "batch_prediction_calls", "failed_batch_prediction_calls",
                "optimization_failed", "fallback_used", "initial_candidate_generation_failed")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_sha(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, "Invalid " + name)
    return value


def _portable(value):
    """Reject absolute host paths even in otherwise JSON-compatible metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str), "Metadata keys must be strings")
            _portable(key); _portable(item)
    elif isinstance(value, list):
        for item in value:
            _portable(item)
    elif isinstance(value, str):
        _require(not value.startswith(("/", "\\", "~")) and not re.match(r"^[A-Za-z]:[\\/]", value)
                 and not any(fragment in value for fragment in ("/home/", "/data/", "/tmp/", "/private/")),
                 "Private/absolute path in plan metadata")


def validate_actions(actions, expected_count=None):
    value = np.asarray(actions)
    _require(value.dtype == np.dtype("<f4") and value.ndim == 2 and value.shape[1] == 8
             and value.shape[0] > 0, "Actions must be a nonempty little-endian float32 (N,8) array")
    _require(expected_count is None or value.shape[0] == expected_count, "Action count mismatch")
    _require(np.isfinite(value).all() and np.all(value >= -1) and np.all(value <= 1),
             "Actions must be finite and normalized to [-1,1]; never silently clip an invalid artifact")
    return value


def actions_sha(actions):
    """Portable content hash: fixed schema, shape and little-endian C-order bytes."""
    actions = validate_actions(actions)
    header = canonical_json({"dtype": "<f4", "shape": list(actions.shape), "order": "C"}).encode()
    return hashlib.sha256(header + b"\n" + actions.tobytes(order="C")).hexdigest()


def generation_budget(source):
    count = source["clock_audit"]["policy_steps"]
    horizon = source["effective_cem_config"]["horizon_steps"]
    totals = source["solver_counts"]
    return {"source_control_decisions": count, "source_plant_control_steps": count,
            "source_plant_physics_substeps": count * 4,
            "source_candidate_evaluations": totals["candidate_evaluations"],
            "source_scenario_evaluations": totals["scenario_evaluations"],
            "source_prediction_horizon_steps": horizon, "physics_substeps_per_control": 4,
            "source_predicted_physics_substeps": totals["scenario_evaluations"] * horizon * 4,
            "source_batch_prediction_calls": totals["batch_prediction_calls"], "scope": BUDGET_SCOPE}


def validate_metadata(metadata):
    """Validate the portable schema, including provenance and budget coherence."""
    _require(isinstance(metadata, dict) and set(metadata) == {
        "schema_version", "kind", "actions_file", "actions_file_sha256", "actions_content_sha256",
        "action_contract", "identity", "source_generation", "offline_generation_budget", "use_scope"},
        "Unexpected model-plan metadata fields")
    canonical_json(metadata); _portable(metadata)
    _require(type(metadata["schema_version"]) is int and metadata["schema_version"] == 1
             and metadata["kind"] == KIND and metadata["actions_file"] == "actions.npz"
             and metadata["use_scope"] == USE_SCOPE, "Unsupported model-plan version/kind/use")
    _require(_sha(metadata["actions_file_sha256"]) and _sha(metadata["actions_content_sha256"]),
             "Missing action file/content SHA-256")
    action = metadata["action_contract"]
    _require(isinstance(action, dict) and set(action) == {"action_joint_names", "action_count", "action_dimension",
        "dtype", "normalized_range", "physics_dt_s", "control_dt_s", "first_action_start_s",
        "last_action_start_s", "applied_interval_end_s", "tail_rule"}, "Invalid action contract fields")
    count = _integer(action["action_count"], "action count", 1)
    _require(action["action_joint_names"] == list(JOINT_NAMES) and action["action_dimension"] == 8
             and action["dtype"] == "float32" and action["normalized_range"] == [-1., 1.]
             and action["physics_dt_s"] == .005 and action["control_dt_s"] == .02
             and action["first_action_start_s"] == 0.
             and action["last_action_start_s"] == (count - 1) * .02
             and action["applied_interval_end_s"] == count * .02 and action["tail_rule"] == TAIL_RULE,
             "Unsupported action order, bounds, clock or tail semantics")
    identity = metadata["identity"]
    _require(isinstance(identity, dict) and set(identity) == {"reference_id", "reference_spec", "motion_sha256",
        "robot_reference_sha256", "model_sha256", "actuator_contract_sha256", "task_profile_sha256",
        "objective_contract", "objective_contract_sha256"}, "Invalid model-plan identity fields")
    _require(isinstance(identity["reference_id"], str) and bool(identity["reference_id"])
             and all(_sha(identity[name]) for name in identity if name.endswith("sha256")), "Invalid plan identity hashes")
    spec = identity["reference_spec"]
    _require(isinstance(spec, dict) and spec.get("reference_id") == identity["reference_id"]
             and spec.get("motion_npz_sha256") == identity["motion_sha256"]
             and spec.get("robot_reference_npz_sha256") == identity["robot_reference_sha256"]
             and spec.get("frames") == count and spec.get("fps") == 50., "Reference specification/clock mismatch")
    objective = identity["objective_contract"]
    _require(isinstance(objective, dict) and json_sha(objective) == identity["objective_contract_sha256"]
             and objective.get("dt_s") == .02 and objective.get("joint_names") == list(JOINT_NAMES)
             and objective.get("absolute_height_target") is False
             and objective.get("reference_vertical_velocity_target") is False
             and objective.get("height_regularization", {}).get("mode") == "vertical_acceleration_only"
             and len(objective.get("weights", {})) == 15, "Require the matching 15-term height-smooth objective")
    source = metadata["source_generation"]
    _require(isinstance(source, dict) and set(source) == {"kind", "engine", "status", "nominal_plant_mu",
        "prediction_mu_scenarios", "effective_cem_config", "action_parameterization", "source_artifact_sha256",
        "reported_source_sha256", "input_sha256", "initial_candidate_source_counts", "initial_plan",
        "declarations", "clock_audit", "solver_counts", "native_prediction_identity", "provenance_scope"},
        "Invalid source-generation fields")
    _require(source["kind"] == "completed_closed_loop_model_generated_NMPC" and source["engine"] == "MuJoCo"
             and source["status"] == "completed" and source["nominal_plant_mu"] == .7
             and source["prediction_mu_scenarios"] == [.7] and source["provenance_scope"] == PROVENANCE_SCOPE,
             "Require one fixed nominal MuJoCo model-generated donor")
    declarations = source["declarations"]
    _require(isinstance(declarations, dict) and set(declarations) == set(DECLARATIONS)
             and all(value is False for value in declarations.values()), "Unsupported learned/replay/root-force source")
    _require(isinstance(source["initial_plan"], str) and bool(source["initial_plan"]), "Missing source initialization disclosure")
    artifacts = source["source_artifact_sha256"]
    _require(isinstance(artifacts, dict) and set(artifacts) == {"report", "trace", "solver"}
             and all(_sha(value) for value in artifacts.values()), "Missing donor report/trace/solver hashes")
    hashes = source["reported_source_sha256"]
    _require(isinstance(hashes, list) and bool(hashes) and hashes == sorted(hashes)
             and all(_sha(value) for value in hashes), "Invalid reported implementation identities")
    inputs = source["input_sha256"]
    _require(isinstance(inputs, dict) and set(inputs) == {"motion", "robot_reference", "model", "actuator_contract",
        "task_profile", "settings", "reference_catalog"} and all(_sha(value) for value in inputs.values()),
        "Missing source input hashes")
    _require(all(inputs[name] == identity[name + "_sha256"] for name in
                 ("motion", "robot_reference", "model", "actuator_contract", "task_profile")), "Source input identity mismatch")
    _require(all(inputs[name] in hashes for name in ("actuator_contract", "task_profile", "settings", "reference_catalog")),
             "Source report does not identify required input bytes")
    clock = source["clock_audit"]
    _require(clock == {"policy_steps": count, "native_samples": 4 * count + 1, "start_s": 0.,
                       "end_s": count * .02, "four_substep_action_hold_verified": True,
                       "held_reference_phase_verified": True}, "Incomplete donor clock audit")
    cfg = source["effective_cem_config"]
    _require(isinstance(cfg, dict) and cfg.get("reference_convention") == "held_pre_action"
             and type(cfg.get("seed")) is int and cfg["seed"] >= 0, "Invalid source CEM identity/seed")
    for name in ("horizon_steps", "control_blocks", "candidates", "iterations", "elite_count"):
        _integer(cfg.get(name), "source " + name, 1)
    _require(cfg["control_blocks"] <= cfg["horizon_steps"] and cfg["elite_count"] <= cfg["candidates"],
             "Invalid source CEM dimensions")
    mode = source["action_parameterization"]
    _require(isinstance(mode, dict) and mode.get("mode") == cfg.get("parameterization")
             and mode.get("learned_baseline") is False, "Source parameterization identity mismatch")
    totals = source["solver_counts"]
    _require(isinstance(totals, dict) and set(totals) == set(COUNT_FIELDS), "Missing source solver counts")
    for name, value in totals.items():
        _integer(value, "source " + name)
    _require(totals["candidate_evaluations"] == totals["finite_candidates"] > 0
             and totals["scenario_evaluations"] == totals["candidate_evaluations"]
             and all(totals[name] == 0 for name in COUNT_FIELDS if name.startswith("failed_") or name in
                     ("optimization_failed", "fallback_used", "initial_candidate_generation_failed")),
             "Require a finite failure-free donor solver episode")
    initial = source["initial_candidate_source_counts"]
    _require(isinstance(initial, dict) and bool(initial)
             and set(initial) <= {"bounded_previous", "shifted_previous_plan", "physics_only_initial_candidate"},
             "Unrecognized donor initialization source")
    for name, value in initial.items():
        _integer(value, "initial source " + name, 1)
    _require(sum(initial.values()) == count,
             "Donor must have a fully identified model-only initialization; no external seed")
    native = source["native_prediction_identity"]
    _require(isinstance(native, dict) and set(native) == {"mujoco_version", "header_version", "abi_version",
        "source_sha256", "mujoco_header_sha256", "mujoco_library_sha256", "native_library_sha256", "model_snapshot_sha256"}
        and native["mujoco_version"] == "3.8.1" and native["header_version"] == 3008001 and native["abi_version"] == 1
        and all(_sha(value) for key, value in native.items() if key.endswith("sha256")), "Invalid donor native identity")
    _require(metadata["offline_generation_budget"] == generation_budget(source), "Incorrect offline generation budget")
    return metadata


def _load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


@dataclass(frozen=True)
class ModelNominalPlan:
    _action_bytes: bytes
    _metadata_json: str
    metadata_sha256: str

    @property
    def generation_budget(self):
        return json.loads(self._metadata_json)["offline_generation_budget"]

    def contract(self):
        """Detached portable metadata; mutations cannot alter the runtime plan."""
        return json.loads(self._metadata_json)

    def report_contract(self):
        return {"enabled": True, "artifact_sha256": self.metadata_sha256,
                "contract": self.contract(), "offline_generation_budget": self.generation_budget}

    def plan(self, start_index, horizon_steps):
        """Copy a normalized Hx8 candidate; saturate indices, not bad actions."""
        _require(isinstance(start_index, (int, np.integer)) and not isinstance(start_index, (bool, np.bool_))
                 and start_index >= 0, "start_index must be a nonnegative integer")
        _require(isinstance(horizon_steps, (int, np.integer)) and not isinstance(horizon_steps, (bool, np.bool_))
                 and horizon_steps > 0, "horizon_steps must be a positive integer")
        actions = np.frombuffer(self._action_bytes, dtype="<f4").reshape(-1, 8)
        # Clamp before arange/add to avoid overflow for arbitrarily large Python indices.
        start = min(int(start_index), len(actions) - 1)
        indices = np.minimum(np.arange(int(horizon_steps), dtype=np.int64), len(actions) - 1 - start) + start
        return actions[indices].astype(np.float64)

    def initial_candidates(self, state, start_index, *, previous_action, previous_vz, horizon_steps):
        """Duck-compatible initializer; intentionally ignores all plant state."""
        return {"model_nominal_plan": self.plan(start_index, horizon_steps)}


def load_model_nominal_plan(path, *, expected_reference_id, expected_motion_sha256, expected_model_sha256,
                            expected_robot_reference_sha256, expected_actuator_contract_sha256,
                            expected_objective_contract, control_dt_s=.02, action_joint_names=JOINT_NAMES):
    """Load a no-pickle action-only artifact, requiring deployment identities."""
    path = Path(path)
    _require(path.is_file(), "Model plan metadata file does not exist")
    metadata = validate_metadata(_load_json(path))
    identity = metadata["identity"]
    expected = {"reference_id": expected_reference_id, "motion_sha256": expected_motion_sha256,
                "robot_reference_sha256": expected_robot_reference_sha256, "model_sha256": expected_model_sha256,
                "actuator_contract_sha256": expected_actuator_contract_sha256}
    _require(all(identity[name] == value for name, value in expected.items()), "Model plan deployment identity mismatch")
    _require(identity["objective_contract"] == json.loads(canonical_json(expected_objective_contract))
             and identity["objective_contract_sha256"] == json_sha(expected_objective_contract),
             "Model plan objective mismatch")
    action = metadata["action_contract"]
    _require(control_dt_s == .02 and tuple(action_joint_names) == tuple(action["action_joint_names"]),
             "Model plan deployment clock/action order mismatch")
    actions_path = path.parent / metadata["actions_file"]
    _require(actions_path.is_file() and file_sha(actions_path) == metadata["actions_file_sha256"],
             "Model plan actions file SHA-256 mismatch")
    with np.load(actions_path, allow_pickle=False) as archive:
        _require(archive.files == ["actions"], "Model plan NPZ must contain actions only")
        actions = validate_actions(archive["actions"], action["action_count"])
    _require(actions_sha(actions) == metadata["actions_content_sha256"], "Model plan actions content SHA-256 mismatch")
    return ModelNominalPlan(actions.tobytes(order="C"), canonical_json(metadata), file_sha(path))
