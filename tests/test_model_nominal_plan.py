"""Synthetic plan/donor fixtures only: no licensed asset, physics or network."""
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import model_nominal_plan as plan_module
import export_model_nominal_plan as exporter
from controller_references import get_reference
from nmpc_tracking_objective import BODY_NAMES, HeightSmoothTrackingObjective


def components(count=7):
    h = {name: str(k) * 64 for k, name in enumerate(
        ("motion", "robot_reference", "model", "actuator_contract", "task_profile", "settings", "reference_catalog"), 1)}
    objective = HeightSmoothTrackingObjective(json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())).contract()
    spec = {"reference_id": "synthetic_asset_free", "frames": count, "fps": 50.,
            "motion_npz_sha256": h["motion"], "robot_reference_npz_sha256": h["robot_reference"]}
    identity = {"reference_id": spec["reference_id"], "reference_spec": spec,
                **{name + "_sha256": h[name] for name in
                   ("motion", "robot_reference", "model", "actuator_contract", "task_profile")},
                "objective_contract": objective, "objective_contract_sha256": plan_module.json_sha(objective)}
    source = {"kind": "completed_closed_loop_model_generated_NMPC", "engine": "MuJoCo", "status": "completed",
        "nominal_plant_mu": .7, "prediction_mu_scenarios": [.7], "effective_cem_config": copy.deepcopy(exporter.DONOR_CONFIG),
        "action_parameterization": {"mode": "warm_start_residual_blocks", "learned_baseline": False},
        "source_artifact_sha256": {name: "a" * 64 for name in ("report", "trace", "solver")},
        "reported_source_sha256": sorted(h.values()), "input_sha256": h,
        "initial_candidate_source_counts": {"bounded_previous": count}, "initial_plan": exporter.INITIAL_PLAN,
        "declarations": {name: False for name in plan_module.DECLARATIONS},
        "clock_audit": {"policy_steps": count, "native_samples": 4 * count + 1, "start_s": 0., "end_s": count * .02,
                        "four_substep_action_hold_verified": True, "held_reference_phase_verified": True},
        "solver_counts": {name: 0 for name in plan_module.COUNT_FIELDS},
        "native_prediction_identity": {"mujoco_version": "3.8.1", "header_version": 3008001, "abi_version": 1,
            **{name: "b" * 64 for name in ("source_sha256", "mujoco_header_sha256", "mujoco_library_sha256",
                                          "native_library_sha256", "model_snapshot_sha256")}},
        "provenance_scope": plan_module.PROVENANCE_SCOPE}
    source["solver_counts"].update(candidate_evaluations=count * 10, scenario_evaluations=count * 10,
                                    finite_candidates=count * 10, batch_prediction_calls=count)
    actions = np.linspace(-1, 1, count * 8, dtype=np.float32).reshape(count, 8)
    return actions, identity, source


def artifact(tmp_path):
    actions, identity, source = components()
    path = exporter.write_artifact(tmp_path / "artifact", actions, identity=identity, source_generation=source)
    return path, actions, identity, source


def expected(identity):
    return {"expected_" + key: identity[key] for key in (
        "reference_id", "motion_sha256", "model_sha256", "robot_reference_sha256", "actuator_contract_sha256", "objective_contract")}


def load(path, identity, **overrides):
    return plan_module.load_model_nominal_plan(path, **(expected(identity) | overrides))


def edit_metadata(path, change):
    metadata = json.loads(path.read_text()); change(metadata)
    path.write_text(json.dumps(metadata))


def test_roundtrip_index_slice_last_padding_and_detached_metadata(tmp_path):
    path, actions, identity, source = artifact(tmp_path)
    plan = load(path, identity)
    np.testing.assert_array_equal(plan.plan(2, 3), actions[2:5].astype(float))
    np.testing.assert_array_equal(plan.plan(5, 5), actions[[5, 6, 6, 6, 6]])
    np.testing.assert_array_equal(plan.plan(10 ** 100, 2), np.repeat(actions[-1:], 2, axis=0))
    assert plan.plan(0, 1).dtype == np.float64
    changed = plan.plan(0, 7); changed[:] = 0
    np.testing.assert_array_equal(plan.plan(0, 7), actions)
    metadata = plan.contract(); metadata["identity"]["model_sha256"] = "f" * 64
    assert plan.contract()["identity"]["model_sha256"] == identity["model_sha256"]
    assert plan.report_contract()["artifact_sha256"] == plan_module.file_sha(path)
    assert plan.report_contract()["offline_generation_budget"] == plan.generation_budget
    assert "unverified" in plan.contract()["action_contract"]["tail_rule"]
    assert str(tmp_path) not in json.dumps(plan.report_contract())
    np.testing.assert_array_equal(plan.initial_candidates(object(), 1, previous_action=object(), previous_vz=object(),
                                                         horizon_steps=2)["model_nominal_plan"], actions[1:3])


@pytest.mark.parametrize("start,horizon", [(-1, 2), (True, 2), (1.5, 2), (0, 0), (0, -1), (0, True), (0, 2.5)])
def test_invalid_plan_queries_rejected(tmp_path, start, horizon):
    path, _, identity, _ = artifact(tmp_path)
    with pytest.raises(ValueError): load(path, identity).plan(start, horizon)


@pytest.mark.parametrize("argument,value", [("expected_reference_id", "other"), ("expected_motion_sha256", "c" * 64),
    ("expected_robot_reference_sha256", "c" * 64), ("expected_model_sha256", "c" * 64),
    ("expected_actuator_contract_sha256", "c" * 64), ("control_dt_s", .01),
    ("action_joint_names", plan_module.JOINT_NAMES[::-1])])
def test_wrong_deployment_identity_rejected(tmp_path, argument, value):
    path, _, identity, _ = artifact(tmp_path)
    with pytest.raises(ValueError): load(path, identity, **{argument: value})


def test_changed_objective_rejected_even_with_same_weight_count(tmp_path):
    path, _, identity, _ = artifact(tmp_path)
    objective = copy.deepcopy(identity["objective_contract"]); objective["weights"]["drift_speed"] += .01
    with pytest.raises(ValueError): load(path, identity, expected_objective_contract=objective)


@pytest.mark.parametrize("problem", ["version", "extra", "private", "dtype", "clock", "tail", "order", "shape", "budget",
    "learned", "replay", "force", "plant", "scenario", "source_hash", "initial", "native", "height", "objective_digest"])
def test_corrupt_or_unsupported_metadata_rejected(tmp_path, problem):
    path, _, identity, _ = artifact(tmp_path)
    def change(m):
        if problem == "version": m["schema_version"] = 2
        elif problem == "extra": m["root_states"] = []
        elif problem == "private": m["source_generation"]["initial_plan"] = "/home/private/file"
        elif problem == "dtype": m["action_contract"]["dtype"] = "float64"
        elif problem == "clock": m["action_contract"]["control_dt_s"] = .04
        elif problem == "tail": m["action_contract"]["tail_rule"] = "certified replay"
        elif problem == "order": m["action_contract"]["action_joint_names"].reverse()
        elif problem == "shape": m["action_contract"]["action_count"] += 1
        elif problem == "budget": m["offline_generation_budget"]["source_predicted_physics_substeps"] += 1
        elif problem in ("learned", "replay", "force"):
            key = {"learned": "learned_policy_used", "replay": "action_replay", "force": "external_root_forces_used"}[problem]
            m["source_generation"]["declarations"][key] = True
        elif problem == "plant": m["source_generation"]["nominal_plant_mu"] = .6
        elif problem == "scenario": m["source_generation"]["prediction_mu_scenarios"] = [.2, .7]
        elif problem == "source_hash": m["source_generation"]["source_artifact_sha256"]["report"] = "x"
        elif problem == "initial": m["source_generation"]["initial_candidate_source_counts"] = {"explicit_physics_only_seed": 7}
        elif problem == "native": m["source_generation"]["native_prediction_identity"]["mujoco_version"] = "3.9.0"
        elif problem == "height": m["identity"]["objective_contract"]["absolute_height_target"] = True
        elif problem == "objective_digest": m["identity"]["objective_contract_sha256"] = "f" * 64
    edit_metadata(path, change)
    with pytest.raises(ValueError): load(path, identity)


@pytest.mark.parametrize("problem", ["filehash", "contenthash", "float64", "bigendian", "shape", "nan", "bound", "extra", "pickle"])
def test_invalid_npz_never_silently_clipped_or_unpickled(tmp_path, problem):
    path, actions, identity, _ = artifact(tmp_path)
    payload = {"actions": actions.copy()}
    if problem == "filehash":
        with (path.parent / "actions.npz").open("ab") as stream: stream.write(b"changed")
    elif problem == "contenthash": payload["actions"][0, 0] = 0.
    elif problem == "float64": payload["actions"] = actions.astype(np.float64)
    elif problem == "bigendian": payload["actions"] = actions.astype(">f4")
    elif problem == "shape": payload["actions"] = actions[:, :7]
    elif problem == "nan": payload["actions"][0, 0] = np.nan
    elif problem == "bound": payload["actions"][0, 0] = 1.01
    elif problem == "extra": payload["root_states"] = np.zeros((7, 3))
    elif problem == "pickle": payload["actions"] = np.array([object()], dtype=object)
    if problem != "filehash":
        np.savez_compressed(path.parent / "actions.npz", **payload)
        edit_metadata(path, lambda m: m.update(actions_file_sha256=plan_module.file_sha(path.parent / "actions.npz")))
    with pytest.raises(ValueError): load(path, identity)


def test_duplicate_json_nonfinite_and_actions_path_escape_rejected(tmp_path):
    path, _, identity, _ = artifact(tmp_path)
    original = path.read_text()
    path.write_text('{"schema_version":1,' + original.lstrip()[1:])
    with pytest.raises(ValueError, match="Duplicate"): load(path, identity)
    path.write_text(original.replace('"schema_version": 1', '"schema_version": NaN'))
    with pytest.raises(ValueError): load(path, identity)
    path.write_text(original)
    edit_metadata(path, lambda m: m.update(actions_file="../actions.npz"))
    with pytest.raises(ValueError): load(path, identity)


def test_fresh_output_only_and_portable_content_hash(tmp_path):
    path, actions, identity, source = artifact(tmp_path)
    other = exporter.write_artifact(tmp_path / "moved", actions.copy(order="F"), identity=identity, source_generation=source)
    assert load(path, identity).contract()["actions_content_sha256"] == load(other, identity).contract()["actions_content_sha256"]
    with pytest.raises(FileExistsError): exporter.write_artifact(path.parent, actions, identity=identity, source_generation=source)


def test_runtime_module_works_without_simulator_export_or_analysis_modules():
    code = """
import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'mujoco','torch','isaaclab','nmpc_validation_contract','analyze_controller_pair','export_model_nominal_plan'}:
   raise RuntimeError('Forbidden runtime import '+fullname)
sys.meta_path.insert(0,Block())
import model_nominal_plan
assert model_nominal_plan.JOINT_NAMES[-1]=='wheel_R_Joint'
"""
    process = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "scripts", capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


def donor_fixture(tmp_path):
    """Full synthetic349/1397 evidence, with no simulator/model asset."""
    from test_nmpc_validation_contract import fixture, write_episode
    args = fixture(tmp_path)
    spec = args["reference_spec"]
    directory = Path(args["nominal"]["mujoco"]["_directory"])
    report = json.loads((directory / "report.json").read_text())
    with np.load(directory / "rollout.npz", allow_pickle=False) as data:
        raw = {name: data[name] for name in data.files}
    report.update(effective_cem_config=copy.deepcopy(exporter.DONOR_CONFIG), initial_plan=exporter.INITIAL_PLAN,
                  prediction_backend="native", reference_catalog_sha256=args["bindings"]["reference_catalog"],
                  old_qr_used=False)
    report["action_parameterization"]["learned_baseline"] = False
    for suffix, digest in exporter.AUDITED_REPORTED_SOURCES.items(): report["source_sha256"]["/private/" + suffix] = digest
    report["source_sha256"]["/private/catalog"] = args["bindings"]["reference_catalog"]
    native = components()[2]["native_prediction_identity"]
    report["native_prediction_backend"] = {"learned_initialization": False, "plant_mu_used_for_prediction": False,
                                           "terminal_cost": None, "workers": {"0.7": native}}
    raw["action_joint_names"] = np.array(plan_module.JOINT_NAMES)
    log = []
    for k in range(349):
        info = {name: 0 for name in plan_module.COUNT_FIELDS}
        info.update(optimizer="bounded_cem_shooting", horizon_steps=30, control_blocks=5,
            prediction_mu_scenarios=[.7], reference_convention="held_pre_action", shared_full_control_sequence=True,
            action_parameterization="warm_start_residual_blocks", residual_baseline_fixed_per_solve=True,
            learned_policy_fallback=False, candidate_evaluations=194 if k == 0 else 195,
            finite_candidates=194 if k == 0 else 195, scenario_evaluations=194 if k == 0 else 195,
            batch_prediction_calls=5 if k == 0 else 6, optimization_failed=False, fallback_used=False,
            initial_candidate_generation_failed=False, initial_candidate_source="bounded_previous")
        log.append({"frame": k, "reference_frame_for_reward": k, "time_s": (k + 1) * .02,
                    "action": [0.] * 8, "compute_s": .4, "solver": info})
    write_episode(directory, report, raw, log)
    binding = args["bindings"]
    inputs = {"motion": binding["motion"], "robot_reference": binding["reference"], "model": binding["model"],
              "actuator_contract": binding["contract"], **{name: binding[name] for name in ("task_profile", "settings", "reference_catalog")}}
    return directory, spec, inputs, report["objective_contract"]


def test_full_synthetic_donor_audit_and_exact_budget(tmp_path):
    directory, spec, inputs, objective = donor_fixture(tmp_path)
    actions, source = exporter.audit_source_episode(directory, reference_spec=spec, input_sha256=inputs, objective_contract=objective)
    assert actions.shape == (349, 8) and actions.dtype == np.float32
    budget = plan_module.generation_budget(source)
    assert budget["source_scenario_evaluations"] == 68054 and budget["source_predicted_physics_substeps"] == 8166480
    assert budget["source_plant_control_steps"] == 349 and budget["source_plant_physics_substeps"] == 1396
    assert budget["source_batch_prediction_calls"] == 2093
    assert "/private/" not in json.dumps(source)


@pytest.mark.parametrize("problem", ["source", "old_seed", "plan_used", "partial", "clock", "raw_action", "external_force",
    "learned", "model", "actuator", "objective", "counts", "phase", "order"])
def test_donor_audit_rejects_wrong_source_clock_identity_and_budget(tmp_path, problem):
    from test_nmpc_validation_contract import write_episode
    directory, spec, inputs, objective = donor_fixture(tmp_path)
    report = json.loads((directory / "report.json").read_text())
    log = json.loads((directory / "solver_log.json").read_text())
    with np.load(directory / "rollout.npz", allow_pickle=False) as data: raw = {name: data[name] for name in data.files}
    if problem == "source": report["source_sha256"]["/private/scripts/nmpc_shooting.py"] = "c" * 64
    elif problem == "old_seed": log[0]["solver"]["initial_candidate_source"] = "explicit_physics_only_seed"
    elif problem == "plan_used": report["model_generated_nominal_plan"] = {"enabled": True}
    elif problem == "partial": report["completed_policy_steps"] -= 1
    elif problem == "clock": log[0]["time_s"] += .02
    elif problem == "raw_action": raw["actions"][0, 0] = .123456789
    elif problem == "external_force": report["external_root_forces_used"] = True
    elif problem == "learned": report["learned_policy_used"] = True
    elif problem == "model": inputs["model"] = "c" * 64
    elif problem == "actuator": inputs["actuator_contract"] = "c" * 64
    elif problem == "objective": objective = copy.deepcopy(objective); objective["weights"]["drift_speed"] = 3.
    elif problem == "counts": log[0]["solver"]["scenario_evaluations"] += 1
    elif problem == "phase": raw["reference_frame_for_action"][1] = 1
    elif problem == "order": raw["action_joint_names"] = raw["action_joint_names"][::-1]
    write_episode(directory, report, raw, log)
    with pytest.raises(ValueError):
        actions, source = exporter.audit_source_episode(directory, reference_spec=spec, input_sha256=inputs, objective_contract=objective)
        # Source-selection validation also runs before any writer creates output.
        identity = {"reference_id": spec["reference_id"], "reference_spec": spec,
            **{name + "_sha256": inputs[name] for name in ("motion", "robot_reference", "model", "actuator_contract", "task_profile")},
            "objective_contract": objective, "objective_contract_sha256": plan_module.json_sha(objective)}
        exporter.write_artifact(tmp_path / "must_not_export", actions, identity=identity, source_generation=source)
