"""Bounded observed-action INPUT probe for the one failed Isaac I6 archive.

Exactly four recorded pre-action states and two predeclared horizons are used.
Each horizon gets one fresh isolated worker and one batch, with no retry, CEM,
new closed loop, actor, state conversion or prescribed reference/root state.
This cannot reconstruct the original, unrecorded failing candidate population.
Imports are CPU/NumPy-only until all file/scope guards have passed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from probe_nmpc_isaac_prediction import archived_source_hash, validate_trace


ROOT = Path(__file__).resolve().parents[1]
FRAMES = (136, 137, 138, 139)
HORIZONS = (1, 30)
EXPECTED_ARCHIVE = {
    "report.json": "88c8cde0734687209ffb0487e055a16acd305a1c824e1de4a3e8dd43e3a4a01a",
    "rollout.npz": "5f7a251fe47e694a03e477a327298ea7bae02b520d0c7e531e8e59e869f764c4",
    "solver_log.json": "89dbb8b6408124d43c8a3cf426cd1b17d3215931a49bb48fe7f531b776eae120",
}
MODEL_SHA256 = "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"
SNAPSHOT_SHA256 = "28ebe2555a1f7511f882c5e62e914ece71bee6855a7ab4425e6fd614bfbe4b9e"
NATIVE_IDENTITY_FIELDS = ("source_sha256", "mujoco_version", "header_version", "abi_version",
    "mujoco_header_sha256", "mujoco_library_sha256", "model_snapshot_sha256")
SOURCE_HELPERS = ("scripts/eval_tracking_mujoco.py", "scripts/eval_tracking.py",
    "training/tron1_domain_randomization.py", "training/tron1_contact_slip.py",
    "training/tron1_physics_clock.py", "training/tron1_drift.py")
RUNTIME_FILES = ("scripts/nmpc_native_batch.py", "scripts/nmpc_tracking_objective.py", "native/nmpc_batch.cpp")


class ScopeError(ValueError):
    """Identity/scope failures abort instead of authorizing another batch."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_unchanged(paths, hashes):
    require({name: sha(path) for name, path in paths.items()} == hashes,
            "Probe inputs/source/library bytes changed; no further prediction authorized")


def load_archive(directory):
    """Pure read-only archive and complete physical-clock validation."""
    directory = Path(directory)
    for name, expected in EXPECTED_ARCHIVE.items():
        require(sha(directory/name) == expected, "Probe is bound to the exact I6 archive: " + name)
    report = json.loads((directory/"report.json").read_text())
    solver = json.loads((directory/"solver_log.json").read_text())
    with np.load(directory/"rollout.npz", allow_pickle=False) as archive:
        trace = {name: archive[name].copy() for name in archive.files}
    require(all(value.dtype.kind not in "biufc" or np.isfinite(value).all() for value in trace.values()),
            "Nonfinite numeric native trace field")
    require(report.get("status") == "completed" and report.get("engine") == "Isaac/PhysX"
            and report.get("reference_id") == "drift_drl_traj4" and report.get("plant_mu") == .7
            and report.get("prediction_mu_scenarios") == [.7] and report.get("prediction_backend") == "native"
            and report.get("prediction_receives_plant_mu") is False and report.get("learned_policy_used") is False
            and report.get("tvlqr_used") is False and report.get("action_replay") is False
            and report.get("root_state_prescribed_after_reset") is False
            and report.get("external_root_forces_used") is False and report.get("pose_height_failure_gate") is False,
            "Unsupported I6 nominal episode declarations")
    cfg = report.get("effective_cem_config", {})
    require(all(cfg.get(name) == value for name, value in (
        ("horizon_steps", 30), ("control_blocks", 5), ("iterations", 6), ("candidates", 64),
        ("elite_count", 8), ("seed", 42), ("parameterization", "warm_start_residual_blocks"))),
        "Probe requires the recorded H30/B5/I6 controller identity")
    require(report.get("rollout_sha256") == EXPECTED_ARCHIVE["rollout.npz"]
            and report.get("solver_log_sha256") == EXPECTED_ARCHIVE["solver_log.json"],
            "Report-owned artifact hashes disagree")
    require(sha(ROOT/"scripts/nmpc_tracking_objective.py") == archived_source_hash(report, "scripts/nmpc_tracking_objective.py"),
            "Pure trace validator's named-body/action constants differ from the archive")
    validate_trace(trace, solver)
    packed = np.asarray(trace["packed_controller_states"], dtype=float)
    require(np.allclose(np.linalg.norm(packed[:, 3:7], axis=1), 1., atol=1e-5, rtol=0),
            "Recorded packed root quaternions are not unit wxyz")
    require(solver[137]["solver"].get("failed_batch_prediction_calls") == 1
            and solver[137]["solver"].get("failed_candidates") == 64
            and all(row["solver"].get("optimization_failed") is True
                    and row["solver"].get("fallback_used") is True for row in solver[138:]),
            "Recorded partial-batch then persistent-fallback onset differs")
    return report, trace, solver


def observed_inputs(trace, solver, horizon):
    """Preserve recorded 111D rows/history and exact future APPLIED actions."""
    require(type(horizon) is int and horizon in HORIZONS, "Only predeclared H1/H30 are authorized")
    packed = np.asarray(trace["packed_controller_states"], dtype=float)[list(FRAMES)].copy()
    actions = np.stack([trace["actions"][k:k+horizon] for k in FRAMES]).copy()
    require(packed.shape == (4, 111) and actions.shape == (4, horizon, 8)
            and np.isfinite(packed).all() and np.isfinite(actions).all()
            and np.max(np.abs(actions)) <= 1.
            and np.array_equal(actions, actions.astype(np.float32).astype(float)),
            "Malformed recorded state/applied-action batch")
    require(np.allclose(np.linalg.norm(packed[:, 3:7], axis=1), 1., atol=1e-5, rtol=0)
            and np.array_equal(packed[:, -1], np.zeros(4)), "Unexpected quaternion/reset inputs")
    previous = np.stack([trace["actions"][k-1] for k in FRAMES]).copy()
    vz = np.array([solver[k-1]["actual_isaac_objective"]["base_com_vz"] for k in FRAMES], dtype=float)
    require(previous.shape == (4, 8) and np.isfinite(previous).all() and np.max(np.abs(previous)) <= 1.
            and vz.shape == (4,) and np.isfinite(vz).all(), "Invalid recorded objective memory")
    require(np.array_equal(previous, trace["held_action"][4*np.asarray(FRAMES)])
            and np.array_equal(vz, np.asarray(trace["base_com_velocity_w"])[4*np.asarray(FRAMES), 2]),
            "Recorded previous-action/vz memory differs from physical pre-action boundary")
    return {"qpos": packed[:, :15].copy(), "qvel": packed[:, 15:29].copy(),
            "history": packed[:, 29:110].reshape(4, 3, 9, 3).copy(), "actions": actions,
            "previous_actions": previous, "previous_base_com_vz": vz}


@contextmanager
def evaluator(source_root):
    """Use the same real mesh-free loader/configurer, without Isaac or a plant."""
    source_root = Path(source_root).resolve()
    prefixes = ("training", "eval_tracking", "_failed_input_evaluator")
    matches = lambda name: any(name == prefix or name.startswith(prefix+".") for prefix in prefixes)
    saved_path = sys.path[:]
    saved_modules = {name: module for name, module in sys.modules.items() if matches(name)}
    for name in saved_modules: del sys.modules[name]
    try:
        sys.path[:0] = [str(source_root/"scripts"), str(source_root)]
        spec = importlib.util.spec_from_file_location("_failed_input_evaluator", source_root/SOURCE_HELPERS[0])
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        yield module
    finally:
        for name in list(sys.modules):
            if matches(name): del sys.modules[name]
        sys.modules.update(saved_modules); sys.path[:] = saved_path


@contextmanager
def original_runtime(runtime_root):
    """Canonical module name/path must survive until spawned workers close."""
    runtime_root = Path(runtime_root).resolve()
    name = "nmpc_native_batch"
    saved_path = sys.path[:]
    previous = sys.modules.pop(name, None)
    try:
        sys.path.insert(0, str(runtime_root/"scripts"))
        importlib.invalidate_caches()
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != runtime_root/"scripts/nmpc_native_batch.py":
            raise ScopeError("Original native wrapper import resolved outside the bound runtime snapshot")
        yield module.NativeBatchTransition
    finally:
        sys.modules.pop(name, None)
        if previous is not None: sys.modules[name] = previous
        sys.path[:] = saved_path


def run_batches(model, q0, contract, library, original_worker, batches, *, worker_factory, guard, save):
    """Exactly two planned fresh-worker scopes; never retry or salvage a batch."""
    rows = []
    for horizon in HORIZONS:
        guard()
        inputs = batches[horizon]
        row = {"horizon_steps": horizon, "batch_size": 4, "requested_physics_substeps": 4*horizon*4,
               "worker_construction_attempts": 1, "rollout_calls": 0,
               "complete_batch_success": False, "failed_batch_localization": None}
        try:
            with worker_factory(model, q0, contract, library) as worker:
                for name in NATIVE_IDENTITY_FIELDS:
                    if worker.metadata.get(name) != original_worker.get(name):
                        raise ScopeError("Local native identity differs from original worker: " + name)
                row["local_worker_identity"] = {name: worker.metadata[name] for name in NATIVE_IDENTITY_FIELDS}
                row["local_native_library_sha256"] = worker.metadata.get("native_library_sha256")
                row["rollout_calls"] = 1
                prediction = worker.rollout(inputs["qpos"], inputs["qvel"], inputs["actions"],
                                             initial_history=inputs["history"])
            require(np.asarray(prediction["qpos"]).shape == (4, horizon, 4, 15)
                    and np.asarray(prediction["qvel"]).shape == (4, horizon, 4, 14)
                    and all(np.isfinite(value).all() for value in prediction.values()),
                    "Malformed/nonfinite returned native batch")
            row.update(complete_batch_success=True,
                maximum_absolute_predicted_qvel=float(np.max(np.abs(prediction["qvel"]))),
                finite_returned_arrays=True)
        except ScopeError:
            raise
        except Exception as error:
            row.update(error_type=type(error).__name__, error_message=str(error),
                failed_batch_localization="Unknown: no failed row/candidate/substep is inferred from a batch failure")
        rows.append(row)
        save(rows)
        guard()
    return rows


def analyze(args):
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserving existing failed-input probe output")
    report, trace, solver = load_archive(args.episode)
    from controller_references import DEFAULT_CATALOG, get_reference
    spec = get_reference("drift_drl_traj4")
    require(sha(args.model) == MODEL_SHA256 == report["mujoco_model_sha256"]
            and sha(args.motion) == spec["motion_npz_sha256"] == report["motion_sha256"]
            and report["reference_spec"] == spec and sha(DEFAULT_CATALOG) == report["reference_catalog_sha256"],
            "Exact model/motion/reference identity is required")
    require(sha(args.contract) == archived_source_hash(report, "config/nominal_actuator_contract.json"),
            "Actual actuator contract bytes differ")
    paths = {"episode:"+name: Path(args.episode)/name for name in EXPECTED_ARCHIVE}
    paths.update(model=Path(args.model), motion=Path(args.motion), actuator_contract=Path(args.contract),
        native_library=Path(args.native_library), native_build_manifest=Path(args.native_library).with_suffix(".json"),
        reference_catalog=DEFAULT_CATALOG, probe_source=Path(__file__),
        trace_validation_helper=ROOT/"scripts/probe_nmpc_isaac_prediction.py",
        trace_validation_objective=ROOT/"scripts/nmpc_tracking_objective.py")
    runtime_root = Path(args.runtime_source_root)
    runtime_manifest = runtime_root/"source_manifest.json"
    manifest = json.loads(runtime_manifest.read_text())
    expected_runtime = {relative: archived_source_hash(report, relative) for relative in RUNTIME_FILES}
    require(manifest.get("kind") == "exact_git_blob_runtime_snapshot"
            and str(manifest.get("git_commit", "")).startswith("0e1d884")
            and manifest.get("files") == expected_runtime, "Original runtime snapshot manifest differs")
    paths["runtime_source_manifest"] = runtime_manifest
    for relative in RUNTIME_FILES:
        path = runtime_root/relative
        require(sha(path) == archived_source_hash(report, relative), "Frozen runtime source differs: "+relative)
        paths["source:"+relative] = path
    for relative in SOURCE_HELPERS:
        path = Path(args.source_root)/relative
        require(sha(path) == archived_source_hash(report, relative), "Frozen evaluator helper differs: "+relative)
        paths["helper:"+relative] = path
    hashes = {name: sha(path) for name, path in paths.items()}
    batches = {horizon: observed_inputs(trace, solver, horizon) for horizon in HORIZONS}
    original = report["prediction_backend_metadata"]["workers"]["0.7"]
    require(original.get("model_snapshot_sha256") == SNAPSHOT_SHA256, "Unexpected original compiled-model snapshot")
    with original_runtime(runtime_root) as factory:
        return execute_prepared(args, report, batches, paths, hashes, original, worker_factory=factory)


def execute_prepared(args, report, batches, paths, hashes, original, *, worker_factory):
    # Reached only inside the guarded original-runtime import context.
    output = Path(args.output)
    import mujoco
    with evaluator(args.source_root) as api:
        contract = api.validate_contract(json.loads(Path(args.contract).read_text()))
        model, removed = api.load_model(args.model, no_visual_mesh=True)
        api.configure_model(model, contract)
        model.geom_friction[:, 0] = .7
        reference = api.MotionReference(args.motion, contract)
        q0 = np.array(reference.q0, dtype=float, copy=True)
    require((model.nq, model.nv, model.nu, model.na) == (15, 14, 8, 0)
            and q0.shape == (8,) and np.isfinite(q0).all()
            and hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest() == SNAPSHOT_SHA256,
            "Loaded/configured model or MotionReference q0 differs from recorded native identity")
    verify_unchanged(paths, hashes)
    result = {"schema_version": 1, "kind": "bounded_failed_isaac_I6_observed_action_input_probe",
        "selected_policy_frames": list(FRAMES), "predeclared_horizons": list(HORIZONS),
        "maximum_planned_physics_substeps": 496, "maximum_native_rollout_calls": 2,
        "status": "inputs_validated_not_yet_predicted", "input_sha256": hashes,
        "original_report_status": report["status"], "original_app_terminal_success_established": False,
        "runtime_module_import": "Canonical nmpc_native_batch from bound original snapshot; context retained until both independent workers close",
        "original_native_identity": {name: original[name] for name in (*NATIVE_IDENTITY_FIELDS, "native_library_sha256")},
        "compiled_model_snapshot_sha256": SNAPSHOT_SHA256, "visual_geometries_removed": removed,
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "new_closed_loop_episode": False, "new_cem_search": False, "new_isaac_app": False,
        "native_build_performed": False, "automatic_worker_restart": False,
        "recorded_sensor_history_reordered_or_zeroed": False, "root_reference_prescribed": False,
        "objective_cost_evaluator_exercised": False,
        "scope": "Observed-input hypothesis only; exact recorded packed states and future APPLIED float32 actions, not the unrecorded failed CEM candidates or a new controller run.",
        "history": "Exact PhysX newest-first 3x9x3 memory; unmodified even though native output uses chronological order.",
        "previous_action_vz_scope": "Validated from recorded applied actions and authoritative prior-step actual objective; not consumed by the dynamics-only worker.",
        "failure_scope": "A failed batch does not identify its failing row, candidate, substep, contact mode or external process cause.",
        "interpretation": "Successful observed-action batches cannot explain or reproduce the original sampled-population failure. A failure cannot by itself establish invalid input, OOM, timeout or worker death. Objective-adapter exceptions are not tested.",
        "rows": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream: stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    def save(rows):
        result["rows"] = rows
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    try:
        rows = run_batches(model, q0, contract, args.native_library, original, batches,
            worker_factory=worker_factory, guard=lambda: verify_unchanged(paths, hashes), save=save)
        result["status"] = "completed_observed_action_input_probe" if all(row["complete_batch_success"] for row in rows) else "completed_with_failed_predeclared_batch"
    except BaseException as error:
        result.update(status="aborted_preserving_predeclared_scope", error_type=type(error).__name__, error_message=str(error))
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        raise
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "model", "motion", "contract", "source-root", "runtime-source-root", "native-library", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    return p


if __name__ == "__main__":
    result = analyze(parser().parse_args())
    print(json.dumps({"status": result["status"], "rows": result["rows"]}, indent=2))
    raise SystemExit(0 if result["status"] == "completed_observed_action_input_probe" else 2)
