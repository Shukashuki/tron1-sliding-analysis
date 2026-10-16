"""Pure bounded-scope fixtures; never construct MuJoCo or native workers."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"scripts"), str(Path(__file__).parent)]
import probe_nmpc_isaac_failed_inputs as probe
from test_probe_nmpc_isaac_prediction import synthetic_trace


def recorded_inputs():
    trace, solver = synthetic_trace()
    trace["actions"] = np.linspace(-.5, .5, 349*8, dtype=np.float32).reshape(349, 8).astype(float)
    trace["held_action"][1:] = np.repeat(trace["actions"], 4, axis=0)
    trace["packed_controller_states"][:, 3] = 1.
    for k, row in enumerate(solver):
        row["action"] = trace["actions"][k].tolist()
        row["actual_isaac_objective"] = {"base_com_vz": .3}
    return trace, solver


def native_identity():
    return {name: "fixed-"+name for name in probe.NATIVE_IDENTITY_FIELDS}


def test_inputs_use_exact_fixed_packed_rows_newest_history_and_future_applied_actions():
    trace, solver = recorded_inputs(); old = copy.deepcopy(trace)
    for horizon in probe.HORIZONS:
        inputs = probe.observed_inputs(trace, solver, horizon)
        packed = trace["packed_controller_states"][list(probe.FRAMES)]
        np.testing.assert_array_equal(inputs["qpos"], packed[:, :15])
        np.testing.assert_array_equal(inputs["qvel"], packed[:, 15:29])
        np.testing.assert_array_equal(inputs["history"], packed[:, 29:110].reshape(4, 3, 9, 3))
        assert np.max(inputs["history"]) > 0.
        for row, frame in enumerate(probe.FRAMES):
            np.testing.assert_array_equal(inputs["actions"][row], trace["actions"][frame:frame+horizon])
            np.testing.assert_array_equal(inputs["previous_actions"][row], trace["actions"][frame-1])
        for value in inputs.values(): value[:] = 0.
        for name in old: np.testing.assert_array_equal(trace[name], old[name])
    assert sum(4*h*4 for h in probe.HORIZONS) == 496


@pytest.mark.parametrize("problem", ["horizon", "quat", "nan", "raw_precision", "bounds", "reset", "previous_vz", "previous_action"])
def test_malformed_scope_or_recorded_inputs_fail_before_any_worker(problem):
    trace, solver = recorded_inputs(); horizon = 30
    if problem == "horizon": horizon = 31
    elif problem == "quat": trace["packed_controller_states"][137, 3:7] = 0.
    elif problem == "nan": trace["packed_controller_states"][137, 15] = np.nan
    elif problem == "raw_precision": trace["actions"][137, 0] = .1234567890123
    elif problem == "bounds": trace["actions"][137, 0] = 1.1
    elif problem == "reset": trace["packed_controller_states"][137, -1] = 1.
    elif problem == "previous_vz": solver[136]["actual_isaac_objective"]["base_com_vz"] = .4
    else: trace["held_action"][4*137, 0] += .1
    with pytest.raises(ValueError): probe.observed_inputs(trace, solver, horizon)


def fake_workers(calls, *, fail_horizon=None, bad_identity=False):
    class Worker:
        def __init__(self, *args):
            self.metadata = native_identity() | {"native_library_sha256": "local-binary-not-remote"}
            if bad_identity: self.metadata["model_snapshot_sha256"] = "other-model"
            calls.append(("created", id(self)))
        def __enter__(self): return self
        def __exit__(self, *args): calls.append(("closed", id(self)))
        def rollout(self, qpos, qvel, actions, *, initial_history):
            horizon = actions.shape[1]
            calls.append(("rollout", id(self), horizon, actions.copy(), initial_history.copy()))
            if horizon == fail_horizon: raise RuntimeError("Synthetic native whole-batch failure; cause unknown")
            return {"qpos": np.tile(qpos[:, None, None], (1, horizon, 4, 1)),
                    "qvel": np.tile(qvel[:, None, None], (1, horizon, 4, 1))}
    return Worker


@pytest.mark.parametrize("failed_horizon", [None, 1, 30])
def test_exactly_one_fresh_worker_and_one_batch_per_predeclared_horizon_no_retry(failed_horizon):
    trace, solver = recorded_inputs(); calls, guards, saved = [], [], []
    batches = {h: probe.observed_inputs(trace, solver, h) for h in probe.HORIZONS}
    rows = probe.run_batches(object(), np.zeros(8), {}, "explicit-prebuilt-library", native_identity(), batches,
        worker_factory=fake_workers(calls, fail_horizon=failed_horizon), guard=lambda: guards.append(True),
        save=lambda value: saved.append(copy.deepcopy(value)))
    assert [call[2] for call in calls if call[0] == "rollout"] == [1, 30]
    assert len([call for call in calls if call[0] == "created"]) == 2
    assert len([call for call in calls if call[0] == "closed"]) == 2
    assert len(guards) == 4 and len(saved) == 2
    assert sum(row["requested_physics_substeps"] for row in rows) == 496
    for row in rows:
        assert row["rollout_calls"] == 1
        assert row["complete_batch_success"] is (row["horizon_steps"] != failed_horizon)
        if not row["complete_batch_success"]:
            assert row["error_type"] == "RuntimeError" and "cause unknown" in row["error_message"]
            assert row["failed_batch_localization"].startswith("Unknown:")
            assert "maximum_absolute_predicted_qvel" not in row


def test_wrong_native_identity_aborts_before_prediction_and_without_retry():
    trace, solver = recorded_inputs(); calls = []
    with pytest.raises(probe.ScopeError, match="model_snapshot"):
        probe.run_batches(object(), np.zeros(8), {}, "prebuilt", native_identity(),
            {h: probe.observed_inputs(trace, solver, h) for h in probe.HORIZONS},
            worker_factory=fake_workers(calls, bad_identity=True), guard=lambda: None, save=lambda rows: None)
    assert len([call for call in calls if call[0] == "created"]) == 1
    assert not any(call[0] == "rollout" for call in calls)


def test_changed_bound_bytes_abort_before_any_worker(tmp_path):
    path = tmp_path/"synthetic-input"; path.write_text("original")
    hashes = {"input": probe.sha(path)}; path.write_text("changed")
    with pytest.raises(ValueError, match="bytes changed"):
        probe.verify_unchanged({"input": path}, hashes)


def test_existing_output_preserved_before_loading_inputs_or_importing_mujoco(tmp_path):
    output = tmp_path/"existing.json"; output.write_text("user owned")
    with pytest.raises(FileExistsError, match="Preserving"):
        probe.analyze(SimpleNamespace(output=output))
    assert output.read_text() == "user owned"


def test_archive_tamper_rejected_before_loading_native_runtime(tmp_path):
    (tmp_path/"report.json").write_text("not the locked archive")
    with pytest.raises(ValueError, match="exact I6 archive"):
        probe.load_archive(tmp_path)


def test_import_and_help_do_not_import_simulators_or_build_a_worker():
    code = "import sys;sys.path.insert(0,'scripts');import probe_nmpc_isaac_failed_inputs as p;assert not {'mujoco','torch','isaaclab','omni','nmpc_native_batch'} & set(sys.modules);assert '--native-library' in p.parser().format_help()"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_original_runtime_has_canonical_spawn_module_path_and_restores_imports(tmp_path, monkeypatch):
    directory = tmp_path/"scripts"; directory.mkdir()
    (directory/"nmpc_native_batch.py").write_text("class NativeBatchTransition: pass\n")
    previous = SimpleNamespace(marker="must be restored")
    monkeypatch.setitem(sys.modules, "nmpc_native_batch", previous)
    prior_path = sys.path[:]
    with probe.original_runtime(tmp_path) as factory:
        assert factory.__module__ == "nmpc_native_batch"
        assert Path(sys.modules[factory.__module__].__file__) == directory/"nmpc_native_batch.py"
        assert sys.path[0] == str(directory)
    assert sys.modules["nmpc_native_batch"] is previous and sys.path == prior_path


def test_prepared_orchestration_records_both_batches_using_only_fake_runtime(tmp_path, monkeypatch):
    trace, solver = recorded_inputs(); calls = []
    contract = tmp_path/"contract.json"; contract.write_text("{}")
    blob = b"synthetic compiled-model identity"
    monkeypatch.setattr(probe, "SNAPSHOT_SHA256", probe.hashlib.sha256(blob).hexdigest())
    monkeypatch.setattr(probe.pickle, "dumps", lambda *a, **k: blob)
    model = SimpleNamespace(nq=15, nv=14, nu=8, na=0, geom_friction=np.ones((2, 3)))
    api = SimpleNamespace(validate_contract=lambda value: value, load_model=lambda *a, **k: (model, 0),
        configure_model=lambda *a: None, MotionReference=lambda *a: SimpleNamespace(q0=np.zeros(8)))
    monkeypatch.setattr(probe, "evaluator", lambda *a: nullcontext(api))
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(__version__="synthetic-test-only"))
    args = SimpleNamespace(output=tmp_path/"probe.json", contract=contract, source_root=tmp_path,
        model=tmp_path/"never-read-model.xml", motion=tmp_path/"never-read-motion.npz", native_library="fake-prebuilt")
    original = native_identity() | {"native_library_sha256": "original-remote-binary"}
    paths = {"fixture_contract": contract}
    result = probe.execute_prepared(args, {"status": "completed"},
        {h: probe.observed_inputs(trace, solver, h) for h in probe.HORIZONS}, paths,
        {name: probe.sha(path) for name, path in paths.items()}, original,
        worker_factory=fake_workers(calls))
    assert result["status"] == "completed_observed_action_input_probe"
    assert [row["horizon_steps"] for row in result["rows"]] == [1, 30]
    assert result["maximum_planned_physics_substeps"] == 496
    assert result["original_app_terminal_success_established"] is False
    assert result["objective_cost_evaluator_exercised"] is False
    assert not result["new_cem_search"] and not result["automatic_worker_restart"]
    assert json.loads(args.output.read_text()) == result


def test_actual_public_probe_exact_bytes_portable_field_allowlist_and_privacy():
    path = ROOT/"results/nmpc_i6_failed_inputs_probe_20261016_v1.json"
    assert probe.sha(path) == "1a64b65731d2a07690917501aa1eb253fce30ccfdb98d3748a7c07bfb2d89904"
    text = path.read_text(); result = json.loads(text)
    assert set(result) == {
        "schema_version", "kind", "selected_policy_frames", "predeclared_horizons",
        "maximum_planned_physics_substeps", "maximum_native_rollout_calls", "status",
        "input_sha256", "original_report_status", "original_app_terminal_success_established",
        "runtime_module_import", "original_native_identity", "compiled_model_snapshot_sha256",
        "visual_geometries_removed", "mujoco_version", "numpy_version", "new_closed_loop_episode",
        "new_cem_search", "new_isaac_app", "native_build_performed", "automatic_worker_restart",
        "recorded_sensor_history_reordered_or_zeroed", "root_reference_prescribed",
        "objective_cost_evaluator_exercised", "scope", "history", "previous_action_vz_scope",
        "failure_scope", "interpretation", "rows",
    }
    assert set(result["input_sha256"]) == {
        "episode:report.json", "episode:rollout.npz", "episode:solver_log.json", "model", "motion",
        "actuator_contract", "native_library", "native_build_manifest", "reference_catalog",
        "probe_source", "trace_validation_helper", "trace_validation_objective",
        "runtime_source_manifest", "source:scripts/nmpc_native_batch.py",
        "source:scripts/nmpc_tracking_objective.py", "source:native/nmpc_batch.cpp",
        "helper:scripts/eval_tracking_mujoco.py", "helper:scripts/eval_tracking.py",
        "helper:training/tron1_domain_randomization.py", "helper:training/tron1_contact_slip.py",
        "helper:training/tron1_physics_clock.py", "helper:training/tron1_drift.py",
    }
    identity_fields = {
        "source_sha256", "mujoco_version", "header_version", "abi_version",
        "mujoco_header_sha256", "mujoco_library_sha256", "model_snapshot_sha256",
    }
    assert set(result["original_native_identity"]) == identity_fields | {"native_library_sha256"}
    row_fields = {
        "horizon_steps", "batch_size", "requested_physics_substeps", "worker_construction_attempts",
        "rollout_calls", "complete_batch_success", "failed_batch_localization", "local_worker_identity",
        "local_native_library_sha256", "maximum_absolute_predicted_qvel", "finite_returned_arrays",
    }
    for row in result["rows"]:
        assert set(row) == row_fields
        assert set(row["local_worker_identity"]) == identity_fields
    for value in result["input_sha256"].values():
        assert len(value) == 64 and set(value) <= set("0123456789abcdef")
    for private in ("/home/", "/data/", "/tmp/", "/private/", "file://", "zengxiong",
                    "14.103.233.29", "117.50.226.213"):
        assert private not in text


def test_actual_public_probe_two_successful_observed_action_batches_scope_and_original_identity():
    result = json.loads((ROOT/"results/nmpc_i6_failed_inputs_probe_20261016_v1.json").read_text())
    assert result["schema_version"] == 1
    assert result["kind"] == "bounded_failed_isaac_I6_observed_action_input_probe"
    assert result["status"] == "completed_observed_action_input_probe"
    assert result["selected_policy_frames"] == [136, 137, 138, 139]
    assert result["predeclared_horizons"] == [1, 30]
    assert result["maximum_planned_physics_substeps"] == 496
    assert result["maximum_native_rollout_calls"] == 2
    rows = result["rows"]
    assert [(row["horizon_steps"], row["batch_size"], row["requested_physics_substeps"])
            for row in rows] == [(1, 4, 16), (30, 4, 480)]
    assert sum(row["rollout_calls"] for row in rows) == 2
    assert sum(row["requested_physics_substeps"] for row in rows) == 496
    original = result["original_native_identity"]
    assert original["source_sha256"] == "deb3afe3367e11c28295e00e4a081e84968818070dcfb7044f06466761875ed2"
    assert original["model_snapshot_sha256"] == result["compiled_model_snapshot_sha256"] == \
        "28ebe2555a1f7511f882c5e62e914ece71bee6855a7ab4425e6fd614bfbe4b9e"
    hashes = result["input_sha256"]
    assert hashes["source:scripts/nmpc_native_batch.py"] == \
        "475ef6fe1712783ea11af2cae64b5ff0d61321e6edee0a596f9977a7a69da8a1"
    assert hashes["source:native/nmpc_batch.cpp"] == original["source_sha256"]
    assert hashes["model"] == "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"
    assert original["native_library_sha256"] == \
        "ce676e8414d38beb0676e8529cd88ebfea980ef9d91494c052e21414c4f89ea5"
    assert hashes["native_library"] == \
        "b13e74df3a532496449d2f6b34b6934784a6bfd4ea5f0574c90cf279048821b7"
    for row in rows:
        assert row["worker_construction_attempts"] == row["rollout_calls"] == 1
        assert row["complete_batch_success"] is True and row["finite_returned_arrays"] is True
        assert row["failed_batch_localization"] is None
        assert np.isfinite(row["maximum_absolute_predicted_qvel"])
        assert row["local_native_library_sha256"] == hashes["native_library"]
        assert row["local_native_library_sha256"] != original["native_library_sha256"]
        assert row["local_worker_identity"] == {
            key: value for key, value in original.items() if key != "native_library_sha256"
        }
    for flag in ("new_closed_loop_episode", "new_cem_search", "new_isaac_app", "native_build_performed",
                 "automatic_worker_restart", "recorded_sensor_history_reordered_or_zeroed",
                 "root_reference_prescribed", "objective_cost_evaluator_exercised"):
        assert result[flag] is False
    assert result["original_report_status"] == "completed"
    assert result["original_app_terminal_success_established"] is False
    assert "future APPLIED float32 actions" in result["scope"]
    assert "unrecorded failed CEM candidates" in result["scope"]
    assert "newest-first" in result["history"]
    assert "Successful observed-action batches cannot explain or reproduce" in result["interpretation"]
    assert "Objective-adapter exceptions are not tested." in result["interpretation"]
