"""Offline numerical-comparison orchestration tests; no simulator/inference."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_ppo_contact_numerics as a


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "recordings"
    root.mkdir()
    reference = tmp_path / "reference.npz"
    np.savez(reference, synthetic_reference=np.array([1.]))
    digest = a.smooth.sha256(reference)
    monkeypatch.setattr(a.contact, "MOTION_SHA", digest)
    (root / "contact_numerics_status.json").write_text(json.dumps({
        "status": "episodes_complete_requires_contact_and_tracking_review",
        "paired_native_offsets_and_materials_identical": True}))
    time = np.arange(1397) * .005
    raw = {"time_s": time, "wheel_ground_contact_count": np.ones((1397, 1, 2), dtype=np.int32),
        "wheel_ground_contact_force_w_n": np.zeros((1397, 1, 2, 3)),
        "root_linear_velocity_w_m_s": np.zeros((1397, 1, 3)),
        "root_angular_velocity_w_rad_s": np.zeros((1397, 1, 3)),
        "nonwheel_ground_contact_count": np.zeros((1397, 1), dtype=np.int32)}
    raw["wheel_ground_contact_force_w_n"][:, 0, :, 2] = 100.
    for label, iterations in zip(a.ORDER, (4, 8)):
        folder = root / label; folder.mkdir()
        runtime = {"native_shape_readback": {"contact_offsets_m": [[.02]], "rest_offsets_m": [[0.]],
                    "materials_static_dynamic_restitution": [[[.7, .7, 0.]]]},
            "scene_physx_iteration_limits_cfg": {"max_velocity_iteration_count": 255},
            "gain_readback": {"kp_range": [500., 500.], "kd_range": [10., 10.]},
            "cfg_requested": {"enabled_self_collisions": True},
            "usd_authored_solver": {"roots": [{"position": 8, "velocity": iterations,
                                                 "enabled_self_collisions_usd": True}]}}
        contract = {"checkpoint_sha256": a.experiment.base.CHECKPOINT_SHA,
            "reference_file_sha256": digest, "actor_dim": 51, "actor_file": "actor_normalized.pt",
            "leg_kp": 500., "leg_kd": 10., "configured_solver_position_iteration_count": 8,
            "configured_solver_velocity_iteration_count": iterations, "solver_counts_are_native_readback": False,
            "export_validation_max_abs_error": 1e-7 * iterations,
            "contact_numerics_diagnostic": {"runtime_readback": runtime}}
        report = {"motion_file_sha256": digest, "policy_export": contract}
        for key in ("task_profile", "evaluation_task_profile", "contact_material_audit", "physics_recording",
                    "contact_slip_measurement", "domain_randomization_audit", "legacy_joint_friction_audit"):
            report[key] = {"same_in_both": True}
        (folder / "report.json").write_text(json.dumps(report))
        np.savez_compressed(folder / "physics_trajectory.npz", **raw)
    calls = []
    def validate(folder, iterations):
        calls.append((Path(folder).name, iterations))
        report = json.loads((folder / "report.json").read_text())
        return {"recording_sha256": a.smooth.sha256(folder / "physics_trajectory.npz"),
                "report_sha256": a.smooth.sha256(folder / "report.json"),
                "runtime_readback": report["policy_export"]["contact_numerics_diagnostic"]["runtime_readback"]}
    monkeypatch.setattr(a.experiment, "validate_output", validate)
    monkeypatch.setattr(a.smooth, "analyze_trajectory", lambda *args, **kwargs:
        {"windows": {"core": {"tracking": {"xy_rmse_m": .05}}}, "new_simulation": False, "new_inference": False})
    monkeypatch.setattr(a.contact, "analyze_signals", lambda *args, **kwargs:
        {"core": {"physical_samples": len(args[0]), "both_wheels": .9}})
    monkeypatch.setattr(a, "posture_diagnostics", lambda data: {"native_samples": len(data["time_s"])})
    return SimpleNamespace(root=root, reference=reference, raw=raw, calls=calls, output=tmp_path / "publication")


def edit_report(inputs, label, change):
    path = inputs.root / label / "report.json"
    value = json.loads(path.read_text()); change(value)
    path.write_text(json.dumps(value))


def test_completed_matched_pair_uses_validation_and_shared_contract(inputs):
    raw, summary = a.load_comparison(inputs.root, inputs.reference)
    assert inputs.calls == [("solver_v4", 4), ("solver_v8", 8)]
    assert summary["case_order"] == ["solver_v4", "solver_v8"]
    assert summary["checkpoint_sha256"] == a.experiment.base.CHECKPOINT_SHA
    assert summary["reference_sha256"] == a.smooth.sha256(inputs.reference)
    assert summary["additional_training_updates"] == 0
    assert summary["stable_contact_claimed"] is False
    assert summary["height_or_airborne_failure_gate"] is False
    assert summary["zero_sideslip_target"] is False
    assert summary["new_simulation_or_inference_by_analysis"] is False
    assert summary["cases"]["solver_v4"]["runtime_readback"]["usd_authored_solver"]["roots"][0]["velocity"] == 4
    assert summary["cases"]["solver_v8"]["runtime_readback"]["usd_authored_solver"]["roots"][0]["velocity"] == 8
    assert np.array_equal(raw["solver_v4"]["time_s"], inputs.raw["time_s"])


def test_wrong_reference_rejected_before_validation(inputs):
    inputs.reference.write_bytes(b"wrong motion; must not be opened as valid reference")
    with pytest.raises(ValueError, match="Locked reference SHA"):
        a.load_comparison(inputs.root, inputs.reference)
    assert not inputs.calls


def test_report_motion_binding_required(inputs):
    edit_report(inputs, "solver_v8", lambda report: report.update(motion_file_sha256="0" * 64))
    with pytest.raises(ValueError, match="Evaluation motion SHA"):
        a.load_comparison(inputs.root, inputs.reference)


@pytest.mark.parametrize("change", ["status", "paired"])
def test_incomplete_or_unpaired_experiment_rejected(inputs, change):
    path = inputs.root / "contact_numerics_status.json"
    status = json.loads(path.read_text())
    if change == "status": status["status"] = "running_solver_v8"
    else: status["paired_native_offsets_and_materials_identical"] = False
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="both completed"):
        a.load_comparison(inputs.root, inputs.reference)
    assert not inputs.calls


@pytest.mark.parametrize("field", ["task_profile", "evaluation_task_profile", "contact_material_audit",
    "physics_recording", "contact_slip_measurement", "domain_randomization_audit", "legacy_joint_friction_audit"])
def test_other_physical_contact_or_task_setting_change_rejected(inputs, field):
    edit_report(inputs, "solver_v8", lambda report: report.update({field: {"different": True}}))
    with pytest.raises(ValueError, match="Additional setting changed"):
        a.load_comparison(inputs.root, inputs.reference)


def test_different_actor_control_contract_rejected(inputs):
    edit_report(inputs, "solver_v8", lambda report: report["policy_export"].update(leg_kp=250.))
    with pytest.raises(ValueError, match="Actor/control contract"):
        a.load_comparison(inputs.root, inputs.reference)


@pytest.mark.parametrize("parity", [None, -1., .01, float("nan")])
def test_actor_export_parity_must_remain_verified(inputs, parity):
    edit_report(inputs, "solver_v8", lambda report: report["policy_export"].update(export_validation_max_abs_error=parity))
    with pytest.raises(ValueError, match="Actor export parity"):
        a.load_comparison(inputs.root, inputs.reference)


def test_native_offsets_must_match_exactly(inputs):
    def mutate(report):
        report["policy_export"]["contact_numerics_diagnostic"]["runtime_readback"]["native_shape_readback"]["contact_offsets_m"] = [[.03]]
    edit_report(inputs, "solver_v8", mutate)
    with pytest.raises(ValueError, match="Additional native/configured"):
        a.load_comparison(inputs.root, inputs.reference)


def test_output_freshness_guard_precedes_input_loading(inputs, monkeypatch):
    inputs.output.mkdir(); marker = inputs.output / "keep.txt"; marker.write_text("preserve")
    def forbidden(*args): raise AssertionError("Must check output before reading inputs")
    monkeypatch.setattr(a, "load_comparison", forbidden)
    with pytest.raises(ValueError, match="fresh output"):
        a.run(SimpleNamespace(input_root=inputs.root, reference=inputs.reference, output_dir=inputs.output))
    assert marker.read_text() == "preserve"


def test_offline_publication_hashes_and_exact_recorded_arrays(inputs):
    before = {path: path.read_bytes() for path in inputs.root.rglob("*") if path.is_file()}
    summary = a.run(SimpleNamespace(input_root=inputs.root, reference=inputs.reference, output_dir=inputs.output))
    assert summary["artifacts"]["recorded_states.npz"]["sha256"] == a.smooth.sha256(inputs.output / "recorded_states.npz")
    assert summary["artifacts"]["contact_comparison.png"]["sha256"] == a.smooth.sha256(inputs.output / "contact_comparison.png")
    assert json.loads((inputs.output / "summary.json").read_text())["case_order"] == list(a.ORDER)
    with np.load(inputs.output / "recorded_states.npz", allow_pickle=False) as archive:
        for label in a.ORDER:
            for key, value in inputs.raw.items():
                assert np.array_equal(archive[label + "_" + key], value)
    assert before == {path: path.read_bytes() for path in inputs.root.rglob("*") if path.is_file()}
