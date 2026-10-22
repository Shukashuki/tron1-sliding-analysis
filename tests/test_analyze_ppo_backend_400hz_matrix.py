from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_ppo_backend_400hz_matrix as mod
from controller_metrics import heading_signals


def signal_fixture(offset=0.):
    time = np.arange(2793) * .0025
    reference_time = np.arange(349) * .02
    position = np.column_stack((time * .5 + offset, time * 0., time * 0. + .6))
    quaternion = np.tile([1., 0., 0., 0.], (len(time), 1))
    velocity = np.tile([.5, 0., 0.], (len(time), 1))
    trace = {"time_s": time, "physics_dt_s": .0025, "position": position, "quaternion": quaternion,
        "com_velocity": velocity, "angular_velocity": np.column_stack((np.sin(2 * np.pi * 20 * time), time * 0., time * 0.)),
        "contact_count": np.ones((len(time), 2)), "contact_force": np.tile([[0., 0., 10.], [0., 0., 10.]], (len(time), 1, 1)),
        "wheel_pos": position[:, None, :] + np.tile([[0., .2, -.6], [0., -.2, -.6]], (len(time), 1, 1))}
    trace["wheel_quaternion_wxyz"] = np.tile([1., 0., 0., 0.], (len(time), 2, 1))
    reference = {"time_s": reference_time, "position": np.column_stack((reference_time * .5, reference_time * 0., reference_time * 0. + .6)),
        "velocity": np.tile([.5, 0., 0.], (349, 1)), "quaternion": np.tile([1., 0., 0., 0.], (349, 1))}
    signals = heading_signals(quaternion, velocity, np.tile([1., 0., 0., 0.], (len(time), 2, 1)), np.repeat(velocity[:, None], 2, axis=1))
    trace["signals"] = signals
    return trace, reference, signals


def test_core_native_tracking_units_contact_and_common_band():
    trace, reference, signals = signal_fixture(.2)
    result = mod.window_metrics(trace, reference, signals, (1.44, 4.))
    assert result["native_samples"] == 1025
    assert result["tracking"]["xy_rmse_m"] == pytest.approx(.2)
    assert result["tracking"]["xy_peak_m"] == pytest.approx(.2)
    assert result["tracking"]["speed_retention"] == pytest.approx(1.)
    assert result["tracking"]["beta_rmse_deg"] == pytest.approx(0.)
    assert result["tracking"]["beta_valid_samples"] == 1025
    assert result["world_angular_xy_common10_100"]["common_band_rms"] == pytest.approx(.5, abs=.002)
    assert result["contact"]["load_eligibility"]["both"]["sample_duty_fraction"] == 1.
    assert result["contact"]["load_eligibility"]["none"]["sample_duty_fraction"] == 0.
    assert result["height_or_airborne_failure_gate"] is False


def test_full_reference_unavailable_no_endpoint_extrapolation():
    trace, reference, signals = signal_fixture()
    full = mod.window_metrics(trace, reference, signals, (0., 6.98))
    overlap = mod.window_metrics(trace, reference, signals, (0., 6.96))
    assert full["tracking"]["available"] is False
    assert overlap["tracking"]["available"] is True
    assert overlap["native_samples"] == 2785


def test_nonzero_beta_is_preserved_not_targeted_to_zero():
    trace, reference, _ = signal_fixture()
    trace["com_velocity"][:] = [.5, .5, 0.]
    reference["velocity"][:] = [.5, .5, 0.]
    signals = heading_signals(trace["quaternion"], trace["com_velocity"],
        np.tile([1., 0., 0., 0.], (2793, 2, 1)), np.repeat(trace["com_velocity"][:, None], 2, axis=1))
    result = mod.window_metrics(trace, reference, signals, (1.44, 4.))
    assert np.degrees(signals["root_directed_beta_rad"][0]) == pytest.approx(45.)
    assert result["tracking"]["beta_rmse_deg"] == pytest.approx(0.)


def test_persistent_tilt_is_diagnostic_not_failure():
    trace, reference, signals = signal_fixture()
    trace["quaternion"][:] = [np.cos(np.pi / 6), np.sin(np.pi / 6), 0., 0.]
    result = mod.window_metrics(trace, reference, signals, (1.44, 4.))
    assert result["maximum_tilt_deg"] == pytest.approx(60.)
    assert result["persistent_tilt_diagnostics"]["45"]["persistent_event_count"] == 1
    assert result["height_or_airborne_failure_gate"] is False


def test_wheel_spin_invariant_arrows_use_rolling_plane():
    trace, _, signals = signal_fixture()
    angle = np.arange(2793) * .17
    spinning = np.zeros((2793, 2, 4))
    spinning[:, :, 0] = np.cos(angle[:, None] / 2)
    spinning[:, :, 2] = np.sin(angle[:, None] / 2)
    spinning_signals = heading_signals(trace["quaternion"], trace["com_velocity"], spinning,
                                      np.repeat(trace["com_velocity"][:, None], 2, axis=1))
    assert np.allclose(spinning_signals["wheel_rolling_plane_heading_rad"], signals["wheel_rolling_plane_heading_rad"])


def test_geometry_masks_independent_of_speed_and_vertical_body_axis():
    trace, _, _ = signal_fixture()
    trace["com_velocity"][:] = 0.
    body, wheels = mod.heading_geometry_masks(trace)
    assert body.all() and wheels.all()
    trace["quaternion"][:] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
    body, wheels = mod.heading_geometry_masks(trace)
    assert not body.any()
    assert wheels.all()


def test_matrix_incomplete_rejected_before_fk_or_source_loading(tmp_path, monkeypatch):
    (tmp_path / "validation_matrix_manifest.json").write_text(json.dumps({"status": "running"}))
    monkeypatch.setattr(mod, "load_fk", lambda *a: pytest.fail("No model loading before completed matrix guard"))
    with pytest.raises(ValueError, match="not completed"):
        mod.load_matrix(tmp_path, Path("unused"), Path("unused"), Path("unused"))


def test_run_preserves_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_matrix", lambda *a: pytest.fail("Must reject old output first"))
    with pytest.raises(ValueError, match="Preserve"):
        mod.run(NS(output_dir=tmp_path))


def test_reference_pin_rejected_before_loading(tmp_path):
    path = tmp_path / "reference.npz"
    path.write_bytes(b"not the locked reference")
    with pytest.raises(ValueError, match="locked cohort"):
        mod.reference_data(path)


def test_nominal_plot_has_identical_world_bounds_and_no_alignment(tmp_path):
    trace, reference, signals = signal_fixture()
    nominal = {(origin, engine): trace for origin in ("physx", "mjlab") for engine in mod.ENGINE_LABELS}
    path = tmp_path / "nominal.png"
    settings = mod.render_nominal(nominal, reference, {}, path)
    assert path.is_file()
    assert settings["arrow_length_m"] == .18
    assert settings["arrow_timestamps_s"] == [2., 3., 4.]
    assert "spin invariant" in settings["wheel_arrow_kind"]
    assert settings["shared_world_bounds"]["xlim"][0] == pytest.approx(-.35)


def test_source_no_dynamics_step_or_policy_inference():
    source = Path(mod.__file__).read_text()
    assert "mj_step(" not in source
    assert "torch.jit.load(" not in source
    assert "reference_state_initialization" not in source


def test_real_frozen_400hz_context_matches_existing_metrics_and_fk():
    # This OLD actor is ONLY a regression fixture, never a new matrix member.
    pytest.importorskip("mujoco")
    recording = ROOT / "outputs/ppo_contact_timestep_20261022_v1/dt0p0025/physics_trajectory.npz"
    reference_path = ROOT.parent / "tron1-learning-from-human/results/2026-10-14-car-drift/matched-friction-mu0p70/motion.npz"
    model = ROOT / "outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml"
    source = ROOT / "outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_mujoco.py"
    if not all(path.is_file() for path in (recording, reference_path, model, source)):
        pytest.skip("Unpublished historical regression inputs unavailable")
    reference = mod.reference_data(reference_path)
    fk, loader_audit = mod.load_fk(model, source)
    with np.load(recording, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    trace, metrics = mod.analyze_case(raw, "physx", reference, fk)
    core = metrics["windows"]["core"]
    assert core["tracking"]["xy_rmse_m"] == pytest.approx(.055221, abs=2e-6)
    assert core["world_angular_xy_common10_100"]["common_band_rms"] == pytest.approx(.947650, abs=2e-6)
    assert metrics["wheel_heading_FK_audit"]["maximum_wheel_link_position_error_m"] < 2e-4
    assert loader_audit["dynamics_stepping"] is False
    assert trace["time_s"].shape == (2793,)


def matrix_fixture(tmp_path, monkeypatch):
    protocol = mod.verify_protocol()
    reference_file, model_file = tmp_path / "reference.npz", tmp_path / "model.xml"
    reference_file.write_bytes(b"reference")
    model_file.write_bytes(b"model")
    trace, reference, _ = signal_fixture()
    cases, proofs = [], {}
    for phase in ("pipeline_smoke", "formal"):
        for origin in ("physx", "mjlab"):
            for engine in mod.ENGINE_LABELS:
                for mu in ([.7] if phase == "pipeline_smoke" else protocol["validation"]["friction_grid"]):
                    relative = f"{phase}/{origin}/{engine}/mu{mu:.3f}".replace(".", "p")
                    directory = tmp_path / relative
                    directory.mkdir(parents=True)
                    checkpoint = ("a" if origin == "physx" else "b") * 64
                    actor = ("c" if origin == "physx" else "d") * 64
                    report = {"backend_400hz_cohort_evaluation": {"normalized_actor_sha256": actor}, "contact_material_audit": {}}
                    (directory / "report.json").write_text(json.dumps(report))
                    recording = directory / ("physics_trajectory.npz" if engine == "physx" else "rollout.npz")
                    np.savez(recording, placeholder=np.zeros(1))
                    proof = {"smoke_only": phase == "pipeline_smoke", "training_backend": origin,
                        "checkpoint_sha256": checkpoint, "evaluation_contact_friction": mu,
                        "report_sha256": mod.sha256(directory / "report.json"), "recording_sha256": mod.sha256(recording)}
                    proofs[str(directory)] = proof
                    cases.append({"phase": phase, "training_backend": origin, "engine_tag": engine,
                        "validation_backend": mod.ENGINE_LABELS[engine], "mu": mu,
                        "relative_output_dir": relative, "directory": str(directory), "cohort_dir": str(tmp_path / origin),
                        "checkpoint_sha256": checkpoint, "actor_sha256": actor, "verification": proof,
                        "status": "completed_verified"})
    manifest = {"status": "validation_matrix_completed", "protocol_sha256": mod.sha256(mod.DEFAULT_PROTOCOL),
        "input_motion_sha256": mod.sha256(reference_file), "input_model_sha256": mod.sha256(model_file), "cases": cases,
        "counts": {"pipeline_smoke_expected": 4, "pipeline_smoke_completed": 4,
                   "formal_expected": 68, "formal_completed": 68, "runtime_failed": 0}}
    (tmp_path / "validation_matrix_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(mod, "reference_data", lambda *a: reference)
    monkeypatch.setattr(mod, "load_fk", lambda *a: (None, {}))
    monkeypatch.setattr(mod, "validate_output", lambda folder, *a, **k: proofs[str(folder)])
    monkeypatch.setattr(mod, "analyze_case", lambda *a: (trace, {"windows": {}, "xy_screen": {}}))
    monkeypatch.setattr(mod, "material_summary", lambda *a: {})
    return reference_file, model_file, manifest


def test_complete_matrix_inventory_uses68formal_not4smokes(tmp_path, monkeypatch):
    reference, model, manifest = matrix_fixture(tmp_path, monkeypatch)
    nominal, reference_values, summary = mod.load_matrix(tmp_path, reference, model, Path("unused"))
    assert len(summary["rows"]) == 68
    assert len(nominal) == 4
    assert summary["pipeline_smoke_episode_count"] == 4
    assert summary["visual_reviewed"] is False
    assert summary["sample_size_per_cell"] == 1
    assert summary["new_policy_inference"] is False
    assert len({row["checkpoint_sha256"] for row in summary["rows"]}) == 2


@pytest.mark.parametrize("defect", ["duplicate", "escape", "actor", "runtime_failed"])
def test_matrix_guard_rejects_partial_escaped_or_mixed_inventory(tmp_path, monkeypatch, defect):
    reference, model, manifest = matrix_fixture(tmp_path, monkeypatch)
    if defect == "duplicate": manifest["cases"][-1] = manifest["cases"][-2]
    if defect == "escape": manifest["cases"][0]["relative_output_dir"] = "../escape"
    if defect == "actor": manifest["cases"][-1]["actor_sha256"] = "0" * 64
    if defect == "runtime_failed": manifest["counts"]["runtime_failed"] = 1
    (tmp_path / "validation_matrix_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        mod.load_matrix(tmp_path, reference, model, Path("unused"))


def compact_fixture():
    trace, reference, signals = signal_fixture()
    core = mod.window_metrics(trace, reference, signals, (1.44, 4.))
    return {"rows": [{"training_backend": origin, "validation_backend": engine, "mu": mu,
            "windows": {"core": core}, "xy_screen": {"large_xy_error": mu == .03}}
        for origin in ("physx", "mjlab") for engine in mod.ENGINE_LABELS.values() for mu in (.03, .7, 2.)]}


def test_compact_overview_nominal_four_and_discrete_not_continuous_ranges():
    table, overview = mod.compact_metrics(compact_fixture())
    assert len(table) == 12
    assert len(overview["nominal_four_cells"]) == 4
    assert all(row["mu"] == .7 for row in overview["nominal_four_cells"])
    assert overview["sample_size_per_cell"] == 1
    assert overview["visual_reviewed"] is False
    assert overview["tested_mu_lists"][0]["tested_XY_pass_mu"] == [.7, 2.]
    assert overview["tested_mu_lists"][0]["tested_large_XY_mu"] == [.03]
    assert "NOT continuous feasible ranges" in overview["interpretation"]
    assert table[0]["first_core_tilt30_persistent_onset_s"] is None
    assert table[0]["both_wheels_load_time_duty_fraction"] == 1.


def test_compact_csv_json_fresh_and_readable(tmp_path):
    import csv
    mod.write_compact_outputs(compact_fixture(), tmp_path)
    overview = json.loads((tmp_path / "overview.json").read_text())
    with (tmp_path / "metric_table.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert float(rows[0]["xy_rmse_m"]) == pytest.approx(0., abs=1e-15)
    assert overview["all_cells_table"] == "metric_table.csv"
    with pytest.raises(FileExistsError):
        mod.write_compact_outputs(compact_fixture(), tmp_path)
