"""Asset-free checks for the read-only Isaac MPC backend matrix preflight."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from preflight_mpc_backend_matrix import (BACKENDS, ENGINE, SOLVER,
                                          canonical_sha256, main,
                                          objective_semantics, validate_manifest)


def _report(*, plant="isaac_physx", predictor="isaac_physx", historical=False):
    report = {
        "kind": "receding_horizon_nmpc", "status": "completed",
        "reference_id": "drift_drl_traj4", "reference_catalog_sha256": "a" * 64,
        "motion_sha256": "b" * 64, "asset_sha256": "c" * 64,
        "plant_mu": 0.7, "prediction_mu_scenarios": [0.7],
        "physics_dt_s": .005, "control_dt_s": .02, "plant_seed": 42,
        "total_reference_policy_steps": 349, "completed_policy_steps": 349,
        "core_window_s": [1.44, 4.0], "core_coverage_complete": True,
        "truncated_smoke": False, "core_xy_rmse_m": .1,
        "core_speed_retention_mean_ratio": .9,
        "compute_ms_median_p95_max": [10., 11., 12.],
        "effective_cem_config": {"candidates": 64, "seed": 42},
        "objective_contract": {"kind": "numpy", "weights": {"drift_sideslip": 2.},
                               "surrogate_differences": ["implementation note"]},
        "action_parameterization": {"mode": "warm_start_residual_blocks"},
    }
    if historical:
        report.update(engine="Isaac/PhysX", prediction_engine="MuJoCo",
                      prediction_backend="native", prediction_mujoco_version="3.8.1")
    else:
        report.update(engine=ENGINE[plant], prediction_engine=ENGINE[predictor],
                      isaac_lab_version="3.0.0", isaac_lab_revision="f" * 40,
                      plant_physics_backend=plant, predictor_physics_backend=predictor,
                      backend_runtime_audit={
                          "plant": {"backend": plant, "solver": SOLVER[plant],
                                    "runtime_verified": True},
                          "predictor": {"backend": predictor, "solver": SOLVER[predictor],
                                        "runtime_verified": True},
                      })
    return report


def _write_report(folder, report):
    folder.mkdir(parents=True, exist_ok=True)
    rollout = b"test rollout"
    solver_log = b"[]\n"
    (folder / "rollout.npz").write_bytes(rollout)
    (folder / "solver_log.json").write_bytes(solver_log)
    report["rollout_sha256"] = hashlib.sha256(rollout).hexdigest()
    report["solver_log_sha256"] = hashlib.sha256(solver_log).hexdigest()
    (folder / "report.json").write_text(json.dumps(report))


def _manifest(tmp_path):
    baseline = _report(historical=True)
    _write_report(tmp_path / "legacy", baseline)
    lock = {key: baseline[key] for key in (
        "reference_id", "reference_catalog_sha256", "motion_sha256", "asset_sha256",
        "plant_mu", "prediction_mu_scenarios", "physics_dt_s", "control_dt_s",
        "plant_seed", "total_reference_policy_steps", "core_window_s")}
    lock.update(effective_cem_config_sha256=canonical_sha256(baseline["effective_cem_config"]),
                objective_semantics_sha256=canonical_sha256(objective_semantics(baseline["objective_contract"])),
                action_parameterization_sha256=canonical_sha256(baseline["action_parameterization"]))
    cells = [{"plant_backend": plant, "predictor_backend": predictor,
              "report": f"{plant}_{predictor}/report.json"}
             for plant in BACKENDS for predictor in BACKENDS]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "comparison_id": "test",
                                "lock": lock, "historical_baseline_report": "legacy/report.json",
                                "cells": cells}))
    return path, cells


def test_missing_newton_and_physx_cells_are_explicit(tmp_path):
    manifest, _ = _manifest(tmp_path)
    result = validate_manifest(manifest)
    assert result["status"] == "incomplete"
    assert result["historical_baseline"]["status"] == "valid"
    assert [row["status"] for row in result["cells"]] == ["missing"] * 4
    assert main([str(manifest), "--require-complete"]) == 3


def test_complete_crossed_matrix_and_objective_annotation_exclusion(tmp_path):
    manifest, cells = _manifest(tmp_path)
    for cell in cells:
        report = _report(plant=cell["plant_backend"], predictor=cell["predictor_backend"])
        report["objective_contract"]["surrogate_differences"] = ["backend-specific prose"]
        _write_report(tmp_path / Path(cell["report"]).parent, report)
    result = validate_manifest(manifest)
    assert result["status"] == "ready"
    assert all(row["status"] == "valid" for row in result["cells"])


def test_mujoco_cpu_report_cannot_be_relabelled_as_mjwarp(tmp_path):
    manifest, cells = _manifest(tmp_path)
    cell = next(row for row in cells if row["predictor_backend"] == "isaac_newton_mjwarp")
    _write_report(tmp_path / Path(cell["report"]).parent, _report(historical=True))
    result = validate_manifest(manifest)
    row = next(row for row in result["cells"] if row["report"].endswith(cell["report"]))
    assert row["status"] == "invalid"
    assert any("prediction_engine" in error for error in row["errors"])
    assert any("predictor_physics_backend" in error for error in row["errors"])


@pytest.mark.parametrize("field,value,expected", [
    ("asset_sha256", "d" * 64, "asset_sha256"),
    ("physics_dt_s", .01, "physics_dt_s"),
    ("plant_mu", .4, "plant_mu"),
    ("effective_cem_config", {"candidates": 32}, "effective_cem_config"),
    ("objective_contract", {"weights": {"drift_sideslip": 0.}}, "objective_contract"),
])
def test_locked_contract_mismatch_is_rejected(tmp_path, field, value, expected):
    manifest, cells = _manifest(tmp_path)
    cell = cells[0]
    report = _report()
    report[field] = value
    _write_report(tmp_path / Path(cell["report"]).parent, report)
    result = validate_manifest(manifest)
    assert result["status"] == "invalid"
    assert any(expected in error for error in result["cells"][0]["errors"])


def test_missing_runtime_audit_and_tampered_rollout_are_rejected(tmp_path):
    manifest, cells = _manifest(tmp_path)
    cell = cells[0]
    folder = tmp_path / Path(cell["report"]).parent
    report = _report()
    report.pop("backend_runtime_audit")
    _write_report(folder, report)
    (folder / "rollout.npz").write_bytes(b"tampered")
    row = validate_manifest(manifest)["cells"][0]
    assert row["status"] == "invalid"
    assert any("runtime backend/solver audit" in error for error in row["errors"])
    assert any("rollout.npz SHA-256 mismatch" in error for error in row["errors"])


def test_mixed_isaac_lab_revisions_are_not_called_backend_effects(tmp_path):
    manifest, cells = _manifest(tmp_path)
    for index, cell in enumerate(cells):
        report = _report(plant=cell["plant_backend"], predictor=cell["predictor_backend"])
        if index == 1:
            report["isaac_lab_revision"] = "e" * 40
        _write_report(tmp_path / Path(cell["report"]).parent, report)
    result = validate_manifest(manifest)
    assert result["status"] == "invalid"
    assert all(row["status"] == "invalid" for row in result["cells"])
    assert all(any("version/revision differs" in error for error in row["errors"])
               for row in result["cells"])


def test_malformed_matrix_is_rejected(tmp_path):
    manifest, cells = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["cells"] = cells[:-1] + [cells[0]]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Duplicate or unknown"):
        validate_manifest(manifest)
