"""Read-only provenance and comparability check for the Isaac MPC backend matrix.

The four experimental cells cross an Isaac PhysX or Newton/MuJoCo-Warp
predictor with an Isaac PhysX or Newton/MuJoCo-Warp plant.  The historical
MuJoCo CPU predictor / PhysX plant run is reported separately.  This checker
does not launch a simulator, infer a backend from a filename, or certify that
a self-reported runtime audit reflects the actual hardware execution.

Each new run report must contain the ordinary CEM report fields plus:

  plant_physics_backend, predictor_physics_backend
  engine, prediction_engine
  isaac_lab_version, isaac_lab_revision
  backend_runtime_audit: {
    plant: {backend, solver, runtime_verified: true},
    predictor: {backend, solver, runtime_verified: true}}

The canonical backend strings are ``isaac_physx`` and
``isaac_newton_mjwarp``; the corresponding engine strings are
``Isaac/PhysX`` and ``Isaac/Newton/MJWarp``.  In particular, the existing
``prediction_backend: native`` field describes MuJoCo CPU execution mode,
not Newton or MJWarp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


BACKENDS = ("isaac_physx", "isaac_newton_mjwarp")
ENGINE = {"isaac_physx": "Isaac/PhysX",
          "isaac_newton_mjwarp": "Isaac/Newton/MJWarp"}
SOLVER = {"isaac_physx": "physx",
          "isaac_newton_mjwarp": "mujoco_warp"}
PAIRS = {(plant, predictor) for plant in BACKENDS for predictor in BACKENDS}
LOCK_SCALARS = ("reference_id", "reference_catalog_sha256", "motion_sha256",
                "asset_sha256", "plant_mu", "prediction_mu_scenarios",
                "physics_dt_s", "control_dt_s", "plant_seed",
                "total_reference_policy_steps", "core_window_s")
FINGERPRINT_FIELDS = {
    "effective_cem_config_sha256": "effective_cem_config",
    "objective_semantics_sha256": "objective_contract",
    "action_parameterization_sha256": "action_parameterization",
}


def canonical_sha256(value: object) -> str:
    """Hash JSON semantics, independent of whitespace and object key order."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def objective_semantics(contract: object) -> object:
    """Exclude implementation label and backend-specific caveat prose only."""
    if not isinstance(contract, dict):
        return contract
    return {key: value for key, value in contract.items()
            if key not in ("kind", "surrogate_differences")}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _check_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected schema_version 1")
    if not isinstance(manifest.get("comparison_id"), str) or not manifest["comparison_id"]:
        raise ValueError("Missing comparison_id")
    lock = manifest.get("lock")
    if not isinstance(lock, dict) or any(key not in lock for key in
                                         (*LOCK_SCALARS, *FINGERPRINT_FIELDS)):
        raise ValueError("Incomplete locked reference/friction/clock/objective/controller/asset contract")
    if not isinstance(manifest.get("historical_baseline_report"), str):
        raise ValueError("historical_baseline_report must be a path")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != len(PAIRS):
        raise ValueError("Manifest must contain exactly the four predictor x plant cells")
    observed = set()
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("report"), str):
            raise ValueError("Each cell needs a report path")
        pair = (cell.get("plant_backend"), cell.get("predictor_backend"))
        if pair not in PAIRS or pair in observed:
            raise ValueError("Duplicate or unknown predictor x plant cell")
        observed.add(pair)
    if observed != PAIRS:
        raise ValueError("Incomplete predictor x plant matrix")


def _lock_errors(report: dict, lock: dict) -> list[str]:
    errors = []
    for field in LOCK_SCALARS:
        if report.get(field) != lock[field]:
            errors.append(f"{field} differs from locked experiment")
    for fingerprint, field in FINGERPRINT_FIELDS.items():
        value = report.get(field)
        if value is None:
            errors.append(f"{field} is missing")
            continue
        if field == "objective_contract":
            value = objective_semantics(value)
        try:
            actual = canonical_sha256(value)
        except (TypeError, ValueError):
            errors.append(f"{field} is not finite JSON")
        else:
            if actual != lock[fingerprint]:
                errors.append(f"{field} differs from locked experiment")
    return errors


def _backend_errors(report: dict, plant: str, predictor: str,
                    *, historical: bool) -> list[str]:
    if historical:
        errors = []
        if report.get("engine") != "Isaac/PhysX":
            errors.append("historical plant is not Isaac/PhysX")
        if report.get("prediction_engine") != "MuJoCo" or report.get("prediction_backend") not in ("python", "native"):
            errors.append("historical predictor is not the recorded MuJoCo CPU path")
        if not isinstance(report.get("prediction_mujoco_version"), str):
            errors.append("historical MuJoCo CPU version is missing")
        return errors

    errors = []
    for field in ("isaac_lab_version", "isaac_lab_revision"):
        if not isinstance(report.get(field), str) or not report[field]:
            errors.append(f"missing {field} for same-framework comparison")
    for role, backend, engine_key in (("plant", plant, "engine"),
                                      ("predictor", predictor, "prediction_engine")):
        if report.get(f"{role}_physics_backend") != backend:
            errors.append(f"{role}_physics_backend does not match matrix cell")
        if report.get(engine_key) != ENGINE[backend]:
            errors.append(f"{engine_key} is not {ENGINE[backend]}")
        audit = report.get("backend_runtime_audit")
        entry = audit.get(role) if isinstance(audit, dict) else None
        if (not isinstance(entry, dict) or entry.get("backend") != backend
                or entry.get("solver") != SOLVER[backend]
                or entry.get("runtime_verified") is not True):
            errors.append(f"{role} lacks matching runtime backend/solver audit")
    return errors


def _completion_errors(report: dict, folder: Path) -> list[str]:
    errors = []
    if (report.get("status") != "completed" or report.get("truncated_smoke") is not False
            or report.get("core_coverage_complete") is not True
            or report.get("completed_policy_steps") != report.get("total_reference_policy_steps")):
        errors.append("report is not a complete full-horizon core-covered episode")
    for metric in ("core_xy_rmse_m", "core_speed_retention_mean_ratio"):
        value = report.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"missing or nonfinite {metric}")
    timings = report.get("compute_ms_median_p95_max")
    if (not isinstance(timings, list) or len(timings) != 3 or
            any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in timings)):
        errors.append("missing or nonfinite MPC latency quantiles")
    for filename, field in (("rollout.npz", "rollout_sha256"),
                            ("solver_log.json", "solver_log_sha256")):
        target = folder / filename
        expected = report.get(field)
        if not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"missing {field}")
        elif not target.is_file():
            errors.append(f"missing {filename}")
        elif _file_sha256(target) != expected:
            errors.append(f"{filename} SHA-256 mismatch")
    return errors


def _inspect_report(path: Path, lock: dict, plant: str, predictor: str,
                    *, historical: bool = False) -> dict:
    entry = {"report": str(path), "plant_backend": plant,
             "predictor_backend": predictor}
    if not path.is_file():
        return {**entry, "status": "missing", "errors": []}
    try:
        report = _read_json(path)
        errors = (_lock_errors(report, lock)
                  + _backend_errors(report, plant, predictor, historical=historical)
                  + _completion_errors(report, path.parent))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return {**entry, "status": "invalid", "errors": [f"Cannot validate report: {error}"]}
    result = {**entry, "status": "valid" if not errors else "invalid", "errors": errors}
    if not errors:
        result["core_xy_rmse_m"] = report["core_xy_rmse_m"]
        result["core_speed_retention_mean_ratio"] = report["core_speed_retention_mean_ratio"]
        result["compute_ms_median_p95_max"] = report["compute_ms_median_p95_max"]
        if not historical:
            result["isaac_lab_version"] = report["isaac_lab_version"]
            result["isaac_lab_revision"] = report["isaac_lab_revision"]
    return result


def validate_manifest(path: Path) -> dict:
    """Read-only matrix preflight; absent run artifacts remain explicitly missing."""
    path = Path(path).resolve()
    manifest = _read_json(path)
    _check_manifest(manifest)
    lock = manifest["lock"]
    base = path.parent
    historical = _inspect_report((base / manifest["historical_baseline_report"]).resolve(),
                                 lock, "isaac_physx", "mujoco_cpu", historical=True)
    cells = [_inspect_report((base / cell["report"]).resolve(), lock,
                             cell["plant_backend"], cell["predictor_backend"])
             for cell in manifest["cells"]]
    # A PhysX 2.x result and a Newton 3.x result would confound the physics
    # backend with the framework upgrade, even if the task-level locks match.
    identities = {(entry["isaac_lab_version"], entry["isaac_lab_revision"])
                  for entry in cells if entry["status"] == "valid"}
    if len(identities) > 1:
        for entry in cells:
            if entry["status"] == "valid":
                entry["status"] = "invalid"
                entry["errors"].append("Isaac Lab version/revision differs across backend cells")
    entries = [historical, *cells]
    status = ("invalid" if any(entry["status"] == "invalid" for entry in entries)
              else "ready" if all(entry["status"] == "valid" for entry in cells)
              else "incomplete")
    return {"comparison_id": manifest["comparison_id"], "status": status,
            "historical_baseline": historical, "cells": cells,
            "note": "Report provenance and artifact digests checked; self-reported runtime backend audits are not independently re-executed."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-complete", action="store_true",
                        help="Return nonzero when any of the four cells is missing")
    args = parser.parse_args(argv)
    try:
        result = validate_manifest(args.manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        print(json.dumps({"status": "invalid_manifest", "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] == "invalid":
        return 2
    if args.require_complete and result["status"] != "ready":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
