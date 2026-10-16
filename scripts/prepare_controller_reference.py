"""Reproduce selected car references offline with explicitly supplied assets.

Uses only pinned source CSVs/code and local licensed model/URDF files. No CARLA,
network requests, animation, training or time-stepped dynamics are executed.
MuJoCo FK, Jacobians and the historical bounded kinematic solve are reused.
Output hashes are compared to the catalog, never silently substituted into it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
import zlib

import numpy as np

from controller_references import (ROOT, PRIMARY_REFERENCE_ID, SUPPLEMENT_REFERENCE_ID,
                                   SELECTED_REFERENCES, get_reference, verify_reference_pair)


DEFAULT_SOURCE_ROOT = ROOT / "upstream/tron1-learning-from-human"
PINNED_SOURCE_FILES = {
    "scripts/capture_car_drift.py": "c54779a2754f308e9e63f3079ddf80c9650fc297ea525da09c1bafbf10870adc",
    "scripts/prepare_car_drift_reference.py": "04fab9f4178b66e302702563ba4cf30001dfb8dbbea9f2a58f9e1a6f0a155f0e",
    "scripts/prepare_drift_drl_turn_reference.py": "3c89ed739b5bc7ad7acba33fc82abc61ca94246d17b2705e920e865336719ac9",
    "scripts/export_tracking_motion.py": "f959db0b0c0080b8a9ee2b20e3b274003dbfbea4c81907c9ab7e69fde61662ca",
    "config/drift_sources.json": "d2fbbb27b96e57b7208c1a6f588d0dee64b290fc480f845a82bfc87d5a13e0fb",
    "config/drift_turn_source.json": "5a2130b616f4d800934a21fa775877c0f38459d4cd07bf22532578fd39f0904d",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_file_hash(path, expected, label):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _git(repository, *arguments):
    result = subprocess.run(["git", "-C", str(repository), *arguments],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Cannot verify pinned source Git checkout at {repository}")
    return result.stdout


def verify_source_inputs(reference_id=PRIMARY_REFERENCE_ID, *, source_root=None):
    """Verify local source checkout, CSV, license and adaptation config only."""
    spec = get_reference(reference_id)
    source_root = Path(source_root or DEFAULT_SOURCE_ROOT).resolve()
    source = spec["source"]
    repository = ROOT / "upstream" / ("drift_drl" if reference_id == PRIMARY_REFERENCE_ID else "DOA")
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    if head != source["commit"]:
        raise ValueError("Source submodule HEAD differs from the selected reference commit")
    csv_file = repository / source["csv_path"]
    verify_file_hash(csv_file, source["csv_sha256"], "source CSV")
    payload = csv_file.read_bytes()
    if payload != _git(repository, "show", f"{head}:{source['csv_path']}"):
        raise ValueError("Source CSV differs from its pinned Git blob")
    license_file = repository / "LICENSE"
    license_payload = license_file.read_bytes()
    if license_payload != _git(repository, "show", f"{head}:LICENSE"):
        raise ValueError("Source license differs from its pinned Git blob")
    config_name = "drift_turn_source.json" if reference_id == PRIMARY_REFERENCE_ID else "drift_sources.json"
    config_file = source_root / "config" / config_name
    verify_file_hash(config_file, PINNED_SOURCE_FILES[f"config/{config_name}"], "source configuration")
    config = json.loads(config_file.read_text())
    if config["repository"] != source["repository"] or config["commit"] != head:
        raise ValueError("Source configuration and selected catalog provenance differ")
    if reference_id == PRIMARY_REFERENCE_ID:
        if config["path"] != source["csv_path"] or config["sha256"] != source["csv_sha256"]:
            raise ValueError("Primary source configuration and selected CSV differ")
    else:
        entries = [entry for entry in config["files"] if entry["id"] == "sa_traj29"]
        if (len(entries) != 1 or entries[0]["path"] != source["csv_path"]
                or entries[0]["sha256"] != source["csv_sha256"]):
            raise ValueError("Supplementary source configuration and selected CSV differ")
        license_entry = next(entry for entry in config["files"] if entry["id"] == "license")
        verify_file_hash(license_file, license_entry["sha256"], "source license")
    return {"spec": spec, "source_root": source_root, "repository": repository,
            "head": head, "csv_file": csv_file, "payload": payload,
            "license_file": license_file, "license_payload": license_payload,
            "configuration_file": config_file, "configuration": config}


@contextmanager
def _source_modules(source_root):
    """Bind pinned sibling imports process-locally, restoring previous modules."""
    source_root = Path(source_root)
    for name, expected in PINNED_SOURCE_FILES.items():
        verify_file_hash(source_root / name, expected, f"pinned {name}")
    names = ("export_tracking_motion", "capture_car_drift",
             "prepare_car_drift_reference", "prepare_drift_drl_turn_reference")
    previous = {name: sys.modules.get(name) for name in names}
    modules = {}
    try:
        for name in names:
            path = source_root / "scripts" / (name + ".py")
            module_spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
            modules[name] = module
        yield SimpleNamespace(export=modules[names[0]], capture=modules[names[1]],
                              car=modules[names[2]], turn=modules[names[3]])
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def adaptation_options(spec, configuration):
    """No new tuning: use the historical source-space and robot-time mapping."""
    if spec["reference_id"] == PRIMARY_REFERENCE_ID:
        options = {"spatial_scale": configuration["spatial_scale"],
                   "time_scale": configuration["time_scale"],
                   "root_height": configuration["root_height_m"],
                   "recovery_s": configuration["recovery_s"],
                   "stationary_tail_s": configuration["stationary_tail_s"]}
    else:
        options = {"spatial_scale": spec["spatial_scale"], "time_scale": spec["time_scale"],
                   "root_height": .75, "recovery_s": 1.5, "stationary_tail_s": .5}
    options["output_fps"] = spec["fps"]
    if (options["spatial_scale"] != spec["spatial_scale"] or options["time_scale"] != spec["time_scale"]
            or options["recovery_s"] * spec["fps"] != spec["synthetic_recovery_frames"]
            or options["stationary_tail_s"] * spec["fps"] != spec["stationary_tail_frames"]):
        raise ValueError("Adaptation options differ from selected reference timing/scaling")
    return options


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _offline_doa_capture(inputs, output, modules):
    """Create only the selected compatible capture, without downloader/preview."""
    root = output / "source_capture"
    raw = root / "raw"
    captured = root / "sa_traj29"
    raw.mkdir(parents=True)
    captured.mkdir()
    (raw / inputs["csv_file"].name).write_bytes(inputs["payload"])
    (raw / "LICENSE").write_bytes(inputs["license_payload"])
    arrays = modules.capture.decode_csv(inputs["payload"])
    capture_path = captured / "captured_source.npz"
    with capture_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    source = inputs["spec"]["source"]
    audit = {"schema_version": 1, "status": "offline_selected_source_capture",
             "source_repository": source["repository"], "source_commit": inputs["head"],
             "configuration_sha256": sha256(inputs["configuration_file"]),
             "capture_script_sha256": sha256(inputs["source_root"] / "scripts/capture_car_drift.py"),
             "selected_for_export": "sa_traj29", "network_requests": 0,
             "animation_generated": False, "simulation_steps": 0,
             "captures": {"sa_traj29": {"source_path": source["csv_path"],
                                        "source_sha256": source["csv_sha256"]}}}
    _write_json(root / "source_audit.json", audit)
    # Reuse the pinned independent CSV/NPZ/audit verification, not just a copy.
    return modules.car.load_verified_source(capture_path)


def _versions():
    versions = {"python": platform.python_version(), "zlib": zlib.ZLIB_RUNTIME_VERSION}
    for package in ("numpy", "scipy", "mujoco"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def prepare_reference(*, model, urdf, output_dir, reference_id=PRIMARY_REFERENCE_ID, source_root=None):
    """Create a new export; keep mismatching output and report it honestly."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Preserving existing reference output: " + str(output))
    inputs = verify_source_inputs(reference_id, source_root=source_root)
    spec = inputs["spec"]
    model, urdf = Path(model).resolve(), Path(urdf).resolve()
    model_sha = verify_file_hash(model, spec["motion_contract"]["export_model_sha256"], "licensed export MJCF")
    urdf_sha = verify_file_hash(urdf, spec["motion_contract"]["export_urdf_sha256"], "licensed original URDF")
    options = adaptation_options(spec, inputs["configuration"])
    report = {"schema_version": 1, "reference_id": reference_id, "stratum": spec["stratum"],
              "status": "initializing", "pooled_success_statistics": False,
              "network_requests": 0, "simulation_steps": 0, "animation_generated": False,
              "physics_validated": False, "policy_trained": False,
              "model_file": str(model), "model_sha256": model_sha,
              "urdf_file": str(urdf), "urdf_sha256": urdf_sha,
              "wrapper_sha256": sha256(Path(__file__)), "versions": _versions(),
              "pinned_source_files_sha256": dict(PINNED_SOURCE_FILES),
              "source_repository": spec["source"]["repository"], "source_commit": inputs["head"],
              "source_csv_path": spec["source"]["csv_path"], "source_csv_sha256": spec["source"]["csv_sha256"],
              "source_license_sha256": hashlib.sha256(inputs["license_payload"]).hexdigest(),
              "adaptation_options": options,
              "expected_hashes": {"motion_npz_sha256": spec["motion_npz_sha256"],
                                  "robot_reference_npz_sha256": spec["robot_reference_npz_sha256"]},
              "warnings": ["NumPy/SciPy/MuJoCo or compression-version changes can change bytes or kinematic solve rounding. A mismatch is not permission to replace the catalog lock.",
                           "Source clocks are reconstructed and source cars are CARLA simulations, not physical mocap.",
                           "Wheel rolling-plane separation is geometric, not measured contact-patch slip."]}
    with _source_modules(inputs["source_root"]) as modules:
        output.mkdir(parents=True)
        try:
            (output / "source_LICENSE").write_bytes(inputs["license_payload"])
            if reference_id == PRIMARY_REFERENCE_ID:
                full = modules.turn.decode_csv(inputs["payload"])
                source, source_evidence = modules.turn.selected_source(full, inputs["configuration"])
                provenance = {"source_repository": spec["source"]["repository"],
                              "source_commit": inputs["head"], "source_csv_path": spec["source"]["csv_path"],
                              "source_csv_sha256": spec["source"]["csv_sha256"],
                              "source_configuration_sha256": sha256(inputs["configuration_file"]),
                              "source_is_physical_car_capture": False,
                              "source_clock_status": spec["source"]["clock"],
                              "source_row_interval_inclusive": spec["source"]["row_interval_inclusive"]}
                (output / "source.csv").write_bytes(inputs["payload"])
                report["source_selection_evidence"] = source_evidence
            else:
                source, provenance = _offline_doa_capture(inputs, output, modules)
                report["capture_audit_note"] = "New offline selected-source audit; not a byte copy of the historical multi-source/video audit. Audit paths/hashes do not enter the motion arrays."
            intent = modules.car.adapt_source(source, **options)
            if reference_id == PRIMARY_REFERENCE_ID:
                intent["source_row_numbers"] += inputs["configuration"]["source_row_start_inclusive"]
            arrays, diagnostics = modules.car.solve_reference(intent, model_path=model)
            if reference_id == PRIMARY_REFERENCE_ID:
                report["exported_reference_turn_evidence"] = modules.turn.reference_turn_evidence(arrays, inputs["configuration"])
            robot_path = output / "robot_reference.npz"
            with robot_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            exported, metadata = modules.export.convert_motion(
                robot_path, model_path=model, urdf_path=urdf, output_fps=50.,
                prepend_s=0., append_s=0., z_offset=0.)
            for key in ("demonstration_time_s_estimated", "source_phase", "root_yaw_rad"):
                if len(exported["time_s"]) != len(arrays[key]):
                    raise ValueError("Exporter changed the declared 50 Hz source-phase clock")
                exported[key] = arrays[key]
            if len(exported["time_s"]) != spec["frames"]:
                raise ValueError("Regenerated frame count differs from selected reference")
            motion_path = output / "motion.npz"
            with motion_path.open("xb") as stream:
                np.savez_compressed(stream, **exported)
            metadata.update(reference_id=reference_id, source_car_provenance=provenance,
                            upstream_task_adaptation=intent["metadata"], recovery_is_demonstrated=False,
                            source_controls_are_robot_targets=False,
                            wheel_spin_reference="Geometric rolling-direction contact Jacobian; lateral slip retained; wheel spin unobserved and tracking masked",
                            output_sha256=sha256(motion_path))
            metadata["warnings"] = [warning for warning in metadata.get("warnings", [])
                                    if not warning.startswith(("No coordinate reset", "Endpoint holds"))]
            metadata["warnings"].append("Explicit 0.08x space/2x time car adaptation and synthetic recovery/tail; no dynamics validation.")
            _write_json(output / "motion.json", metadata)
            actual = {"motion_npz_sha256": sha256(motion_path), "robot_reference_npz_sha256": sha256(robot_path)}
            matches = {name: value == report["expected_hashes"][name] for name, value in actual.items()}
            matched = all(matches.values())
            if matched:
                verify_reference_pair(motion_path, robot_path, reference_id)
            report.update(status="hash_matched" if matched else "hash_mismatch", actual_hashes=actual,
                          hash_matches=matches, catalog_lock_matched=matched,
                          frames=len(exported["time_s"]), fps=50., sample_span_s=float(exported["time_s"][-1]),
                          provenance=provenance, calibration_and_adaptation=intent["metadata"], diagnostics=diagnostics)
            adaptation = {**report, "robot_reference_sha256": actual["robot_reference_npz_sha256"],
                          "motion_sha256": actual["motion_npz_sha256"]}
            _write_json(output / "adaptation.json", adaptation)
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            _write_json(output / "reproduction_report.json", report)
            raise
    _write_json(output / "reproduction_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-id", choices=tuple(SELECTED_REFERENCES), default=PRIMARY_REFERENCE_ID)
    parser.add_argument("--model", type=Path, required=True, help="Explicit licensed export MJCF matching the catalog hash")
    parser.add_argument("--urdf", type=Path, required=True, help="Explicit original unmerged URDF matching the catalog hash")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing paths are preserved")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
                        help="Pinned TRON1 source checkout; car CSVs come from this repo's upstream submodules")
    args = parser.parse_args(argv)
    report = prepare_reference(reference_id=args.reference_id, model=args.model, urdf=args.urdf,
                               output_dir=args.output_dir, source_root=args.source_root)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["catalog_lock_matched"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
