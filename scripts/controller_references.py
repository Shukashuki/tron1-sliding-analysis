"""Hash-locked car references for separate controller-comparison strata.

No motion data, private filesystem paths, model assets or download logic are
stored here. Reference files remain caller supplied. Primary and supplementary
results must never be pooled into one success statistic.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "config/controller_references.json"
PRIMARY_REFERENCE_ID = "drift_drl_traj4"
SUPPLEMENT_REFERENCE_ID = "doa_sa29_scale2"
SELECTED_REFERENCES = {
    PRIMARY_REFERENCE_ID: ("primary", "large_angle_turn"),
    SUPPLEMENT_REFERENCE_ID: ("separate_supplement", "high_sideslip_s_avoidance"),
}
MOTION_JOINT_NAMES = (
    "abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
    "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint",
)
REFERENCE_JOINT_NAMES = (
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint",
)
MOTION_BODY_NAMES = (
    "base_Link", "abad_L_Link", "hip_L_Link", "knee_L_Link", "wheel_L_Link",
    "abad_R_Link", "hip_R_Link", "knee_R_Link", "wheel_R_Link", "limx_imu",
)


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"Invalid SHA-256 for {name}")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric {name}")
    return float(value)


def _interval(value, name, *, lower, upper):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Expected two-element {name}")
    start, end = (_number(item, name) for item in value)
    if not lower <= start < end <= upper:
        raise ValueError(f"{name} must stay within demonstrated source motion, not synthetic stop/tail")
    return start, end


def load_catalog(path=None):
    """Read and validate the two explicitly selected, separately scored refs."""
    with Path(path or DEFAULT_CATALOG).open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    if (not isinstance(catalog, dict) or catalog.get("schema_version") != 1
            or catalog.get("default_reference_id") != PRIMARY_REFERENCE_ID
            or catalog.get("pooled_success_statistics") is not False
            or catalog.get("heading_separation_is_measured_contact_slip") is not False):
        raise ValueError("Expected selected-reference schema 1 with primary default and no pooled success or measured-slip claim")
    references = catalog.get("references")
    if not isinstance(references, dict) or set(references) != set(SELECTED_REFERENCES):
        raise ValueError("Catalog must contain exactly the selected primary and separate supplement")
    for reference_id, (role, stratum) in SELECTED_REFERENCES.items():
        spec = references[reference_id]
        if (not isinstance(spec, dict) or spec.get("reference_id") != reference_id
                or spec.get("role") != role or spec.get("stratum") != stratum):
            raise ValueError(f"Reference identity/stratum mismatch: {reference_id}")
        source = spec.get("source", {})
        if (source.get("kind") != "CARLA_simulation" or source.get("is_physical_car_capture") is not False
                or source.get("clock") != "reconstructed_estimate_not_original_timestamps"):
            raise ValueError("Car provenance must distinguish simulation and reconstructed timestamps")
        csv_path = source.get("csv_path")
        if (not isinstance(csv_path, str) or PurePosixPath(csv_path).is_absolute()
                or ".." in PurePosixPath(csv_path).parts or "\\" in csv_path):
            raise ValueError("Source CSV path must be repository-relative, not a private local path")
        for name in ("motion_npz_sha256", "robot_reference_npz_sha256"):
            _digest(spec.get(name), name)
        _digest(source.get("csv_sha256"), "source CSV")
        for name in ("frames", "source_demo_frames", "synthetic_recovery_frames", "stationary_tail_frames"):
            value = spec.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Expected positive integer {name}")
        frames, fps = spec["frames"], _number(spec.get("fps"), "fps")
        if fps != 50.0 or spec["source_demo_frames"] + spec["synthetic_recovery_frames"] + spec["stationary_tail_frames"] != frames:
            raise ValueError("Expected 50 Hz and disjoint demonstrated/recovery/tail frames")
        span = _number(spec.get("sample_span_s"), "sample_span_s")
        if not math.isclose(span, (frames - 1) / fps, abs_tol=1e-9):
            raise ValueError("sample_span_s must describe samples, not frame-count playback duration")
        demo_end = (spec["source_demo_frames"] - 1) / fps
        start, end = _interval(spec.get("source_demo_interval_s"), "source_demo_interval_s", lower=0., upper=span)
        if start != 0. or not math.isclose(end, demo_end, abs_tol=1e-9):
            raise ValueError("source_demo_interval_s must exclude synthetic stop/tail")
        _interval(spec.get("evaluation_core_interval_s"), "evaluation_core_interval_s", lower=start, upper=end)
        events = spec.get("supplementary_event_windows_s")
        if not isinstance(events, list):
            raise ValueError("supplementary_event_windows_s must be explicit")
        for event in events:
            _interval(event, "supplementary event window", lower=start, upper=end)
    return copy.deepcopy(catalog)


def get_reference(reference_id=PRIMARY_REFERENCE_ID, *, catalog_path=None):
    """Return an independent spec; omission pins the existing traj4 primary."""
    catalog = load_catalog(catalog_path)
    if reference_id not in catalog["references"]:
        raise ValueError(f"Unknown selected reference_id: {reference_id!r}; choose {tuple(SELECTED_REFERENCES)}")
    spec = copy.deepcopy(catalog["references"][reference_id])
    spec["pooled_success_statistics"] = False
    spec["heading_separation_definition"] = catalog["heading_separation_definition"]
    spec["heading_separation_is_measured_contact_slip"] = False
    spec["wheel_heading_definition"] = catalog["wheel_heading_definition"]
    spec["motion_contract"] = copy.deepcopy(catalog["motion_contract"])
    return spec


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_motion_identity(motion_file, reference_id=PRIMARY_REFERENCE_ID, *, catalog_path=None):
    """Check selected motion BYTE IDENTITY only, before starting a simulator.

    This does not load NPZ arrays, verify the paired robot reference, validate
    assets or establish physical feasibility. Use verify_reference_pair for
    the stronger two-artifact structural contract.
    """
    spec = get_reference(reference_id, catalog_path=catalog_path)
    if _sha256(motion_file) != spec["motion_npz_sha256"]:
        raise ValueError(f"motion_npz_sha256 mismatch for selected reference_id={reference_id}")
    spec["verification_scope"] = "motion_byte_identity_only_not_reference_pair_or_asset_validation"
    return copy.deepcopy(spec)


def _numeric(archive, name, shape):
    if name not in archive:
        raise ValueError(f"Reference artifact lacks {name}")
    value = archive[name]
    if value.shape != shape or value.dtype.kind not in "biuf" or not np.isfinite(value).all():
        raise ValueError(f"Invalid reference array {name}: expected finite shape {shape}")
    return value


def _names(archive, name, expected):
    if name not in archive or archive[name].shape != (len(expected),) or tuple(archive[name].tolist()) != expected:
        raise ValueError(f"Unexpected named {name} order in reference artifact")


def _clock_and_phases(archive, spec):
    frames = spec["frames"]
    if "fps" not in archive:
        raise ValueError("Reference artifact lacks fps")
    raw_fps = archive["fps"]
    if raw_fps.size != 1 or raw_fps.dtype.kind not in "iuf" or float(raw_fps.reshape(-1)[0]) != spec["fps"]:
        raise ValueError("Reference artifact fps differs from selected 50 Hz contract")
    clock = _numeric(archive, "time_s", (frames,))
    if not np.allclose(clock, np.arange(frames) / spec["fps"], rtol=0., atol=1e-7):
        raise ValueError("Reference time_s must be the complete uniform 50 Hz sample clock")
    phase = _numeric(archive, "source_phase", (frames,))
    expected = np.repeat([0, 1, 2], [spec["source_demo_frames"], spec["synthetic_recovery_frames"], spec["stationary_tail_frames"]])
    if not np.array_equal(phase, expected):
        raise ValueError("source_phase mixes demonstrated frames with synthetic stop/tail")


def verify_reference_pair(motion_file, reference_file, reference_id=PRIMARY_REFERENCE_ID, *, catalog_path=None):
    """Verify byte identity AND simulator-facing arrays; return the selected spec.

    Export-model hashes are provenance, not a claim that a runtime asset has
    been verified. This function never opens or rewrites a robot model.
    """
    spec = get_reference(reference_id, catalog_path=catalog_path)
    for path, key in ((motion_file, "motion_npz_sha256"), (reference_file, "robot_reference_npz_sha256")):
        if _sha256(path) != spec[key]:
            raise ValueError(f"{key} mismatch for selected reference_id={reference_id}")
    frames = spec["frames"]
    with np.load(motion_file, allow_pickle=False) as motion:
        _clock_and_phases(motion, spec)
        _names(motion, "joint_names", MOTION_JOINT_NAMES)
        _names(motion, "body_names", MOTION_BODY_NAMES)
        for name, shape in {
            "joint_pos": (frames, 8), "joint_vel": (frames, 8),
            "body_pos_w": (frames, 10, 3), "body_quat_w": (frames, 10, 4),
            "body_lin_vel_w": (frames, 10, 3), "body_ang_vel_w": (frames, 10, 3),
            "qpos_mujoco": (frames, 15), "qvel_mujoco": (frames, 14),
        }.items():
            _numeric(motion, name, shape)
        if not np.allclose(np.linalg.norm(motion["body_quat_w"], axis=-1), 1., atol=1e-5):
            raise ValueError("Motion body quaternions must be normalized wxyz")
        for name, expected in {
            "joint_tracking_mask": [True] * 6 + [False] * 2,
            "body_orientation_tracking_mask": [True, True, True, True, False, True, True, True, False, True],
            "body_angular_velocity_tracking_mask": [True, True, True, True, False, True, True, True, False, True],
        }.items():
            mask = _numeric(motion, name, (len(expected),))
            if not np.array_equal(mask, expected):
                raise ValueError(f"Wheel spin/orientation tracking masks differ: {name}")
    with np.load(reference_file, allow_pickle=False) as reference:
        _clock_and_phases(reference, spec)
        _names(reference, "joint_names", REFERENCE_JOINT_NAMES)
        for name, shape in {
            "qpos": (frames, 15), "qvel": (frames, 14),
            "joint_positions_rad": (frames, 8), "joint_velocities_rad_s": (frames, 8),
            "root_target_position_m": (frames, 3), "root_target_quat_wxyz": (frames, 4),
            "wheel_center_positions_world_m": (frames, 2, 3),
        }.items():
            _numeric(reference, name, shape)
        if not np.allclose(np.linalg.norm(reference["root_target_quat_wxyz"], axis=-1), 1., atol=1e-5):
            raise ValueError("Reference root quaternions must be normalized wxyz")
    return copy.deepcopy(spec)
