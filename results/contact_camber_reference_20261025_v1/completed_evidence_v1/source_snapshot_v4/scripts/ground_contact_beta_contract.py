"""Shared G provenance checks; no simulator, policy, weights or GPU imports."""
from __future__ import annotations

import hashlib
from pathlib import Path

BETA_SOURCE_SHA256 = "d5da59c7af58f8a45e3deeb8f69fb0af8d382335e7b0974a5637106fc8c56bb6"
BETA_QUALITY_SOURCE = "training.tron1_drift.sideslip_tracking_unweighted"


def require(value, message):
    if not value: raise ValueError(message)


def verify_beta_source(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Require direct existing beta source file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == BETA_SOURCE_SHA256, "Existing beta source SHA differs")
    return {"path": str(path.resolve()), "sha256": digest}


def verify_beta_runtime(audit, *, expected_envs, expected_windows=None):
    require(type(expected_envs) is int and expected_envs > 0, "Invalid expected environment count")
    for key, expected in {
        "beta_quality_source": BETA_QUALITY_SOURCE, "beta_quality_source_sha256": BETA_SOURCE_SHA256,
        "beta_quality_std_rad": .12, "beta_low_reference_speed_m_s": .3,
        "beta_num_envs": expected_envs, "device_held_reference_phase_health": True,
        "device_low_reference_speed_gate_health": True,
        "beta_quality_source_verified": True, "cpu_fixture_only": False,
        "beta_quality_is_unweighted_existing_term": True,
        "reference_beta_not_zero_beta_is_intended": True,
    }.items():
        require(audit.get(key) == expected, "G beta runtime proof differs: "+key)
    windows = audit.get("steps_completed")
    require(type(windows) is int and windows > 0 and windows == audit.get("steps_started")
            == audit.get("unique_reward_windows") == audit.get("beta_unique_reward_reads"),
            "G beta unique window count differs")
    if expected_windows is not None:
        require(type(expected_windows) is int and windows == expected_windows, "G beta expected window count differs")
    counts = [audit.get(key) for key in ("beta_moving_reference_samples", "beta_low_reference_samples")]
    require(all(type(n) is int and n >= 0 for n in counts) and sum(counts) == expected_envs*windows,
            "G beta moving/low-reference sample accounting differs")
    require(audit.get("beta_environment_reward_samples") == expected_envs*windows,
            "G beta environment reward sample count differs")
    return audit
