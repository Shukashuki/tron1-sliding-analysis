"""J native angular provenance; no Isaac/GPU import or data mutation."""
from __future__ import annotations
import hashlib
import inspect
import math
from numbers import Real
from pathlib import Path

ARTICULATION_DATA_SHA256 = "bfe53518d1c511455b3538dcee487845ccfe1b62ddbd9c2cd92794137d390994"
ANGULAR_SOURCE = "isaaclab.assets.articulation.articulation_data.ArticulationData.root_com_ang_vel_w"
SOURCE_PROPERTIES = ("root_com_ang_vel_w", "root_com_vel_w")


def require(value, message):
    if not value: raise ValueError(message)


def verify_angular_source_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Require direct native angular source")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == ARTICULATION_DATA_SHA256, "Native angular API source differs")
    return {"path": str(path.resolve()), "sha256": digest}


def verify_angular_data_source(data):
    cls = type(data)
    require(cls.__module__+"."+cls.__name__ == "isaaclab.assets.articulation.articulation_data.ArticulationData",
            "Require native ArticulationData angular semantics")
    source = inspect.getsourcefile(cls)
    require(source is not None, "Native angular source missing")
    for name in SOURCE_PROPERTIES:
        require(isinstance(inspect.getattr_static(cls, name, None), property), "Require pinned native angular properties")
    return verify_angular_source_file(source)


def verify_angular_runtime(audit, *, expected_envs, expected_windows=None):
    require(type(expected_envs) is int and expected_envs > 0, "Invalid angular environment count")
    for key, value in {"angular_source": ANGULAR_SOURCE,
            "angular_source_sha256": ARTICULATION_DATA_SHA256, "angular_source_verified": True,
            "cpu_fixture_only": False, "angular_signal": "root_COM_worldXY_rad_s",
            "angular_num_envs": expected_envs, "angular_all8_actual_substeps": True,
            "angular_start_is_current_physical_state": True, "angular_velocity_is_pose_finite_difference": False,
            "angular_end_state_repeated": False, "angular_cache_same_refresh_not_absolute_time": True,
            "angular_cache_refresh_health": True, "device_finite_angular_health": True}.items():
        require(audit.get(key) == value, "J angular proof differs: "+key)
    path = audit.get("angular_source_path")
    require(isinstance(path, str) and Path(path).is_absolute(), "Angular source path missing")
    windows = audit.get("steps_completed")
    require(type(windows) is int and windows > 0 and windows == audit.get("steps_started")
            == audit.get("unique_reward_windows") == audit.get("angular_unique_reward_windows")
            == audit.get("angular_boundary_samples"), "Angular unique window/start count differs")
    if expected_windows is not None:
        require(type(expected_windows) is int and windows == expected_windows, "Angular expected windows differ")
    require(audit.get("angular_physics_samples") == audit.get("physics_samples") == 8*windows
            and audit.get("angular_environment_physics_samples") == expected_envs*8*windows
            and audit.get("angular_environment_boundary_samples") == expected_envs*windows,
            "Angular native physics/start sample counts differ")
    penalty = audit.get("mean_added_angular_penalty")
    require(isinstance(penalty, Real) and not isinstance(penalty, bool)
            and math.isfinite(float(penalty)) and 0. <= penalty <= .500001,
            "Angular penalty summary invalid")
    return audit
