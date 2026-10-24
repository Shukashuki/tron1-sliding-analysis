"""H native-COM provenance checks; no simulator/GPU import or data mutation."""
from __future__ import annotations
import hashlib
import inspect
from numbers import Real
from pathlib import Path
import math

ARTICULATION_DATA_SHA256 = "bfe53518d1c511455b3538dcee487845ccfe1b62ddbd9c2cd92794137d390994"
VELOCITY_SOURCE = "isaaclab.assets.articulation.articulation_data.ArticulationData.root_com_lin_vel_w"
SOURCE_PROPERTIES = ("root_com_lin_vel_w", "root_com_vel_w")

def require(value, message):
    if not value: raise ValueError(message)

def verify_velocity_source_file(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(), "Require direct existing ArticulationData source")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest==ARTICULATION_DATA_SHA256, "Native COM velocity API source differs")
    return {"path":str(path.resolve()),"sha256":digest}

def verify_velocity_data_source(data):
    cls=type(data)
    require(cls.__module__+"."+cls.__name__=="isaaclab.assets.articulation.articulation_data.ArticulationData",
            "Require actual native ArticulationData COM semantics")
    source=inspect.getsourcefile(cls)
    require(source is not None, "Native COM velocity source file is unavailable")
    for name in SOURCE_PROPERTIES:
        require(isinstance(inspect.getattr_static(cls,name,None),property),
                "Require source-pinned COM velocity properties")
    return verify_velocity_source_file(source)

def verify_velocity_runtime(audit, *, expected_envs, expected_windows=None):
    require(type(expected_envs) is int and expected_envs>0, "Invalid expected velocity environment count")
    for key,value in {
        "velocity_source":VELOCITY_SOURCE, "velocity_source_sha256":ARTICULATION_DATA_SHA256,
        "velocity_source_verified":True, "cpu_fixture_only":False,
        "velocity_signal":"base_COM_worldZ_m_s", "velocity_all8_actual_substeps":True,
        "velocity_is_finite_difference":False, "velocity_end_state_repeated":False,
        "velocity_cache_epoch_same_refresh_not_absolute_sim_time":True,
        "velocity_cache_refresh_health":True, "device_finite_com_vz_health":True,
        "velocity_num_envs":expected_envs,
    }.items():
        require(audit.get(key)==value, "H native velocity proof differs: "+key)
    path=audit.get("velocity_source_path")
    require(isinstance(path,str) and Path(path).is_absolute(), "Native velocity source path missing")
    windows=audit.get("steps_completed")
    require(type(windows) is int and windows>0 and windows==audit.get("steps_started")
            ==audit.get("unique_reward_windows")==audit.get("velocity_unique_reward_windows"),
            "H velocity unique window count differs")
    if expected_windows is not None:
        require(type(expected_windows) is int and expected_windows==windows, "H velocity expected windows differ")
    require(audit.get("velocity_physics_samples")==audit.get("physics_samples")==8*windows
            and audit.get("velocity_environment_samples")==expected_envs*8*windows,
            "H velocity/force native sample counts differ")
    penalty=audit.get("mean_added_velocity_penalty")
    require(isinstance(penalty,Real) and not isinstance(penalty,bool)
            and math.isfinite(float(penalty)) and 0.<=penalty<=.500001,
            "H velocity penalty summary is invalid")
    return audit

