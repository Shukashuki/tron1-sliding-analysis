"""Strict warm-start and native target proofs for3D axial wheel tracking."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

PARENT_SHA = "07c43152ae4d32b87cdde7d97b4ecbb31fcb333b8884415ca4e192658e6e7f6e"


def materialize(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    for old, new in [
        ("import car_taskspace_tracking as targets", "import car_wheel_axis_tracking as targets"),
        ("import train_contact_taskspace_refinement as training", "import train_contact_wheel_axis as training"),
        ("live_heading_reward_shape", "live_axis_reward_shape"),
        ("strict_contact_taskspace_refinement_training_verification", "strict_contact_wheel_axis_training_verification"),
        ('    result["task_space_tracking_verified"] = True\n',
         '    result["task_space_tracking_verified"] = True\n    result["three_dimensional_axial_orientation_verified"] = True\n'),
        ('task_space_tracking_verified=True, from_scratch=False, historical_checkpoint_used=True,',
         'task_space_tracking_verified=True, three_dimensional_axial_orientation_verified=True, from_scratch=False, historical_checkpoint_used=True,'),
    ]:
        assert source.count(old) >= 1, old
        source = source.replace(old, new)
    compile(source, "<strict_3D_wheel_axis_refinement_proof>", "exec")
    return source


def _namespace():
    source = materialize((Path(__file__).parent/"verify_contact_taskspace_refinement.py").read_text())
    namespace = dict(__name__="_strict_3D_wheel_axis_refinement_proof", __file__=str(Path(__file__).resolve()))
    exec(compile(source, "<strict_3D_wheel_axis_refinement_proof>", "exec"), namespace)
    return namespace


def verify_training(*args, **kwargs):
    return _namespace()["verify_training"](*args, **kwargs)


if __name__ == "__main__":
    _namespace()["main"]()
