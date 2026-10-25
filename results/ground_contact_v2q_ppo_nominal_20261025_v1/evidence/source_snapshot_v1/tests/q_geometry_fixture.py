"""Synthetic geometry provenance only; never imported by production code."""
import hashlib
from pathlib import Path


def geometry_audit_fixture(num_envs):
    root = Path(__file__).resolve().parents[1]/'scripts'
    return dict(kind='readonly_authored_cylinder_gap', verified_environment_count=num_envs,
        actual_wheels_per_environment=2, cpu_fixture_only=False, original_one_cylinder_per_wheel=True,
        no_geometry_or_physics_writes=True, distance_is_authored_geometry_not_native_manifold=True,
        all_environment_frames_checked=True,
        specs=[dict(radius_m=.127,half_width_m=.025,local_center_m=[0.,0.,0.],local_axis=[0.,-1.,0.]) for _ in range(2)],
        native_wheel_offsets=dict(contact_m=[.001,.001],rest_m=[0.,0.]),
        startup_CPU_Torch_max_gap_difference_m=0.,
        source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                      for n in ('wheel_ground_geometry.py','wheel_ground_geometry_torch.py')})
