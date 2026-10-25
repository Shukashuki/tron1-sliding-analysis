"""Read-only native point separations plus authored-wheel geometry before steps.

The point API's manifold/separation and the current-pose authored support gap
are distinct signals. Missing native points remain missing, never zero gap.
"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import wheel_ground_geometry as geometry

WHEELS = ('wheel_L_Link', 'wheel_R_Link')
SENSORS = ('tire_L_ground', 'tire_R_ground')


def plain(value): return value.detach().cpu().numpy().copy()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContactManifoldProbe:
    def __init__(self, env, geometry_rows):
        assert env.num_envs == 1 and env.physics_dt == .0025 and env.step_dt == .02
        assert len(geometry_rows) == 2 and all(r['level'] == 'authored' for r in geometry_rows)
        self.env, self.robot, self.geometry_rows = env, env.scene['robot'], geometry_rows
        self.ids = [self.robot.body_names.index(n) for n in WHEELS]
        self.sensors = [env.scene.sensors[n] for n in SENSORS]
        for sensor in self.sensors:
            assert sensor.contact_physx_view.max_contact_data_count == 128
        self.sdk = inspect.getfile(type(self.sensors[0].contact_physx_view))
        self.sources = {str(Path(p).resolve()): sha(p) for p in (__file__, geometry.__file__, self.sdk)}
        self.rows = []
        self.original_write = env.scene.write_data_to_sim
        env.scene.write_data_to_sim = self.write

    def write(self, *args, **kwargs):
        counter = int(self.env._sim_step_counter)
        if counter == 0: return self.original_write(*args, **kwargs)
        assert counter == len(self.rows)+1, 'Consecutive physical pre-write samples required'
        data = self.robot.data
        pos = plain(data.body_link_pos_w[0, self.ids]); quat = plain(data.body_link_quat_w[0, self.ids])
        com = plain(data.body_com_pos_w[0, self.ids]); vel = plain(data.body_com_lin_vel_w[0, self.ids])
        omega = plain(data.body_com_ang_vel_w[0, self.ids])
        centers, axes, low, gaps, low_velocity = [], [], [], [], []
        counts, matrices, sums = [], [], []
        forces = np.full((2, 128), np.nan); points = np.full((2, 128, 3), np.nan)
        normals = np.full((2, 128, 3), np.nan); distances = np.full((2, 128), np.nan)
        point_velocity = np.full((2, 128, 3), np.nan)
        for k, (row, sensor) in enumerate(zip(self.geometry_rows, self.sensors, strict=True)):
            assert row['body'].endswith('/'+WHEELS[k])
            matrix = np.asarray(row['original_cylinder_to_body_matrix'])
            np.testing.assert_allclose(matrix[:3, :3]@matrix[:3, :3].T, np.eye(3), rtol=0, atol=1e-6)
            local_axis = matrix[{'X': 0, 'Y': 1, 'Z': 2}[row['original_axis']], :3]
            center = pos[k]+geometry.rotate(quat[k], matrix[3, :3])
            axis = geometry.rotate(quat[k], local_axis)
            surface = geometry.cylinder_support(center, axis, row['radius_m'], row['total_width_m']/2.)
            centers.append(center); axes.append(axis); low.append(surface['authored_lowest_point_w_m'])
            gaps.append(surface['authored_signed_gap_m'])
            low_velocity.append(geometry.point_velocity(low[-1], com[k], vel[k], omega[k]))
            view = sensor.contact_physx_view
            values = tuple(map(plain, view.get_contact_data(self.env.physics_dt)))
            assert len(values) == 6
            f, p, n, d, c, start = values
            assert f.shape == d.shape == (128, 1) and p.shape == n.shape == (128, 3)
            assert c.shape == start.shape == (1, 1)
            count, begin = int(c[0, 0]), int(start[0, 0]); assert 0 <= count < 128
            assert count == 0 or (0 <= begin and begin+count <= 128)
            selected = slice(begin, begin+count) if count else slice(0, 0)
            forces[k, :count] = f[selected, 0]; points[k, :count] = p[selected]
            normals[k, :count] = n[selected]; distances[k, :count] = d[selected, 0]
            if count:
                assert all(np.isfinite(v).all() for v in (f[selected], p[selected], n[selected], d[selected]))
                point_velocity[k, :count] = geometry.point_velocity(p[selected], com[k], vel[k], omega[k])
            sums.append((f[selected]*n[selected]).sum(axis=0))
            direct = plain(view.get_contact_force_matrix(self.env.physics_dt)); assert direct.shape == (1, 1, 3)
            matrices.append(direct[0, 0]); counts.append(count)
        self.rows.append(dict(env_counter=counter, native_counter_before=int(self.env.sim.current_time_step_index),
            wheel_link_position_w_m=pos, wheel_link_quaternion_wxyz=quat, wheel_COM_position_w_m=com,
            wheel_COM_velocity_w_m_s=vel, wheel_angular_velocity_w_rad_s=omega,
            cylinder_center_w_m=np.asarray(centers), cylinder_axis_w=np.asarray(axes),
            authored_lowest_point_w_m=np.asarray(low), authored_signed_gap_m=np.asarray(gaps),
            authored_lowest_point_velocity_w_m_s=np.asarray(low_velocity),
            native_point_count=np.asarray(counts), native_point_force_n=forces,
            native_point_position_w_m=points, native_point_normal_w=normals,
            native_point_separation_m=distances, native_point_velocity_w_m_s=point_velocity,
            native_point_force_sum_w_n=np.asarray(sums), native_direct_matrix_w_n=np.asarray(matrices)))
        return self.original_write(*args, **kwargs)

    def save(self, output):
        assert len(self.rows) == 2792 and self.env.scene.write_data_to_sim == self.write
        assert all(sha(p) == digest for p, digest in self.sources.items())
        arrays = {key: np.asarray([r[key] for r in self.rows]) for key in self.rows[0]}
        arrays['time_before_step_s'] = np.arange(2792)*.0025
        path = output/'manifold_trajectory.npz'
        np.savez_compressed(path, **arrays)  # Keep full evidence before numeric assertions.
        with np.load(output/'tire_trajectory.npz', allow_pickle=False) as z: tire = {k:z[k] for k in z.files}
        with np.load(output/'physics_trajectory.npz', allow_pickle=False) as z: raw = {k:z[k] for k in z.files}
        pair_error = float(np.abs(arrays['native_point_force_sum_w_n']-arrays['native_direct_matrix_w_n']).max())
        normal = arrays['native_direct_matrix_w_n'][..., 2]
        legacy = raw['wheel_ground_contact_force_w_n'][:-1, 0, :, 2]
        audit = dict(kind='K_readonly_PhysX_native_contact_manifold_probe', native_pre_step_samples=2792,
            absolute_native_counter_at_first_sample=int(arrays['native_counter_before'][0]),
            parent_clock_is_relative_to_initial_recording_sample=True,
            initial_and_final_physics_states_retained_in_parent_recording=True, padded_inactive_points_are_NaN=True,
            max_point_to_matrix_vector_error_n=pair_error,
            max_matrix_to_parent_pre_step_Fz_error_n=float(np.abs(normal-legacy).max()),
            source_sha256=self.sources, native_trace_sha256=sha(path), geometry=self.geometry_rows,
            authored_geometry_gap_is_not_native_manifold_separation=True,
            native_contact_report_geometry_is_backend_timing_not_resimulated_post_pose=True,
            body_state_writes_action_reward_model_force_changes=False, task_success_claimed=False)
        (output/'manifold_numeric_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        np.testing.assert_array_equal(arrays['env_counter'], np.arange(1, 2793))
        np.testing.assert_array_equal(arrays['native_counter_before']-arrays['native_counter_before'][0], raw['simulation_step_counter'][:-1])
        np.testing.assert_allclose(arrays['native_point_force_sum_w_n'], arrays['native_direct_matrix_w_n'], rtol=1e-5, atol=.001)
        np.testing.assert_allclose(normal, tire['preceding_normal_load_n'], rtol=1e-5, atol=.001)
        np.testing.assert_allclose(normal, legacy, rtol=1e-5, atol=.001)
        np.testing.assert_allclose(arrays['wheel_link_position_w_m'], raw['wheel_position_m'][:-1, 0], rtol=0, atol=1e-7)
        return audit
