"""Actual post-forward MuJoCo manifold and authored cylinder geometry; read only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import wheel_ground_geometry as geometry

ACTIVE = None


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContactManifoldProbe:
    def __init__(self, measured):
        self.measured = measured; self.rows = []
        m, mj = measured.model, measured.mj
        assert m.opt.timestep == .0025
        self.geoms, self.geometry = [], []
        for body in measured.wheels:
            ids = [i for i in range(m.ngeom) if m.geom_bodyid[i] == body and (m.geom_contype[i] or m.geom_conaffinity[i])]
            assert len(ids) == 1 and m.geom_type[ids[0]] == mj.mjtGeom.mjGEOM_CYLINDER
            index = ids[0]; self.geoms.append(index)
            self.geometry.append(dict(body=m.body(int(body)).name, geom_id=index,
                radius_m=float(m.geom_size[index, 0]), half_width_m=float(m.geom_size[index, 1]),
                geom_body_position_m=m.geom_pos[index].tolist(), geom_body_quaternion_wxyz=m.geom_quat[index].tolist()))

    def observe(self):
        state = self.measured; m, d, mj = state.model, state.data, state.mj
        centers, axes, lows, gaps, low_vels, coms, velocities, omegas = [], [], [], [], [], [], [], []
        for body, geom in zip(state.wheels, self.geoms, strict=True):
            center = d.geom_xpos[geom].copy(); axis = d.geom_xmat[geom].reshape(3, 3)[:, 2].copy()
            com = d.xipos[body].copy(); jp, jr = np.empty((3, m.nv)), np.empty((3, m.nv))
            mj.mj_jacBodyCom(m, d, jp, jr, int(body)); velocity, omega = jp@d.qvel, jr@d.qvel
            surface = geometry.cylinder_support(center, axis, m.geom_size[geom, 0], m.geom_size[geom, 1])
            centers.append(center); axes.append(axis); coms.append(com); velocities.append(velocity); omegas.append(omega)
            lows.append(surface['authored_lowest_point_w_m']); gaps.append(surface['authored_signed_gap_m'])
            low_vels.append(geometry.point_velocity(lows[-1], com, velocity, omega))
        per_side = [[], []]
        for index in range(d.ncon):
            contact = d.contact[index]; b1, b2 = m.geom_bodyid[[contact.geom1, contact.geom2]]
            if 0 not in (b1, b2): continue
            for side, body in enumerate(state.wheels):
                if body not in (b1, b2): continue
                wrench = np.zeros(6); mj.mj_contactForce(m, d, index, wrench)
                normal = contact.frame[:3].copy()*(1. if b2 == body else -1.)
                point = contact.pos.copy()
                per_side[side].append((float(wrench[0]), point, normal, float(contact.dist),
                    geometry.point_velocity(point, coms[side], velocities[side], omegas[side])))
        counts = np.asarray([len(v) for v in per_side]); assert (counts < 128).all()
        forces = np.full((2, 128), np.nan); points = np.full((2, 128, 3), np.nan)
        normals = np.full((2, 128, 3), np.nan); distances = np.full((2, 128), np.nan)
        point_velocity = np.full((2, 128, 3), np.nan); summed = np.zeros((2, 3))
        for side, rows in enumerate(per_side):
            for i, (force, point, normal, distance, v) in enumerate(rows):
                forces[side, i] = force; points[side, i] = point; normals[side, i] = normal
                distances[side, i] = distance; point_velocity[side, i] = v
                summed[side] += force*normal
        self.rows.append(dict(time_s=float(d.time), wheel_link_position_w_m=d.xpos[state.wheels].copy(),
            wheel_link_quaternion_wxyz=d.xquat[state.wheels].copy(), wheel_COM_position_w_m=np.asarray(coms),
            wheel_COM_velocity_w_m_s=np.asarray(velocities), wheel_angular_velocity_w_rad_s=np.asarray(omegas),
            cylinder_center_w_m=np.asarray(centers), cylinder_axis_w=np.asarray(axes),
            authored_lowest_point_w_m=np.asarray(lows), authored_signed_gap_m=np.asarray(gaps),
            authored_lowest_point_velocity_w_m_s=np.asarray(low_vels), native_point_count=counts,
            native_point_force_n=forces, native_point_position_w_m=points, native_point_normal_w=normals,
            native_point_separation_m=distances, native_point_velocity_w_m_s=point_velocity,
            native_point_force_sum_w_n=summed))

    def save(self, output, raw):
        assert len(self.rows) == 2793
        arrays = {key: np.asarray([r[key] for r in self.rows]) for key in self.rows[0]}
        path = output/'manifold_trajectory.npz'; np.savez_compressed(path, **arrays)
        error = float(np.abs(arrays['native_point_force_sum_w_n']-raw['wheel_ground_contact_force_w_n']).max())
        audit = dict(kind='K_readonly_MuJoCo_native_contact_manifold_probe', native_states=2793,
            padded_inactive_points_are_NaN=True, max_point_to_parent_force_vector_error_n=error,
            native_trace_sha256=sha(path), geometry=self.geometry,
            native_points_source='actual data.contact after the existing mj_step + mj_forward and before parent record',
            no_new_step_forward_or_state_write=True, authored_geometry_gap_is_not_native_manifold_separation=True,
            body_state_writes_action_reward_model_force_changes=False, task_success_claimed=False)
        (output/'manifold_numeric_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        np.testing.assert_allclose(arrays['time_s'], raw['time_s'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(arrays['wheel_link_position_w_m'], raw['wheel_pos'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(arrays['native_point_force_sum_w_n'], raw['wheel_ground_contact_force_w_n'], rtol=1e-5, atol=.001)
        return audit


def observe(measured):
    global ACTIVE
    if ACTIVE is None: ACTIVE = ContactManifoldProbe(measured)
    assert ACTIVE.measured is measured
    ACTIVE.observe()
