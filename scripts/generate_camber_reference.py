"""Generate TRON1 camber references from car motion and rigid-wheel kinematics.

This is a forward kinematic trajectory generator, not a friction law or a
physics rollout. The signed local+Y axle convention is fixed by the robot.
Only the current yaw-only car mapping and original cylinder asset are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WHEELS = ("wheel_L_Link", "wheel_R_Link")
ABADS = ("abad_L_Joint", "abad_R_Joint")
SPINS = ("wheel_L_Joint", "wheel_R_Joint")
SOURCE_SHA = "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
MODEL_SHA = "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoothstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0., 1.)
    return x*x*x*(10.+x*(-15.+6.*x))


def camber_profile(time, coefficients_deg, timing):
    """C2 envelope times a cubic Bernstein profile; convex hull bounds angles."""
    t = np.asarray(time, dtype=float)
    c = np.asarray(coefficients_deg, dtype=float)
    knots = np.asarray(timing, dtype=float)
    if (t.ndim != 1 or c.shape != (2, 4) or knots.shape != (4,)
            or not all(np.isfinite(a).all() for a in (t, c, knots))
            or not np.all(np.diff(knots) > 0) or np.max(abs(c)) > 15.):
        raise ValueError("Require finite times,2x4 coefficients within +/-15deg,and four increasing ramp times")
    u = np.clip((t-knots[0])/(knots[-1]-knots[0]), 0., 1.)
    basis = np.stack(((1-u)**3, 3*u*(1-u)**2, 3*u*u*(1-u), u**3), axis=-1)
    envelope = smoothstep((t-knots[0])/(knots[1]-knots[0]))*(1-smoothstep((t-knots[2])/(knots[3]-knots[2])))
    return np.deg2rad(basis@c.T)*envelope[:, None]


def cylinder_lowest_point(center, axle, radius, halfwidth):
    """Lowest point of the ideal cylinder; horizontal axles have a support line.

    At zero camber select that line's center. At nonzero camber select its lower
    rim. This branch change is intrinsic to a rigid cylinder, not smoothed away.
    This geometrical support point is NOT a measured contact or known load.
    """
    center = np.asarray(center, dtype=float)
    axle = np.asarray(axle, dtype=float)
    if (center.shape[-1:] != (3,) or axle.shape != center.shape
            or not np.isfinite(center).all() or not np.isfinite(axle).all()
            or not np.isfinite([radius, halfwidth]).all() or radius <= 0 or halfwidth <= 0):
        raise ValueError("Require finite matching vectors and positive cylinder dimensions")
    length = np.linalg.norm(axle, axis=-1, keepdims=True)
    if np.any(length < 1e-10):
        raise ValueError("Zero wheel axle")
    a = axle/length
    az = a[..., 2]
    radial = np.array([0., 0., 1.])-az[..., None]*a
    rn = np.linalg.norm(radial, axis=-1)
    if np.any(rn < 1e-8):
        raise ValueError("Near-horizontal wheel plane has an undefined rolling direction")
    branch = np.where(abs(az) <= 1e-10, 0., np.sign(az))
    point = center-radius*radial/rn[..., None]-halfwidth*branch[..., None]*a
    return point, branch


def differentiate(model, qpos, dt):
    import mujoco
    velocity = np.empty((len(qpos), model.nv))
    for i in range(len(qpos)):
        low, high = max(0, i-1), min(len(qpos)-1, i+1)
        mujoco.mj_differentiatePos(model, velocity[i], (high-low)*dt, qpos[low], qpos[high])
    return velocity


class Generator:
    def __init__(self, model, source):
        import mujoco
        self.mj, self.model, self.data, self.source = mujoco, model, mujoco.MjData(model), source
        self.time = np.asarray(source["time_s"], dtype=float)
        if len(self.time) < 5 or not np.allclose(np.diff(self.time), .02, rtol=0, atol=1e-10):
            raise ValueError("Require the unscaled50Hz car reference")
        self.dt = .02
        self.wheels = [model.body(n).id for n in WHEELS]
        self.geoms = []
        for body in self.wheels:
            matches = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body
                       and model.geom_contype[i] and model.geom_conaffinity[i]]
            if len(matches) != 1 or model.geom_type[matches[0]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
                raise ValueError("Require one original cylinder per wheel")
            g = matches[0]
            # Cylinder localZ must be parallel to the authored wheel+Y axle.
            mat = np.empty(9); mujoco.mju_quat2Mat(mat, model.geom_quat[g])
            if abs(mat.reshape(3,3)[1,2]) < 1-1e-10:
                raise ValueError("Cylinder and motor axes differ")
            self.geoms.append(g)
        self.abads = [model.joint(n).qposadr.item() for n in ABADS]
        self.spin_q = [model.joint(n).qposadr.item() for n in SPINS]
        self.spin_v = [model.joint(n).dofadr.item() for n in SPINS]
        self.body_ids = [model.body(str(n)).id for n in source["body_names"]]
        self.joint_ids = [model.joint(str(n)).id for n in source["joint_names"]]
        self.jq = model.jnt_qposadr[self.joint_ids]
        self.jv = model.jnt_dofadr[self.joint_ids]
        self.jp, self.jr = np.empty((3, model.nv)), np.empty((3, model.nv))
        q = source["qpos_mujoco"]
        # The direct abduction parameterization is exact for this yaw-only map.
        if not np.allclose(q[:,4:6], 0., rtol=0, atol=1e-10) or not np.allclose(q[:,self.abads], 0., atol=1e-10):
            raise ValueError("This model-based parameterization requires yaw-only roots and zero initial abduction")
        finite = differentiate(model, q, self.dt)
        np.testing.assert_allclose(finite[:,:6], source["qvel_mujoco"][:,:6], rtol=0, atol=1e-8)

    def pose(self, q):
        self.data.qpos[:] = q
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_comPos(self.model, self.data)
        assert self.data.time == 0., "No physics integration permitted"

    def wheel_geometry(self):
        points, axes, branches = [], [], []
        for b, g in zip(self.wheels, self.geoms):
            axis = self.data.xmat[b].reshape(3,3)[:,1].copy()
            p, branch = cylinder_lowest_point(self.data.geom_xpos[g], axis, *self.model.geom_size[g,:2])
            points.append(p); axes.append(axis); branches.append(branch)
        return np.asarray(points), np.asarray(axes), np.asarray(branches)

    def build(self, coefficients_deg, timing):
        m, d = self.model, self.data
        gamma = camber_profile(self.time, coefficients_deg, timing)
        q = self.source["qpos_mujoco"].copy()
        q[:,self.abads] = gamma
        for frame in range(len(q)):
            self.pose(q[frame])
            p, _, _ = self.wheel_geometry()
            # A geometric support envelope, not an absolute base-height target.
            q[frame,2] -= p[:,2].min()
        v = differentiate(m, q, self.dt)
        desired_spin = np.zeros((len(q),2))
        for frame in range(len(q)):
            self.pose(q[frame]); points, axes, _ = self.wheel_geometry()
            v[frame,self.spin_v] = 0.
            for i, b in enumerate(self.wheels):
                normal = np.array([0.,0.,1.]); forward = np.cross(axes[i], normal)
                forward /= np.linalg.norm(forward)
                self.mj.mj_jac(m, d, self.jp, self.jr, points[i], b)
                coupling = forward@self.jp[:,self.spin_v[i]]
                if abs(coupling) < .01:
                    raise ValueError("Degenerate wheel spin Jacobian")
                desired_spin[frame,i] = -(forward@self.jp@v[frame])/coupling
        phase = np.zeros_like(desired_spin)
        phase[0] = self.source["qpos_mujoco"][0,self.spin_q]
        phase[1:] = phase[0]+np.cumsum(.5*(desired_spin[:-1]+desired_spin[1:])*self.dt, axis=0)
        q[:,self.spin_q] = phase
        v = differentiate(m, q, self.dt)
        output = {k: a.copy() for k,a in self.source.items()}
        records = {k: [] for k in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
            "wheel_camber_rad", "wheel_axle_w", "wheel_center_w", "wheel_support_point_w",
            "wheel_geometric_gap_m", "wheel_support_branch", "wheel_point_velocity_w_m_s",
            "wheel_point_lateral_velocity_m_s", "wheel_point_longitudinal_velocity_m_s")}
        clearance = []
        wheel_geom_set = set(self.geoms)
        for frame in range(len(q)):
            self.pose(q[frame]); points, axes, branches = self.wheel_geometry()
            records["body_pos_w"].append(d.xpos[self.body_ids].copy())
            records["body_quat_w"].append(d.xquat[self.body_ids].copy())
            linear, angular = [], []
            for b in self.body_ids:
                self.mj.mj_jacBodyCom(m, d, self.jp, self.jr, b)
                linear.append(self.jp@v[frame]); angular.append(self.jr@v[frame])
            records["body_lin_vel_w"].append(linear); records["body_ang_vel_w"].append(angular)
            pv, lateral, longitudinal = [], [], []
            for i,b in enumerate(self.wheels):
                forward = np.cross(axes[i], [0.,0.,1.]); forward /= np.linalg.norm(forward)
                side = np.cross([0.,0.,1.], forward)
                self.mj.mj_jac(m, d, self.jp, self.jr, points[i], b)
                point_v = self.jp@v[frame]
                pv.append(point_v); lateral.append(side@point_v); longitudinal.append(forward@point_v)
            for key, value in dict(wheel_camber_rad=np.arcsin(axes[:,2].clip(-1,1)), wheel_axle_w=axes,
                    wheel_center_w=d.geom_xpos[self.geoms].copy(), wheel_support_point_w=points,
                    wheel_geometric_gap_m=points[:,2], wheel_support_branch=branches,
                    wheel_point_velocity_w_m_s=pv, wheel_point_lateral_velocity_m_s=lateral,
                    wheel_point_longitudinal_velocity_m_s=longitudinal).items():
                records[key].append(value)
            lowest = []
            for g in range(m.ngeom):
                if g in wheel_geom_set or not m.geom_bodyid[g] or not m.geom_contype[g] or not m.geom_conaffinity[g]:
                    continue
                mat, size = d.geom_xmat[g].reshape(3,3), m.geom_size[g]
                if m.geom_type[g] == self.mj.mjtGeom.mjGEOM_BOX:
                    z = d.geom_xpos[g,2]-abs(mat[2])@size
                elif m.geom_type[g] == self.mj.mjtGeom.mjGEOM_CYLINDER:
                    z = cylinder_lowest_point(d.geom_xpos[g], mat[:,2], *size[:2])[0][2]
                else:
                    raise ValueError("Unsupported nonwheel collider in clearance audit")
                lowest.append(z)
            clearance.append(min(lowest))
        output.update({k:np.asarray(a) for k,a in records.items()})
        output.update(qpos_mujoco=q, qvel_mujoco=v, joint_pos=q[:,self.jq], joint_vel=v[:,self.jv],
            camber_command_rad=gamma, wheel_geometrically_touching=output["wheel_geometric_gap_m"] < 1e-7,
            nonwheel_geometric_floor_clearance_m=np.asarray(clearance),
            desired_longitudinal_rolling_spin_rad_s=desired_spin)
        np.testing.assert_array_equal(q[:,:2], self.source["qpos_mujoco"][:,:2])
        np.testing.assert_array_equal(q[:,3:7], self.source["qpos_mujoco"][:,3:7])
        np.testing.assert_allclose(v[:,:2], self.source["qvel_mujoco"][:,:2], atol=1e-10, rtol=0)
        np.testing.assert_allclose(output["wheel_camber_rad"], gamma, atol=1e-10, rtol=0)
        assert output["wheel_geometrically_touching"].any(-1).all()
        assert output["wheel_geometric_gap_m"].min() > -1e-9
        if min(clearance) < -1e-7:
            raise ValueError("Generated reference intersects the floor with a nonwheel collider")
        for j in self.joint_ids:
            if m.jnt_limited[j] and np.any((q[:,m.jnt_qposadr[j]] < m.jnt_range[j,0]) | (q[:,m.jnt_qposadr[j]] > m.jnt_range[j,1])):
                raise ValueError("Generated joint reference violates an authored limit")
        return output


def describe(output, source):
    t = output["time_s"]; core = (t >= 1.44-1e-9) & (t <= 4.+1e-9)
    touch = output["wheel_geometrically_touching"]
    joint_names = output["joint_names"].astype(str).tolist()
    leg = [i for i,n in enumerate(joint_names) if n not in SPINS]
    wheels = []
    for i,name in enumerate(("left", "right")):
        valid = core & touch[:,i]
        a = output["wheel_point_lateral_velocity_m_s"][valid,i]
        wheels.append(dict(wheel=name, core_geometric_support_fraction=float(touch[core,i].mean()),
            core_touching_signed_lateral_velocity_m_s_median=None if not len(a) else float(np.median(a)),
            core_touching_abs_lateral_velocity_m_s_p95=None if not len(a) else float(np.quantile(abs(a),.95)),
            whole_max_geometric_gap_m=float(output["wheel_geometric_gap_m"][:,i].max()),
            whole_max_abs_camber_deg=float(np.rad2deg(abs(output["wheel_camber_rad"][:,i])).max())))
    return dict(wheels=wheels, core_interval_s=[1.44,4.],
        whole_max_abs_named_leg_velocity_rad_s=float(abs(output["joint_vel"][:,leg]).max()),
        whole_max_abs_longitudinal_point_velocity_m_s=float(abs(output["wheel_point_longitudinal_velocity_m_s"]).max()),
        root_height_range_m=[float(output["qpos_mujoco"][:,2].min()),float(output["qpos_mujoco"][:,2].max())],
        nonwheel_min_geometric_floor_clearance_m=float(output["nonwheel_geometric_floor_clearance_m"].min()),
        max_wheel_center_XY_displacement_from_old_reference_m=float(np.linalg.norm(
            output["body_pos_w"][:,[list(source["body_names"]).index(n) for n in WHEELS],:2]
            -source["body_pos_w"][:,[list(source["body_names"]).index(n) for n in WHEELS],:2], axis=-1).max()),
        base_XY_path_speed_and_orientation_preserved=True, geometric_touch_is_not_measured_load=True,
        friction_or_dynamic_feasibility_verified=False, training_or_task_success_claimed=False)


def run(a):
    import mujoco
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    if sha(a.reference) != SOURCE_SHA or sha(a.model) != MODEL_SHA:
        raise ValueError("Require the declared original car reference and robot asset")
    config = json.loads(a.config.read_text())
    if config["kind"] != "TRON1_camber_kinematic_reference_v1":
        raise ValueError("Wrong reference-generator configuration")
    with np.load(a.reference, allow_pickle=False) as f:
        source = {k:f[k].copy() for k in f.files}
    generator = Generator(mujoco.MjModel.from_xml_path(str(a.model)), source)
    payload = [(row, generator.build(row["coefficients_deg"], config["ramp_times_s"])) for row in config["cases"]]
    a.output_dir.mkdir(parents=True)
    rows = []
    for spec, data in payload:
        path = a.output_dir/(spec["id"]+".npz")
        np.savez_compressed(path, **data)
        rows.append(dict(id=spec["id"], title=spec["title"], coefficients_deg=spec["coefficients_deg"],
                         reference_sha256=sha(path), metrics=describe(data, source)))
    proof = dict(kind=config["kind"], status="kinematic_references_generated", rows=rows,
        generator_sha256=sha(__file__), config_sha256=sha(a.config), original_reference_sha256=sha(a.reference),
        model_sha256=sha(a.model), mujoco_version=mujoco.__version__, frames=len(generator.time), fps=50,
        integration_steps=0, policy_training_steps=0, contact_forces_available=False,
        camber_is_signed_asin_worldZ_dot_authored_wheel_localY=True,
        root_height_from_lowest_wheel_envelope=True, absolute_height_gate_added=False,
        longitudinal_spin_from_rolling_Jacobian_trapezoid_then_differentiate=True,
        limits=["These are hypothetical rigid-surface point velocities,not measured friction/contact forces.",
            "Nonzero camber alone does not uniquely command side slip;abduction rate and whole-body motion enter the Jacobian.",
            "Touching is geometrical only;one supporting candidate does not prove positive normal force.",
            "Zero-camber support is a line;tilted rigid cylinders select a rim,without tread deformation or camber thrust.",
            "Camber parameterization changes wheel-center XY;those shifts are reported and must replace the old wheel targets in any future learning cohort.",
            "Reference generation does not establish dynamic feasibility,motor torque sufficiency,or completion of stable-contact turning."])
    (a.output_dir/"generation.json").write_text(json.dumps(proof, indent=2, allow_nan=False)+"\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("reference", "model", "config", "output-dir"):
        p.add_argument("--"+n, type=Path, required=True)
    run(p.parse_args())
