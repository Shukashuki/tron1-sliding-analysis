"""Conditional centroidal contact audit of a kinematic reference, no rollout.

The ideal saturated kinetic-Coulomb calculation is a stated hypothesis, not a
PhysX force reconstruction, identified tire model, or task impossibility proof.
No reference, reward, controller, acceptance gate or native asset is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


WHEELS = ("wheel_L_Link", "wheel_R_Link")
LEGS = ("abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
        "knee_L_Joint", "knee_R_Joint")


def closest_load_allocation(required_xy, force_per_normal, total_normal_n):
    """Closest two-contact force with nonnegative loads summing to total Fz.

    With unit-load force vectors d_L/d_R, Fxy = f_L*d_L + (N-f_L)*d_R,
    0 <= f_L <= N. This is Euclidean projection onto a line segment; unequal
    allocation and single-wheel support are allowed. No yaw-moment constraint.
    """
    required = np.asarray(required_xy, dtype=float)
    directions = np.asarray(force_per_normal, dtype=float)
    normal = np.asarray(total_normal_n, dtype=float)
    if required.shape[-1:] != (2,) or directions.shape != required.shape[:-1]+(2, 2):
        raise ValueError("Require XY forces and exactly two unit-load XY vectors")
    if normal.shape != required.shape[:-1] or np.any(normal <= 0):
        raise ValueError("Require positive total normal force per sample")
    if not all(np.isfinite(x).all() for x in (required, directions, normal)):
        raise ValueError("All force inputs must be finite")
    start = normal[..., None]*directions[..., 1, :]
    delta = normal[..., None]*(directions[..., 0, :]-directions[..., 1, :])
    denominator = np.sum(delta*delta, axis=-1)
    fraction = np.divide(np.sum((required-start)*delta, axis=-1), denominator,
                         out=np.full_like(denominator, .5), where=denominator > 1e-24)
    fraction = fraction.clip(0., 1.)
    closest = start+fraction[..., None]*delta
    return closest, fraction, np.linalg.norm(required-closest, axis=-1)


def reference_kinematics(model, reference):
    """Recompute COM and instantaneous material-point velocities by Jacobians.

    Exported body positions are link origins; exported linear velocities are
    COM velocities. They must not be combined as if measured at the same point.
    mj_forward updates kinematics here; mj_step is never called.
    """
    import mujoco

    names = reference["body_names"].astype(str).tolist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate reference bodies")
    ids = [names.index(model.body(i).name) for i in range(1, model.nbody)]
    wheel_ids = [model.body(n).id for n in WHEELS]
    geoms = []
    for body in wheel_ids:
        active = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body
                  and model.geom_contype[i] and model.geom_conaffinity[i]]
        if len(active) != 1 or model.geom_type[active[0]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError("Require one original cylindrical collider per wheel")
        geoms.append(active[0])
    data = mujoco.MjData(model)
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    records = {k: [] for k in ("COM_position_w_m", "COM_velocity_w_m_s",
        "wheel_surface_velocity_w_m_s", "wheel_surface_lateral_velocity_m_s",
        "wheel_lowest_surface_z_m", "wheel_axle_w", "body_forward_w")}
    max_position_error = max_velocity_error = max_axle_vertical = 0.
    for frame in range(len(reference["time_s"])):
        data.qpos[:] = reference["qpos_mujoco"][frame]
        data.qvel[:] = reference["qvel_mujoco"][frame]
        mujoco.mj_forward(model, data)
        velocities = []
        for i, ref_id in zip(range(1, model.nbody), ids):
            np.testing.assert_allclose(data.xpos[i], reference["body_pos_w"][frame, ref_id], rtol=0, atol=2e-6)
            q = reference["body_quat_w"][frame, ref_id].astype(float)
            q /= np.linalg.norm(q)
            assert abs(float(q@data.xquat[i])) > 1-1e-6
            mujoco.mj_jac(model, data, jp, jr, data.xipos[i], i)
            velocity = jp@data.qvel
            error = np.max(np.abs(velocity-reference["body_lin_vel_w"][frame, ref_id]))
            max_velocity_error = max(max_velocity_error, float(error))
            assert error < 2e-6, "Exported velocity must be at COM, not link origin"
            max_position_error = max(max_position_error,
                float(np.max(np.abs(data.xpos[i]-reference["body_pos_w"][frame, ref_id]))))
            velocities.append(velocity)
        records["COM_position_w_m"].append(np.average(data.xipos[1:], axis=0, weights=model.body_mass[1:]))
        records["COM_velocity_w_m_s"].append(np.average(velocities, axis=0, weights=model.body_mass[1:]))
        contact_v, contact_z, axes, lateral = [], [], [], []
        for body, geom in zip(wheel_ids, geoms):
            axis = data.xmat[body].reshape(3, 3)[:, 1]
            max_axle_vertical = max(max_axle_vertical, abs(float(axis[2])))
            if abs(axis[2]) > 1e-5:
                raise ValueError("This centerline flat-floor diagnostic requires horizontal wheel axles")
            # For a horizontal axle the lowest cylindrical surface point is
            # exactly its geometric center minus radius along world Z.
            point = data.geom_xpos[geom]+[0., 0., -model.geom_size[geom, 0]]
            mujoco.mj_jac(model, data, jp, jr, point, body)
            velocity = jp@data.qvel
            contact_v.append(velocity)
            contact_z.append(point[2])
            lateral.append(float(velocity@axis))
            axes.append(axis.copy())
        records["wheel_surface_velocity_w_m_s"].append(contact_v)
        records["wheel_surface_lateral_velocity_m_s"].append(lateral)
        records["wheel_lowest_surface_z_m"].append(contact_z)
        records["wheel_axle_w"].append(axes)
        records["body_forward_w"].append(data.xmat[model.body("base_Link").id].reshape(3, 3)[:, 0].copy())
        assert data.time == 0., "No integration or replay correction is allowed"
    return {k: np.asarray(v) for k, v in records.items()}, dict(
        link_origin_FK_max_abs_error_m=max_position_error,
        exported_COM_velocity_max_abs_error_m_s=max_velocity_error,
        max_wheel_axle_vertical_component=max_axle_vertical,
        body_names_in_model_order=[model.body(i).name for i in range(1, model.nbody)],
        body_masses_kg=model.body_mass[1:].tolist(), total_mass_kg=float(model.body_mass.sum()))


def run(args):
    import mujoco

    output = args.output_dir
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(args.mu) or args.mu <= 0:
        raise ValueError("Require a positive nominal diagnostic coefficient")
    with np.load(args.reference, allow_pickle=False) as archive:
        ref = {n: archive[n].copy() for n in archive.files}
    time = ref["time_s"]
    if len(time) < 5 or not np.isfinite(time).all() or not np.allclose(np.diff(time), .02, rtol=0, atol=1e-8):
        raise ValueError("Require the original uniformly sampled 50 Hz reference")
    core = (time >= 1.44-1e-9) & (time <= 4.+1e-9)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    assert np.array_equal(model.opt.gravity, [0., 0., -9.81])
    raw, proof = reference_kinematics(model, ref)
    mass, gravity = proof["total_mass_kg"], 9.81
    acc = np.gradient(raw["COM_velocity_w_m_s"], time, axis=0, edge_order=2)
    required = mass*acc[:, :2]
    total_normal = mass*(gravity+acc[:, 2])
    surface_v = raw["wheel_surface_velocity_w_m_s"][..., :2]
    speed = np.linalg.norm(surface_v, axis=-1)
    valid = (speed > 1e-4).all(axis=-1) & (total_normal > 0.)
    assert valid[core].all(), "Core must satisfy both-sliding assumptions"
    directions = -surface_v[valid]/speed[valid, :, None]
    closest, fraction, residual = closest_load_allocation(required[valid], args.mu*directions, total_normal[valid])
    # A restricted equal-load, scalar curve fit; this is not an identified road mu.
    unit_mu_force = total_normal[valid, None]*directions.mean(axis=1)
    denominator = np.sum(unit_mu_force**2, axis=-1)
    fitted = np.maximum(np.divide(np.sum(required[valid]*unit_mu_force, axis=-1), denominator,
                                  out=np.zeros_like(denominator), where=denominator > 1e-24), 0.)
    residual_all, mu_all = np.full(len(time), np.nan), np.full(len(time), np.nan)
    residual_all[valid], mu_all[valid] = residual/(mass*gravity), fitted
    names = ref["joint_names"].astype(str).tolist()
    leg_ids = [names.index(n) for n in LEGS]
    quantiles = lambda a: np.quantile(a, [.5, .95, 1.], axis=0).tolist()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = dict(kind="conditional_reference_centroidal_kinetic_friction_audit", physics_rollouts=0,
        integration_steps=0, task_success_claimed=False, task_impossibility_claimed=False,
        controller_or_reward_or_reference_changed=False, mu=args.mu,
        reference_sha256=sha(args.reference), model_sha256=sha(args.model), analyzer_sha256=sha(__file__),
        mujoco_version=mujoco.__version__, kinematic_reconstruction=proof, core_interval_s=[1.44, 4.],
        core_samples=int(core.sum()), all_core_samples_two_sliding_wheels=True,
        quantile_order=[.5, .95, 1.],
        core_surface_abs_lateral_speed_m_s_quantiles_L_R=quantiles(abs(raw["wheel_surface_lateral_velocity_m_s"][core])),
        core_required_COM_acceleration_XY_over_g_quantiles=quantiles(np.linalg.norm(acc[core, :2], axis=-1)/gravity),
        core_required_total_normal_over_mg_range=[float(total_normal[core].min()/(mass*gravity)), float(total_normal[core].max()/(mass*gravity))],
        core_best_load_split_force_residual_over_mg_quantiles=quantiles(residual_all[core]),
        core_equal_load_scalar_mu_fit_quantiles=quantiles(mu_all[core]),
        core_reference_lowest_surface_z_range_m=[float(raw["wheel_lowest_surface_z_m"][core].min()), float(raw["wheel_lowest_surface_z_m"][core].max())],
        reference_named_leg_max_abs_velocity_rad_s=np.max(abs(ref["joint_vel"][:, leg_ids]), axis=0).tolist(),
        assumed_law="Fully sliding isotropic kinetic Coulomb: Ft_i=-mu*Fn_i*surface_vxy_i/|surface_vxy_i|; FnL+FnR=m*(g+aCOMz), Fn_i>=0.",
        assumptions=["Flat floor, horizontal reference wheel axles, original rigid cylinders.",
            "Surface-point velocity includes the actual reference wheel spin via the Jacobian.",
            "Only gravity and wheel-ground external forces; no air force or other-body contact.",
            "Kinetic friction is saturated and exactly opposite the instantaneous XY material-point velocity.",
            "Best nonnegative load split allows a single supporting wheel; no bilateral constraint.",
            "Reference derivatives are at 50 Hz, not measured contact forces.",
            "The scalar mu fit assumes equal wheel loads and is not a road/tire coefficient estimate."],
        limits=["A centroidal force-only check omits angular momentum, joint limits, motor authority and discrete contact-solver behavior.",
            "Actual policies may move legs, shift height/lean/wheel headings and change surface slip while retaining task-level tracking.",
            "This conditional calculation cannot prove task infeasibility or attribute native PhysX/MuJoCo support gaps.",
            "No friction change, reference scaling or relaxed acceptance follows from this audit."],
        next_experiment="Test fresh task-space K/M learning with identical scratch seed/budget; previous low-rate warm refinements retained the old static-leg imitation checkpoint.",
        primary_API_documentation="https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac")
    output.mkdir(parents=True)
    (output/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(output/"reference_diagnostics.npz", time_s=time, **raw,
        COM_acceleration_w_m_s2=acc, required_XY_force_n=required, required_total_normal_n=total_normal,
        valid_two_sliding_wheels=valid, conditional_force_residual_over_mg=residual_all,
        conditional_equal_load_mu_fit=mu_all, best_left_load_fraction_on_valid_samples=fraction,
        closest_conditional_XY_force_n_on_valid_samples=closest)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), layout="constrained")
    for i, name in enumerate(("Left", "Right")):
        axes[0].plot(time, raw["wheel_surface_lateral_velocity_m_s"][:, i], label=name)
    axes[0].set(title="Reference surface lateral velocity", ylabel="m/s"); axes[0].legend()
    axes[1].plot(time, np.linalg.norm(acc[:, :2], axis=-1)/gravity)
    axes[1].set(title="Required COM horizontal acceleration", ylabel="acceleration / g")
    axes[2].plot(time, residual_all)
    axes[2].set(title=f"Conditional force residual, mu={args.mu:g}", ylabel="Best residual / mg", ylim=(0., None))
    axes[3].plot(time, mu_all)
    axes[3].axhline(args.mu, color="tab:red", ls="--", label="Nominal diagnostic mu")
    axes[3].set(title="Equal-load scalar fit (not measured mu)", ylabel="Restricted fitted coefficient"); axes[3].legend()
    for ax in axes:
        ax.axvspan(1.44, 4., color="gray", alpha=.12); ax.set(xlabel="Reference time (s)"); ax.grid(alpha=.2)
    fig.suptitle("Reference kinematics and conditional Coulomb hypothesis | no physics rollout or task impossibility claim")
    fig.savefig(output/"comparison.png", dpi=140); plt.close(fig)
    print(json.dumps({k: result[k] for k in ("core_surface_abs_lateral_speed_m_s_quantiles_L_R",
        "core_required_COM_acceleration_XY_over_g_quantiles", "core_best_load_split_force_residual_over_mg_quantiles",
        "core_equal_load_scalar_mu_fit_quantiles")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--mu", type=float, default=.7)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
