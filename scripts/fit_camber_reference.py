"""Bounded kinematic shooting for a camber-driven surface-velocity reference.

Positive common camber selects the right wheel's geometric support candidate.
The left wheel may lift,as explicitly permitted;its velocity is not scored as
ground slip. This fits kinematics only,never a road coefficient or tire force.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import generate_camber_reference as forward


def predict_lateral(generator, coefficients_deg, timing):
    """Same geometry/Jacobians as export,omitting spin-independent body arrays."""
    q = generator.source["qpos_mujoco"].copy()
    gamma = forward.camber_profile(generator.time, coefficients_deg, timing)
    q[:,generator.abads] = gamma
    for i in range(len(q)):
        generator.pose(q[i]); points,_,_ = generator.wheel_geometry()
        q[i,2] -= points[:,2].min()
    v = forward.differentiate(generator.model, q, generator.dt)
    v[:,generator.spin_v] = 0.
    lateral, gaps = [], []
    for i in range(len(q)):
        generator.pose(q[i]); points,axes,_ = generator.wheel_geometry()
        values = []
        for p,a,b in zip(points,axes,generator.wheels):
            side = a-np.array([0.,0.,a[2]])
            side /= np.linalg.norm(side)
            generator.mj.mj_jac(generator.model,generator.data,generator.jp,generator.jr,p,b)
            values.append(side@generator.jp@v[i])
        lateral.append(values);gaps.append(points[:,2])
    return np.asarray(lateral),np.asarray(gaps),gamma


def run(a):
    import mujoco
    import scipy
    from scipy.optimize import least_squares

    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    if forward.sha(a.reference) != forward.SOURCE_SHA or forward.sha(a.model) != forward.MODEL_SHA:
        raise ValueError("Require the original declared reference/model")
    config = json.loads(a.config.read_text())
    if config["kind"] != "camber_kinematic_slip_shooting_v1":
        raise ValueError("Wrong shooting configuration")
    with np.load(a.reference,allow_pickle=False) as f:
        source = {k:f[k].copy() for k in f.files}
    generator = forward.Generator(mujoco.MjModel.from_xml_path(str(a.model)),source)
    timing = config["ramp_times_s"]
    zero,_,_ = predict_lateral(generator,np.zeros((2,4)),timing)
    core = (generator.time >= 1.44-1e-9)&(generator.time <= 4.+1e-9)
    target = zero[:,1]*config["right_geometric_support_velocity_ratio"]
    calls = 0
    def residual(coefficients):
        nonlocal calls
        calls += 1
        lateral,gap,gamma = predict_lateral(generator,np.tile(coefficients,(2,1)),timing)
        if np.max(abs(gap[core,1])) > 1e-7:
            raise ValueError("The scored right wheel is not a geometric support candidate")
        error = lateral[core,1]-target[core]
        # Fixed prospective regularizers,no task acceptance/gate is modified.
        magnitude = config["coefficient_regularization_m_s"]*coefficients/15.
        return np.r_[error,magnitude]
    initial = np.asarray(config["initial_coefficients_deg"],dtype=float)
    first = residual(initial)
    result = least_squares(residual,initial,bounds=(0.,15.),diff_step=1e-3,
                           max_nfev=config["max_nfev"],ftol=1e-7,xtol=1e-7,gtol=1e-7)
    coeff = np.tile(result.x,(2,1))
    data = generator.build(coeff,timing)
    fast,gap,_ = predict_lateral(generator,coeff,timing)
    np.testing.assert_allclose(fast,data["wheel_point_lateral_velocity_m_s"],atol=2e-10,rtol=0)
    actual = data["wheel_point_lateral_velocity_m_s"][:,1]
    a.output_dir.mkdir(parents=True)
    spec = dict(kind="TRON1_camber_kinematic_reference_v1",ramp_times_s=timing,cases=[
        dict(id="zero",title="Zero camber",coefficients_deg=np.zeros((2,4)).tolist()),
        dict(id="constant10",title="Constant +10 deg",coefficients_deg=np.full((2,4),10.).tolist()),
        dict(id="sweep",title="Camber sweep -10 to+10 deg",coefficients_deg=np.tile(np.linspace(-10,10,4),(2,1)).tolist()),
        dict(id="fitted",title="Kinematic slip-target fit",coefficients_deg=coeff.tolist())])
    (a.output_dir/"fitted_reference_config.json").write_text(json.dumps(spec,indent=2)+"\n")
    np.savez_compressed(a.output_dir/"fitted_reference.npz",**data,
        right_surface_lateral_velocity_target_m_s=target,slip_target_active=core)
    proof = dict(kind=config["kind"],status="kinematic_shooting_finished",solver="scipy.optimize.least_squares,TRF,bounded numerical Jacobian",
        scipy_version=scipy.__version__,optimizer_success=bool(result.success),message=str(result.message),
        objective_evaluations=calls,nfev=int(result.nfev),initial_coefficients_deg=initial.tolist(),
        fitted_common_coefficients_deg=result.x.tolist(),coefficient_bounds_deg=[0,15],
        right_geometric_support_velocity_ratio=config["right_geometric_support_velocity_ratio"],
        initial_core_target_RMSE_m_s=float(np.sqrt(np.mean(first[:int(core.sum())]**2))),
        final_core_target_RMSE_m_s=float(np.sqrt(np.mean((actual[core]-target[core])**2))),
        zero_camber_core_signed_slip_m_s_median=float(np.median(zero[core,1])),
        target_core_signed_slip_m_s_median=float(np.median(target[core])),
        fitted_core_signed_slip_m_s_median=float(np.median(actual[core])),
        scored_right_wheel_core_geometric_support_fraction=float(np.mean(abs(gap[core,1])<1e-7)),
        left_wheel_core_geometric_support_fraction=float(np.mean(abs(gap[core,0])<1e-7)),
        raw_target_achieved_exactly=False,kinematic_report=forward.describe(data,source),
        config_sha256=forward.sha(a.config),reference_sha256=forward.sha(a.reference),model_sha256=forward.sha(a.model),
        forward_generator_sha256=forward.sha(forward.__file__),fitter_sha256=forward.sha(__file__),
        fitted_reference_sha256=forward.sha(a.output_dir/"fitted_reference.npz"),
        no_physics_integration=True,no_reward_or_policy_changed=True,task_success_claimed=False,
        limit="Positive common camber family,one geometric supporting candidate;not a general inverse dynamics solution or proof of feasible loads.")
    (a.output_dir/"fit.json").write_text(json.dumps(proof,indent=2,allow_nan=False)+"\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("reference","model","config","output-dir"):
        p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
