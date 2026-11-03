"""Time-scale existing GMR poses with FK; fixed XY distance, no new IK or physics."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, RotationSpline


def clock(scale):
    if scale not in (1.5, 2., 2.5, 3.):
        raise ValueError('Require a predeclared speed level')
    stop_frame = round(90/scale)
    if not np.isclose(stop_frame*.02*scale, 1.8):
        raise ValueError('Stopping phase must fall on the 50 Hz grid')
    start_frame = int(np.ceil(20/scale-1e-10))
    end_frame = stop_frame+48
    return dict(scale=scale, brake_start_s=start_frame*.02, brake_end_s=stop_frame*.02,
                kinematic_brake_start_s=.4/scale, stop_hold_s=.96,
                frames=end_frame+1, endpoint_s=end_frame*.02)


def scaled_reset(state, scale):
    result = {k:np.array(v, copy=True) for k,v in state.items()}
    # Scale every generalized/rigid velocity, retaining their kinematic relation.
    for key in ('root_com_vel','root_omega','joint_vel','body_com_vel','body_omega'):
        result[key] *= scale
    return result


def reference_pair(model, original, diagnostics, scale):
    import mujoco
    spec = clock(scale)
    t = np.arange(spec['frames'])*.02
    source_t = diagnostics['time_s']
    qold = diagnostics['qpos']
    if qold.shape != (139, model.nq) or not np.allclose(source_t, np.arange(139)*.02):
        raise ValueError('Frozen reference clock/model mismatch')
    names = original['joint_names'].astype(str).tolist()
    bodies = original['body_names'].astype(str).tolist()
    qids = [int(model.jnt_qposadr[model.joint(name).id]) for name in names]
    bids = [model.body('base_Link' if name=='limx_imu' else name).id for name in bodies]
    data = mujoco.MjData(model)
    scalar = PchipInterpolator(source_t, qold, axis=0, extrapolate=False)
    orientation = RotationSpline(source_t, Rotation.from_quat(qold[:,3:7], scalar_first=True))

    def pose(time, behavior):
        source = min(max(time*scale,0.), float(source_t[-1]))
        q = scalar(source)
        q[3:7] = orientation(source).as_quat(scalar_first=True)
        if behavior == 'direct':
            q[2:] = qold[0,2:]
        return q

    def fk(q):
        data.qpos[:] = q
        data.qvel[:] = 0.
        mujoco.mj_forward(model,data)
        return data.xpos[bids].copy(), data.xquat[bids].copy(), data.xipos[bids].copy()

    # Verify the exact source asset kinematics before synthesizing new samples.
    maximum = 0.
    for i,q in enumerate(qold):
        p,rot,_ = fk(q)
        maximum = max(maximum,float(abs(p-original['body_pos_w'][i]).max()))
        np.testing.assert_allclose(abs((rot*original['body_quat_w'][i]).sum(-1)),1.,atol=1e-6,rtol=0)
    if maximum > 2e-6:
        raise ValueError('Source body poses do not match the frozen FK model')

    pair, diagnostics_out = {}, {}
    for behavior in ('direct','hockey'):
        positions,quats,velocities,omegas,joints,dq,poses = [],[],[],[],[],[],[]
        for time in t:
            q = pose(time,behavior)
            minus,plus = max(0.,time-1e-5),time+1e-5
            qm,qp = pose(minus,behavior),pose(plus,behavior)
            p,rot,com = fk(q)
            pm,rm,cm = fk(qm);pp,rp,cp = fk(qp)
            omega = (Rotation.from_quat(rp,scalar_first=True)*Rotation.from_quat(rm,scalar_first=True).inv()).as_rotvec()/(plus-minus)
            positions.append(p);quats.append(rot);velocities.append((cp-cm)/(plus-minus));omegas.append(omega)
            joints.append(q[qids]);dq.append((qp[qids]-qm[qids])/(plus-minus));poses.append(q)
        motion = dict(fps=np.array(50.),joint_pos=np.array(joints),joint_vel=np.array(dq),
            body_pos_w=np.array(positions),body_quat_w=np.array(quats),
            body_lin_vel_w=np.array(velocities),body_ang_vel_w=np.array(omegas),
            joint_names=original['joint_names'].copy(),body_names=original['body_names'].copy())
        if np.max(abs(motion['joint_vel'][:,:6])) >= 15.:
            raise ValueError('Reference leg speed reaches the existing motor no-load speed')
        pair[behavior] = motion
        diagnostics_out[behavior] = dict(qpos=np.array(poses),time_s=t,
            source_time_s=np.minimum(t*scale,source_t[-1]))
    np.testing.assert_array_equal(pair['direct']['body_pos_w'][:,0,:2],pair['hockey']['body_pos_w'][:,0,:2])
    np.testing.assert_allclose(pair['hockey']['body_pos_w'][-1,0,:2],original['body_pos_w'][-1,0,:2],atol=1e-10,rtol=0)
    np.testing.assert_allclose(pair['direct']['joint_pos'][0],pair['hockey']['joint_pos'][0],atol=1e-12,rtol=0)
    return pair, diagnostics_out, dict(**spec, maximum_source_FK_position_error_m=maximum,
        body_linear_velocity_semantics='derivative of world COM positions from FK',
        angular_velocity_semantics='world angular velocity from nearby FK rotations',
        motion_timing='uniform compression of original pose sequence, followed by stationary extension to retain 0.96 s post-brake hold',
        reference_max_leg_speed_rad_s={k:float(abs(v['joint_vel'][:,:6]).max()) for k,v in pair.items()},
        physics_steps=0,new_IK_calls=0,dynamic_feasibility_claimed=False)
