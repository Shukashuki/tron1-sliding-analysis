"""Generate a full-body reference by native-dynamics MPPI, then replay it.

Both abduction commands and both wheel torques are independent. Contact loads
are outputs of MuJoCo, never optimization variables. This is offline trajectory
generation with a NEW explicit planning cost, not a matched-PPO reward study.
The existing MPPI implementation deploys its final weighted mean, not an elite.
Prediction cold-resets every 20 ms; the actual plant integrates continuously.
An optional explicitly declared CEM action seed is not an independent baseline.
No training, GPU, native compilation or change to historical assets occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from mppi_shooting import MPPIConfig, MPPIController
from nmpc_native_ground_contact_400hz import (
    ACTUATOR_SETTINGS, BODY_NAMES, JOINT_NAMES, NativeBatchTransition,
    _validated_library, ground_contact_readout,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = 'bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513'
REFERENCE_SHA = 'da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254'
MATERIAL_REPORT_SHA = '220edda86e0e4fab5bfbcf4b18864ce6229addf781a462f5daf96a14e1f3d6b9'
SEED_SHA = '0542ab5d8c8e6a64a1b04d16118811fcec7cd1a87e651871373344a0c98374d7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def angle(x):
    return np.arctan2(np.sin(x), np.cos(x))


def quaternion_heading(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def quaternion_up_z(q):
    return 1-2*(q[..., 1]**2+q[..., 2]**2)


def wheel_angles(q):
    """Authored wheel-local +Y axle; rolling direction is axle cross world Z."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    axle = np.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)
    return np.arcsin(axle[..., 2].clip(-1, 1)), np.arctan2(-axle[..., 0], axle[..., 1])


def load_model(model_path, material_report):
    """Remove visual meshes only; restore the recorded native evaluation plant."""
    root = ET.fromstring(Path(model_path).read_text())
    for parent in root.iter():
        for item in list(parent):
            if item.tag == 'geom' and item.get('class') == 'visual':
                parent.remove(item)
    for asset in root.findall('asset'):
        for item in list(asset.findall('mesh')):
            asset.remove(item)
    root.find('compiler').attrib.pop('meshdir', None)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    audit = material_report['plant_contact_material_audit']
    ids = [model.geom(name).id for name in audit['colliding_geom_names']]
    for field in ('geom_friction', 'geom_priority', 'geom_condim', 'geom_solref', 'geom_solimp'):
        getattr(model, field)[ids] = audit[field]
    # Same self-collision masks as the historical native evaluation configuration.
    for geom in ids:
        if model.geom_bodyid[geom]:
            model.geom_contype[geom], model.geom_conaffinity[geom] = 1, 3
    readback = audit['native_plant_readback']
    assert [model.body(i).name for i in range(model.nbody)] == readback['body_names']
    for field, row in readback['readbacks'].items():
        getattr(model, field)[:] = np.asarray(row['values'])
    for field, value in readback['solver_configuration'].items():
        original = getattr(model.opt, field)
        setattr(model.opt, field, int(value) if isinstance(original, int) else value)
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model


def torque(model, data, q0, action):
    qids = [model.joint(n).qposadr.item() for n in JOINT_NAMES]
    vids = [model.joint(n).dofadr.item() for n in JOINT_NAMES]
    dq = data.qvel[vids]
    a = np.clip(action, -1., 1.)
    requested = np.r_[500*(q0[:6]+a[:6]-data.qpos[qids[:6]])-10*dq[:6], 12*a[6:]]
    effort, speed = np.r_[np.full(6, 80.), [12., 12.]], np.r_[np.full(6, 15.), [100., 100.]]
    curve = np.clip(dq, -2*speed, 2*speed)
    lower = np.maximum(effort*(-1-curve/speed), -effort)
    upper = np.minimum(effort*(1-curve/speed), effort)
    return np.clip(requested, lower, upper)


class Planner:
    def __init__(self, native, model, source, cfg, *, seed_actions=None):
        self.native, self.model, self.source, self.cfg = native, model, source, cfg
        self.weight = float(model.body_mass.sum())*9.81
        self.q0 = source['joint_pos'][0, [list(source['joint_names']).index(n) for n in JOINT_NAMES]]
        self.joint_qids = [model.joint(n).qposadr.item() for n in JOINT_NAMES]
        self.joint_vids = [model.joint(n).dofadr.item() for n in JOINT_NAMES]
        self.seed_actions = seed_actions
        self.nominal = []
        d = mujoco.MjData(model)
        for q, v in zip(source['qpos_mujoco'], source['qvel_mujoco']):
            mujoco.mj_resetData(model, d); d.qpos[:] = q; d.qvel[:] = v
            mujoco.mj_forward(model, d)
            self.nominal.append(np.r_[(q[self.joint_qids[:6]]-self.q0[:6])+
                                      (d.qfrc_bias[self.joint_vids[:6]]+10*v[self.joint_vids[:6]])/500,
                                      [0., 0.]])
        self.nominal = np.clip(self.nominal, -1, 1)

    def initial_candidate(self, state, start_index, *, horizon_steps, **kwargs):
        indices = np.minimum(start_index+np.arange(horizon_steps), len(self.nominal)-1)
        return (self.nominal if self.seed_actions is None else self.seed_actions)[indices].copy()

    def rollout_batch(self, state, actions, start_index, *, mu, previous_action, previous_vz):
        if mu != self.cfg['mu']:
            raise ValueError('Prediction material is fixed explicitly; no implicit friction oracle')
        n = len(actions)
        raw = self.native.rollout(np.tile(state[:15], (n, 1)), np.tile(state[15:], (n, 1)), actions,
                                  initial_history=np.zeros((n, 3, 9, 3)))
        return self.cost(raw, actions, start_index, previous_action, previous_vz)

    def cost(self, raw, actions, start_index, previous_action, previous_vz, *, return_terms=False):
        # Score all eight physical post-action samples against the corresponding
        # interpolated original reference, rather than a shifted/held XY path.
        horizon = actions.shape[1]
        t = (start_index+np.arange(horizon))[:, None]*.02+np.arange(1, 9)[None]*.0025
        times = self.source['time_s']
        def interp(values):
            return np.stack([np.interp(t, times, values[:, i]) for i in range(values.shape[-1])], -1)
        ref_xy = interp(self.source['qpos_mujoco'][:, :2])
        # The input ends while moving. Extend ONLY prediction times beyond its
        # end, so the last horizon does not introduce a fictitious stopping goal.
        extra = np.maximum(t-times[-1], 0.)
        ref_xy += extra[..., None]*self.source['qvel_mujoco'][-1, :2]
        base = list(self.source['body_names']).index('base_Link')
        ref_v = interp(self.source['body_lin_vel_w'][:, base, :2])
        ref_yaw = np.interp(t, times, np.unwrap(quaternion_heading(self.source['body_quat_w'][:, base])))
        target_beta = angle(np.arctan2(ref_v[..., 1], ref_v[..., 0])-ref_yaw)
        pos = raw['qpos'][..., :2]
        vel = raw['body_lin_vel_w'][..., 0, :2]
        quat = raw['body_quat_w'][..., 0, :]
        yaw = quaternion_heading(quat)
        speed, target_speed = np.linalg.norm(vel, axis=-1), np.linalg.norm(ref_v, axis=-1)
        beta = angle(np.arctan2(vel[..., 1], vel[..., 0])-yaw)
        w, s = self.cfg['cost_weights'], self.cfg['cost_scales']
        terms = {}
        terms['xy'] = np.mean(np.sum((pos-ref_xy[None])**2, axis=-1)/s['xy_m']**2, -1)
        terms['speed'] = np.mean(((speed/np.maximum(target_speed, .1)-1)/s['speed_ratio'])**2, -1)
        terms['beta'] = np.mean((angle(beta-target_beta)/s['beta_rad'])**2, -1)
        terms['yaw'] = np.mean((angle(yaw-ref_yaw)/s['yaw_rad'])**2, -1)
        # Max over wheels permits single-wheel support. Mean/max over SUBSTEPS
        # measures both duty and the worst physical sample, not force at50Hz.
        best_fz = np.maximum(raw['ground_normal_force_w'][..., 2], 0).max(-1)
        target = self.cfg['support_target_weight_fraction']*self.weight
        margin = np.clip((target-best_fz)/(target-self.cfg['support_threshold_n']), 0, 1)
        terms['support_mean'] = np.mean(margin**2, -1)
        terms['support_worst'] = np.max(margin**2, -1)
        terms['upright'] = np.mean((1-quaternion_up_z(quat))**2, -1)
        nonwheel = np.linalg.norm(raw['body_net_forces_w'][..., :7, :], axis=-1).max(-1)
        terms['nonwheel_contact'] = np.mean(np.minimum(nonwheel/self.weight, 5)**2, -1)
        angular = raw['body_ang_vel_w'][..., 0, :2]
        terms['angular_xy'] = np.mean(np.sum(angular**2, -1), -1)/s['angular_xy_rad_s']**2
        vz = raw['body_lin_vel_w'][..., 0, 2]
        preceding = np.concatenate((np.full((len(actions), 1), previous_vz), vz.reshape(len(actions), -1)[:, :-1]), -1)
        az = (vz.reshape(len(actions), -1)-preceding)/.0025
        terms['vertical_acceleration'] = np.mean(np.minimum((az/s['vertical_acceleration_m_s2'])**2, 25).reshape(vz.shape), -1)
        old = np.concatenate((np.broadcast_to(previous_action, (len(actions), 1, 8)), actions[:, :-1]), 1)
        terms['action_rate'] = np.sum((actions-old)**2, -1)
        stage = sum(w[k]*terms[k] for k in w)
        result = .02*np.sum(stage, -1)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite native MPPI planning cost')
        return (result, terms) if return_terms else result


def record(model, data, applied):
    ids = [model.body(n).id for n in BODY_NAMES]
    qids = [model.joint(n).qposadr.item() for n in JOINT_NAMES]
    vids = [model.joint(n).dofadr.item() for n in JOINT_NAMES]
    velocity = np.empty((len(ids), 6))
    for i, body in enumerate(ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity[i], 0)
    ground = ground_contact_readout(model, data)
    # Only floor pairs with non-wheel bodies, not self-contact force sums.
    nonwheel = 0.
    floor = model.geom('floor').id
    wheels = {model.body('wheel_'+s+'_Link').id for s in ('L', 'R')}
    for c in range(data.ncon):
        geoms = data.contact[c].geom
        if floor in geoms:
            other = geoms[1] if geoms[0] == floor else geoms[0]
            if model.geom_bodyid[other] not in wheels:
                wrench = np.empty(6); mujoco.mj_contactForce(model, data, c, wrench)
                nonwheel += float(np.linalg.norm(wrench[:3]))
    return dict(time_s=np.asarray(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
        root_pos=data.qpos[:3].copy(), root_quaternion_wxyz=data.qpos[3:7].copy(),
        joint_pos=data.qpos[qids].copy(), joint_vel=data.qvel[vids].copy(),
        body_pos_w=data.xpos[ids].copy(), body_quat_w=data.xquat[ids].copy(),
        body_lin_vel_w=velocity[:, 3:].copy(), body_ang_vel_w=velocity[:, :3].copy(),
        base_com_velocity_w=velocity[0, 3:].copy(), base_angular_velocity_w=velocity[0, :3].copy(),
        applied_joint_torque_nm=np.array(applied),
        ground_normal_force_w=ground['ground_normal_force_w'],
        ground_contact_count=ground['ground_contact_count'],
        nonwheel_ground_contact_force_norm_n=np.asarray(nonwheel))


def export_motion(model, raw, source, output):
    """Recompute every reference tensor, including IMU; force channels excluded."""
    frames = np.arange(0, len(raw['time_s'])-1, 8)
    n = min(len(frames), len(source['time_s'])); frames = frames[:n]
    d = mujoco.MjData(model); ids = [model.body(str(name)).id for name in source['body_names']]
    qids = [model.joint(str(name)).qposadr.item() for name in source['joint_names']]
    vids = [model.joint(str(name)).dofadr.item() for name in source['joint_names']]
    motion = {k:v.copy() for k,v in source.items() if k not in ('qpos_mujoco','qvel_mujoco')}
    for key in ('body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w','joint_pos','joint_vel'):
        motion[key] = motion[key][:n].copy()
    for key in ('time_s','source_time_s','demonstration_time_s_estimated','source_phase','root_yaw_rad'):
        if key in motion and motion[key].ndim and len(motion[key]) == len(source['time_s']):
            motion[key] = motion[key][:n].copy()
    for i, frame in enumerate(frames):
        d.qpos[:] = raw['qpos'][frame]; d.qvel[:] = raw['qvel'][frame]; mujoco.mj_forward(model, d)
        motion['body_pos_w'][i] = d.xpos[ids]; motion['body_quat_w'][i] = d.xquat[ids]
        for j, body in enumerate(ids):
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            motion['body_lin_vel_w'][i,j] = velocity[3:]; motion['body_ang_vel_w'][i,j] = velocity[:3]
        motion['joint_pos'][i] = d.qpos[qids]; motion['joint_vel'][i] = d.qvel[vids]
    motion['qpos_mujoco'] = raw['qpos'][frames].copy(); motion['qvel_mujoco'] = raw['qvel'][frames].copy()
    motion['root_yaw_rad'] = quaternion_heading(motion['qpos_mujoco'][:,3:7])
    # Orientation tracking masks are inherited as tracking CHOICES, not measured states.
    np.savez_compressed(output, **motion)


def run(a):
    if a.output_dir.exists():
        raise FileExistsError('Fresh result directory required')
    if sha(a.model) != MODEL_SHA or sha(a.reference) != REFERENCE_SHA or sha(a.material_report) != MATERIAL_REPORT_SHA:
        raise ValueError('Model/reference/material provenance differs')
    cfg = json.loads(a.config.read_text())
    if (cfg['kind'] != 'native_mppi_contact_reference_v1' or cfg['mu'] != .7
            or cfg['physics_dt_s'] != .0025 or cfg['control_dt_s'] != .02
            or cfg['absolute_height_cost'] or cfg['wheel_xy_cost']):
        raise ValueError('Unexpected reference planning contract')
    config = MPPIConfig(**cfg['mppi']); contract = dict(ACTUATOR_SETTINGS, actuator_model='dc_motor', physics_dt=.0025, decimation=8)
    material_report = json.loads(a.material_report.read_text())
    model = load_model(a.model, material_report)
    library, _, _ = _validated_library(a.native_library)
    with np.load(a.reference, allow_pickle=False) as f: source = {k:f[k].copy() for k in f.files}
    seed = None
    if a.seed_rollout is not None:
        if sha(a.seed_rollout) != SEED_SHA:
            raise ValueError('Only the declared historical pure-model CEM seed is allowed')
        with np.load(a.seed_rollout, allow_pickle=False) as f: seed = f['actions'].copy()
    d = mujoco.MjData(model)
    d.qpos[:] = material_report['initial_state']['qpos']; d.qvel[:] = material_report['initial_state']['qvel']
    mujoco.mj_forward(model, d)
    q0 = source['joint_pos'][0, [list(source['joint_names']).index(n) for n in JOINT_NAMES]]
    aids = [model.actuator(n).id for n in JOINT_NAMES]
    steps = a.steps or len(source['time_s'])
    if not 1 <= steps <= len(source['time_s']): raise ValueError('Invalid episode length')
    a.output_dir.mkdir(parents=True)
    report = dict(kind=cfg['kind'], status='running', config=cfg, mppi=asdict(config),
        source_sha256=sha(__file__), mppi_source_sha256=sha(ROOT/'scripts/mppi_shooting.py'),
        native_source_sha256=sha(ROOT/'native/nmpc_batch_ground_contact_400hz.cpp'),
        native_bridge_sha256=sha(ROOT/'scripts/nmpc_native_ground_contact_400hz.py'),
        config_sha256=sha(a.config), model_sha256=sha(a.model), reference_sha256=sha(a.reference),
        material_report_sha256=sha(a.material_report), native_library_sha256=sha(a.native_library),
        seed_rollout_sha256=None if a.seed_rollout is None else sha(a.seed_rollout),
        initialization='bias feedforward' if seed is None else 'historical pure-model CEM V2k action sequence',
        independent_MPPI_baseline=seed is None, learned_policy_used=False, policy_updates=0,
        independent_L_R_actions=True, force_optimization_variables=False,
        prediction_reset='cold reset each20ms; full qpos/qvel only', plant_reset='initialization only; continuous physical integration',
        physics_dt_s=.0025, control_dt_s=.02, steps=steps, planned_Fz_is_sensor_measurement=False,
        absolute_height_cost=False, wheel_XY_cost=False, PPO_reward_parity_claimed=False,
        post_reference_prediction_XY_extension='constant final root XY velocity; scoring reference unchanged',
        nonwheel_planning_signal='nonwheel body resultant contact forces; includes self contact; actual audit separately uses floor-only contacts',
        nominal_task_success_claimed=False, cross_engine_validation_complete=False)
    write_json(a.output_dir/'report.json', report)
    records=[record(model,d,np.zeros(8))]; previous=np.zeros(8); decisions=[]; started=time.monotonic()
    try:
        with NativeBatchTransition(model,q0,contract,library,timeout_s=60) as native:
            planner=Planner(native,model,source,cfg,seed_actions=seed)
            controller=MPPIController(planner,config,action_low=-np.ones(8),action_high=np.ones(8),mu_scenarios=(.7,))
            for frame in range(steps):
                state=np.r_[d.qpos,d.qvel]
                action,info=controller.action(state,frame,previous_action=previous,
                    previous_vz=float(records[-1]['base_com_velocity_w'][2]))
                decisions.append(dict(frame=frame,action=action.tolist(),**info))
                for substep in range(8):
                    applied=torque(model,d,q0,action);d.ctrl[aids]=applied
                    mujoco.mj_step(model,d);mujoco.mj_forward(model,d)
                    if any(w.number for w in d.warning): raise FloatingPointError('Native plant warning; preserve prefix')
                    if np.any(d.qfrc_applied) or np.any(d.xfrc_applied): raise RuntimeError('Unexpected external/root force')
                    records.append(record(model,d,applied))
                previous=action.copy()
                if frame%10==0 or frame==steps-1:
                    xy=float(np.linalg.norm(d.qpos[:2]-source['qpos_mujoco'][min(frame+1,348),:2]))
                    print(json.dumps(dict(frame=frame+1,total=steps,elapsed_s=time.monotonic()-started,
                        xy_error_m=xy,wheel_Fz_n=records[-1]['ground_normal_force_w'][:,2].tolist(),
                        solve_s=info['compute_time_s'],fallback=info['fallback_used'])),flush=True)
        report['status']='completed'
    except BaseException as error:
        report.update(status='failed_preserved_prefix',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        raw={k:np.asarray([r[k] for r in records]) for k in records[0]}
        raw.update(actions=np.asarray([r['action'] for r in decisions]),action_joint_names=np.asarray(JOINT_NAMES),body_names=np.asarray(BODY_NAMES))
        np.savez_compressed(a.output_dir/'rollout.npz',**raw)
        write_json(a.output_dir/'solver_log.json',decisions)
        report.update(completed_control_intervals=(len(records)-1)//8,physical_samples=len(records),wall_s=time.monotonic()-started,
            rollout_sha256=sha(a.output_dir/'rollout.npz'),solver_log_sha256=sha(a.output_dir/'solver_log.json'))
        write_json(a.output_dir/'report.json',report)
    export_motion(model,raw,source,a.output_dir/'candidate_motion.npz')
    report['candidate_motion_sha256']=sha(a.output_dir/'candidate_motion.npz')
    write_json(a.output_dir/'report.json',report)


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--model',type=Path,default=ROOT/'outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml')
    p.add_argument('--reference',type=Path,default=ROOT/'results/contact_wheel_axis_20261025_v1/reference.npz')
    p.add_argument('--material-report',type=Path,default=ROOT/'outputs/ground_contact_v2k_20261024_v1/cem_actual_nominal_v1/report.json')
    p.add_argument('--native-library',type=Path,default=ROOT/'outputs/native_ground_contact_v2_20261023_v1/libtron1_nmpc_batch_ground_contact_400hz.so')
    p.add_argument('--config',type=Path,default=ROOT/'config/mppi_contact_reference_v1.json')
    p.add_argument('--seed-rollout',type=Path)
    p.add_argument('--steps',type=int)
    p.add_argument('--output-dir',type=Path,required=True)
    return p


if __name__=='__main__':run(parser().parse_args())
