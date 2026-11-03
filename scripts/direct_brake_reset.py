"""Identical physical evaluation reset for equal-distance stopping comparisons.

Training keeps each reference's RSI. Evaluation writes the archived base-link
pose, COM twist and named joint state, before applying the shared perturbation.
No simulation step, wheel lock, force or persistent state override is applied.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def ordered_state(state, joint_names, origin):
    names = state['joint_names'].astype(str).tolist()
    if len(set(names)) != len(names) or set(names) != set(joint_names):
        raise ValueError('Shared reset must contain every robot joint exactly once')
    index = [names.index(name) for name in joint_names]
    pose = np.r_[state['root_pos'] + np.asarray(origin), state['root_quat']]
    twist = np.r_[state['root_com_vel'], state['root_omega']]
    q, dq = state['joint_pos'][index], state['joint_vel'][index]
    if not all(np.isfinite(v).all() for v in (pose, twist, q, dq)):
        raise ValueError('Nonfinite shared reset')
    if not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-5):
        raise ValueError('Invalid shared reset quaternion')
    return pose, twist, q, dq


def apply_common_reset(env, protocol, output):
    import torch
    cfg = (protocol['common_evaluation_initial_state'] if 'common_evaluation_initial_state' in protocol
           else protocol['direct_brake']['common_evaluation_initial_state'])
    path = Path(cfg['path'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != cfg['sha256'] or env.num_envs != 1:
        raise ValueError('Common physical reset identity/world count changed')
    with np.load(path, allow_pickle=False) as archive:
        state = dict(archive)
    robot = env.scene['robot']
    values = ordered_state(state, list(robot.joint_names), env.scene.env_origins[0].cpu().numpy())
    pose, twist, q, dq = [torch.as_tensor(v[None], device=env.device, dtype=torch.float32) for v in values]
    robot.write_root_pose_to_sim(pose)
    robot.write_root_com_velocity_to_sim(twist)
    robot.write_joint_state_to_sim(q, dq)
    env.scene.write_data_to_sim()
    env.scene.update(protocol['physics_dt_s'])
    observed = (torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), -1),
                torch.cat((robot.data.root_com_lin_vel_w, robot.data.root_com_ang_vel_w), -1),
                robot.data.joint_pos, robot.data.joint_vel)
    errors = [float((actual - expected).abs().max())
              for actual, expected in zip(observed, (pose, twist, q, dq))]
    command = env.command_manager.get_term('motion')
    command.time_steps.zero_()
    command._refresh_relative_targets()
    if max(errors) > 5e-5 or bool(env.action_manager.action.any()) or bool(env.action_manager.prev_action.any()):
        raise ValueError('Shared reset readback or previous-action mismatch')
    report = dict(kind='direct_brake_shared_physical_reset_v1', source_sha256=digest,
                  maximum_state_error=max(errors), field_errors=errors,
                  joint_order=list(robot.joint_names), phase=0, previous_actions_zero=True,
                  base_COM_entry_speed_m_s=float(torch.linalg.vector_norm(twist[0, :3])),
                  applied_before_paired_perturbation=True, simulation_advanced=False)
    (Path(output) / 'common_reset_audit.json').write_text(json.dumps(report, indent=2) + '\n')


def instrument_trainer(text):
    """Small, count-checked evaluation-only insertion into a frozen trainer."""
    marker = '        env.scene.update(.0025)\n        if case is not None:'
    if text.count(marker) != 1:
        raise ValueError('Frozen trainer reset insertion point changed')
    text = text.replace(marker, '        env.scene.update(.0025)\n'
                        '        from direct_brake_reset import apply_common_reset\n'
                        '        apply_common_reset(env, protocol, output)\n'
                        '        if case is not None:')
    marker = 'files = ["final_rollout.npz", "native_contact_buffers.npz"]'
    if text.count(marker) != 1:
        raise ValueError('Frozen trainer artifact insertion point changed')
    text = text.replace(marker, 'files = ["final_rollout.npz", "native_contact_buffers.npz", "common_reset_audit.json"]')
    # Archive actuator effort and other-body contact evidence for later analysis.
    marker = '        state.update(contact)'
    if text.count(marker) != 1:
        raise ValueError('Frozen trainer telemetry insertion point changed')
    text = text.replace(marker, '        state["applied_joint_torque_nm"] = robot.data.applied_torque[:, jid]\n'
                        '        sensor = env.scene.sensors["contact_forces"]\n'
                        '        state["all_body_net_contact_force_w"] = sensor.data.net_forces_w\n'
                        + marker)
    marker = 'joint_names=np.array(ACTION_JOINT_NAMES), body_names=np.array(body_names),'
    if text.count(marker) != 1:
        raise ValueError('Frozen trainer name insertion point changed')
    text = text.replace(marker, marker + '\n        contact_sensor_body_names=np.array(env.scene.sensors["contact_forces"].body_names),')
    return text
