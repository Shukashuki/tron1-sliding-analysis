"""Guards against biased initialization, uncontrolled launches and reward drift."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from direct_brake_reset import ordered_state, instrument_trainer, apply_common_reset
from prepare_direct_brake_training import direct_protocol, build
from dispatch_direct_brake_training import gpu_rows, inspect, bind_parent, sha, write
from hockey_training_continuation import matching_recipe


def test_shared_reset_preserves_com_twist_and_joint_order():
    state = dict(np.load(ROOT/'results/hockey_equal_distance_20261102_v1/common_evaluation_initial_state.npz'))
    names = state['joint_names'].astype(str).tolist()[::-1]
    pose, twist, q, dq = ordered_state(state, names, [10, -5, 0])
    np.testing.assert_allclose(pose[:3], state['root_pos'] + [10, -5, 0])
    np.testing.assert_array_equal(twist, np.r_[state['root_com_vel'], state['root_omega']])
    np.testing.assert_array_equal(q, state['joint_pos'][::-1])
    np.testing.assert_array_equal(dq, state['joint_vel'][::-1])
    with pytest.raises(ValueError):
        ordered_state(state, names[:-1], [0, 0, 0])


def test_protocol_changes_only_declared_behavior_fields():
    old = ROOT/'results/hockey_contact_reward_search_20261028_v1/source_snapshot_v7/protocols/native_equal_smooth_stop_xyz_huber_continuity.json'
    original = json.loads(old.read_text())
    new = direct_protocol(original, 43, 'reference', 'reset')
    assert new['reward_weights']['beta'] == 0
    assert new['bilateral_sliding']['weights']['sliding'] == 0
    assert 'warm_start' not in new and 'training_continuation' not in new
    # Restore only the declared substitutions; all other hyperparameters match.
    for key in ('seed','reference_sha256','status','initialization','evaluation_cases'):
        if key in original:
            new[key] = original[key]
        else:
            new.pop(key, None)
    new.pop('direct_brake')
    new['warm_start'] = original['warm_start']
    new['reward_weights']['beta'] = original['reward_weights']['beta']
    new['bilateral_sliding']['seed'] = original['bilateral_sliding']['seed']
    new['bilateral_sliding']['weights']['sliding'] = original['bilateral_sliding']['weights']['sliding']
    assert new == original


def xml_gpu(*, used=0, util=0, process=''):
    return f'<nvidia_smi_log><gpu><product_name>NVIDIA GeForce RTX 4090 D</product_name><uuid>GPU-x</uuid><fb_memory_usage><used>{used} MiB</used></fb_memory_usage><utilization><gpu_util>{util} %</gpu_util></utilization><processes>{process}</processes></gpu></nvidia_smi_log>'


def test_gpu_must_be_empty_not_just_zero_utilization():
    assert gpu_rows(xml_gpu())[0]['eligible']
    assert not gpu_rows(xml_gpu(used=24000))[0]['eligible']
    assert not gpu_rows(xml_gpu(util=1))[0]['eligible']
    for kind in ('C', 'G'):
        process = f'<process_info><pid>123</pid><type>{kind}</type></process_info>'
        assert not gpu_rows(xml_gpu(process=process))[0]['eligible']
    assert not gpu_rows(xml_gpu(used='N/A'))[0]['eligible']
    with pytest.raises(ValueError):
        gpu_rows('<nvidia_smi_log/>')


def test_bundle_and_continuation_bind_only_own_completed_parent(tmp_path):
    build(tmp_path/'experiment')
    bundle = tmp_path/'experiment/source_snapshot_v1'
    plan, _ = inspect(bundle, verify_runtime=False)
    assert plan['unique_formal_transitions_total'] == 5*(12288000+2*(3072000+9216000))
    assert len(list((bundle/'protocols').glob('*.json'))) == 15
    template = json.loads((bundle/'protocols/seed43_equal.json').read_text())
    cfg = tmp_path/'cfg.json'
    write(cfg, template)
    folder = tmp_path/'completed'
    folder.mkdir()
    (folder/'model_final.pt').write_bytes(b'test-checkpoint')
    (folder/'actor_normalized.pt').write_bytes(b'test-actor')
    report = dict(status='training_completed_not_task_success', transitions=3072000,
                  config_sha256=sha(cfg), checkpoint_sha256=sha(folder/'model_final.pt'),
                  actor_sha256=sha(folder/'actor_normalized.pt'), bilateral_audit={'fault_seen':False})
    write(folder/'report.json', report)
    target = bind_parent(template, folder, cfg, continuation=True)
    matching_recipe(template, target)
    assert target['warm_start']['checkpoint'] == str(folder/'model_final.pt')
    assert target['formal']['iterations'] == 750
    report['status'] = 'failed'
    write(folder/'report.json', report)
    with pytest.raises(ValueError):
        bind_parent(template, folder, cfg)
    (bundle/'motion.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError):
        inspect(bundle, verify_runtime=False)


def test_reset_adapter_readback_and_zero_previous_action(tmp_path):
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace as NS
    path = ROOT/'results/hockey_equal_distance_20261102_v1/common_evaluation_initial_state.npz'
    state = dict(np.load(path))
    data = NS()
    class Robot:
        joint_names = state['joint_names'].astype(str).tolist()[::-1]
        def write_root_pose_to_sim(self, pose):
            data.root_pos_w, data.root_quat_w = pose[:, :3].clone(), pose[:, 3:].clone()
        def write_root_com_velocity_to_sim(self, velocity):
            data.root_com_lin_vel_w, data.root_com_ang_vel_w = velocity[:, :3].clone(), velocity[:, 3:].clone()
        def write_joint_state_to_sim(self, q, dq):
            data.joint_pos, data.joint_vel = q.clone(), dq.clone()
    robot = Robot()
    robot.data = data
    class Scene(dict):
        env_origins = torch.tensor([[2., 3., 0.]])
        def write_data_to_sim(self): pass
        def update(self, dt): assert dt == .0025
    command = NS(time_steps=torch.ones(1), _refresh_relative_targets=lambda:None)
    env = NS(scene=Scene(robot=robot), num_envs=1, device='cpu',
             action_manager=NS(action=torch.zeros(1, 8), prev_action=torch.zeros(1, 8)),
             command_manager=NS(get_term=lambda name:command))
    cfg = dict(physics_dt_s=.0025, direct_brake={'common_evaluation_initial_state':{'path':str(path),'sha256':sha(path)}})
    apply_common_reset(env, cfg, tmp_path)
    assert json.loads((tmp_path/'common_reset_audit.json').read_text())['maximum_state_error'] == 0
    assert not command.time_steps.any()
    env.action_manager.prev_action[0, 0] = 1
    with pytest.raises(ValueError):
        apply_common_reset(env, cfg, tmp_path)


def test_instrumentation_does_not_change_training_branch():
    path = ROOT/'results/hockey_speed_success_20261028_v1/source_snapshot_v1/train_hockey_ppo.py'
    old = path.read_text()
    new = instrument_trainer(old)
    assert old[old.index('def main():'):] == new[new.index('def main():'):]
    assert 'apply_common_reset(env, protocol, output)' in new
    with pytest.raises(ValueError):
        instrument_trainer(new)
