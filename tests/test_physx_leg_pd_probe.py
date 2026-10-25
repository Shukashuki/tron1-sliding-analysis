"""CPU actuator-boundary fixtures, never substitutes for native PhysX evidence."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from physx_leg_pd_probe import LegPDProbe,PROFILES
import eval_physx_leg_pd_probe as evaluator


class Motor:
    def __init__(self,ids,kp,kd,limit,speed):
        self.joint_indices=torch.tensor(ids);shape=(1,len(ids))
        self.stiffness=torch.full(shape,float(kp));self.damping=torch.full(shape,float(kd))
        self.effort_limit=torch.full(shape,float(limit));self.velocity_limit=torch.full(shape,float(speed))
        self._saturation_effort=float(limit);self.cfg=SimpleNamespace(stiffness=kp,damping=kd)
        self._vel_at_effort_lim=self.velocity_limit*(1+self.effort_limit/self._saturation_effort)
    def compute(self,control_action,joint_pos,joint_vel):
        self.computed_effort=self.stiffness*(control_action.joint_positions-joint_pos)+self.damping*(control_action.joint_velocities-joint_vel)+control_action.joint_efforts
        self.applied_effort=self.computed_effort.clamp(-self.effort_limit,self.effort_limit)
        control_action.joint_efforts=self.applied_effort;control_action.joint_positions=None;control_action.joint_velocities=None
        return control_action


def fixture(profile):
    robot=SimpleNamespace(actuators={'legs':Motor(list(range(6)),500.,10.,80.,15.),'wheels':Motor([6,7],0.,0.,12.,100.)},joint_names=[f'leg{i}' for i in range(6)]+['wl','wr'])
    env=SimpleNamespace(scene={'robot':robot},num_envs=1,device='cpu',physics_dt=.0025,step_dt=.02,_sim_step_counter=0)
    term=SimpleNamespace(_asset=robot,_joint_ids=list(range(6)),apply_actions=lambda:None)
    env.action_manager=SimpleNamespace(get_term=lambda name:term)
    return env,LegPDProbe(env,profile,allow_cpu_fixture=True)


def apply(env,owner):
    env._sim_step_counter+=1
    owner.term.apply_actions()
    command=SimpleNamespace(joint_positions=torch.full((1,6),.1),joint_velocities=torch.full((1,6),.05),joint_efforts=torch.ones((1,6)))
    return owner.legs.compute(command,torch.full((1,6),.03),torch.full((1,6),-.1))


@pytest.mark.parametrize('profile',PROFILES)
def test_actual_gain_change_records_unmodified_commands_and_original_motor_output(profile):
    env,owner=fixture(profile);result=apply(env,owner);kp,kd=PROFILES[profile]
    expected=kp*(.1-.03)+kd*(.05+.1)+1.
    torch.testing.assert_close(result.joint_efforts,torch.full((1,6),expected))
    np.testing.assert_allclose(owner.rows[0]['target_pos_rad'],.1)
    assert owner.wheels.stiffness.sum()==0 and owner.wheels.damping.sum()==0
    assert owner.gain_readback()==dict(kp_range=[kp,kp],kd_range=[kd,kd])


def test_non_consecutive_motor_calls_and_partial_episode_rejected(tmp_path):
    env,owner=fixture('soft_40_2p5');apply(env,owner)
    env._sim_step_counter-=1
    with pytest.raises(AssertionError,match='consecutive'):apply(env,owner)
    with pytest.raises(AssertionError):owner.save(tmp_path/'partial.npz')


def test_complete_record_validates_actual_PD_and_does_not_relabel_training(tmp_path):
    env,owner=fixture('soft_40_2p5')
    for _ in range(2792):apply(env,owner)
    proof=owner.save(tmp_path/'actual.npz')
    assert proof['training_leg_kp']==500. and proof['actual_gain_readback']['kp_range']==[40.,40.]
    assert proof['inference_only'] and not proof['retrained_for_modified_gains']
    assert proof['physical_motor_computations']==2792
    assert proof['PD_reconstruction_max_abs_error_nm']<2e-4
    owner.wheels.effort_limit.fill_(13.)
    with pytest.raises(AssertionError):owner.save(tmp_path/'invalid.npz')


def test_parent_is_pinned_and_replacements_counted():
    path=Path(evaluator.__file__).with_name('eval_physx_wheel_contact_models.py');source=path.read_text()
    generated=evaluator.materialize(source)
    assert generated.count('_gain_install(env,current,original)')==1
    assert generated.count('_gain_finish(metadata,report,contract,current)')==1
    assert path.read_text()==source
    with pytest.raises(AssertionError):evaluator.materialize(source+'\n')


def test_native_recorder_single_world_axis_and_before_after_physics_alignment():
    names=np.array([f'j{i}' for i in range(8)]);ids=[7,4,1,2,0,3]
    actual={'joint_names':names,'applied_joint_torque_nm':np.arange(2793*8).reshape(2793,1,8).astype(float),
        'joint_position_rad':np.ones((2793,1,8)),'joint_velocity_rad_s':np.full((2793,1,8),2.)}
    measured={'leg_joint_names':names[ids],
        'applied_effort_nm':actual['applied_joint_torque_nm'][1:,0,ids].copy(),
        'joint_pos_rad':actual['joint_position_rad'][:-1,0,ids].copy(),
        'joint_vel_rad_s':actual['joint_velocity_rad_s'][:-1,0,ids].copy()}
    evaluator.verify_motor_recording(measured,actual)
    measured['applied_effort_nm'][0,0]+=1.
    with pytest.raises(AssertionError):evaluator.verify_motor_recording(measured,actual)


@pytest.mark.parametrize('profile',PROFILES)
def test_initialization_callbacks_preserve_original_gains_and_are_not_physics_samples(profile):
    env,owner=fixture(profile)
    for _ in range(3):
        command=SimpleNamespace(joint_positions=torch.ones((1,6)),joint_velocities=torch.zeros((1,6)),joint_efforts=torch.zeros((1,6)))
        owner.legs.compute(command,torch.zeros((1,6)),torch.zeros((1,6)))
        assert bool((owner.legs.stiffness==500.).all())
    assert not owner.rows and owner.action_count==0
    assert len(owner.initialization_callbacks)==3
    apply(env,owner)
    assert len(owner.rows)==owner.action_count==1
    assert owner.gains_activated
