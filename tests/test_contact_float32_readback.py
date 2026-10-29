from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_physx_contact_float32 import replay_contact_readback
from evaluate_hockey_physx import reconstruct_contacts


def fixture():
    n=3
    q=Rotation.from_euler('xyz',[89.99,0,30],degrees=True).as_quat()[[3,0,1,2]].astype(np.float32)
    q*=np.float32(1+2e-7)
    tr=dict(root_pos=np.zeros((n,3),np.float32),root_quat=np.tile(np.array([1,0,0,0],np.float32),(n,1)),
        root_com_vel=np.zeros((n,3),np.float32),body_com_pos=np.zeros((n,2,3),np.float32),
        body_com_vel=np.tile(np.array([1,2,0],np.float32),(n,2,1)),body_omega=np.zeros((n,2,3),np.float32),
        body_quat=np.tile(q,(n,2,1)),body_names=np.array(['wheel_L_Link','wheel_R_Link']),
        environment_origin_w=np.zeros(3,np.float32),physics_dt_s=.0025,policy_dt_s=.02)
    for k in ['normal_fz','lateral_slip_rms','utilization_proxy','friction_anchor_power']:
        tr[k]=np.zeros((n,2),np.float32)
    tr['friction_force_w']=np.zeros((n,2,3),np.float32)
    raw=dict(physical_sample_indices=np.arange(1,n))
    for side in (0,1):
        values=(np.array([[100.]],np.float32),np.zeros((1,3),np.float32),
                np.array([[0,0,1]],np.float32),np.zeros((1,1),np.float32),
                np.array([[1]],np.int32),np.array([[0]],np.int32))
        fr=(np.array([[0,-70,0]],np.float32),np.zeros((1,3),np.float32),values[4],values[5])
        for kind,vs in [('normal',values),('friction',fr)]:
            for k,v in enumerate(vs):raw[f'w{side}_{kind}_buffer_{k}']=np.stack([v,v])
    mapping=dict(normal_fz='normal_fz',normal_point_lateral_rms='lateral_slip_rms',anchor_power='friction_anchor_power',friction_force='friction_force_w',aggregate_utilization='utilization_proxy')
    for key,value in replay_contact_readback(tr,raw,.7).items():tr[mapping[key]]=value
    return tr,raw


def test_near_vertical_axle_keeps_physical_float64_result_but_audits_recorded_arithmetic():
    tr,raw=fixture()
    with pytest.raises(ValueError,match='Native readback mismatch normal_point_lateral_rms'):
        reconstruct_contacts(tr,raw,.7)
    result,audit=reconstruct_contacts(tr,raw,.7,readback_mode='recorded_float32')
    assert audit['maximum_readback_errors']['normal_point_lateral_rms']<1e-7
    assert audit['normalized_float64_differences']['normal_point_lateral_rms']>4e-5
    assert np.max(abs(result['normal_point_lateral_rms']-tr['lateral_slip_rms']))>4e-5


def test_recorded_arithmetic_does_not_accept_corrupted_slip_or_buffer_range():
    tr,raw=fixture();tr['lateral_slip_rms'][1,0]+=.01
    with pytest.raises(ValueError,match='Native readback mismatch'):
        reconstruct_contacts(tr,raw,.7,readback_mode='recorded_float32')
    tr,raw=fixture();raw['w0_normal_buffer_5'][0]=1
    with pytest.raises(ValueError,match='range'):
        replay_contact_readback(tr,raw,.7)
