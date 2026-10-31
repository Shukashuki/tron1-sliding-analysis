import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_matched_contact_geometry import cylinder_gaps
from nmpc_tracking_objective import quaternion_rotate


def test_camber_and_opposite_collider_offsets_do_not_create_false_gap():
    angle=np.deg2rad(20)
    quat=np.array([np.cos(angle/2),np.sin(angle/2),0.,0.])
    geom_quat=np.array([np.sqrt(.5),np.sqrt(.5),0.,0.])
    geometry=[dict(radius_m=.127,half_width_m=.025,local_center_m=[0,sign*.00707,0],
                   local_quaternion_wxyz=geom_quat) for sign in (1,-1)]
    extent=.127*np.cos(angle)+.025*np.sin(angle)
    positions=np.array([[[0,0,extent-.00707*np.sin(angle)],[0,0,extent+.00707*np.sin(angle)]]])
    trace=dict(wheel_pos=positions,wheel_quaternion_wxyz=np.tile(quat,(1,2,1)))
    np.testing.assert_allclose(cylinder_gaps(trace,geometry,quaternion_rotate),np.zeros((1,2)),atol=1e-12)
    # A link-centered calculation would incorrectly report millimeter gaps.
    assert np.max(np.abs(positions[:,:,2]-extent))>.002
