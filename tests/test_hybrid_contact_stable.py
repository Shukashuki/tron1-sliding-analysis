"""Conservative separation certificates and invariance to rigid orientation."""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from hybrid_contact_reference_stable import cylinder_separation_certificate


def test_parallel_cylinder_axial_and_radial_separation_are_exact():
    size=np.array([.127,.025]);origin=np.zeros(3);axis=np.array([0.,1.,0.])
    assert abs(cylinder_separation_certificate(origin,axis,size,[0,.31114,0],axis,size)-.26114)<1e-12
    assert abs(cylinder_separation_certificate(origin,axis,size,[.5,0,0],axis,size)-.246)<1e-12
    assert cylinder_separation_certificate(origin,axis,size,origin,axis,size)<0.


def test_certificate_invariant_to_common_rotation_and_axis_sign():
    from scipy.spatial.transform import Rotation
    a=np.array([.2,.3,.8]);a/=np.linalg.norm(a);b=np.array([.1,-.7,.2]);b/=np.linalg.norm(b)
    c=np.array([.5,.1,-.3]);s=np.array([.05,.1])
    original=cylinder_separation_certificate(np.zeros(3),a,s,c,b,s)
    for rot in Rotation.from_euler('xyz',[[.1,.2,.3],[1.,2.,3.],[-.5,.2,-.4]]).as_matrix():
        actual=cylinder_separation_certificate(np.zeros(3),-rot@a,s,rot@c,rot@b,s)
        assert abs(actual-original)<1e-12


def test_near_parallel_axes_have_no_centimeter_scale_finite_difference_jump():
    size=np.array([.05,.025]);a=np.array([1.,0.,0.]);delta=np.array([0.,.21,0.])
    base=cylinder_separation_certificate(np.zeros(3),a,size,delta,a,size)
    perturbed=np.array([1.,1e-8,0.]);perturbed/=np.linalg.norm(perturbed)
    value=cylinder_separation_certificate(np.zeros(3),a,size,delta,perturbed,size)
    assert abs(value-base)<1e-8
