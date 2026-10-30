"""Analytic support-locus cases prevent arbitrary wheel-edge comparisons."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from compare_hockey_gmr_contact_paths import bottom_locus,point_segment_distance


def test_upright_wheel_has_a_line_of_minima():
    center=np.array([[0.,0.,.127]]);axis=np.array([[0.,1.,0.]])
    point,gap,segment=bottom_locus(center,axis,.127,.025)
    np.testing.assert_allclose(gap,0,atol=1e-12)
    np.testing.assert_allclose(point,[[0,0,0]],atol=1e-12)
    np.testing.assert_allclose(segment[0],[[0,-.025,0],[0,.025,0]],atol=1e-12)
    # Every point across the 50 mm width can be a valid bottom point.
    distance=point_segment_distance(np.array([[0,-.02],[0,.02],[.1,0],[0,.05]]),segment[0,:,:2])
    np.testing.assert_allclose(distance,[0,0,.1,.025],atol=1e-12)


def test_cambered_wheel_support_agrees_with_analytic_height():
    angle=np.pi/6;r=.127;h=.025
    axis=np.array([[0,np.cos(angle),np.sin(angle)]])
    center=np.array([[0,0,r*np.cos(angle)+h*np.sin(angle)]])
    point,gap,segment=bottom_locus(center,axis,r,h)
    np.testing.assert_allclose(gap,0,atol=1e-12)
    np.testing.assert_allclose(point,[[0,r*np.sin(angle)-h*np.cos(angle),0]],atol=1e-12)
    np.testing.assert_allclose(segment[0,:,2],[0,.0001],atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(segment[0,1]-segment[0,0]),.0002,atol=1e-12)


def test_tiny_camber_sign_change_retains_width_within_tolerance():
    # Picking only the exact minimum jumps 50 mm as camber crosses zero;
    # the near-minimum geometric locus keeps both edges within tolerance.
    axes=np.array([[0,np.sqrt(1-1e-16),1e-8],[0,np.sqrt(1-1e-16),-1e-8]])
    center=np.array([[0,0,.127],[0,0,.127]])
    point,gap,segment=bottom_locus(center,axes,.127,.025)
    assert np.linalg.norm(point[0,:2]-point[1,:2])>.049
    np.testing.assert_allclose(segment[0,:,:2],segment[1,:,:2],atol=3e-9)
    np.testing.assert_allclose(np.linalg.norm(segment[:,1]-segment[:,0],axis=-1),.05,atol=1e-12)
