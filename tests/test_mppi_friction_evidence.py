"""Friction geometry, power and same-point evidence rather than coincident maxima."""
from pathlib import Path
import json,sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_mppi_friction_evidence as a


@pytest.fixture
def cfg():
    return json.loads((a.g.ROOT/'config/mppi_friction_evidence_v1.json').read_text())


def test_native_pyramid_face_can_be_saturated_inside_circular_cone():
    r=a.friction_metrics(100,[-35,-35],[1,1],[.7,.7])
    assert r['rho_pyramid']==pytest.approx(1)
    assert r['rho_ellipse']==pytest.approx(1/np.sqrt(2))
    assert r['dissipation_efficiency']==pytest.approx(1)


def test_native_pyramid_maximum_dissipation_need_not_be_antiparallel():
    r=a.friction_metrics(100,[-70,0],[1,.8],[.7,.7])
    assert r['dissipation_efficiency']==pytest.approx(1)
    assert r['opposition_cosine']<.8


def test_stationary_saturated_force_is_not_evidence_of_velocity_slip():
    r=a.friction_metrics(100,[-70,0],[0,0],[.7,.7])
    assert r['rho_pyramid']==1 and r['dissipated_power_w']==0
    assert np.isnan(r['dissipation_efficiency'])


def test_interior_cone_slip_retains_positive_dissipation_but_has_weak_force_corroboration():
    r=a.friction_metrics(100,[-7,0],[1,0],[.7,.7])
    assert r['rho_pyramid']==pytest.approx(.1)
    assert r['dissipated_power_w']==7 and r['dissipation_efficiency']==pytest.approx(.1)


def points(fn,slip,rho,eta):
    n=len(fn)
    return dict(sample_index=np.zeros(n,dtype=int),wheel_index=np.zeros(n,dtype=int),Fn_n=np.asarray(fn),
        signed_lateral_m_s=np.asarray(slip),rho_pyramid=np.asarray(rho),dissipated_power_w=np.ones(n),dissipation_efficiency=np.asarray(eta))


def test_slip_and_saturation_at_different_points_cannot_create_joint_evidence(cfg):
    p=points([100.,100.],[0.,.2],[1.,.1],[1.,.1])
    fraction,denom,mask=a.matched_load_fraction(p,1,cfg)
    assert denom[0,0]==200 and not mask.any() and fraction[0,0]==0


def test_small_sliding_contact_cannot_represent_majority_wheel_load(cfg):
    p=points([6.,100.],[.2,0.],[1.,.1],[1.,.1])
    fraction,_,mask=a.matched_load_fraction(p,1,cfg)
    assert mask.tolist()==[True,False]
    assert fraction[0,0]==pytest.approx(6/106)
    assert fraction[0,0]<cfg['matched_point_normal_load_fraction_min']


def test_negative_friction_dissipation_is_flagged_not_called_force_corroborated_slip(cfg):
    p=points([100.],[.2],[1.],[-1.]);p['dissipated_power_w'][:]=-1
    fraction,_,_=a.matched_load_fraction(p,1,cfg)
    assert fraction[0,0]==0


def test_pyramid_support_function_matches_explicit_vertices_for_anisotropic_contact():
    fn=37.;mu=np.array([.4,.8]);velocity=np.array([-.3,.7])
    vertices=np.array([[mu[0],0],[-mu[0],0],[0,mu[1]],[0,-mu[1]]])*fn
    best=vertices[np.argmax(-vertices@velocity)]
    r=a.friction_metrics(fn,best,velocity,mu)
    assert r['maximum_dissipative_power_w']==pytest.approx(np.max(-vertices@velocity))
    assert r['dissipation_efficiency']==pytest.approx(1)


def test_power_and_cone_invariant_to_contact_tangent_sign_and_axis_permutation():
    f,v,mu=np.array([-12.,5.]),np.array([.4,-.3]),np.array([.4,.7])
    before=a.friction_metrics(100,f,v,mu)
    after=a.friction_metrics(100,np.array([f[1],-f[0]]),np.array([v[1],-v[0]]),mu[::-1])
    for key in before:assert after[key]==pytest.approx(before[key])


def test_complete_joint_event_and_report_use_same_field_contract(cfg):
    t=1.44+np.arange(41)*.0025
    raw=dict(time_s=t,base_com_velocity_w=np.tile([1.,0,0],(41,1)),ground_normal_force_w=np.zeros((41,2,3)))
    raw['ground_normal_force_w'][:,0,2]=100.
    p=points(np.full(41,100.),np.full(41,.2),np.ones(41),np.ones(41))
    p.update(sample_index=np.arange(41),rho_ellipse=np.ones(41),dissipated_power_w=np.full(41,14.))
    w,_=a.aggregate(p,raw,cfg);rows,sensitivity=a.summaries(p,w,raw,cfg)
    assert len(rows[0]['persistent_events'])==1
    assert rows[0]['joint_corroborated_fraction']==1
    assert rows[1]['joint_corroborated_fraction'] is None
    assert len(sensitivity)==54
    document=a.html_report(dict(rows=[dict(id='test',title='Synthetic test',rows=rows)]),'source')
    assert '<table>' in document and '100.0%' in document


def test_weak_point_is_retained_but_excluded_from_primary_joint_estimator(cfg):
    p=points([4.],[.3],[1.],[1.])
    fraction,denom,mask=a.matched_load_fraction(p,1,cfg)
    assert not mask.any() and denom[0,0]==0 and fraction[0,0]==0
