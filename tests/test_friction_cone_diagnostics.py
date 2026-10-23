"""Recorded-force algebra only: no simulator, reward, training or acceptance edits."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import friction_cone_diagnostics as mod


def diag(fn,ft,mu=(1.,1.),**kwargs):
    return mod.contact_diagnostics(fn,ft,mu,force_units="N",**kwargs)


def test_elliptic_anisotropic_formula_and_engine_boundary_are_distinct():
    d=diag([10.],[[3.,4.]],[.5,1.])
    assert d["coulomb_reference_rho_l2"][0]==pytest.approx(np.sqrt(6**2+4**2)/10)
    assert np.isnan(d["engine_rho"]).all() and not d["engine_boundary_available"].any()
    assert d["engine_boundary"] is None and not d["task_acceptance_changed"]
    assert d["thresholds"]["normal_force_n_strict"]==1e-6


def test_mujoco_condim3_pyramid_diagonal_differs_from_elliptic_reference():
    d=diag([1.],[[.7,.7]],engine_boundary="mujoco_pyramidal_condim3",condim=3,engine_tangents_aligned=True)
    assert d["coulomb_reference_rho_l2"][0]==pytest.approx(np.sqrt(.98))
    assert d["engine_rho"][0]==pytest.approx(1.4)
    assert d["coulomb_reference_near_boundary"][0] and not d["coulomb_reference_above_boundary"][0]
    assert d["engine_near_boundary"][0] and d["engine_above_boundary"][0]
    e=diag([1.],[[.7,.7]],engine_boundary="elliptic_condim3",condim=3,engine_tangents_aligned=True)
    np.testing.assert_array_equal(e["engine_rho"],e["coulomb_reference_rho_l2"])
    assert np.isnan(e["mujoco_aligned_condim3_rho_l1"]).all()


@pytest.mark.parametrize("condim,aligned",[(None,True),(4,True),(6,True),(3,False)])
def test_only_verified_aligned_condim3_allows_engine_boundary(condim,aligned):
    with pytest.raises(ValueError,match="condim3"):
        diag([1.],[[.1,.1]],engine_boundary="mujoco_pyramidal_condim3",condim=condim,engine_tangents_aligned=aligned)


def test_resultant_lower_bound_never_claims_actual_point_or_engine_boundary():
    d=diag([10.],[[3.,4.]],data_scope="wheel_resultant_lower_bound",tangent_basis="world_xy_flat_floor",normal_threshold_n=5.)
    assert d["resultant_reference_lower_bound_only"] and not d["actual_per_contact_measurement_claimed"]
    assert d["coulomb_reference_rho_l2"][0]==.5 and np.isnan(d["engine_rho"]).all()
    with pytest.raises(ValueError,match="resultant"):
        diag([10.],[[3.,4.]],data_scope="wheel_resultant_lower_bound",engine_boundary="elliptic_condim3",condim=3,engine_tangents_aligned=True)


def test_isotropic_tangent_rotation_frame_invariance_and_anisotropic_axis_permutation():
    rng=np.random.default_rng(42);ft=rng.normal(size=(20,2));slip=rng.normal(size=(20,2));fn=np.full(20,4.)
    theta=.61;rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    a=diag(fn,ft,[.7,.7],slip_m_s=slip);b=diag(fn,ft@rotation.T,[.7,.7],slip_m_s=slip@rotation.T)
    np.testing.assert_allclose(a["coulomb_reference_rho_l2"],b["coulomb_reference_rho_l2"],rtol=3e-15)
    np.testing.assert_allclose(a["slip_total_speed_m_s"],b["slip_total_speed_m_s"],rtol=3e-15)
    a=diag(fn,ft,[.3,.9]);b=diag(fn,ft[:,::-1],[.9,.3])
    np.testing.assert_allclose(a["coulomb_reference_rho_l2"],b["coulomb_reference_rho_l2"],rtol=0,atol=0)


def test_unilateral_wheel_support_and_zero_force_are_missing_ratio_not_zero():
    d=diag([100.,0.],[[60.,0.],[0.,0.]])
    assert d["loaded"].tolist()==[True,False]
    assert d["cone_valid_loaded"].tolist()==[True,False] and np.isnan(d["coulomb_reference_rho_l2"][1])
    a=mod.aggregate_wheels(d,["L","R"])
    assert a["wheels"]["L"]["maximum_coulomb_reference_rho_l2"]==.6
    assert np.isnan(a["wheels"]["R"]["maximum_coulomb_reference_rho_l2"])
    assert np.isnan(a["wheels"]["R"]["normal_load_weighted_coulomb_reference_rho_l2"])


def test_max_and_normal_load_weighted_aggregation_never_cancel_opposing_forces():
    d=diag([10.,10.,30.],[[8.,0.],[-8.,0.],[3.,0.]])
    a=mod.aggregate_wheels(d,["L","L","R"])
    assert a["wheels"]["L"]["maximum_coulomb_reference_rho_l2"]==.8
    assert a["wheels"]["L"]["normal_load_weighted_coulomb_reference_rho_l2"]==.8
    b=mod.aggregate_wheels(d,["L","R","L"])
    assert b["wheels"]["L"]["normal_load_weighted_coulomb_reference_rho_l2"]==pytest.approx(.275)
    assert not a["vector_cancellation_used"] and not a["temporal_averaging_performed"]


def test_small_load_coverage_numerical_threshold_is_not_wheel_support_5N():
    fn=np.array([[0.,1e-6,1.01e-6,2.]])
    d=diag(fn,np.zeros((1,4,2)))
    assert d["loaded"].tolist()==[[False,False,True,True]]
    a=mod.aggregate_wheels(d,["L","L","R","R"])
    assert a["wheels"]["L"]["excluded_small_load_contacts"].tolist()==[2]
    assert a["wheels"]["L"]["excluded_small_load_normal_force_n"][0]==1e-6
    assert a["wheels"]["R"]["loaded_contacts"].tolist()==[2]


@pytest.mark.parametrize("bad_mu",[[0.,1.],[-1.,1.],[np.nan,1.],[np.inf,1.]])
def test_zero_invalid_or_nonfinite_mu_no_epsilon_fabrication(bad_mu):
    d=diag([10.],[[1.,1.]],bad_mu)
    assert not d["effective_mu_valid"].any() and not d["cone_valid_loaded"].any()
    assert np.isnan(d["coulomb_reference_rho_l2"]).all()


def test_nonfinite_or_negative_forces_have_explicit_masks_and_nan_ratios():
    d=diag([np.nan,-1.,10.,10.],[[1.,1.],[1.,1.],[np.nan,0.],[1.,np.inf]])
    assert d["normal_valid"].tolist()==[False,False,True,True]
    assert not d["cone_valid_loaded"].any() and np.isnan(d["coulomb_reference_rho_l2"]).all()
    a=mod.aggregate_wheels(d,["L","L","R","R"])
    assert a["wheels"]["L"]["invalid_normal_contacts"]==2
    assert a["wheels"]["R"]["invalid_tangent_force_contacts"]==2


def test_missing_required_data_empty_contacts_and_missing_optional_slip_not_fabricated():
    for fn,ft,mu in ((None,[[0.,0.]],[1.,1.]),([1.],None,[1.,1.]),([1.],[[0.,0.]],None)):
        d=mod.contact_diagnostics(fn,ft,mu,force_units="N")
        assert not d["available"] and "missing" in d["reason"] and "coulomb_reference_rho_l2" not in d
        assert not mod.aggregate_wheels(d,["L"])["available"]
    d=diag(np.zeros((0,)),np.zeros((0,2)))
    assert not d["available"] and d["reason"]=="no contact samples"
    d=diag([100.],[[95.,0.]])
    assert np.isnan(d["slip_total_speed_m_s"]).all() and not d["slip_total_valid"].any()
    assert d["reference_near_boundary_total_slip_unavailable"].all()
    assert not d["reference_near_boundary_low_measured_total_slip"].any()


def test_force_boundary_and_motion_are_independent_not_sliding_equals_saturation():
    d=diag([100.,100.,100.],[[95.,0.],[10.,0.],[95.,0.]],slip_m_s=[[0.,0.],[.1,0.],[.1,0.]])
    assert d["reference_near_boundary_low_measured_total_slip"].tolist()==[True,False,False]
    assert d["observed_total_slip"].tolist()==[False,True,True]
    assert d["reference_near_boundary_with_total_slip"].tolist()==[False,False,True]
    assert not d["static_friction_state_inferred"] and not d["task_acceptance_changed"]


def test_contact_tangent2_is_not_wheel_lateral_without_explicit_basis_or_measurement():
    d=diag([10.,10.],[[1.,0.],[1.,0.]],slip_m_s=[[0.,.1],[0.,0.]])
    assert d["observed_total_slip"].tolist()==[True,False]
    assert not d["wheel_lateral_slip_valid"].any() and np.isnan(d["wheel_lateral_slip_abs_m_s"]).all()
    e=diag([10.,10.],[[1.,0.],[1.,0.]],slip_m_s=[[0.,.1],[0.,0.]],tangent_basis="wheel_longitudinal_lateral")
    assert e["observed_wheel_lateral_slip"].tolist()==[True,False]
    f=diag([10.,10.],[[1.,0.],[1.,0.]],wheel_lateral_slip_m_s=[np.nan,.1])
    assert f["wheel_lateral_slip_valid"].tolist()==[False,True]
    assert not f["slip_total_valid"].any() and np.isnan(f["slip_total_speed_m_s"]).all()
    assert f["observed_wheel_lateral_slip"].tolist()==[False,True]


def test_strict_thresholds_optional_nonfinite_slip_and_input_immutability():
    fn=np.array([10.,10.]);ft=np.array([[9.,0.],[9.,0.]]);copy=ft.copy()
    d=diag(fn,ft,slip_m_s=[[.05,0.],[np.nan,0.]])
    assert d["coulomb_reference_near_boundary"].all() and not d["observed_total_slip"].any()
    assert d["slip_total_valid"].tolist()==[True,False]
    np.testing.assert_array_equal(ft,copy)


def test_force_units_are_required_and_impulses_are_never_silently_converted():
    with pytest.raises(TypeError):mod.contact_diagnostics([1.],[[0.,0.]],[1.,1.])
    with pytest.raises(ValueError,match="not auto-converted"):
        mod.contact_diagnostics([1.],[[0.,0.]],[1.,1.],force_units="N*s")


@pytest.mark.parametrize("kwargs",[{"normal_threshold_n":-1.},{"slip_threshold_m_s":np.nan},{"saturation_threshold":1.1},
    {"normal_threshold_n":True},{"tangent_basis":"guessed_wheel_axis"},{"data_scope":"unknown"},{"engine_boundary":"general_condim6"}])
def test_bad_configuration_rejected(kwargs):
    with pytest.raises(ValueError):diag([1.],[[0.,0.]],**kwargs)


def test_shapes_explicit_mu2_and_wheel_assignment_rejected():
    for fn,ft,mu in ((1.,[0.,0.],[1.,1.]),([1.],[[0.,0.]],1.),([1.],[[0.,0.,0.]],[1.,1.])):
        with pytest.raises(ValueError):diag(fn,ft,mu)
    d=diag([1.],[[0.,0.]])
    with pytest.raises(ValueError):mod.aggregate_wheels(d,["unknown"])
    with pytest.raises(ValueError):diag([1.],[[0.,0.]],slip_m_s=[0.,0.])
    with pytest.raises(ValueError):diag([1.],[[0.,0.]],wheel_lateral_slip_m_s=0.)

