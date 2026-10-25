"""Loaded slip excludes unknown zero placeholders and off-core samples."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_contact_taskspace_scratch as analysis
import render_contact_taskspace_scratch as render


def fixture():
    force = np.zeros((4, 2, 3)); force[:,:,2] = [[10.,0.],[10.,20.],[0.,20.],[100.,100.]]
    return dict(time_s=np.array([1.44, 1.4425, 1.445, 4.0025]), floor_normal_force=force,
        contact_count=np.array([[0,0],[1,1],[0,1],[2,2]]), contact_slip=np.array([[0.,0.],[.2,.05],[0.,.4],[9.,9.]]))


def test_no_point_zero_is_unknown_and_outside_core_not_included():
    result = analysis.slip_summary(fixture()); left = result["wheels"]["left"]; right = result["wheels"]["right"]
    assert left["core_loaded_wheel_samples"] == 2 and left["core_identified_loaded_point_samples"] == 1
    assert left["core_loaded_samples_without_identified_strong_point"] == 1
    assert left["core_lateral_point_RMS_m_s_median"] == .2
    assert left["core_identified_loaded_fraction_RMS_over_0p1_m_s"] == 1.
    assert right["core_lateral_point_RMS_m_s_median"] == pytest.approx(.225)
    assert right["core_identified_loaded_fraction_RMS_over_0p1_m_s"] == .5
    assert not result["new_task_gate"] and not result["signed_slip_or_individual_point_vectors_retained"]


def test_strict_load_threshold_and_missing_measurements_do_not_imply_stick():
    t = fixture(); t["floor_normal_force"][:] = 0.; t["floor_normal_force"][:,:,2] = 5.
    result = analysis.slip_summary(t)
    for side in result["wheels"].values():
        assert side["core_identified_loaded_point_samples"] == 0 and side["core_lateral_point_RMS_m_s_median"] is None


def test_invalid_placeholder_or_negative_RMS_rejected():
    t = fixture(); t["contact_slip"][0,0] = .1
    with pytest.raises(AssertionError): analysis.slip_summary(t)
    t = fixture(); t["contact_slip"][1,0] = -.1
    with pytest.raises(AssertionError): analysis.slip_summary(t)


def test_frozen_renderer_integration_preserves_native_pose_and_fixed_camera():
    p = Path(__file__).resolve().parents[1]/"scripts/render_contact_warm_refinement.py"
    source = render.materialize(p.read_text())
    assert "paired_scratch_taskspace_K_M_contact_dual_engine" in source
    assert "camera = graphics.prepare_camera" in source
    assert "camera_yaw_follow=False, camera_zoom=False, robot_mesh_alpha=1." in source
    assert "same K checkpoint" not in source
    with pytest.raises(AssertionError): render.materialize(p.read_text()+'\n')


def test_body_contact_separates_initial_buffer_without_height_failure_gate():
    trace = dict(time_s=np.array([0., .0025, 6.96]),
        quaternion=np.array([[1.,0.,0.,0.]]*3), position=np.array([[0.,0.,1.],[0.,0.,.5],[0.,0.,.25]]))
    result = analysis.body_integrity({"nonwheel_ground_contact_force_norm_n":np.array([[60.],[0.],[7.]])}, trace)
    assert result["initial_nonwheel_force_norm_n"] == 60.
    assert result["first_postinitial_nonwheel_force_over5N_time_s"] == 6.96
    assert result["postinitial_nonwheel_force_over5N_samples"] == 1
    assert result["root_height_range_m"] == [.25,1.]
    assert not result["absolute_height_or_flight_gate_added"]
