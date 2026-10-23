"""Exact native frame selections and any-wheel contact overlays, no simulation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import render_ground_contact_v2 as mod


def test_native_frames_select_existing_samples_without_interpolation():
    indices=mod.native_indices()
    assert len(indices)==350 and indices[0]==0 and indices[-1]==2792
    assert np.array_equal(np.diff(indices),np.full(349,8))


def test_eight_substep_overlay_allows_one_or_alternating_wheel():
    forces=np.zeros((2793,2,3));forces[:,0,2]=10
    trace={"contact_force":forces};first=mod.contact_window(trace,16)
    forces[::2,0,2]=0;forces[::2,1,2]=10
    second=mod.contact_window(trace,16)
    assert first["supported_samples"]==second["supported_samples"]==8
    assert first["native_indices"].tolist()==list(range(9,17))
    assert second["single_wheel_lift_allowed"] is True and second["flight_inferred"] is False


def test_threshold_strict_and_initial_frame_not_fabricated_eight_samples():
    forces=np.ones((2793,2,3))*5;trace={"contact_force":forces}
    assert mod.contact_window(trace,8)["supported_samples"]==0
    first=mod.contact_window(trace,0)
    assert first["native_indices"].tolist()==[0] and first["samples"]==1


@pytest.mark.parametrize("index",[-1,2793,1.2,True])
def test_invalid_native_sample_refused(index):
    with pytest.raises(ValueError):mod.contact_window({"contact_force":np.zeros((2793,2,3))},index)


def test_existing_output_refused_before_loading_or_rendering(tmp_path):
    with pytest.raises(ValueError,match="fresh output"):
        mod.run(SimpleNamespace(output_dir=tmp_path))


def test_render_source_does_not_step_dynamics_or_load_policies():
    source=Path(mod.__file__).read_text()
    assert "mj_step(" not in source and "torch.load(" not in source and "torch.jit.load(" not in source
    assert "prepare_camera(runtime,traces,qposes,order,native_indices())" in source
    assert '"camera_distance_same_for_all_cases_frames":True' in source
