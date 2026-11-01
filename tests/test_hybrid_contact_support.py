"""Exact geometric support and independent per-time force dependencies."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import mujoco

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from hybrid_contact_reference_support import SupportProblem


@pytest.fixture
def problem():
    with np.load(ROOT/'results/contact_wheel_axis_20261025_v1/reference.npz',allow_pickle=False) as f:s={k:f[k].copy() for k in f.files}
    m=mujoco.MjModel.from_xml_path(str(ROOT/'outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml'))
    cfg=json.loads((ROOT/'config/hybrid_contact_reference_search_v4.json').read_text())
    return SupportProblem(m,s,cfg)


def test_moving_body_and_legs_keep_rolling_wheel_geometrically_supported(problem):
    p=problem;c=p.C.copy();c[5,0]=.1;c[6,4]=-.12;c[5,7]=.9
    _,record=p.evaluate(c,save=True);gap=record['points'][:,2*p.rolling:2*p.rolling+2,2].min(-1)
    np.testing.assert_allclose(gap,0.,atol=1e-12,rtol=0)
    assert np.ptp(record['qpos'][:,2])>.001
    assert abs(record['qpos'][0]-p.source['qpos_mujoco'][0]).max()<1e-12


def test_a_single_force_sample_has_no_cross_time_effect_and_sparsity_covers_it(problem):
    p=problem;x=p.vector();initial=p.residual(x)
    index=p.state_size+120*12+2;changed=x.copy();changed[index]+=.001
    after=p.residual(changed);allowed=p.sparsity()[:,index].toarray().ravel().astype(bool)
    assert np.max(abs(after[~allowed]-initial[~allowed]))<1e-12
    assert np.max(abs(after[allowed]-initial[allowed]))>0
    lo,hi=p.bounds();assert lo.shape==hi.shape==x.shape
