"""P peak-impact cost must remain observable when positive tracking reward is zero."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ground_contact_reward_v2p as p


def test_one_wheel_and_alternating_support_below3mg_exactly_preserve_K():
    for force in (np.tile([200., 0.], (8, 1)), np.array([[200., 0.], [0., 200.]]*4)):
        old = p.previous.reward_numpy(force, 200., .8, .7, .6, 1.)
        new = p.reward_numpy(force, 200., .8, .7, .6, 1.)
        assert new['raw_reward'] == old['raw_reward'] and new['ungated_peak_impact_cost'] == 0.


def test_impact_cost_is_ungated_and_not_diluted_over16_samples():
    f = np.tile([200., 0.], (8, 1)); f[2, 0] = 1200.; f[5] = 0.
    result = p.reward_numpy(f, 200., 0., 0., 0., 1.)
    assert result['G_gated_support_bonus'] == 0.
    assert result['ungated_peak_impact_cost'] == pytest.approx(1-np.exp(-1.))
    assert result['raw_reward'] == pytest.approx(-1.-(1-np.exp(-1.)))
    assert result['impact_gate'] == pytest.approx(np.exp(-1./16.))


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_torch_numpy_batched_masses_match_without_hot_path_scalar_reads(dtype):
    rng = np.random.default_rng(94)
    f = rng.uniform(0, 1600, (3, 8, 2)).astype(dtype)
    mg = np.array([150., 200., 250.], dtype=dtype)
    expected = p.reward_numpy(f, mg, .5, .4, .3, 1.)
    with patch.object(Path, 'read_bytes', side_effect=AssertionError('No hot-path file reads')), \
            patch.object(torch.Tensor, 'item', side_effect=AssertionError('No hot-path scalar reads')):
        actual = p.reward_torch(torch.from_numpy(f), torch.from_numpy(mg), .5, .4, .3, 1., validate=False)
    for key in expected:
        np.testing.assert_allclose(actual[key].numpy(), expected[key], rtol=2e-6, atol=3e-7)
    assert ((expected['raw_reward'] >= -2.) & (expected['raw_reward'] <= 1.)).all()


def test_invalid_force_poisons_every_component_without_hiding_nan():
    f = torch.ones((2, 8, 2))*200.; f[1, 3, 0] = float('nan')
    r = p.reward_torch(f, 200., 1., 1., 1., 1., validate=False)
    assert all(torch.isnan(v[1]).all() for v in r.values())


def test_real_collector_applies_cost_once_per_complete_native_window():
    from test_ppo_ground_contact_v2p import GroundContactV2PTest
    fixture = GroundContactV2PTest(); fixture.setUp()
    try:
        fixture.env.force_sequence[3] = torch.tensor([[600., 0.], [1200., 0.]])
        owner = fixture.install()
        fixture.env.read_twice = True
        result = fixture.env.step()
        expected = p.reward_torch(owner.forces, owner.mg, fixture.anchor, fixture.speed,
            fixture.beta, torch.ones(2))['raw_reward']
        torch.testing.assert_close(result, expected)
        assert owner.unique_reads == 1 and owner.audit(health=True)['ungated_peak_impact_weight'] == 1.
    finally:
        fixture.doCleanups()
