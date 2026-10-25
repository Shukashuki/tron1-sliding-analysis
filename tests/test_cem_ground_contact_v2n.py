"""Actual stage and discounted CEM horizon change only grounded cost."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts")); sys.path.insert(0, str(ROOT/"tests"))
import nmpc_batch_ground_contact_v2k_400hz as parent
import nmpc_batch_ground_contact_v2n_400hz as candidate
import run_cem_ground_contact_v2k as parent_runner
import run_cem_ground_contact_v2n as runner
from test_nmpc_batch_ground_contact_v2e_400hz import fixture, evaluate
from test_cem_ground_contact_v2m import objective


@pytest.mark.parametrize("gamma", [.8, .99, 1.])
def test_grounded_only_horizon_delta(gamma):
    native, reference, profile, actions = fixture()
    frozen = copy.deepcopy(native)
    values = [evaluate(module.BatchGroundContactTrackingObjective400Hz(
        profile, model_weight_n=100., discount_factor=gamma), native, reference, actions)
        for module in (parent, candidate)]
    before, after = values
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        assert len(after[channel]) == 15
        for name in before[channel]:
            if name != "drift_grounded":
                np.testing.assert_array_equal(before[channel][name], after[channel][name])
    unsupported = np.any(np.max(native["ground_normal_force_w"][..., 2], axis=-1) <= 5., axis=-1)
    delta = .02*.4*3.*unsupported
    np.testing.assert_allclose(after["stage_cost"]-before["stage_cost"], delta, atol=4e-16, rtol=1e-12)
    np.testing.assert_allclose(after["candidate_costs"]-before["candidate_costs"],
                              (delta*gamma**np.arange(5)).sum(-1), atol=6e-16, rtol=1e-12)
    for name in native: np.testing.assert_array_equal(native[name], frozen[name])


@pytest.mark.parametrize("missing", [0, 1, 8])
@pytest.mark.parametrize("load", [5., 5.0001, 25., 100.])
@pytest.mark.parametrize("beta,refspeed", [(0., 1.), (.37, 1.), (0., .3)])
def test_actual_stage_conditional_cost_and_intended_beta(missing, load, beta, refspeed):
    legacy = objective(); legacy["raw_terms"]["drift_sideslip"] = beta
    force = np.zeros((8, 2, 3)); force[:, 0, 2] = load; force[:missing] = 0.
    before = parent_runner.replace_actual_grounded(legacy, force, 100., reference_speed=refspeed)
    after = runner.replace_actual_grounded(legacy, force, 100., reference_speed=refspeed)
    gap = bool(missing or load <= 5.)
    components = after["ground_contact_v2"]
    assert components["beta_contact_gate"] == (beta if refspeed > .3 else 1.)
    assert components["support_gap_penalty"] == pytest.approx(
        (1. if missing else np.clip((50.-load)/45., 0., 1.)) + 3.*gap)
    assert after["stage_cost"]-before["stage_cost"] == pytest.approx(.02*.4*3.*gap, abs=3e-16)
    for channel in ("raw_terms", "weighted_terms", "term_costs"):
        for name in before[channel]:
            if name != "drift_grounded": assert after[channel][name] == before[channel][name]


def test_runner_pins_current_N_protocol_and_avoids_native_build():
    assert runner.sha256(ROOT/"config/ppo_ground_contact_v2n_400hz_v1.json") == runner.PROTOCOL_SHA256
    source = Path(runner.__file__).read_text()
    for forbidden in ("torch.load(", "torch.jit.load(", "build_native_library("):
        assert forbidden not in source
