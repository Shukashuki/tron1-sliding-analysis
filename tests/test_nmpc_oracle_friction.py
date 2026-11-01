"""The privileged friction input is explicit and leaves ordinary MPC unchanged."""

from argparse import Namespace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_nmpc_tracking import select_prediction_friction


def args(**changes):
    values = dict(plant_mu=.2, oracle_plant_friction=False,
                  prediction_mu_scenarios=None, adaptive_contact_config=None)
    values.update(changes)
    return Namespace(**values)


def test_default_prediction_stays_predeclared():
    assert select_prediction_friction(args(), (.7,)) == ((.7,), "predeclared")
    assert select_prediction_friction(args(prediction_mu_scenarios=[.3, .7]), (.3, .7)) == (
        (.3, .7), "predeclared")


def test_oracle_uses_exact_plant_value_even_at_nominal():
    assert select_prediction_friction(args(oracle_plant_friction=True), (.7,)) == (
        (.2,), "plant_friction_oracle")
    assert select_prediction_friction(args(plant_mu=.7, oracle_plant_friction=True), (.7,)) == (
        (.7,), "plant_friction_oracle")


@pytest.mark.parametrize("changes", [
    dict(prediction_mu_scenarios=[.7]),
    dict(adaptive_contact_config=Path("adaptation.json")),
    dict(plant_mu=-.1),
    dict(plant_mu=float("nan")),
])
def test_oracle_rejects_conflicting_or_invalid_inputs(changes):
    with pytest.raises(ValueError):
        select_prediction_friction(args(oracle_plant_friction=True, **changes), (.7,))
