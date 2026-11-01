import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "upstream/tron1-learning-from-human"))

from nmpc_tracking_objective import HeightSmoothTrackingObjective
from training.tron1_drift import validate_task_profile


def test_no_sideslip_profile_changes_only_one_reward_weight():
    original = json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text())
    ablated = json.loads((ROOT / "config/car_drift_height_smooth_no_sideslip_task.json").read_text())
    assert validate_task_profile(ablated) == ablated
    assert {k: v for k, v in ablated.items() if k not in ("reward_weights", "description")} == {
        k: v for k, v in original.items() if k not in ("reward_weights", "description")
    }
    expected = dict(original["reward_weights"])
    expected["sideslip"] = 0.0
    assert ablated["reward_weights"] == expected

    old = HeightSmoothTrackingObjective(original).contract()
    new = HeightSmoothTrackingObjective(ablated).contract()
    assert old["weights"]["drift_sideslip"] == pytest.approx(2.0)
    assert new["weights"]["drift_sideslip"] == pytest.approx(0.0)
    assert {k: v for k, v in old["weights"].items() if k != "drift_sideslip"} == {
        k: v for k, v in new["weights"].items() if k != "drift_sideslip"
    }
