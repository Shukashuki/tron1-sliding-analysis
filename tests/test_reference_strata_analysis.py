from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from analyze_friction_grid import summarize_new_grid


def test_s_turn_uses_own_demo_and_signed_event_windows_not_primary_turn_core():
    time = np.arange(258)*.02
    position = np.column_stack((time, np.zeros_like(time)))
    trace = {"reference_t": time, "reference_xy": position,
             "case_isaac_t": time, "case_isaac_xy": position,
             "case_mujoco_t": time, "case_mujoco_xy": position}
    provenance = {"reference_spec": {
        "reference_id": "doa_sa29_scale2", "stratum": "high_sideslip_s_avoidance",
        "evaluation_core_interval_s": [0., 3.14],
        "supplementary_event_windows_s": [[.78, 1.54], [1.88, 3.04]]}}
    result = summarize_new_grid(trace, provenance, {"case": (.7, "")})
    assert result["core_interval_s"] == [0., 3.14]
    assert result["pooled_success_statistics"] is False
    assert result["task_stratum"] == "high_sideslip_s_avoidance"
    assert [event["interval_s"] for event in result["supplementary_event_results"]] == [[.78, 1.54], [1.88, 3.04]]
    assert all(event["rows"][0]["sim2sim"]["full_core_xy_rmse_m"] == 0.
               for event in result["supplementary_event_results"])


def test_legacy_grid_without_catalog_keeps_original_metrics_layout():
    time = np.array([0., 1.44, 2., 4., 5.])
    position = np.column_stack((time, np.zeros_like(time)))
    trace = {"reference_t": time, "reference_xy": position,
             "case_isaac_t": time, "case_isaac_xy": position,
             "case_mujoco_t": time, "case_mujoco_xy": position}
    result = summarize_new_grid(trace, {}, {"case": (.7, "")})
    assert result["core_interval_s"] == [1.44, 4.]
    assert "task_stratum" not in result
