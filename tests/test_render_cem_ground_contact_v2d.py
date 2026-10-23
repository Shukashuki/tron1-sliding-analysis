"""CEM renderer proof/clock/force/caption tests; no rendering or physics jobs."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import render_cem_ground_contact_v2d as mod
from nmpc_tracking_objective import JOINT_NAMES


@pytest.fixture(scope="module")
def actual():
    """Arithmetic-only strict loading/recomputation of already recorded episodes."""
    args = mod.parser().parse_args(["--output-dir", "never-created-by-tests"])
    reference = mod.graphics.metrics.reference_data(args.reference)
    return args, reference, mod.load_cases(args, reference)


def raw_report():
    n = 2793
    raw = {"qpos": np.zeros((n, 15)), "qvel": np.zeros((n, 14)), "actions": np.zeros((349, 8)),
        "time_s": np.arange(n)*.0025, "physics_dt_s": np.asarray(.0025), "action_joint_names": np.asarray(JOINT_NAMES),
        "root_pos": np.zeros((n, 3)), "root_quaternion_wxyz": np.tile([1., 0., 0., 0.], (n, 1)),
        "base_com_velocity_w": np.zeros((n, 3)), "base_angular_velocity_w": np.zeros((n, 3)),
        "joint_pos": np.zeros((n, 8)), "joint_vel": np.zeros((n, 8)), "wheel_pos": np.zeros((n, 2, 3)),
        "wheel_ground_contact_count": np.zeros((n, 2)), "wheel_ground_contact_force_w_n": np.zeros((n, 2, 3)),
        "wheel_lateral_contact_slip_m_s": np.zeros((n, 2)), "ground_normal_force_w": np.zeros((n, 2, 3))}
    raw["ground_normal_force_w"][:, 0, 2] = 20.
    report = {"status": "completed", "smoke_only": False, "truncated_smoke": False, "completed_policy_steps": 349,
        "native_samples": 2793, "condition": "beta_on", "plant_mu": .7, "prediction_mu_scenarios": [.7],
        "prediction_friction_known": True, "motion_sha256": "ref", "model_sha256": "model"}
    return raw, report


def test_frozen_source_and_actual_analysis_pins():
    assert mod.sha256(mod.graphics.__file__) == mod.GRAPHICS_SHA256
    assert mod.sha256(mod.audit.__file__) == mod.ANALYZER_SHA256
    assert mod.sha256(mod.DEFAULT_ANALYSIS) == mod.ANALYSIS_SHA256
    for name in ("native_indices", "contact_window", "heading_glyphs", "case_contact_caption", "draw_map"):
        assert getattr(mod, name) is getattr(mod.graphics, name)


def test_actual_complete_four_cases_are_not_B_or_PPO(actual):
    args, reference, (traces, cases, evidence, order, binding) = actual
    assert order == ["original", "v2a", "v2c", "v2d"]
    assert len(traces) == len(cases) == len(evidence) == 4
    assert binding["all_four_metrics_recomputed_and_bound"]
    assert binding["B_excluded_from_display"]
    assert all("CEM" in case["title"] and "PPO" not in case["title"] for case in cases.values())
    assert all(case["engine"] == "native CPU MuJoCo" for case in cases.values())
    assert cases["original"]["new_nominal_screen"] is None
    assert cases["original"]["baseline_screen_not_applied_to_itself"]
    assert "B" not in order and "v2b" not in cases


def test_actual_clock_full_native_tail_and_default_late_poster(actual):
    args, _, (traces, _, _, _, _) = actual
    assert args.poster_frame == 348
    assert mod.native_indices()[348] == 2784
    assert mod.native_indices()[349] == 2792
    assert mod.FRAMES == 350 and mod.FPS == 50
    for tr in traces.values():
        assert len(tr["time_s"]) == 2793
        np.testing.assert_allclose(tr["time_s"], np.arange(2793)*.0025, rtol=0, atol=1e-9)
        assert tr["time_s"][-1] == pytest.approx(6.98)


def test_all_actual_force_arrays_are_floor_normal_not_net_body(actual):
    _, _, (traces, _, evidence, _, _) = actual
    assert evidence["original"]["floor_force_channel"] == "wheel_ground_contact_force_w_n"
    for key, trace in traces.items():
        assert evidence[key]["floor_force_channel"] == ("wheel_ground_contact_force_w_n" if key == "original" else "ground_normal_force_w")
        assert np.array_equal(trace["contact_force"], trace["floor_normal_force"])
        assert trace["contact_force"].shape == (2793, 2, 3)


def test_d_actual_late_departure_not_cropped_or_height_corrected(actual):
    _, reference, (traces, _, _, _, _) = actual
    index = int(mod.native_indices()[348])
    value = mod.graphics.diagnostics(traces["v2d"], reference, index)
    assert value["xy_error_m"] == pytest.approx(1.383, abs=.001)
    assert traces["v2d"]["position"].shape == (2793, 3)


def test_actual_per_case_contact_captions_not_hardcoded_cem_footer(actual):
    _, _, (_, cases, _, order, _) = actual
    expected = {"original": "74.68%", "v2a": "70.47%", "v2c": "64.31%", "v2d": "70.36%"}
    for key in order:
        assert expected[key] in mod.case_contact_caption(cases[key])
    text = mod.outcome_caption(cases, order)
    assert text.startswith("CEM ONLY") and "Original = historical baseline" in text
    assert "V2A: FAIL 2" in text and "V2C: FAIL 7" in text and "V2D: FAIL 5" in text
    assert "B omitted" in text and "NOT claimed" in text
    assert "missing" not in text.lower() and "PPO" not in text and "70.47" not in text


@pytest.mark.parametrize("field,value", [("status", "running"), ("status", "pending"), ("smoke_only", True),
    ("truncated_smoke", True), ("completed_policy_steps", 348), ("native_samples", 2785)])
def test_pending_smoke_or_prefix_rejected(field, value):
    raw, report = raw_report(); report[field] = value
    with pytest.raises(ValueError, match="terminal349-step"): mod.validate_episode(raw, report, "ref", "model")


@pytest.mark.parametrize("field,value", [("condition", "beta_off"), ("plant_mu", .15), ("prediction_mu_scenarios", [.15]),
    ("prediction_friction_known", False), ("motion_sha256", "other"), ("model_sha256", "other")])
def test_wrong_identity_or_known_material_rejected(field, value):
    raw, report = raw_report(); report[field] = value
    with pytest.raises(ValueError, match="reference/model/beta/known material"): mod.validate_episode(raw, report, "ref", "model")


def test_native_shape_splice_clock_and_force_errors_rejected():
    raw, report = raw_report(); raw["actions"] = raw["actions"][:348]
    with pytest.raises(ValueError, match="Incomplete native"): mod.validate_episode(raw, report, "ref", "model")
    raw, report = raw_report(); raw["time_s"][100:] += .0025
    with pytest.raises(ValueError, match="consecutive"): mod.validate_episode(raw, report, "ref", "model")
    raw, report = raw_report(); raw["ground_normal_force_w"][20, 0, 2] = np.nan
    with pytest.raises(ValueError, match="floor-normal"): mod.validate_episode(raw, report, "ref", "model")
    raw, report = raw_report(); raw["ground_normal_force_w"][20, 0, 2] = -.1
    with pytest.raises(ValueError, match="floor-normal"): mod.validate_episode(raw, report, "ref", "model")


def test_force_only_single_support_and_exact_eight_samples_ignore_count_and_netforce():
    raw, report = raw_report()
    # Old net/all-body channel and point counts cannot override explicit floor force.
    raw["wheel_ground_contact_force_w_n"][:] = 10000.
    raw["ground_normal_force_w"][12, :, 2] = 5.
    trace, key = mod.validate_episode(raw, report, "ref", "model")
    assert key == "ground_normal_force_w" and not trace["contact_count"].any()
    window = mod.contact_window(trace, 16)
    assert window["indices"].tolist() == list(range(9, 17))
    assert window["supported_samples"] == 7 and window["samples"] == 8
    assert mod.contact_window(trace, 0)["samples"] == 1
    assert np.all(np.delete(trace["contact_force"][:, 1], 12, axis=0) == 0.)
    assert trace["contact_force"][12, 1, 2] == 5.  # Exactly5N remains unsupported.


def test_frozen_summary_mutation_or_pending_rejected_before_graphics(actual, monkeypatch):
    args, reference, _ = actual
    monkeypatch.setattr(mod, "ANALYSIS_SHA256", "0"*64)
    with pytest.raises(ValueError, match="analysis summary changed"): mod.load_cases(args, reference)


def test_metric_binding_cannot_relabel_missing_or_changed_D(actual):
    _, _, (_, cases, evidence, _, _) = actual
    summary = mod.read_json(mod.DEFAULT_ANALYSIS)
    metrics = {key: value["analysis"] for key, value in cases.items()}
    hashes = {key: value["input_sha256"] for key, value in evidence.items()}
    rewards = {key: value["actual_reward_audit"] for key, value in evidence.items() if key != "original"}
    mod.assert_analysis(summary, metrics, hashes, rewards, cases["v2d"]["new_nominal_screen"])
    changed = copy.deepcopy(summary); changed["status"] = "running"
    with pytest.raises(ValueError, match="completed ACTUAL D analysis"):
        mod.assert_analysis(changed, metrics, hashes, rewards, cases["v2d"]["new_nominal_screen"])
    changed = copy.deepcopy(metrics); changed["v2d"] = changed["v2a"]
    with pytest.raises(ValueError, match="D actual trace/reward/screen differs"):
        mod.assert_analysis(summary, changed, hashes, rewards, cases["v2d"]["new_nominal_screen"])


def test_synthetic_canvas_cem_header_and_outcome_override_no_stale_ppo(actual, monkeypatch):
    from PIL import Image, ImageDraw
    _, _, (_, cases, _, order, _) = actual
    calls = []; canvas = Image.new("RGB", (2560, 850))
    monkeypatch.setattr(mod.graphics, "render_frame", lambda *args: canvas)
    class Draw:
        def rectangle(self, box, **kwargs): calls.append(("rectangle", box))
        def text(self, point, text, **kwargs): calls.append(("text", text))
    monkeypatch.setattr(ImageDraw, "Draw", lambda image: Draw())
    assert mod.render_frame(NS(small=None), {}, {}, {}, cases, order, {}, None, 348) is canvas
    text = [value for kind, value in calls if kind == "text"]
    assert "CEM ONLY Original/A/C/D" in text[0] and "t=6.96s" in text[0]
    assert "B excluded" in text[0] and "native MuJoCo" in text[0]
    assert text[1].startswith("CEM ONLY") and "No policy result claimed" in text[2]
    assert all("New PPO" not in value and "NOT DISPLAYED" not in value for value in text)
    assert [value for kind, value in calls if kind == "rectangle"] == [(0, 0, 2560, 29), (0, 763, 2560, 786), (0, 829, 2560, 849)]


def test_overwrite_rejected_and_static_geometry_contracts_preserved(tmp_path):
    with pytest.raises(ValueError, match="fresh output"): mod.run(NS(output_dir=tmp_path))
    source = Path(mod.__file__).read_text()
    for forbidden in ("torch.load(", "torch.jit.load(", "mj_step(", "render_ground_contact_v2 as"):
        assert forbidden not in source
    assert 'graphics.prepare_camera(' in source and 'graphics.world_bounds(traces, reference)' in source
    assert 'graphics.build_qpos(' in source and 'graphics.replay_fk_audit(' in source
    assert '"shared_world_bounds_all_panels": True' in source
    assert '"camera_same_distance_all_cases_frames": True' in source
    assert '"camera_yaw_follow": False' in source and '"camera_zoom": False' in source
    assert '"smoothing_interpolation_alignment_height_correction": False' in source
    assert '"CEM_only_no_PPO_result": True' in source and '"goal_completion_claimed": False' in source
    assert '"robot_mesh_alpha": 1.' in source

