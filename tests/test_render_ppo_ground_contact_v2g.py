"""Synthetic-only F renderer checks: no F actor, result, graphics or physics."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import render_ppo_ground_contact_v2g as mod


def trace():
    n = 2793; force = np.zeros((n, 2, 3)); force[:, 0, 2] = 20.
    return {"position": np.zeros((n, 3)), "contact_force": force, "floor_normal_force": force.copy(),
        "contact_count": np.zeros((n, 2)), "wheel_pos": np.zeros((n, 2, 3)),
        "body_heading_geometry_valid": np.ones(n, dtype=bool), "wheel_heading_geometry_valid": np.ones((n, 2), dtype=bool),
        "signals": {"body_heading_rad": np.zeros(n), "course_rad": np.zeros(n), "speed_m_s": np.ones(n),
                    "wheel_rolling_plane_heading_rad": np.zeros((n, 2))}}


def args():
    return NS(physx_dir="Ephysx", mujoco_dir="Emujoco", subset=None, training_audit=Path("explicit_E_training.json"),
        protocol=mod.audit.DEFAULT_PROTOCOL, baseline_root="oldroot", baseline_summary="oldsummary")


def fake_audit(monkeypatch, *, wrong_binding=False):
    calls = []; training = {"exported_policy_sha256": "a"*64, "checkpoint_sha256": "b"*64}
    monkeypatch.setattr(mod.audit, "load_protocol", lambda path: {})
    monkeypatch.setattr(mod.audit, "load_training_audit", lambda path, protocol: (training, {"sha256": "c"*64}))
    def new(folder, engine, protocol, *, training_audit):
        calls.append(("G", engine, training_audit))
        return {}, {"contract": {}, "actor_sha256": "a"*64, "checkpoint_sha256": "b"*64,
            "formal_training_source_evidence": {"sha256": ("d" if wrong_binding else "c")*64}}
    monkeypatch.setattr(mod.audit, "load_completed", new)
    monkeypatch.setattr(mod.audit, "load_baseline", lambda *a: ({}, {"contract": {}}))
    monkeypatch.setattr(mod.audit, "_compare_profiles", lambda *a: None)
    monkeypatch.setattr(mod.audit, "recorded_metrics", lambda *a: ({}, trace()))
    monkeypatch.setattr(mod.audit, "nominal_screen", lambda *a: {})
    return calls


def test_explicit_E_training_audit_required_in_cli_and_direct_loader():
    with pytest.raises(SystemExit): mod.parser().parse_args(["--output-dir", "fresh"])
    with pytest.raises(ValueError, match="Explicit --training-audit"):
        a = args(); a.training_audit = None; mod.load_cases(a, {})


def test_strict_E_audit_bound_to_both_engines_and_explicit_order(monkeypatch):
    calls = fake_audit(monkeypatch)
    traces, cases, evidence, order, training = mod.load_cases(args(), {})
    assert order == ["baseline_physx", "v2g_physx", "baseline_mujoco", "v2g_mujoco"]
    assert len(traces) == len(cases) == len(evidence) == 4
    assert calls == [("G", "physx", Path("explicit_E_training.json")), ("G", "mujoco", Path("explicit_E_training.json"))]
    assert training["sha256"] == "c"*64
    assert "G contact-V2g beta-gated bonus/gap-cost" in cases["v2g_physx"]["title"]
    assert cases["v2g_physx"]["G_cohort_id"] == mod.audit.evaluator.COHORT_ID
    assert cases["baseline_physx"]["title"].startswith("Original PPO")


def test_wrong_training_binding_rejected_before_any_graphics(monkeypatch):
    fake_audit(monkeypatch, wrong_binding=True)
    with pytest.raises(ValueError, match="bound to explicit completed training audit"): mod.load_cases(args(), {})


def test_defaults_point_only_to_E_and_missing_engine_not_faked():
    a = NS(physx_dir=None, mujoco_dir=None, subset=None)
    resolved = mod.resolve_directories(a)
    assert resolved.physx_dir == mod.audit.DEFAULT_INPUT_ROOT/"physx"
    assert "ground_contact_v2g_" in str(resolved.physx_dir)
    a = args(); a.mujoco_dir = None
    with pytest.raises(ValueError, match="explicit --subset"): mod.selected_engines(a)
    a.subset = "physx"; assert mod.selected_engines(a) == ("physx",)


def test_frozen_visual_functions_and_native_clock_are_reused_exactly():
    assert mod.sha256(mod.graphics.__file__) == mod.GRAPHICS_SHA256
    assert mod.sha256(mod.audit.__file__) == mod.ANALYZER_SHA256
    for name in ("native_indices", "contact_window", "heading_glyphs", "case_contact_caption", "draw_map"):
        assert getattr(mod, name) is getattr(mod.graphics, name)
    assert mod.native_indices().tolist() == list(range(0, 2793, 8))
    t = trace(); t["contact_force"][12, :, 2] = 5.
    assert mod.contact_window(t, 16)["supported_samples"] == 7
    assert mod.heading_glyphs(t, 10)[-2]["force_only_loaded"] is True
    assert not t["contact_count"].any()


def test_actual_case_captions_and_E_outcome_never_inherit_A_or_DEM_metrics():
    case = {"variant": "v2g", "engine": "physx", "new_nominal_screen": {
        "numeric_nominal_episode_pass": False, "failed_checks": ["measured_contact"]},
        "analysis": {"any_wheel_ground_support": {"full_contact": {
            "any_wheel_support_sample_fraction": .1234, "longest_no_support_observed_s": .0125}}}}
    assert "12.34%" in mod.case_contact_caption(case)
    text = mod.outcome_caption({"v2g_physx": case}, ["v2g_physx"])
    assert "G PPO(beta-gated bonus, ungated gaps)" in text and "physx: FAIL 1" in text and "NOT DISPLAYED: mujoco" in text
    assert "NOT claimed" in text and "70.47" not in text and "86.21" not in text


def test_no_existing_results_overwrite_or_speculative_graphics(tmp_path):
    with pytest.raises(ValueError, match="fresh output"): mod.run(NS(output_dir=tmp_path))
    source = Path(mod.__file__).read_text()
    assert "torch.load(" not in source and "mj_step(" not in source and "torch.jit.load(" not in source
    assert '"goal_completion_claimed": False' in source
    assert '"training_audit_required_and_bound": True' in source
    assert '"camera_same_distance_all_cases_frames": True' in source
    assert 'graphics.render_frame(' in source and 'graphics.prepare_camera(' in source
    assert 'beta-gate only A positive bonus by existing reference-beta quality; neutral1 at reference XY speed<=.3; support-gap cost unchanged and ungated' in source
    assert '"support_gap_weight": 1.' in source


def test_synthetic_canvas_replaces_only_identity_captions_with_E_mean8_only(monkeypatch):
    from PIL import Image, ImageDraw
    calls = []
    canvas = Image.new("RGB", (2560, 850))
    monkeypatch.setattr(mod.graphics, "render_frame", lambda *args: canvas)
    class Draw:
        def rectangle(self, box, **kwargs): calls.append(("rectangle", box))
        def text(self, point, text, **kwargs): calls.append(("text", text))
    monkeypatch.setattr(ImageDraw, "Draw", lambda image: Draw())
    result = mod.render_frame(NS(small=None), {}, {}, {}, {}, [], {}, None, 150)
    assert result is canvas
    texts = [value for kind, value in calls if kind == "text"]
    assert "GcontactV2g(beta-gated bonus, ungated gaps)" in texts[0] and "t=3.00s" in texts[0]
    assert texts[1].startswith("G PPO(beta-gated bonus, ungated gaps)")
    assert [box for kind, box in calls if kind == "rectangle"] == [(0, 0, 2560, 29), (0, 763, 2560, 786)]




