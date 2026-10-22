from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_ppo_beta_nominal as mod


def test_historical_indices200hz_not400hz():
    indices = mod.native_indices()
    assert len(indices) == 350 and indices[-1] == 1396
    assert np.array_equal(np.diff(indices), np.full(349, 4))
    assert indices[150] == 600
    assert np.allclose(indices * .005, np.arange(350) / 50.)


def test_pins_two_distinct_checkpoints_and_original_native_reports():
    assert mod.beta.IDENTITIES["beta_on"]["checkpoint"] != mod.beta.IDENTITIES["beta_off"]["checkpoint"]
    assert all(len(value["trajectory"]) == 64 and len(value["report"]) == 64 for value in mod.MJ_PINS.values())
    assert mod.ORDER == (("beta_on", "physx"), ("beta_off", "physx"), ("beta_on", "mujoco"), ("beta_off", "mujoco"))


def test_existing_output_rejected_before_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_historical", lambda *a: pytest.fail("Must guard fresh output first"))
    with pytest.raises(ValueError, match="Preserve previous"):
        mod.run(NS(output_dir=tmp_path))


def test_200hz_native_clock_rejects400hz_schema(monkeypatch):
    trace = {"time_s": np.arange(2793)*.0025, "physics_dt_s": .0025}
    monkeypatch.setattr(mod, "native_trace", lambda *a: trace)
    with pytest.raises(ValueError, match="original200Hz"):
        mod.validate_native({"policy_dt_s": np.array(.02)}, "physx")


def test_no_new400_loader_no_actor_inference_or_physics():
    source = Path(mod.__file__).read_text()
    assert "load_nominal(" not in source
    assert "torch.jit.load(" not in source
    assert "mj_step(" not in source and "mj_forward(" not in source
    assert "setup_smoke_steps" in source
    assert "causal_beta_only_intervention_claimed\": False" in source


def test_original_four_historical_records_load_without_mutation():
    pytest.importorskip("mujoco")
    paths = [mod.DEFAULT_ON / "isaac/physics_trajectory.npz", mod.DEFAULT_OFF / "isaac/physics_trajectory.npz",
        mod.DEFAULT_ON / "mujoco/rollout.npz", mod.DEFAULT_OFF / "mujoco/rollout.npz", mod.beta.DEFAULT_REFERENCE,
        mod.DEFAULT_MODEL, mod.DEFAULT_LEGACY]
    if not all(path.is_file() for path in paths):
        pytest.skip("Historical private/local recordings are unavailable in a clean checkout")
    old = {str(path): mod.sha256(path) for path in paths}
    traces, reference, summary = mod.load_historical()
    assert set(traces) == set(mod.ORDER)
    assert len(summary["rows"]) == 4
    assert summary["training_physics_hz"] == 200
    assert summary["action_rate_weight"] == -.1
    assert summary["new400hz_cohort"] is False
    assert summary["causal_beta_only_intervention_claimed"] is False
    assert summary["same_initial_state_ON_OFF_per_engine"] is True
    assert summary["setup_smoke_steps"] == {"beta_on": 0, "beta_off": 8}
    assert all(len(trace["time_s"]) == 1397 for trace in traces.values())
    assert all(row["core_tracking"]["samples"] == 513 for row in summary["rows"])
    assert reference["time_s"][-1] == pytest.approx(6.96)
    assert old == {str(path): mod.sha256(path) for path in paths}


def test_four_horizontal_frame_fixed_camera_and_native600(monkeypatch):
    from PIL import ImageFont
    n = 1397
    traces, poses, targets = {}, {}, {}
    for column, key in enumerate(mod.ORDER):
        traces[key] = {"time_s": np.arange(n)*.005, "position": np.tile([column, 0., .75], (n, 1)),
            "quaternion": np.tile([1., 0., 0., 0.], (n, 1)), "com_velocity": np.tile([1., 0., 0.], (n, 1)),
            "contact_count": np.ones((n, 2)), "contact_force": np.tile([[0., 0., 10.], [0., 0., 10.]], (n, 1, 1))}
        poses[key] = np.full((n, 15), column)
        targets[key] = traces[key]["position"][mod.native_indices()] + [0., 0., -.28]
    camera = NS(lookat=np.zeros(3))
    runtime = NS(model=object(), data=object(), option=object(), cameras=[camera], font=ImageFont.load_default(), small=ImageFont.load_default(),
        renderer=NS(update_scene=lambda *a, **k: None, render=lambda: np.zeros((360, 640, 3), np.uint8)))
    plan = {"lookat_by_case": targets, "distance_m": 1.8, "azimuth_deg": 135., "elevation_deg": -30.}
    seen = []
    monkeypatch.setattr(mod, "_pose", lambda model, data, pose: seen.append((pose[0], camera.distance, camera.lookat.copy())))
    monkeypatch.setattr(mod, "_map_panel", lambda *a: None)
    reference = {"time_s": np.arange(349)*.02, "position": np.tile([0., 0., .75], (349, 1))}
    frame = mod.render_frame(runtime, traces, poses, reference, {"world_bounds": [[0., 0.], [1., 1.]]}, plan, 150)
    assert frame.size == (2560, 720)
    assert [row[0] for row in seen] == [0, 1, 2, 3]
    assert all(row[1] == 1.8 for row in seen)
    assert all(np.array_equal(row[2], targets[key][150]) for row, key in zip(seen, mod.ORDER))
