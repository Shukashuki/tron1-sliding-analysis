"""Actor-free validation arithmetic only: no simulator stepping or GPU jobs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("request_replay", ROOT/"scripts/replay_physx_actions_mujoco.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def fixture():
    n = 2793
    actions = np.arange(349*8, dtype=np.float32).reshape(349, 8)/3000
    held = np.concatenate((np.zeros((1, 8)), np.repeat(actions, 8, axis=0)))[:, None]
    return {"time_s": np.arange(n)*.0025, "physics_dt_s": np.array(.0025),
        "policy_dt_s": np.array(.02), "joint_names": np.asarray(replay.JOINTS),
        "valid_mask": np.ones((n, 1), dtype=bool),
        "action_was_applied": np.r_[False, np.ones(n-1, dtype=bool)][:, None],
        "reference_frame_for_action": np.r_[0, np.repeat(np.arange(349), 8)][:, None],
        "actions_raw": held.copy(), "actions_clipped": np.clip(held, -1, 1)}


def test_extract_preserves_349_requests_and_never_reconstructs_an_actor():
    d = fixture()
    actions = replay.extract_actions(d)
    assert actions.shape == (349, 8)
    np.testing.assert_array_equal(actions, d["actions_raw"][1::8, 0])


@pytest.mark.parametrize("field", ["actions_raw", "actions_clipped"])
def test_reject_substep_request_mutation(field):
    d = fixture()
    d[field][3, 0, 4] += .01
    with pytest.raises(ValueError, match="eight-substep"):
        replay.extract_actions(d)


@pytest.mark.parametrize("field", ["valid_mask", "action_was_applied"])
def test_reject_missing_or_reset_action_coverage(field):
    d = fixture()
    d[field][7, 0] = False
    with pytest.raises(ValueError, match="coverage"):
        replay.extract_actions(d)


def test_reject_wrong_clock():
    d = fixture()
    d["physics_dt_s"] = np.array(.005)
    with pytest.raises(ValueError, match="clocks"):
        replay.extract_actions(d)


def test_reject_frame_phase_mismatch():
    d = fixture()
    d["reference_frame_for_action"][9, 0] = 8
    with pytest.raises(ValueError, match="frames"):
        replay.extract_actions(d)


def test_reject_wrong_joint_order():
    d = fixture()
    d["joint_names"] = d["joint_names"][::-1]
    with pytest.raises(ValueError, match="order"):
        replay.extract_actions(d)


def native_pair(steps=10):
    p = fixture()
    n = 1+steps*8
    shapes = {"root_position_m": (3,), "root_quaternion_wxyz": (4,),
        "root_linear_velocity_w_m_s": (3,), "root_angular_velocity_w_rad_s": (3,),
        "joint_position_rad": (8,), "joint_velocity_rad_s": (8,),
        "applied_joint_torque_nm": (8,), "wheel_position_m": (2, 3),
        "wheel_ground_contact_count": (2,), "wheel_ground_contact_force_w_n": (2, 3),
        "wheel_lateral_contact_slip_m_s": (2,), "nonwheel_ground_contact_count": (),
        "nonwheel_ground_contact_force_norm_n": ()}
    m = {"time_s": p["time_s"][:n].copy(), "physics_dt_s": np.array(.0025),
        "policy_dt_s": np.array(.02), "joint_names": p["joint_names"].copy(),
        "actions_raw": p["actions_raw"][1::8, 0][:steps].copy(),
        "actions_clipped": p["actions_clipped"][1::8, 0][:steps].copy()}
    for left, right in replay.FIELD_MAP.items():
        p[left] = np.zeros((2793, 1, *shapes[left]))
        m[right] = p[left][:n, 0].copy()
    return p, m


def test_native_compare_records_different_torques_and_contact_initialization():
    p, m = native_pair()
    m["joint_torque_nm"][:] = 12
    m["wheel_ground_contact_force_w_n"][0] = 100
    m["root_pos"][1:, 0] = .02
    arrays, audit = replay.compare_native(p, m, 10, replay.extract_actions(p))
    assert audit["native_states"] == 81
    assert audit["initial_contact_force_difference_max_abs_n"] == 100
    assert audit["same_time_interengine_xy_peak_m"] == .02
    assert np.max(arrays["mujoco_applied_joint_torque_nm"]) == 12
    assert audit["requests_equal_recorded_source_prefix"]


@pytest.mark.parametrize("field", replay.INITIAL_STATE_FIELDS)
def test_reject_initial_observable_mismatch(field):
    p, m = native_pair()
    m[replay.FIELD_MAP[field]][0, ...] = .001
    with pytest.raises(ValueError, match="Initial observable"):
        replay.compare_native(p, m, 10, replay.extract_actions(p))


def test_reject_changed_requests_despite_valid_native_recording():
    p, m = native_pair()
    m["actions_raw"][2, 0] += .1
    with pytest.raises(ValueError, match="requests differ"):
        replay.compare_native(p, m, 10, replay.extract_actions(p))


def test_reject_mujoco_native_clock_change():
    p, m = native_pair()
    m["physics_dt_s"] = np.array(.005)
    with pytest.raises(ValueError, match="clocks"):
        replay.compare_native(p, m, 10, replay.extract_actions(p))


def test_all_existing_input_and_source_pins_match_read_only():
    source = ROOT/"outputs/controller_evaluation_source_20261015_v2"
    if not source.exists():
        pytest.skip("Optional original source capture unavailable")
    for relative, expected in replay.SOURCE_PINS.items():
        assert replay.sha256(source/relative) == expected, relative
    for name, path in {
        "physx_physics": ROOT/"outputs/ppo_contact_timestep_20261022_v1/dt0p0025/physics_trajectory.npz",
        "physx_report": ROOT/"outputs/ppo_contact_timestep_20261022_v1/dt0p0025/report.json",
        "motion": ROOT.parent/"tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2/motion.npz",
        "model": ROOT/"outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml",
    }.items():
        assert replay.sha256(path) == replay.INPUT_PINS[name], name


def test_fresh_process_imports_only_pinned_cpu_helpers_not_torch_or_physics():
    source = ROOT/"outputs/controller_evaluation_source_20261015_v2/scripts/eval_tracking_mujoco.py"
    if not source.exists():
        pytest.skip("Optional original source capture unavailable")
    code = ("import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('adapter',{str(ROOT/'scripts/replay_physx_actions_mujoco.py')!r}); "
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
        f"e=m.load_evaluator({str(source)!r}); "
        "assert 'torch' not in sys.modules;assert 'mujoco' not in sys.modules; "
        "assert callable(e.run_evaluation); "
        f"assert m.audit_module_origins({str(source.parent.parent)!r}); "
        "print('pinned CPU helpers imported; no Torch, MuJoCo runtime or physics stepping')")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr


def test_foreign_runtime_module_rejected(monkeypatch):
    import types
    root = Path("/tmp/request-replay-test-source")
    monkeypatch.setitem(sys.modules, "eval_tracking", types.SimpleNamespace(__file__="/tmp/foreign.py"))
    with pytest.raises(ValueError, match="Foreign runtime module"):
        replay.audit_module_origins(root)
