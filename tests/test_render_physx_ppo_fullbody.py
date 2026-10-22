"""Kinematic full-body replay guards; no graphics, dynamics, or inference."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_physx_ppo_fullbody as fullbody


RAW = ROOT / "outputs/policy_primary_grid_raw_20261016_v1/PPO/mu0p150/isaac"
PUBLIC = ROOT / "results/policy_primary_friction_dynamics_20261016_v1"
MODEL = ROOT / "outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml"
CHECKPOINT = "6ec411efeaea37fc61389285978b377aa7c5be80e326ce7f18581efc9acdf27b"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_local_recording():
    if not all((RAW/name).is_file() for name in ("physics_trajectory.npz", "report.json")):
        pytest.skip("Original local PhysX capture is required for raw-recording integration tests")


@pytest.fixture(scope="module")
def raw():
    require_local_recording()
    with np.load(RAW / "physics_trajectory.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@pytest.fixture(scope="module")
def recording():
    require_local_recording()
    return fullbody.load_recording()


@pytest.fixture(scope="module")
def model():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.is_file():
        pytest.skip("The original local TRON1 display model and open-source meshes are required")
    return mujoco.MjModel.from_xml_path(str(MODEL))


def test_recording_bytes_bind_to_same_published_physx_ppo_case(raw):
    summary = json.loads((PUBLIC / "summary.json").read_text())
    run = summary["runs"]["PPO_mu0p150"]
    assert sha256(RAW / "physics_trajectory.npz") == run["input_sha256"]["isaac_trace"]
    assert sha256(RAW / "report.json") == run["input_sha256"]["isaac_report"]
    report = json.loads((RAW / "report.json").read_text())
    assert report["simulator"] == "IsaacLab/PhysX"
    assert report["checkpoint_sha256"] == run["training"]["checkpoint_sha256"] == CHECKPOINT
    assert report["evaluation_contact_friction_override"]["evaluation_contact_friction"] == .15
    assert sha256(MODEL) == summary["input_sha256"]["model"]
    with np.load(PUBLIC / "policy_dynamics_trace.npz", allow_pickle=False) as archive:
        for source, target in (("root_position_m", "position"),
                               ("root_quaternion_wxyz", "quaternion"),
                               ("root_linear_velocity_w_m_s", "com_velocity"),
                               ("wheel_position_m", "wheel_pos")):
            np.testing.assert_array_equal(raw[source][:, 0], archive["PPO_mu0p150_isaac_" + target])


def test_raw_qpos_is_link_origin_wxyz_and_named_joint_order_not_model_order(raw):
    qpos = raw["qpos"][:, 0]
    np.testing.assert_array_equal(qpos[:, :3], raw["root_position_m"][:, 0])
    np.testing.assert_array_equal(qpos[:, 3:7], raw["root_quaternion_wxyz"][:, 0])
    np.testing.assert_array_equal(qpos[:, 7:], raw["joint_position_rad"][:, 0])
    np.testing.assert_allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., atol=1e-6, rtol=0)
    assert raw["qpos_joint_names"].tolist() == raw["joint_names"].tolist() == [
        "abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint",
        "knee_L_Joint", "knee_R_Joint", "wheel_L_Joint", "wheel_R_Joint"]


def test_replay_uses_350_original_native_samples_including_terminal_once(raw):
    time = raw["time_s"]
    np.testing.assert_allclose(time, np.arange(1397) * .005, atol=1e-12, rtol=0)
    indices = np.arange(0, len(time), 4)
    assert len(indices) == 350 and (indices[0], indices[-1]) == (0, 1396)
    assert time[indices[0]] == 0. and time[indices[-1]] == pytest.approx(6.98, abs=1e-12)
    assert raw["valid_mask"].all()
    assert np.flatnonzero(raw["terminal_sample"][:, 0]).tolist() == [1396]
    assert int(raw["terminal_sample"][indices, 0].sum()) == 1


def test_renderer_contains_no_dynamics_step_or_policy_execution():
    tree = ast.parse(Path(fullbody.__file__).read_text())
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls & {"mj_step", "mj_step1", "mj_step2", "mj_forward", "learn", "predict", "act_inference"}
    imports = {alias.name.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    assert not imports & {"torch", "isaaclab", "isaacsim", "rsl_rl"}


def test_loader_preserves_recorded_link_pose_joint_angles_and_source_files(raw):
    paths = [RAW / "physics_trajectory.npz", RAW / "report.json", PUBLIC / "summary.json", MODEL]
    before = {str(path): sha256(path) for path in paths}
    trace, _provenance = fullbody.load_recording()
    np.testing.assert_array_equal(trace["position"], raw["root_position_m"][:, 0])
    np.testing.assert_array_equal(trace["quaternion"], raw["root_quaternion_wxyz"][:, 0])
    np.testing.assert_array_equal(trace["joint_pos"], raw["joint_position_rad"][:, 0])
    np.testing.assert_array_equal(trace["wheel_pos"], raw["wheel_position_m"][:, 0])
    np.testing.assert_array_equal(trace["time_s"], raw["time_s"])
    assert tuple(trace["joint_names"]) == tuple(raw["joint_names"].tolist())
    assert {str(path): sha256(path) for path in paths} == before


def test_loader_provenance_retains_original_recording_policy_and_model_bindings(recording):
    _trace, provenance = recording
    source = json.loads((PUBLIC / "summary.json").read_text())
    run = source["runs"]["PPO_mu0p150"]
    assert provenance["recording_sha256"] == run["input_sha256"]["isaac_trace"]
    assert provenance["report_sha256"] == run["input_sha256"]["isaac_report"]
    assert provenance["checkpoint_sha256"] == CHECKPOINT
    assert provenance["model_sha256_expected"] == sha256(MODEL)
    assert provenance["core_metrics"] == run["windows"]["core"]["isaac"]
    assert provenance["posture_diagnostics"] == run["posture_diagnostics"]["isaac"]


def test_named_qpos_mapping_reconstructs_full_body_not_direct_raw_array(model, recording, raw):
    trace, _provenance = recording
    qpos, qids = fullbody.build_qpos(model, trace)
    assert qpos.shape == (1397, 15)
    assert list(qids) == [7, 11, 8, 12, 9, 13, 10, 14]
    np.testing.assert_array_equal(qpos[:, :3], trace["position"])
    np.testing.assert_array_equal(qpos[:, 3:7], trace["quaternion"])
    np.testing.assert_array_equal(qpos[:, qids], trace["joint_pos"])
    assert not np.array_equal(qpos, raw["qpos"][:, 0])


def test_named_joint_column_permutation_reconstructs_identical_qpos(model, recording):
    trace, _provenance = recording
    expected, _ = fullbody.build_qpos(model, trace)
    permutation = [7, 2, 4, 0, 6, 1, 5, 3]
    permuted = copy.deepcopy(trace)
    permuted["joint_names"] = tuple(np.asarray(trace["joint_names"])[permutation])
    permuted["joint_pos"] = trace["joint_pos"][:, permutation]
    actual, qids = fullbody.build_qpos(model, permuted)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, qids], permuted["joint_pos"])


def test_all_native_frames_fk_match_measured_wheels_without_mutating_source(model, recording):
    mujoco = pytest.importorskip("mujoco")
    trace, _provenance = recording
    copies = {name: value.copy() for name, value in trace.items() if isinstance(value, np.ndarray)}
    qpos, _ = fullbody.build_qpos(model, trace)
    before_qpos = qpos.copy()
    # Run the production audit, then independently check its geometrical basis.
    audit = fullbody.validate_fk(model, trace, qpos, tolerance_m=2e-6)
    assert audit["samples_checked"] == 1397
    assert audit["max_wheel_origin_error_m"] < audit["tolerance_m"] == 2e-6
    assert audit["physics_stepping"] is False
    assert audit["kinematic_functions"] == ["mj_kinematics", "mj_comPos", "mj_camlight"]
    data = mujoco.MjData(model)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
    wheels = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
              for name in ("wheel_L_Link", "wheel_R_Link")]
    largest_error = 0.
    for index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(data.xpos[root], trace["position"][index], rtol=0, atol=1e-12)
        expected_quaternion = trace["quaternion"][index]
        expected_quaternion = expected_quaternion / np.linalg.norm(expected_quaternion)
        np.testing.assert_allclose(data.xquat[root], expected_quaternion, rtol=0, atol=1e-7)
        largest_error = max(largest_error, float(np.max(np.abs(data.xpos[wheels] - trace["wheel_pos"][index]))))
    assert largest_error <= 2e-6
    assert largest_error > 1e-7  # Recorded PhysX float32/FK rounding is not exact zero.
    np.testing.assert_array_equal(qpos, before_qpos)
    for name, value in copies.items():
        np.testing.assert_array_equal(trace[name], value)


def test_incorrect_joint_mapping_is_rejected_by_fk_audit(model, recording):
    trace, _provenance = recording
    qpos, _ = fullbody.build_qpos(model, trace)
    incorrect = qpos.copy()
    incorrect[:, 7:] = trace["joint_pos"]
    with pytest.raises(ValueError, match="FK|kinematic|wheel|recorded"):
        fullbody.validate_fk(model, trace, incorrect, tolerance_m=2e-6)


def test_nonfinite_recorded_quaternion_is_rejected(model, recording):
    trace, _provenance = recording
    changed = copy.deepcopy(trace)
    changed["quaternion"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite|unit wxyz quaternions"):
        fullbody.build_qpos(model, changed)


def test_nonunit_recorded_quaternion_is_rejected(model, recording):
    trace, _provenance = recording
    changed = copy.deepcopy(trace)
    changed["quaternion"][0] *= 2.
    with pytest.raises(ValueError, match="unit wxyz quaternions"):
        fullbody.build_qpos(model, changed)


def test_xyzw_misinterpretation_is_rejected_by_geometry_audit(model, recording):
    trace, _provenance = recording
    qpos, _ = fullbody.build_qpos(model, trace)
    wrong = qpos.copy()
    wrong[:, 3:7] = wrong[:, [4, 5, 6, 3]]
    with pytest.raises(ValueError, match="FK|kinematic|wheel|recorded"):
        fullbody.validate_fk(model, trace, wrong, tolerance_m=2e-6)


def test_duplicate_joint_names_are_rejected_before_mapping(model, recording):
    trace, _provenance = recording
    changed = dict(trace)
    changed["joint_names"] = (trace["joint_names"][0],) * 8
    with pytest.raises(ValueError, match="eight unique named joints"):
        fullbody.build_qpos(model, changed)


def test_published_fullbody_states_and_artifacts_preserve_original_pose_and_provenance():
    directory = ROOT/"results/physx_ppo_mu015_fullbody_20261021_v1"
    summary = json.loads((directory/"summary.json").read_text())
    assert summary["training_engine"] == summary["evaluation_engine"] == "PhysX"
    assert summary["graphics_engine"] == "MuJoCo" and summary["mu"] == .15
    assert summary["checkpoint_sha256"] == CHECKPOINT
    assert summary["analysis_source_sha256"] == sha256(fullbody.__file__)
    for flag in ("dynamics_stepping", "contact_solver_called", "new_policy_inference", "new_training",
                 "retargeting", "root_height_corrected", "height_or_flight_failure_gate", "tilt_failure_gate"):
        assert summary[flag] is False
    assert summary["video_frames"] == 350 and summary["video_fps"] == 50
    assert summary["frame_size_pixels"] == [1280, 622]
    assert summary["robot_mesh_alpha"] == 1.
    assert summary["fk_validation"]["max_wheel_origin_error_m"] < 2e-6
    for name, entry in summary["artifacts"].items():
        assert sha256(directory/name) == entry["sha256"]
    with np.load(directory/"replay_states.npz", allow_pickle=False) as states:
        assert states["qpos"].shape == (1397, 15)
        np.testing.assert_allclose(states["time_s"], np.arange(1397)*.005, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(states["qpos"][:, :3], states["original_root_position_m"])
        np.testing.assert_array_equal(states["qpos"][:, 3:7], states["original_root_quaternion_wxyz"])
        np.testing.assert_array_equal(states["named_joint_qpos_addresses"], [7, 11, 8, 12, 9, 13, 10, 14])
