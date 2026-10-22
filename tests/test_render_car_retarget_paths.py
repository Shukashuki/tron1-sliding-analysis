"""Offline car-to-robot path comparison: fixed mapping, not dynamic tracking."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_car_retarget_paths as renderer


CSV = ROOT / "upstream/drift_drl/code/ref_trajectory/traj_4.csv"
ORIGINAL = Path("/home/roy/tron1-learning-from-human/outputs/drift-drl-turn-reference-20261013-v2")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def expected():
    """Independently decode the pinned CARLA schema and historical transform."""
    with CSV.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    columns = {key: np.asarray([float(row[key]) for row in records])
               for key in ("world_x", "world_y", "world_heading", "local_vx", "local_vy", "steer")}
    left_yaw = np.deg2rad(columns["world_heading"])
    positions = np.column_stack((columns["world_x"], -columns["world_y"]))
    velocities = np.column_stack((columns["local_vx"] * np.cos(left_yaw) - columns["local_vy"] * np.sin(left_yaw),
                                  -(columns["local_vx"] * np.sin(left_yaw) + columns["local_vy"] * np.cos(left_yaw))))
    yaw = np.unwrap(-left_yaw)
    displacement = np.diff(positions, axis=0)
    midpoint = .5 * (velocities[:-1] + velocities[1:])
    intervals = np.maximum(np.sum(displacement * midpoint, axis=1) /
                           np.maximum(np.sum(midpoint ** 2, axis=1), 1e-12), 0.)
    intervals[np.linalg.norm(displacement, axis=1) < 1e-7] = 0.
    clock = np.r_[0., np.cumsum(intervals)]
    selected = slice(1535, 1596)
    source_time = clock[selected] - clock[1535]
    initial = yaw[1535]
    matrix = np.array([[np.cos(initial), np.sin(initial)], [-np.sin(initial), np.cos(initial)]])
    source_xy = .08 * (positions[selected] - positions[1535]) @ matrix.T
    source_vel = .04 * velocities[selected] @ matrix.T
    robot_time = np.arange(249) / 50.
    query = robot_time / 2.
    return {"time": robot_time, "source_time": source_time, "matrix": matrix,
            "xy": np.column_stack([np.interp(query, source_time, source_xy[:, axis]) for axis in range(2)]),
            "velocity": np.column_stack([np.interp(query, source_time, source_vel[:, axis]) for axis in range(2)]),
            "yaw": np.interp(query, source_time, yaw[selected]) - initial,
            "initial_yaw": initial}


@pytest.fixture(scope="module")
def replay():
    if not all((ORIGINAL / name).is_file() for name in ("motion.npz", "robot_reference.npz")):
        pytest.skip("Optional original hash-locked private reference pair is unavailable")
    return renderer.load_data(reference_file=ORIGINAL / "robot_reference.npz", motion_file=ORIGINAL / "motion.npz")


@pytest.mark.skipif(not all((ORIGINAL / name).is_file() for name in ("motion.npz", "robot_reference.npz")),
                    reason="Optional original hash-locked private reference pair is unavailable")
def test_locked_historical_pair_not_new_numeric_reproduction(expected):
    lock = json.loads((ROOT / "config/car_reference_lock.json").read_text())
    assert sha256(CSV) == lock["source_csv_sha256"]
    assert sha256(ORIGINAL / "motion.npz") == lock["motion_npz_sha256"]
    assert sha256(ORIGINAL / "robot_reference.npz") == lock["robot_reference_npz_sha256"]
    with np.load(ORIGINAL / "robot_reference.npz", allow_pickle=False) as archive:
        phase = archive["source_phase"]
        assert np.flatnonzero(phase == 0).tolist() == list(range(249))
        assert np.count_nonzero(phase == 1) == 75 and np.count_nonzero(phase == 2) == 25
        np.testing.assert_allclose(archive["root_target_position_m"][:249, :2], expected["xy"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(archive["root_yaw_rad"][:249], expected["yaw"], atol=1e-12, rtol=0)
    with np.load(ORIGINAL / "motion.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["body_pos_w"][:249, 0, :2], expected["xy"], atol=2e-7, rtol=0)


def test_renderer_does_not_retarget_train_or_run_physics():
    tree = ast.parse(Path(renderer.__file__).read_text())
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert not calls & {"mj_step", "mj_forward", "least_squares", "solve_reference", "prepare_reference",
                        "prepare_controller_reference", "learn", "act_inference", "predict"}
    imports = {alias.name.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    assert not imports & {"isaacsim", "isaaclab", "torch", "rsl_rl", "scipy"}


def test_data_uses_only_demonstrated_phase_and_same_fixed_mapping(replay, expected):
    arrays, _summary = replay
    np.testing.assert_allclose(arrays["time_s"], expected["time"], atol=1e-12, rtol=0)
    assert arrays["time_s"].shape == (249,)
    assert arrays["time_s"][0] == 0. and arrays["time_s"][-1] == 4.96
    np.testing.assert_allclose(arrays["source_time_s"], expected["time"] / 2., atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays["car_common_xy_m"], expected["xy"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays["robot_xy_m"], expected["xy"], atol=2e-7, rtol=0)
    np.testing.assert_allclose(arrays["car_body_heading_rad"], expected["yaw"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays["robot_body_heading_rad"], expected["yaw"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(arrays["car_velocity_common_m_s"], expected["velocity"], atol=1e-12, rtol=0)


def test_car_has_body_velocity_but_no_observed_wheel_angles(replay):
    arrays, summary = replay
    assert summary["car_wheel_orientation_available"] is False
    assert not any(name.startswith("car_wheel") for name in arrays)
    assert arrays["car_body_heading_rad"].shape == arrays["car_course_rad"].shape == (249,)
    assert arrays["robot_wheel_heading_rad"].shape == (249, 2)
    assert arrays["robot_wheel_xy_m"].shape == (249, 2, 2)
    text = json.dumps(summary).lower()
    assert "normalized" in text and "steer" in text
    assert "not" in text and "wheel" in text


def test_loading_never_rewrites_pinned_source_or_locked_reference_pair(replay):
    files = [CSV, ORIGINAL / "motion.npz", ORIGINAL / "robot_reference.npz",
             ROOT / "config/car_reference_lock.json"]
    before = {str(path): sha256(path) for path in files}
    renderer.load_data(reference_file=ORIGINAL / "robot_reference.npz", motion_file=ORIGINAL / "motion.npz")
    assert {str(path): sha256(path) for path in files} == before


def test_altered_reference_bytes_are_rejected_not_silently_regenerated(replay, tmp_path):
    changed = tmp_path / "changed_robot_reference.npz"
    changed.write_bytes(b"not the locked reference")
    with pytest.raises(ValueError, match="SHA|hash|mismatch"):
        renderer.load_data(reference_file=changed, motion_file=ORIGINAL / "motion.npz")


def test_main_figure_uses_identical_limits_aspect_and_explicit_shared_scale(replay):
    arrays, summary = replay
    fig, axes = renderer._figure(arrays, summary)
    try:
        assert len(axes) == 2
        settings = summary["plot_settings"]
        assert settings["common_axes_shared"] is True and settings["aspect"] == "equal"
        assert axes[0].get_shared_x_axes().joined(*axes)
        assert axes[0].get_shared_y_axes().joined(*axes)
        for ax in axes:
            assert ax.get_aspect() == 1.
            np.testing.assert_array_equal(ax.get_xlim(), settings["shared_common_bounds"]["xlim"])
            np.testing.assert_array_equal(ax.get_ylim(), settings["shared_common_bounds"]["ylim"])
            assert ax.get_xlabel() == "Robot-scale X [m]"
            assert ax.get_ylabel() == "Robot-scale Y [m]"
        assert "Shared axes / scale / colors / arrow lengths" in fig._suptitle.get_text()
        assert any("Car wheel orientation unavailable" in item.get_text() for item in fig.texts)
    finally:
        renderer._plt().close(fig)


def test_native_scale_figure_explicitly_has_separate_source_and_robot_axes(replay):
    arrays, summary = replay
    fig, axes = renderer._figure(arrays, summary, native=True)
    try:
        settings = summary["plot_settings"]
        assert settings["native_axes_shared"] is False
        assert not axes[0].get_shared_x_axes().joined(*axes)
        assert not axes[0].get_shared_y_axes().joined(*axes)
        assert "axis limits NOT shared" in fig._suptitle.get_text()
        assert "native scale" in axes[0].get_title()
        assert axes[0].get_xlabel() == "Source world X [m]"
        assert axes[1].get_xlabel() == "Robot-scale X [m]"
        for ax, key in zip(axes, ("native_car_bounds", "native_robot_bounds")):
            np.testing.assert_array_equal(ax.get_xlim(), settings[key]["xlim"])
            np.testing.assert_array_equal(ax.get_ylim(), settings[key]["ylim"])
            assert ax.get_aspect() == 1.
        assert not np.allclose(axes[0].get_xlim(), axes[1].get_xlim())
    finally:
        renderer._plt().close(fig)


def test_both_main_panels_use_same_actual_arrow_length_and_colors(replay, tmp_path, monkeypatch):
    from matplotlib.colors import to_rgba
    arrays, summary = replay
    original = renderer._figure
    figures = []

    def capture(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        figures.append((fig, axes))
        monkeypatch.setattr(fig, "savefig", lambda *_args, **_kwargs: None)
        return fig, axes

    monkeypatch.setattr(renderer, "_figure", capture)
    renderer.render_comparison(arrays, summary, tmp_path / "not-written.png")
    _fig, axes = figures[0]
    for ax in axes:
        assert len(ax.collections) == 2
        assert ax.lines[0].get_color() == renderer.COLORS["path"]
        for arrows, color in zip(ax.collections, (renderer.COLORS["body"], renderer.COLORS["velocity"])):
            np.testing.assert_allclose(np.hypot(arrows.U, arrows.V), .18, rtol=0, atol=1e-15)
            np.testing.assert_allclose(arrows.get_facecolors()[0], to_rgba(color), rtol=0, atol=0)
    assert summary["plot_settings"]["common_arrow_length_m"] == renderer.ARROW_LENGTH_M == .18


@pytest.mark.parametrize("skip_video", [False, True])
def test_fresh_output_preserves_arrays_hashes_and_prohibits_overwrite(replay, tmp_path, monkeypatch, skip_video):
    arrays, baseline = replay
    paths = [CSV, ORIGINAL / "motion.npz", ORIGINAL / "robot_reference.npz"]
    before = {str(path): sha256(path) for path in paths}
    calls = []

    def still(_arrays, _summary, output, native=False):
        calls.append((Path(output).name, native))
        Path(output).write_bytes(b"offline-still-test")

    def video(_arrays, _summary, output):
        calls.append((Path(output).name, False))
        Path(output).write_bytes(b"offline-video-test")

    monkeypatch.setattr(renderer, "render_comparison", still)
    monkeypatch.setattr(renderer, "render_video", video)
    output = tmp_path / "comparison"
    argv = ["--reference-file", str(ORIGINAL / "robot_reference.npz"),
            "--motion-file", str(ORIGINAL / "motion.npz"), "--output-dir", str(output)]
    if skip_video:
        argv.append("--skip-video")
    args = renderer.parser().parse_args(argv)
    result = renderer.run(args)
    assert calls == [("comparison.png", False), ("native_scale.png", True)] + (
        [] if skip_video else [("comparison.mp4", False)])
    assert result["mapping_audit"] == baseline["mapping_audit"]
    assert result["mapping_audit"]["source_common_vs_locked_root_target_max_abs_m"] <= 1e-12
    assert result["mapping_audit"]["source_common_vs_export_float32_max_abs_m"] <= 2e-7
    assert result["input_sha256"] == baseline["input_sha256"]
    assert result["mapping"]["additional_alignment_or_time_warp"] is False
    assert result["no_new_retarget"] is True
    for key in ("ik_solver_called", "dynamics_stepping", "new_training", "new_policy_inference",
                "contact_measurement_available", "heading_offset_is_mismatch"):
        assert result[key] is False
    assert result["analysis_source_sha256"] == sha256(renderer.__file__)
    for name, item in result["artifacts"].items():
        assert item["sha256"] == sha256(output / name)
    with np.load(output / "derived_paths.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays)
        for key in arrays:
            np.testing.assert_array_equal(archive[key], arrays[key])
    assert json.loads((output / "summary.json").read_text()) == result
    assert result["frames"] == 249 and result["display_interval_s"] == [0., 4.96]
    assert result["excluded_synthetic_recovery_and_tail_frames"] == 100
    assert result["video"]["frames"] == (0 if skip_video else 249)
    assert result["video"]["encoded_duration_s"] == (None if skip_video else 4.98)
    assert {str(path): sha256(path) for path in paths} == before
    produced = {path.name: sha256(path) for path in output.iterdir()}
    with pytest.raises(ValueError, match="Preserve existing artifacts"):
        renderer.run(args)
    assert {path.name: sha256(path) for path in output.iterdir()} == produced


def test_public_comparison_replays_without_private_reference_files():
    directory = ROOT/"results/car_retarget_paths_20261021_v1"
    summary = json.loads((directory/"summary.json").read_text())
    assert summary["analysis_source_sha256"] == sha256(renderer.__file__)
    assert summary["frames"] == 249 and summary["display_interval_s"] == [0., 4.96]
    assert summary["video"] == {"fps": 50, "frames": 249, "last_sample_time_s": 4.96,
                                "encoded_duration_s": 4.98}
    assert summary["car_wheel_orientation_available"] is False
    assert summary["mapping"]["additional_alignment_or_time_warp"] is False
    for name, item in summary["artifacts"].items():
        assert sha256(directory/name) == item["sha256"]
    with np.load(directory/"derived_paths.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["time_s"], np.arange(249)/50.)
        np.testing.assert_array_equal(arrays["source_time_s"], arrays["time_s"]/2.)
        assert arrays["selected_source_position_xy_m"].shape == (61, 2)
        assert arrays["robot_wheel_xy_m"].shape == (249, 2, 2)
        error = np.max(np.abs(arrays["car_common_xy_m"]-arrays["robot_xy_m"]))
        assert error == summary["mapping_audit"]["source_common_vs_export_float32_max_abs_m"]
        assert error <= 2e-7
        assert not any(name.startswith("car_wheel") for name in arrays.files)
