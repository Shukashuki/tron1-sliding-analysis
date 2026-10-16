from pathlib import Path
import hashlib
import importlib
import json
import sys
from types import ModuleType

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import controller_references as registry


def write_fixture_pair(tmp_path, *, defect=None):
    """Synthetic hashed files test structure independently of the real bytes."""
    catalog = registry.load_catalog()
    spec = catalog["references"][registry.PRIMARY_REFERENCE_ID]
    n = spec["frames"]
    phase = np.repeat([0, 1, 2], [249, 75, 25])
    shared = {"fps": np.array([50.]), "time_s": np.arange(n) / 50., "source_phase": phase}
    quaternion = np.zeros((n, 10, 4)); quaternion[..., 0] = 1.
    root_quaternion = np.zeros((n, 4)); root_quaternion[:, 0] = 1.
    motion = {**shared, "joint_names": np.array(registry.MOTION_JOINT_NAMES),
              "body_names": np.array(registry.MOTION_BODY_NAMES), "joint_pos": np.zeros((n, 8)),
              "joint_vel": np.zeros((n, 8)), "body_pos_w": np.zeros((n, 10, 3)),
              "body_quat_w": quaternion, "body_lin_vel_w": np.zeros((n, 10, 3)),
              "body_ang_vel_w": np.zeros((n, 10, 3)), "qpos_mujoco": np.zeros((n, 15)),
              "qvel_mujoco": np.zeros((n, 14)), "joint_tracking_mask": np.array([True] * 6 + [False] * 2),
              "body_orientation_tracking_mask": np.array([True, True, True, True, False, True, True, True, False, True]),
              "body_angular_velocity_tracking_mask": np.array([True, True, True, True, False, True, True, True, False, True])}
    reference = {**shared, "joint_names": np.array(registry.REFERENCE_JOINT_NAMES),
                 "qpos": np.zeros((n, 15)), "qvel": np.zeros((n, 14)),
                 "joint_positions_rad": np.zeros((n, 8)), "joint_velocities_rad_s": np.zeros((n, 8)),
                 "root_target_position_m": np.zeros((n, 3)), "root_target_quat_wxyz": root_quaternion,
                 "wheel_center_positions_world_m": np.zeros((n, 2, 3))}
    if defect is not None:
        defect(motion, reference)
    motion_path, reference_path = tmp_path / "motion.npz", tmp_path / "robot_reference.npz"
    np.savez(motion_path, **motion)
    np.savez(reference_path, **reference)
    spec["motion_npz_sha256"] = hashlib.sha256(motion_path.read_bytes()).hexdigest()
    spec["robot_reference_npz_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    return motion_path, reference_path, catalog_path


def test_default_primary_and_separate_supplement_keep_actual_hashes_and_scopes():
    main = registry.get_reference()
    supplement = registry.get_reference("doa_sa29_scale2")
    assert main["reference_id"] == "drift_drl_traj4"
    assert main["stratum"] == "large_angle_turn"
    assert supplement["stratum"] == "high_sideslip_s_avoidance"
    assert main["evaluation_core_interval_s"] == [1.44, 4.0]
    assert supplement["evaluation_core_interval_s"] == [0.0, 3.14]
    assert supplement["supplementary_event_windows_s"] == [[0.78, 1.54], [1.88, 3.04]]
    assert main["motion_npz_sha256"] == "da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
    assert main["robot_reference_npz_sha256"] == "afc22f19b813cf10b3e3a90affb0f26f5a7743074f98b262394c69de786e9f6e"
    assert supplement["motion_npz_sha256"] == "7e6b260611e846b47da206b4296aea1eaa6e6e4c56bff8c3d166362cb83a2d10"
    assert supplement["robot_reference_npz_sha256"] == "79bc00a9fed5ba1b9b3bae6605cf337e7dc14b444197c3ae13b91b2491aa20d0"
    for spec in (main, supplement):
        assert spec["pooled_success_statistics"] is False
        assert spec["heading_separation_is_measured_contact_slip"] is False
        assert spec["source"]["kind"] == "CARLA_simulation"
        assert spec["source"]["is_physical_car_capture"] is False
        assert spec["fps"] == 50.
        assert spec["sample_span_s"] == pytest.approx((spec["frames"] - 1) / 50.)
        assert spec["source_demo_frames"] + 75 + 25 == spec["frames"]


def test_returned_specs_are_independent_and_unknown_ids_fail_closed():
    mutated = registry.get_reference()
    mutated["source"]["kind"] = "not the source"
    mutated["evaluation_core_interval_s"][0] = 99.
    assert registry.get_reference()["source"]["kind"] == "CARLA_simulation"
    assert registry.get_reference()["evaluation_core_interval_s"] == [1.44, 4.]
    with pytest.raises(ValueError, match="reference_id"):
        registry.get_reference("sa_traj34")


@pytest.mark.parametrize("field,bad", [
    ("evaluation_core_interval_s", [1.44, 6.0]),
    ("source_demo_interval_s", [0., 6.96]),
    ("supplementary_event_windows_s", [[4.9, 5.1]]),
    ("sample_span_s", 6.98), ("fps", 25.),
])
def test_catalog_cannot_mix_demonstration_with_synthetic_stop(tmp_path, field, bad):
    catalog = registry.load_catalog()
    catalog["references"][registry.PRIMARY_REFERENCE_ID][field] = bad
    path = tmp_path / "invalid_catalog.json"
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError):
        registry.load_catalog(path)


def test_catalog_rejects_pooled_success_and_real_car_relabeling(tmp_path):
    for edit in (lambda cat: cat.update(pooled_success_statistics=True),
                 lambda cat: cat["references"][registry.PRIMARY_REFERENCE_ID]["source"].update(is_physical_car_capture=True)):
        catalog = registry.load_catalog()
        edit(catalog)
        path = tmp_path / "invalid_catalog.json"
        path.write_text(json.dumps(catalog))
        with pytest.raises(ValueError):
            registry.load_catalog(path)


def test_verify_pair_reads_hashes_and_arrays_and_returns_explicit_reference_id(tmp_path):
    motion, reference, catalog = write_fixture_pair(tmp_path)
    spec = registry.verify_reference_pair(motion, reference, catalog_path=catalog)
    assert spec["reference_id"] == registry.PRIMARY_REFERENCE_ID
    assert spec["frames"] == 349
    assert "verified_asset" not in spec


def test_tampered_bytes_and_wrong_selected_reference_are_rejected(tmp_path):
    motion, reference, catalog = write_fixture_pair(tmp_path)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        registry.verify_reference_pair(motion, reference, "doa_sa29_scale2", catalog_path=catalog)
    motion.write_bytes(motion.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        registry.verify_reference_pair(motion, reference, catalog_path=catalog)


def test_motion_identity_only_checks_hash_and_explicitly_disclaims_pair_validation(tmp_path):
    motion = tmp_path / "opaque_motion_bytes"
    motion.write_bytes(b"fixture bytes deliberately not an NPZ")
    catalog = registry.load_catalog()
    catalog["references"][registry.PRIMARY_REFERENCE_ID]["motion_npz_sha256"] = hashlib.sha256(motion.read_bytes()).hexdigest()
    path = tmp_path / "identity_catalog.json"
    path.write_text(json.dumps(catalog))
    spec = registry.verify_motion_identity(motion, catalog_path=path)
    assert spec["reference_id"] == registry.PRIMARY_REFERENCE_ID
    assert spec["verification_scope"] == "motion_byte_identity_only_not_reference_pair_or_asset_validation"
    with pytest.raises(ValueError, match="motion_npz_sha256 mismatch"):
        registry.verify_motion_identity(motion, registry.SUPPLEMENT_REFERENCE_ID, catalog_path=path)
    motion.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="motion_npz_sha256 mismatch"):
        registry.verify_motion_identity(motion, catalog_path=path)


@pytest.mark.parametrize("entry", ["train_beyondmimic_style", "train_sac_tracking_isaac"])
def test_trainers_check_identity_before_app_start_or_output_creation(tmp_path, monkeypatch, entry):
    trainer = importlib.import_module(entry)
    app_module = ModuleType("isaaclab.app")

    class FakeLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            pass

        def __init__(self, *args, **kwargs):
            pytest.fail("AppLauncher must not start before reference identity validation")

    app_module.AppLauncher = FakeLauncher
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    seen = {}

    def reject(motion_file, reference_id):
        seen.update(motion_file=motion_file, reference_id=reference_id)
        raise ValueError("deliberate identity mismatch")

    monkeypatch.setattr(trainer, "verify_motion_identity", reject)
    output = tmp_path / "must_not_exist"
    argv = [entry, "--source-root", str(tmp_path), "--motion-file", str(tmp_path / "motion.npz"),
            "--asset-path", str(tmp_path / "asset.usda"), "--task-profile", str(tmp_path / "task.json"),
            "--output-dir", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="deliberate identity mismatch"):
        trainer.main()
    assert seen == {"motion_file": tmp_path / "motion.npz", "reference_id": registry.PRIMARY_REFERENCE_ID}
    assert not output.exists()


def test_ppo_collection_budget_uses_runner_config_not_a_hardcoded_rollout_length():
    from train_beyondmimic_style import ppo_collection_budget
    assert ppo_collection_budget({"num_steps_per_env": 7}, 3, 5) == {
        "num_steps_per_env": 7, "policy_steps_requested": 35,
        "environment_transitions_requested": 105}
    for bad in (None, True, 0, -1, 2.5):
        with pytest.raises(ValueError, match="num_steps_per_env"):
            ppo_collection_budget({"num_steps_per_env": bad}, 3, 5)


@pytest.mark.parametrize("defect,match", [
    (lambda motion, ref: motion.update(joint_pos=np.zeros((348, 8))), "joint_pos"),
    (lambda motion, ref: motion.update(fps=np.array([25.])), "fps"),
    (lambda motion, ref: motion.update(joint_names=motion["joint_names"][::-1]), "joint_names"),
    (lambda motion, ref: motion["source_phase"].__setitem__(248, 1), "source_phase"),
    (lambda motion, ref: motion["body_quat_w"].__setitem__((0, 0, 0), 2.), "quaternions"),
    (lambda motion, ref: motion["joint_tracking_mask"].__setitem__(7, True), "masks"),
    (lambda motion, ref: ref.update(qpos=np.zeros((349, 14))), "qpos"),
    (lambda motion, ref: ref["root_target_position_m"].__setitem__((0, 0), np.nan), "root_target_position_m"),
])
def test_hash_matched_but_structurally_invalid_artifacts_are_rejected(tmp_path, defect, match):
    motion, reference, catalog = write_fixture_pair(tmp_path, defect=defect)
    with pytest.raises(ValueError, match=match):
        registry.verify_reference_pair(motion, reference, catalog_path=catalog)
