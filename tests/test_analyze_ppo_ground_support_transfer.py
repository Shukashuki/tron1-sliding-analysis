"""CPU synthetic tests; actor files, policy inference and simulation are absent."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_ppo_ground_support_transfer as module
from test_analyze_ppo_ground_support_comparison import cohort, write_json, change_json


@pytest.fixture
def transfer(cohort):
    actor_cases = {}
    for label, isaac_dir, actor_sha in (("baseline", cohort.baseline, "1"*64), ("candidate", cohort.candidate, "2"*64)):
        change_json(isaac_dir, "report.json", lambda doc: doc["policy_export"].update(
            root_prescribed_during_steps=False, domain_randomization="disabled during evaluation; fixed actuator parameters",
            evaluation_contact_friction_override={"evaluation_contact_friction": .7}))
        isaac = module.physical.read_json(isaac_dir/"report.json")
        with np.load(isaac_dir/"physics_trajectory.npz") as archive:
            source = {key: archive[key] for key in archive.files}
        field = module.contact._field
        raw = {"time_s": source["time_s"].copy(), "physics_dt_s": np.array(.005), "joint_names": source["joint_names"].copy(),
            "root_pos": field(source, "root_position_m"), "root_quat_wxyz": field(source, "root_quaternion_wxyz"),
            "root_com_lin_vel_w": field(source, "root_linear_velocity_w_m_s"), "root_ang_vel_w": field(source, "root_angular_velocity_w_rad_s"),
            "joint_pos": field(source, "joint_position_rad"), "joint_vel": field(source, "joint_velocity_rad_s"),
            "wheel_pos": np.zeros((1397, 2, 3)), "wheel_ground_contact_count": field(source, "wheel_ground_contact_count"),
            "wheel_ground_contact_force_w_n": field(source, "wheel_ground_contact_force_w_n"),
            "wheel_lateral_contact_slip_m_s": np.zeros((1397, 2)), "nonwheel_ground_contact_count": np.zeros(1397)}
        mujoco_dir = isaac_dir.parent/"mujoco"
        mujoco_dir.mkdir()
        np.savez(mujoco_dir/"rollout.npz", **raw)
        report = {key: {"same": key} for key in module.SHARED_KEYS}
        report.update(status="completed", engine="MuJoCo", engine_version="3.8.1", completed_full_reference=True,
            full_horizon_diagnostic=True, physics_stepped=True, termination="motion_end", hidden_resets=0, root_state_writes=1,
            physics_steps=1396, policy_steps=349, start_reference_frame=0, final_reference_frame=348,
            motion_sha256=isaac["motion_file_sha256"], model_sha256=module.MODEL_SHA256,
            contract=copy.deepcopy(isaac["policy_export"]), policy_sha256=actor_sha, policy_kind="normalized_torchscript_actor",
            contact_material_audit={"expected_friction": .7, "compiled_friction_range": [.7, .7]},
            domain_parameters_resampled_during_episode=False,
            domain_parameters={"is_nominal": True, "torque_scale": [1.]*8, "velocity_scale": [1.]*8, "wheel_friction_nm": [0., 0.]},
            contact_slip_measurement={"active_point_normal_force_threshold_n": 5., "sample_dt_s": .005})
        write_json(mujoco_dir/"report.json", report)
        actor_cases[label] = {"actor_sha256": actor_sha, "checkpoint_sha256": isaac["checkpoint_sha256"],
            "actor_path": "/private/remote/"+label+"/actor_normalized.pt"}
    actor_audit = cohort.tmp/"actor_audit.json"
    write_json(actor_audit, {"schema_version": 1, "kind": "remote_generated_actor_file_sha256_audit", "cases": actor_cases})
    return SimpleNamespace(**vars(cohort), actor_audit=actor_audit)


def load(data):
    return module.load_transfer(data.candidate, data.training, data.actor_audit, data.reference, data.baseline)


def test_native_transfer_summary_binds_both_checkpoints_without_actor_bytes(transfer):
    result = load(transfer)
    assert result["evaluation_engines"] == ["PhysX", "MuJoCo"]
    assert result["training_engine"] == "PhysX"
    assert result["status"] == "requires_visual_and_tracking_review"
    assert not result["stable_contact_claimed"] and not result["contact_success_threshold_added"]
    assert result["actor_sha_audit_sha256"] == module.smooth.sha256(transfer.actor_audit)
    for label in ("baseline", "candidate"):
        row = result["cases"][label]
        assert row["MuJoCo"]["checkpoint_sha256"] == row["PhysX"]["checkpoint_sha256"]
        assert row["MuJoCo"]["root_state_writes"] == {"frame0_initialization": 1, "during_stepping": 0}
        core = row["MuJoCo"]["windows"]["core"]
        assert core["samples"] == 513
        assert core["tracking"]["xy_rmse_m"] == pytest.approx(.01)
        assert core["base_world_xy_angular_velocity_spectrum"]["high_frequency_rms"] > 0
        assert core["force_z"]["right"]["step_abs_peak_n"] == 97.
        assert not row["MuJoCo"]["windows"]["full"]["tracking"]["available"]


@pytest.mark.parametrize("field,value", [("root_state_writes", 0), ("root_state_writes", 2), ("hidden_resets", 1),
    ("status", "partial"), ("physics_stepped", False), ("full_horizon_diagnostic", False),
    ("completed_full_reference", False), ("policy_steps", 348), ("model_sha256", "9"*64),
    ("motion_sha256", "9"*64), ("policy_sha256", "9"*64)])
def test_reject_invalid_mujoco_identity_and_lifecycle(transfer, field, value):
    change_json(transfer.candidate.parent/"mujoco", "report.json", lambda doc: doc.update({field: value}))
    with pytest.raises(ValueError):
        load(transfer)


def test_reject_changed_contract_even_if_model_matches(transfer):
    change_json(transfer.candidate.parent/"mujoco", "report.json", lambda doc: doc["contract"].update(leg_kd=5.))
    with pytest.raises(ValueError, match="exact PhysX"):
        load(transfer)


@pytest.mark.parametrize("mutation", [
    lambda doc: doc["contact_material_audit"].update(compiled_friction_range=[.15, .15]),
    lambda doc: doc["domain_parameters"].update(is_nominal=False),
    lambda doc: doc["domain_parameters"].update(torque_scale=[.9]*8),
    lambda doc: doc.update(domain_parameters_resampled_during_episode=True),
])
def test_reject_non_nominal_mujoco_material_or_actuator_draw(transfer, mutation):
    change_json(transfer.candidate.parent/"mujoco", "report.json", mutation)
    with pytest.raises(ValueError, match="domain randomization"):
        load(transfer)


def test_reject_changed_actor_audit_checkpoint(transfer):
    change_json(transfer.tmp, transfer.actor_audit.name, lambda doc: doc["cases"]["candidate"].update(checkpoint_sha256="8"*64))
    with pytest.raises(ValueError, match="actor file SHA"):
        load(transfer)


def test_reject_changed_mujoco_physical_backend_configuration(transfer):
    change_json(transfer.candidate.parent/"mujoco", "report.json", lambda doc: doc.update(engine_version="different"))
    with pytest.raises(ValueError, match="Shared MuJoCo"):
        load(transfer)


def test_reject_truncated_native_recording(transfer):
    path = transfer.candidate.parent/"mujoco/rollout.npz"
    with np.load(path) as archive:
        raw = {key: archive[key] for key in archive.files}
    raw = {key: value[:-4] if value.ndim and value.shape[0] == 1397 else value for key, value in raw.items()}
    np.savez(path, **raw)
    with pytest.raises(ValueError, match="original1397"):
        load(transfer)


def test_output_json_only_fresh_and_private_actor_paths_not_published(transfer):
    output = transfer.tmp/"transfer.json"
    args = SimpleNamespace(candidate_dir=transfer.candidate, candidate_training_dir=transfer.training,
        actor_sha_audit=transfer.actor_audit, reference=transfer.reference, baseline_dir=transfer.baseline,
        candidate_status=None, output=output)
    module.run(args)
    assert "/private/remote/" not in output.read_text()
    with pytest.raises(ValueError, match="fresh"):
        module.run(args)


def test_actor_audit_argument_is_required():
    with pytest.raises(SystemExit):
        module.parser().parse_args(["--candidate-dir", "a", "--candidate-training-dir", "b", "--output", "out.json"])
