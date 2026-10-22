"""Synthetic actor/provenance/backend tests only, no GPU/physics/inference."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_ppo_torque_control_v2_transfer as module
from test_analyze_ppo_torque_control_v2 import data, cohort, managers
from test_analyze_ppo_ground_support_comparison import write_json, change_json


@pytest.fixture
def transfer(data):
    audit_cases = {}
    for label, directory, actor_sha in (("baseline", data.baseline, "1"*64), ("candidate", data.candidate, "2"*64)):
        change_json(directory, "report.json", lambda doc: doc["policy_export"].update(root_prescribed_during_steps=False,
            domain_randomization="disabled during evaluation; fixed actuator parameters",
            evaluation_contact_friction_override={"evaluation_contact_friction": .7}))
        isaac = module.physical.shared.read_json(directory/"report.json")
        with np.load(directory/"physics_trajectory.npz") as archive:
            source = {name: archive[name] for name in archive.files}
        field = module.physical.contact._field
        raw = {"time_s": source["time_s"], "physics_dt_s": np.array(.005), "joint_names": source["joint_names"],
            "root_pos": field(source, "root_position_m"), "root_quat_wxyz": field(source, "root_quaternion_wxyz"),
            "root_com_lin_vel_w": field(source, "root_linear_velocity_w_m_s"), "root_ang_vel_w": field(source, "root_angular_velocity_w_rad_s"),
            "joint_pos": field(source, "joint_position_rad"), "joint_vel": field(source, "joint_velocity_rad_s"), "wheel_pos": np.zeros((1397, 2, 3)),
            "wheel_ground_contact_count": field(source, "wheel_ground_contact_count"),
            "wheel_ground_contact_force_w_n": field(source, "wheel_ground_contact_force_w_n"),
            "wheel_lateral_contact_slip_m_s": np.zeros((1397, 2)), "nonwheel_ground_contact_count": np.zeros(1397)}
        folder = directory.parent/"mujoco"; folder.mkdir(); np.savez(folder/"rollout.npz", **raw)
        report = {key: {"same": key} for key in module.mujoco.SHARED_KEYS}
        report.update(status="completed", engine="MuJoCo", engine_version="3.8.1", full_horizon_diagnostic=True,
            completed_full_reference=True, physics_stepped=True, termination="motion_end", hidden_resets=0, root_state_writes=1,
            physics_steps=1396, policy_steps=349, start_reference_frame=0, final_reference_frame=348,
            motion_sha256=isaac["motion_file_sha256"], model_sha256=module.mujoco.MODEL_SHA256,
            contract=copy.deepcopy(isaac["policy_export"]), policy_sha256=actor_sha, policy_kind="normalized_torchscript_actor",
            contact_material_audit={"expected_friction": .7, "compiled_friction_range": [.7, .7]},
            domain_parameters_resampled_during_episode=False,
            domain_parameters={"is_nominal": True, "torque_scale": [1.]*8, "velocity_scale": [1.]*8, "wheel_friction_nm": [0., 0.]},
            contact_slip_measurement={"active_point_normal_force_threshold_n": 5., "sample_dt_s": .005})
        write_json(folder/"report.json", report)
        audit_cases[label] = {"actor_sha256": actor_sha, "checkpoint_sha256": isaac["checkpoint_sha256"], "actor_path": "/private/remote/actor.pt"}
    audit = data.tmp/"actor_audit.json"
    write_json(audit, {"schema_version": 1, "kind": "remote_generated_actor_file_sha256_audit", "cases": audit_cases})
    return SimpleNamespace(**vars(data), actor_sha_audit=audit)


def load(data):
    return module.load_transfer(data.candidate, data.training, data.actor_sha_audit, data.reference, data.baseline)


def test_torque_specific_dual_engine_summary_bound_to_formal_parent(transfer):
    result = load(transfer)
    assert result["kind"] == "recorded_nominal_ppo_torque_control_v2_physx_mujoco_transfer_comparison"
    assert result["original_grounded_reward_preserved"] and result["original_grounded_weight"] == .4
    assert result["candidate_additional_reward"] == {"term": "leg_applied_torque_rate", "weight": -.05}
    assert result["evaluation_engines"] == ["PhysX", "MuJoCo"]
    assert result["status"] == "requires_visual_and_tracking_review" and not result["stable_contact_claimed"]
    assert result["parent_physx_analysis_provenance"]["kind"] == "recorded_nominal_physx_ppo_torque_control_v2_comparison"
    for label in ("baseline", "candidate"):
        row = result["cases"][label]
        assert row["PhysX"]["checkpoint_sha256"] == row["MuJoCo"]["checkpoint_sha256"]
        assert "torque_control" in row["PhysX"]["windows"]["core"]
        for engine in ("PhysX", "MuJoCo"):
            core = row[engine]["windows"]["core"]
            assert core["tracking"]["xy_rmse_m"] == pytest.approx(.01)
            assert core["tracking"]["speed_retention"] == pytest.approx(1.)
            assert core["tracking"]["beta_rmse_deg"] == pytest.approx(0.)
            assert core["tilt"]["peak_deg"] == 0.
            assert core["force_z"]["right"]["step_abs_peak_n"] == 97.
            assert core["base_world_xy_angular_velocity_spectrum"]["high_frequency_rms"] > 0


def test_reject_legacy_ground_support_training_kind_before_using_mujoco_helper(transfer, monkeypatch):
    change_json(transfer.training, "ablation_manifest.json", lambda doc: doc.update(kind="historical_physx_ppo_substep_ground_support_replacement"))
    monkeypatch.setattr(module.mujoco, "load_mujoco_case", lambda *a, **kw: pytest.fail("Must reject wrong training method first"))
    with pytest.raises(ValueError, match="torque-control v2"): load(transfer)


@pytest.mark.parametrize("field,value", [("policy_sha256", "0"*64), ("model_sha256", "0"*64), ("hidden_resets", 1),
    ("root_state_writes", 2), ("full_horizon_diagnostic", False), ("completed_full_reference", False)])
def test_reject_mujoco_actor_or_native_lifecycle(transfer, field, value):
    change_json(transfer.candidate.parent/"mujoco", "report.json", lambda doc: doc.update({field: value}))
    with pytest.raises(ValueError): load(transfer)


def test_reject_actor_hash_audit_wrong_checkpoint(transfer):
    change_json(transfer.tmp, transfer.actor_sha_audit.name, lambda doc: doc["cases"]["candidate"].update(checkpoint_sha256="0"*64))
    with pytest.raises(ValueError, match="actor file SHA"): load(transfer)


def test_reject_changed_mujoco_solver_contract(transfer):
    change_json(transfer.candidate.parent/"mujoco", "report.json", lambda doc: doc["contact_material_audit"].update(compiled_solref_ranges=[[.02, .02], [1., 1.]]))
    with pytest.raises(ValueError, match="Shared MuJoCo"): load(transfer)


def test_fresh_json_only_no_private_actor_path_or_new_simulation(transfer):
    output = transfer.tmp/"transfer_summary.json"
    args = SimpleNamespace(candidate_dir=transfer.candidate, candidate_training_dir=transfer.training, baseline_dir=transfer.baseline,
        reference=transfer.reference, actor_sha_audit=transfer.actor_sha_audit, candidate_status=None, output=output)
    result = module.run(args)
    assert not result["new_simulation"] and not result["new_inference"]
    assert "/private/remote/" not in output.read_text()
    assert result["actor_sha_audit_sha256"] == module.sha256(transfer.actor_sha_audit)
    with pytest.raises(ValueError, match="fresh"): module.run(args)


def test_actor_audit_required_even_without_local_weights():
    with pytest.raises(SystemExit):
        module.parser().parse_args(["--candidate-dir", "a", "--candidate-training-dir", "b", "--output", "out.json"])
