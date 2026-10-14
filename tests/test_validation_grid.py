import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_car_validation", ROOT / "scripts/run_car_validation.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_case_names_and_supported_bounds():
    assert mod.case_name(.3) == "mu0p300"
    assert mod.case_name(1.0) == "mu1p000"
    assert mod.case_name(.01) == "mu0p010"
    assert mod.case_name(2.) == "mu2p000"
    for invalid in (0., .009, 2.01, float("nan")):
        with pytest.raises(ValueError):
            mod.case_name(invalid)


def test_grid_config_has_no_false_beyondmimic_upper_coverage():
    cfg = json.loads((ROOT / "config/evaluation_friction.json").read_text())
    assert cfg["policy_training_friction_randomization"] is True
    assert cfg["beyondmimic_static_range"] == [.3, 1.6]
    assert cfg["beyondmimic_dynamic_range"] == [.3, 1.2]
    assert min(cfg["currently_supported_equal_static_dynamic_grid"]) == .01
    assert max(cfg["currently_supported_equal_static_dynamic_grid"]) == 2.0
    assert cfg["evaluation_mu_bounds"] == [.01, 2.0]
    assert 1.2 in cfg["currently_supported_equal_static_dynamic_grid"]


def test_wide_source_keeps_base_and_training_profile_and_checks_runtime_bounds(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import materialize_wide_friction_source as wide

    source = tmp_path / "base"
    for name in ("training/tron1_drift.py", "scripts/eval_tracking.py"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / "upstream/tron1-learning-from-human" / name).read_bytes())
    old = wide.sha(source / "training/tron1_drift.py")
    output = tmp_path / "wide"
    audit = wide.materialize(source, output)
    assert wide.sha(source / "training/tron1_drift.py") == old
    assert audit["evaluation_mu_bounds"] == [.01, 2.0]
    assert audit["training_unchanged"] is True
    with pytest.raises(FileExistsError):
        wide.materialize(source, output)
    namespace = {}
    exec((output / "training/tron1_drift.py").read_text(), namespace)
    profile = json.loads((ROOT / "config/car_drift_beyondmimic_task.json").read_text())
    assert namespace["validate_task_profile"](profile) == profile
    assert namespace["validate_task_profile"]({**profile, "contact_friction": 2.})["contact_friction"] == 2.
    for mu in (.009, 2.01):
        with pytest.raises(ValueError):
            namespace["validate_task_profile"]({**profile, "contact_friction": mu})
    evaluator = (output / "scripts/eval_tracking_friction_isaac.py").read_text()
    assert '.01 <= args.evaluation_contact_friction <= 2.' in evaluator
    compile(evaluator, "extended_evaluator", "exec")


def test_locked_open_car_source_is_pinned():
    lock = json.loads((ROOT / "config/car_reference_lock.json").read_text())
    source = ROOT / "upstream/drift_drl/code/ref_trajectory/traj_4.csv"
    assert mod.sha(source) == lock["source_csv_sha256"]
    assert lock["left_wheel_peak_abs_separation_deg"] > 20


def test_public_base_materializes_exact_audited_friction_evaluator():
    spec = importlib.util.spec_from_file_location(
        "materialize_friction_evaluator", ROOT / "scripts/materialize_friction_evaluator.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    base = ROOT / "upstream/tron1-learning-from-human/scripts/eval_tracking.py"
    content = generator.generate(base)
    assert __import__("hashlib").sha256(content.encode()).hexdigest() == generator.OUTPUT_SHA256


def test_narrow_drift_patch_matches_formal_training_source(tmp_path):
    base = ROOT / "upstream/tron1-learning-from-human/training/tron1_drift.py"
    target = tmp_path / "training/tron1_drift.py"
    target.parent.mkdir()
    target.write_bytes(base.read_bytes())
    subprocess.run(["patch", "--silent", "-p1", "-i",
                    str(ROOT / "patches/tron1_drift_formal.patch")],
                   cwd=tmp_path, check=True)
    assert mod.sha(target) == "5eb40c9b3c14f5bd5c59c9861c1414ae8ac1b219ff0dd70591f2c6fe0bb5344c"
    profile = json.loads((ROOT / "config/car_drift_beyondmimic_task.json").read_text())
    assert "training_robot_restitution_range" not in profile


def test_generic_grid_rejects_no_data_and_renders_one_valid_case(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_friction_grid as grid

    time = np.array([0., 1.44, 2., 4., 5.])
    ref_pos = np.column_stack((time, np.zeros((5, 2))))
    ref = tmp_path / "reference.npz"
    np.savez_compressed(ref, time_s=time, root_target_position_m=ref_pos)
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"test-motion")
    checkpoint = tmp_path / "model_final.pt"
    checkpoint.write_bytes(b"test-actor")
    trainer = tmp_path / "training.json"
    trainer.write_text(json.dumps({"recipe": {"friction_static": [.3, 1.6],
                                               "friction_dynamic": [.3, 1.2],
                                               "material_buckets": 64},
                                   "motion_sha256": mod.sha(motion)}))
    ia_file = tmp_path / "isaac.npz"
    mj_file = tmp_path / "mujoco.npz"
    np.savez_compressed(ia_file, time_s=time, valid_mask=np.ones((5, 1), dtype=bool),
                        root_position_m=ref_pos[:, None, :])
    np.savez_compressed(mj_file, time_s=time, root_pos=ref_pos)
    ia_report = tmp_path / "ia.json"
    mj_report = tmp_path / "mj.json"
    ia_report.write_text(json.dumps({
        "evaluation_contact_friction_override": {"evaluation_contact_friction": .7},
        "contact_material_audit": {"expected_friction": .7,
            "robot_runtime_friction_range": [.7, .7],
            "ground_colliders": [{"static_friction": .7, "dynamic_friction": .7}]},
        "physics_trajectory_sha256": mod.sha(ia_file),
        "motion_file_sha256": mod.sha(motion),
        "checkpoint_sha256": mod.sha(checkpoint)}))
    mj_report.write_text(json.dumps({"contact_material_audit": {"compiled_friction_range": [.7, .7]},
                                     "motion_sha256": mod.sha(motion)}))
    manifest = tmp_path / "grid.json"
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "policy_training_friction_randomization": True,
                                    "training_manifest": str(trainer), "reference_file": str(ref),
                                    "checkpoint": str(checkpoint),
                                    "cases": [{"case": "mu0p700", "mu": .7,
                                               "isaac_physics": str(ia_file),
                                               "mujoco_rollout": str(mj_file),
                                               "isaac_report": str(ia_report),
                                               "mujoco_report": str(mj_report)}]}))
    result = grid.build(manifest, tmp_path / "output")
    assert result["rows"][0]["sim2sim"]["full_core_xy_rmse_m"] == 0.
    assert (tmp_path / "output/topdown.png").is_file()
    replayed = grid.replay(tmp_path / "output/xy_trace.npz", tmp_path / "replay",
                           tmp_path / "output/summary.json")
    assert replayed == result
    assert (tmp_path / "replay/topdown.png").is_file()
    report = json.loads(ia_report.read_text())
    report["contact_material_audit"]["robot_runtime_friction_range"] = [.7, 1.0]
    ia_report.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="friction/material/trace/reference"):
        grid.build(manifest, tmp_path / "bad_readback")


def test_published_new_grid_replays_without_physics(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_friction_grid as grid

    source = ROOT / "results/car_bm_dr_seed42"
    result = grid.replay(source / "xy_trace.npz", tmp_path / "replayed",
                         source / "summary.json")
    assert len(result["rows"]) == 6
    assert all(row["isaac"]["core_complete"] and row["mujoco"]["core_complete"]
               for row in result["rows"])
