"""Freeze the original-recipe / hockey-recipe speed and success-rate study."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np

from hockey_speed_study import evaluation_cases, scale_reference

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.archive.exists():
        raise FileExistsError("Use a fresh immutable snapshot")
    old = ROOT/"results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2"
    load = lambda name: dict(np.load(old/name, allow_pickle=False))
    motion, diagnostics = load("motion.npz"), load("command_diagnostics.npz")
    original = json.loads((old/"report.json").read_text())
    cfg = json.loads((ROOT/"config/hockey_ppo_learning_v1.json").read_text())
    if sha(old/"motion.npz") != cfg["reference_sha256"]:
        raise ValueError("Parent reference changed")
    a.output.mkdir(parents=True)
    scripts = ("train_hockey_ppo.py", "hockey_ppo_task.py", "hockey_beyondmimic_task.py", "hockey_speed_study.py",
        "height_smoothness.py", "physx_mjlab_recipe.py", "physx_reference_math.py", "evaluate_hockey_physx.py",
        "analyze_hockey_ppo.py", "analyze_hockey_speed_suite.py", "dispatch_hockey_speed_study.py",
        "prepare_hockey_speed_study.py")
    for name in scripts:
        shutil.copyfile(ROOT/"scripts"/name, a.output/name)
    cases = evaluation_cases()
    cells, references = [], []
    for label, factor in (("s100", 1.), ("s150", 1.5), ("s200", 2.)):
        directory = a.output/"references"/label
        directory.mkdir(parents=True)
        if factor == 1:
            for name in ("motion.npz", "command_diagnostics.npz"):
                shutil.copyfile(old/name, directory/name)
        else:
            m, d = scale_reference(motion, diagnostics, factor)
            np.savez_compressed(directory/"motion.npz", **m)
            np.savez_compressed(directory/"command_diagnostics.npz", **d)
        report = dict(original, kind="designed_hockey_speed_reference_v1", speed_scale=factor,
            parent_motion_sha256=sha(old/"motion.npz"), parent_report_sha256=sha(old/"report.json"),
            motion_sha256=sha(directory/"motion.npz"), entry_speed_m_s=factor*original["entry_speed_m_s"],
            designed_root_link_stop_distance_m=factor*original["designed_root_link_stop_distance_m"],
            controller_label="reference for independently trained original-tracking and hockey PPO recipes",
            admitted_for_training=True, existing_actor_has_explicit_velocity_or_braking_command=None,
            speed_scaling="world XY translation and its COM velocity contribution only; posture and phase clocks unchanged",
            posture_clock_changed=False, braking_duration_changed=False,
            physics_run=False, task_success_claimed=False)
        write(directory/"report.json", report)
        references.append(dict(id=label, scale=factor, root_link_entry_speed_m_s=report["entry_speed_m_s"],
            total_designed_root_displacement_m=report["designed_root_link_stop_distance_m"],
            kinetic_energy_ratio_at_equal_mass=factor**2, path=str(directory.relative_to(a.output))))
        for seed in (42, 43, 44):
            for method in ("beyondmimic", "hockey"):
                protocol = copy.deepcopy(cfg)
                protocol.update(kind="hockey_ppo_speed_success_v1", method=method, seed=seed, speed_scale=factor,
                    reference_sha256=sha(directory/"motion.npz"), evaluation_cases=cases,
                    variants={"tracking": 0.}, status="frozen_not_executed")
                if method == "beyondmimic":
                    protocol.update(actor_dim=51, critic_dim=120, extra_observation=[])
                    protocol["ppo"].update(learning_rate=.001, schedule="adaptive")
                    # These fields configure only the hockey branch. Remove
                    # them so the original branch cannot be mislabeled beta-on.
                    protocol.pop("reward_weights")
                    protocol.pop("reward_scales")
                    protocol.pop("no_wheel_XY_or_spin_imitation")
                    protocol.update(wheel_XY_imitation=True, wheel_spin_imitation=False)
                    protocol["baseline_contract"] = dict(
                        exponential_tracking_terms=6, beta_reward=False, stop_hold_reward=False,
                        support_reward=False, explicit_target_velocity_observation=False,
                        wheel_relative_XY_tracking=True, wheel_axial_orientation_tracking=False,
                        height_adaptation="XY-only position/linear velocity; vertical acceleration -0.1",
                        original_motion_end_timeout=True, original_orientation_error_termination_rad=.6)
                identity = f"{method}_{label}_seed{seed}"
                config_path = Path("protocols")/(identity+".json")
                write(a.output/config_path, protocol)
                cells.append(dict(id=identity, method=method, seed=seed, speed_scale=factor,
                    protocol=str(config_path), reference=str(directory.relative_to(a.output)), variant="tracking"))
    provenance_paths = [ROOT/"upstream/tron1-learning-from-human/scripts/train_tracking.py",
        ROOT/"upstream/tron1-learning-from-human/training/tron1_tracking.py",
        ROOT/"upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
        ROOT/"upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
        ROOT/"upstream/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"]
    write(a.output/"upstream_provenance.json", {str(f.relative_to(ROOT)): sha(f) for f in provenance_paths})
    plan = dict(kind="hockey_original_recipe_speed_success_v1", references=references, cells=cells,
        methods=["beyondmimic", "hockey"], seeds=[42, 43, 44], fixed_eval_mu=.7,
        formal_training_runs=18, transitions_per_run=12288000, total_formal_transitions=18*12288000,
        smoke_per_cell=dict(environments=32, iterations=2), formal_per_cell=dict(environments=512, iterations=1000),
        episodes_per_policy=len(cases), random_initial_state_trials_per_policy=32, total_evaluation_episodes=18*len(cases),
        case_draw_seed=20261028, cases=cases, paired_cases_across_methods_and_seeds=True,
        no_in_episode_randomization_at_validation=True, single_environment_validation=True,
        reinitialize_training_from_scratch=True, warm_start=False, hockey_additional_support_weight=0.,
        training_success="nominal full-start task pass for each training seed; report count out of three",
        rollout_success="32 explicitly perturbed initial states per trained policy; nominal excluded",
        confidence="Wilson interval per policy; mean and min/max across training seeds; no pooled IID claim",
        common_adaptations=["PhysX 400 Hz, control 50 Hz", "action-rate -0.5", "shared wheel/ground friction DR [0.3,1.2]",
            "shared encoder bias, COM offsets and pushes", "no absolute height target or height termination"],
        comparison_scope="full recipes, not a causal single-reward ablation: observation, rewards, LR schedule and terminal semantics differ",
        v1_sampling_and_phase_limitations_retained_in_hockey_branch=True,
        resources="one completely idle RTX at a time; serial jobs; no other process changes; stop on infrastructure failure",
        success_gates="same eleven physical stop-and-slide gates in frozen evaluate_hockey_physx.py",
        success_evaluator_sha256=sha(ROOT/"scripts/evaluate_hockey_physx.py"),
        source_speed_is_designed_not_measured=True)
    write(a.output/"plan.json", plan)
    files = [dict(path=str(f.relative_to(a.output)), bytes=f.stat().st_size, sha256=sha(f))
             for f in sorted(a.output.rglob('*')) if f.is_file()]
    manifest = dict(kind="hockey_speed_success_snapshot_v1", files=files, file_count=len(files),
        total_bytes=sum(r["bytes"] for r in files),
        exclusions=["robot models", "actor weights", "credentials", "source video", "SMPL assets"],
        formal_runs=18, total_formal_transitions=18*12288000, nominal_and_perturbed_episodes=18*33)
    write(a.output/"manifest.json", manifest)
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.archive, "w:gz") as archive:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():
                archive.add(f, arcname=str(f.relative_to(a.output)), recursive=False)
    print(json.dumps(dict(files=len(files), bytes=manifest["total_bytes"], archive_bytes=a.archive.stat().st_size,
        archive_sha256=sha(a.archive), references=references, formal_runs=18, episodes=594), indent=2))


if __name__ == "__main__":
    main()
