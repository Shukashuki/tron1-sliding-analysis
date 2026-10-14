"""Build a separate, audited full-horizon fixed-friction evaluation source.

The original pose predicates are observed rather than used to stop rollout.
This snapshot is intended for diagnosing XY errors after a normal early stop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import materialize_wide_friction_source as wide


MUJOCO_BASE_SHA256 = "da6d1da378b5065589a49f2befaebfb0576665dbd2ad5d34a157d11b8984a3aa"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Absent/ambiguous full-horizon patch anchor: {old[:80]}")
    return text.replace(old, new, 1)


def patch_isaac(text):
    for old, new in (
        ("        cfg.seed = args.seed\n        env = TerminalCaptureEnv(cfg=cfg)",
         "        from training.tron1_full_horizon_diagnostic import (install_isaac_diagnostics,\n"
         "            reset_isaac_diagnostics, isaac_diagnostic_report, validate_snapshot)\n"
         "        install_isaac_diagnostics(cfg)\n"
         "        cfg.seed = args.seed\n        env = TerminalCaptureEnv(cfg=cfg)"),
        ("            if task_profile is not None:\n                from training.tron1_contact_slip import sample_raw_isaac_ground_contacts",
         "            validate_snapshot(snapshot)\n"
         "            if task_profile is not None:\n                from training.tron1_contact_slip import sample_raw_isaac_ground_contacts"),
        ("        env.capture_resets = True\n", "        reset_isaac_diagnostics(env)\n        env.capture_resets = True\n"),
        ("        contract[\"trained_task_profile\"] = copy.deepcopy(task_profile)",
         "        contract[\"full_horizon_diagnostic\"] = True\n"
         "        contract[\"trained_task_profile\"] = copy.deepcopy(task_profile)"),
        ("            \"schema_version\": 1, \"status\": \"evaluated\" if not active.any() else \"evaluation_incomplete\",",
         "            \"schema_version\": 1, \"status\": \"evaluated\" if not active.any() else \"evaluation_incomplete\",\n"
         "            \"full_horizon_diagnostic\": True,\n"
         "            \"pose_termination_diagnostics\": isaac_diagnostic_report(env),"),
    ):
        text = replace_once(text, old, new)
    return text


def patch_mujoco(text):
    for old, new in (
        ("    termination, terms = \"motion_end\", []\n",
         "    termination, terms = \"motion_end\", []\n"
         "    diagnostic_mode = bool(contract.get(\"full_horizon_diagnostic\", False))\n"
         "    diagnostic_times, diagnostic_flags = [], []\n"
         "    from training.tron1_full_horizon_diagnostic import POSE_TERMS, summarize_flags\n"),
        ("        state = measured.measure()\n        force, ground_force, _ = measured.contact_forces()",
         "        state = measured.measure()\n"
         "        if not all(np.isfinite(state[key]).all() for key in\n"
         "                   (\"root_pos\", \"root_quat_wxyz\", \"joint_pos\", \"joint_vel\")):\n"
         "            raise FloatingPointError(\"Nonfinite full-horizon MuJoCo state\")\n"
         "        if physical and data.time <= physical[-1][\"time_s\"]:\n"
         "            raise RuntimeError(\"Unexpected MuJoCo time reset; refusing spliced rollout\")\n"
         "        force, ground_force, _ = measured.contact_forces()"),
        ("        terms = training_termination_terms(state, target)\n        if terms:\n",
         "        terms = training_termination_terms(state, target)\n"
         "        diagnostic_times.append(float(data.time))\n"
         "        diagnostic_flags.append([name in terms for name in POSE_TERMS])\n"
         "        if terms and not diagnostic_mode:\n"),
        ("    arrays = {name: np.asarray([sample[name] for sample in physical]) for name in physical[0]}",
         "    if diagnostic_mode:\n        terms = []\n"
         "    arrays = {name: np.asarray([sample[name] for sample in physical]) for name in physical[0]}"),
        ("    arrays.update(joint_names=np.array(reference.action_names), policy_dt_s=np.array(1 / reference.fps),",
         "    arrays.update(pose_predicate_time_s=np.asarray(diagnostic_times),\n"
         "                  pose_predicate_flags=np.asarray(diagnostic_flags, dtype=bool),\n"
         "                  pose_predicate_names=np.asarray(POSE_TERMS))\n"
         "    arrays.update(joint_names=np.array(reference.action_names), policy_dt_s=np.array(1 / reference.fps),"),
        ("        \"status\": \"completed\" if termination == \"motion_end\" else \"terminated\",",
         "        \"status\": \"completed\" if termination == \"motion_end\" else \"terminated\",\n"
         "        \"full_horizon_diagnostic\": diagnostic_mode,\n"
         "        \"pose_termination_diagnostics\": summarize_flags(diagnostic_times, diagnostic_flags),"),
    ):
        text = replace_once(text, old, new)
    return text


def materialize(source: Path, output: Path):
    if wide.sha(source / "scripts/eval_tracking_mujoco.py") != MUJOCO_BASE_SHA256:
        raise ValueError("MuJoCo evaluator differs from the pinned formal/public source")
    _, isaac = wide.extend_sources(source)
    isaac = patch_isaac(isaac)
    mujoco = patch_mujoco((source / "scripts/eval_tracking_mujoco.py").read_text())
    audit = wide.materialize(source, output)
    (output / "scripts/eval_tracking_friction_isaac.py").write_text(isaac)
    (output / "scripts/eval_tracking_mujoco.py").write_text(mujoco)
    shutil.copy2(Path(__file__).with_name("full_horizon_diagnostic.py"),
                 output / "training/tron1_full_horizon_diagnostic.py")
    audit.update(full_horizon_diagnostic=True,
                 observed_pose_predicates=["anchor_pos", "anchor_ori", "ee_body_pos"],
                 stopping_rules="motion_end/time limits; abort on nonfinite state or time reset",
                 materializer_sha256=wide.sha(Path(__file__)))
    names = ["scripts/eval_tracking_friction_isaac.py", "scripts/eval_tracking_mujoco.py",
             "training/tron1_drift.py", "training/tron1_full_horizon_diagnostic.py"]
    audit["changed_output_sha256"] = {name: wide.sha(output / name) for name in names}
    (output / "wide_friction_source.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source_root.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": "materialized", "output": str(args.output_dir),
                      "full_horizon_diagnostic": True}))


if __name__ == "__main__":
    main()
