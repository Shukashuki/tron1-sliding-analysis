"""Generate the audited friction evaluator from the pinned public base script.

This adds only the evaluation-only material override. It refuses a different
base source or an existing output, and verifies the byte-exact generated file.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


BASE_SHA256 = "fd90089810d3d14bfd96210723afdd3a36113a19f667ec60b2df617dd032f992"
OUTPUT_SHA256 = "11cc0b4a1d44ad9dbb49bf260162dd4d0424e8fb4d836befae1eafd26f3f7a3e"

REPLACEMENTS = [
    ("\"\"\"Evaluate one complete deterministic TRON1 tracking episode per environment.",
     "\"\"\"Single-environment Isaac contact-friction diagnostic for a frozen actor."),
    ("import time\n\nimport numpy as np", "import time\nimport traceback\n\nimport numpy as np"),
    ("    parser.add_argument(\"--task-profile\", type=Path, help=\"Must match the checkpoint drift task/material contract\")\n",
     "    parser.add_argument(\"--task-profile\", type=Path, help=\"Must match the checkpoint drift task/material contract\")\n"
     "    parser.add_argument(\"--evaluation-contact-friction\", type=float, required=True,\n"
     "                        help=\"Evaluation-only material friction; training profile remains unchanged\")\n"),
    ("    args = parser.parse_args()\n    deployment_leg_kp = getattr(args, \"deployment_leg_kp\", None)",
     "    args = parser.parse_args()\n"
     "    if (not math.isfinite(args.evaluation_contact_friction)\n"
     "            or not .01 <= args.evaluation_contact_friction <= 1.):\n"
     "        parser.error(\"Evaluation contact friction must be finite and in [0.01, 1.0]\")\n"
     "    if (args.task_profile is None or args.num_envs != 1 or args.seed != 42\n"
     "            or args.terrain_file is not None or args.domain_parameters is not None\n"
     "            or args.deployment_leg_kp is not None or args.deployment_leg_kd is not None\n"
     "            or args.physics_dt_s is not None or not args.record_physics\n"
     "            or not args.verify_unmasked_device or args.contact_threshold_n != 5.0):\n"
     "        parser.error(\"Diagnostic requires one seed42 200Hz episode, original gains, and physics recording\")\n"
     "    deployment_leg_kp = getattr(args, \"deployment_leg_kp\", None)"),
    ("        cfg = make_env_cfg(args.motion_file, args.asset_path, args.num_envs, args.device, eval_mode=True,\n"
     "                           dr_profile={\"mode\": \"fixed\", \"fixed_draw\": domain_parameters} if domain_parameters else None,\n"
     "                           terrain_spec=terrain_spec, task_profile=task_profile)\n",
     "        cfg = make_env_cfg(args.motion_file, args.asset_path, args.num_envs, args.device, eval_mode=True,\n"
     "                           dr_profile={\"mode\": \"fixed\", \"fixed_draw\": domain_parameters} if domain_parameters else None,\n"
     "                           terrain_spec=terrain_spec, task_profile=task_profile)\n"
     "        # Checkpoint/manifest validation above uses the unchanged trained profile.\n"
     "        # Only Isaac's evaluation-time terrain, default, and event materials change.\n"
     "        evaluation_profile = copy.deepcopy(task_profile)\n"
     "        mu = args.evaluation_contact_friction\n"
     "        evaluation_profile[\"contact_friction\"] = mu\n"
     "        cfg.scene.terrain.physics_material.static_friction = mu\n"
     "        cfg.scene.terrain.physics_material.dynamic_friction = mu\n"
     "        cfg.sim.physics_material = cfg.scene.terrain.physics_material\n"
     "        cfg.events.physics_material.params[\"static_friction_range\"] = (mu, mu)\n"
     "        cfg.events.physics_material.params[\"dynamic_friction_range\"] = (mu, mu)\n"),
    ("            contact_material_audit = audit_isaac_materials(env, task_profile)",
     "            contact_material_audit = audit_isaac_materials(env, evaluation_profile)"),
    ("        if args.physics_dt_s is not None:\n"
     "            contract = apply_physics_clock_to_contract(\n"
     "                contract, args.physics_dt_s, trained_contract=trained_manifest[\"contract\"])",
     "        contract[\"trained_task_profile\"] = copy.deepcopy(task_profile)\n"
     "        contract[\"task_profile\"] = copy.deepcopy(evaluation_profile)\n"
     "        contract[\"evaluation_contact_friction_override\"] = {\n"
     "            \"trained_contact_friction\": task_profile[\"contact_friction\"],\n"
     "            \"evaluation_contact_friction\": mu,\n"
     "            \"only_changed_physics_field\": \"contact_friction\",\n"
     "        }\n"
     "        if args.physics_dt_s is not None:\n"
     "            contract = apply_physics_clock_to_contract(\n"
     "                contract, args.physics_dt_s, trained_contract=trained_manifest[\"contract\"])"),
    ("            \"task_profile\": task_profile,\n            \"contact_material_audit\": contact_material_audit,",
     "            \"task_profile\": task_profile,\n"
     "            \"evaluation_task_profile\": evaluation_profile,\n"
     "            \"evaluation_contact_friction_override\": contract[\"evaluation_contact_friction_override\"],\n"
     "            \"contact_material_audit\": contact_material_audit,"),
    ("    finally:\n        # Undo observers in reverse installation order, even on an error.",
     "    except BaseException:\n"
     "        # Isaac's immediate_exit cleanup may otherwise suppress the traceback.\n"
     "        traceback.print_exc()\n"
     "        sys.stderr.flush()\n"
     "        raise\n"
     "    finally:\n        # Undo observers in reverse installation order, even on an error."),
]


def generate(base: Path) -> str:
    content = base.read_bytes()
    if hashlib.sha256(content).hexdigest() != BASE_SHA256:
        raise ValueError("Public base evaluator differs from pinned submodule revision")
    text = content.decode("utf-8")
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError("Friction evaluator patch anchor is absent or ambiguous")
        text = text.replace(old, new, 1)
    if hashlib.sha256(text.encode()).hexdigest() != OUTPUT_SHA256:
        raise ValueError("Generated friction evaluator does not match the audited output hash")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="Defaults to source-root/scripts/eval_tracking_friction_isaac.py")
    args = parser.parse_args()
    base = args.source_root / "scripts/eval_tracking.py"
    output = args.output or args.source_root / "scripts/eval_tracking_friction_isaac.py"
    if output.exists():
        parser.error(f"Preserving existing file: {output}")
    result = generate(base)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result)
    print(f"Generated audited evaluator: {output} ({OUTPUT_SHA256})")


if __name__ == "__main__":
    main()
