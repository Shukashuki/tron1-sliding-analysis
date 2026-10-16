"""Build an isolated full-horizon PPO/SAC frozen-actor evaluation source.

Old policy/model sources remain untouched. SAC is loaded from its hash-verified
deterministic TorchScript export, never interpreted as an RSL PPO checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import materialize_full_horizon_source as full


def patch_controller_entry(text):
    text = full.replace_once(text, "    args = parser.parse_args()\n",
                             "    parser.add_argument(\"--allow-smoke-policy\", action=\"store_true\",\n"
                             "                        help=\"SAC inference-pipeline smoke only; not a trained-policy comparison\")\n"
                             "    args = parser.parse_args()\n")
    old = "        runner = OnPolicyRunner(wrapped, copy.deepcopy(runner_cfg), log_dir=None, device=args.device)"
    new = ("        if runner_cfg.get(\"algorithm\") == \"SAC\":\n"
           "            if args.allow_smoke_policy:\n"
           "                runner_cfg[\"allow_smoke_actor\"] = True\n"
           "            from training.tron1_scripted_actor_runner import ScriptedActorRunner\n"
           "            runner = ScriptedActorRunner(wrapped, runner_cfg, log_dir=None, device=args.device)\n"
           "        else:\n    " + old)
    text = full.replace_once(text, old, new)
    old = '            contract["training_domain_randomization"] = metadata.get("domain_randomization_profile")'
    new = ('            contract["training_domain_randomization"] = metadata.get("recipe", metadata.get("domain_randomization_profile"))\n'
           '            if metadata.get("height_regularization") is not None:\n'
           '                contract["height_regularization"] = metadata["height_regularization"]\n'
           '                contract["historical_pose_predicates_only"] = True\n'
           '        contract["policy_algorithm"] = "SAC" if runner_cfg.get("algorithm") == "SAC" else "PPO"\n'
           '        if getattr(runner, "inference_provenance", None) is not None:\n'
           '            contract["scripted_actor_provenance"] = runner.inference_provenance')
    text = full.replace_once(text, old, new)
    return full.replace_once(text, '"actor_output_is_unclipped": True,',
                             '"actor_output_is_unclipped": runner_cfg.get("algorithm") != "SAC",')


def materialize(source, output):
    helper = Path(__file__).with_name("scripted_actor_runner.py")
    if not helper.is_file():
        raise FileNotFoundError("SAC inference bridge is required before source materialization")
    audit = full.materialize(source, output)
    target = output / "scripts/eval_tracking_friction_isaac.py"
    content = patch_controller_entry(target.read_text())
    compile(content, str(target), "exec")
    target.write_text(content)
    copied = output / "training/tron1_scripted_actor_runner.py"
    shutil.copy2(helper, copied)
    audit.update(controller_algorithms=["PPO", "SAC"],
                 actor_loading="RSL PPO or hash-verified raw-observation deterministic SAC TorchScript",
                 materializer_sha256=full.wide.sha(Path(__file__)))
    audit["changed_output_sha256"].update({
        str(target.relative_to(output)): full.wide.sha(target),
        str(copied.relative_to(output)): full.wide.sha(copied)})
    (output / "wide_friction_source.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    materialize(args.source_root.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": "materialized", "controller_algorithms": ["PPO", "SAC"],
                      "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
