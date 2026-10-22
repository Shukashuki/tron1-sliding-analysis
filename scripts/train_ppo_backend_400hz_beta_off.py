"""Isolated beta-reward OFF adapter around the immutable native400Hz trainer.

Only the explicit body/COM sideslip reward changes, 2 -> 0. PPO, setup/reseed,
DR, clocks, other rewards and the original +0.4 contact reward stay frozen.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE_WRAPPER_SHA256 = "5234070be219e59c61e6c4e4435e0dd0b2ab71df60d9c23886e507ed73692390"
BASELINE_PROTOCOL_SHA256 = "c54cd530884cffa22811ca2232cb2588eba3582deefc91d4d9675e4fca7fe30f"
INITIAL_POLICY_SHA256 = "7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d"
COHORT_ID = "traj4_backend_400hz_actionrate0p5_beta_off_seed42_v1"
DEFAULT_PROTOCOL = ROOT / "config/ppo_backend_400hz_beta_off_v1.json"
BASELINE_PROTOCOL = ROOT / "config/ppo_backend_400hz_comparison_v1.json"


def _load_frozen():
    import hashlib
    path = ROOT / "scripts/train_ppo_backend_400hz.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASELINE_WRAPPER_SHA256:
        raise ValueError("Immutable baseline400Hz wrapper SHA differs")
    spec = importlib.util.spec_from_file_location("_private_beta_off_400hz_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CORE = _load_frozen()
_ORIGINAL = {name: getattr(_CORE, name) for name in ("validate_protocol", "derive_inputs", "prepare_plan",
    "runtime_audit", "validate_runtime_fields", "before_learn", "finalize_contract", "complete", "materialize_source")}
require, sha256, write_json = _CORE.require, _CORE.sha256, _CORE.write_json
BASE_SHA256, NOMINAL_SHA256, PROFILE_SHA256 = _CORE.BASE_SHA256, _CORE.NOMINAL_SHA256, _CORE.PROFILE_SHA256
DEVICE_AUDIT_SHA256, MJLAB_PIN, INPUT_PINS = _CORE.DEVICE_AUDIT_SHA256, _CORE.MJLAB_PIN, _CORE.INPUT_PINS
REWARD_WEIGHTS = {"motion_global_anchor_pos": 2., "motion_global_anchor_ori": 2., "motion_body_pos": 1.,
    "motion_body_ori": 1., "motion_body_lin_vel": 1., "motion_body_ang_vel": 1., "action_rate_l2": -.5,
    "joint_limit": -10., "undesired_contacts": -.1, "drift_sideslip": 0., "drift_yaw_rate": 1.,
    "drift_speed": 1., "drift_grounded": .4, "drift_tilt": -.2, "height_vertical_acceleration": -.1}


def ablation_identity():
    return {"parameter": "reward_weights.sideslip", "baseline_weight": 2., "ablated_weight": 0.,
        "baseline_cohort_id": "traj4_backend_400hz_actionrate0p5_seed42_v1",
        "baseline_protocol_sha256": BASELINE_PROTOCOL_SHA256,
        "expected_initial_policy_state_sha256": INITIAL_POLICY_SHA256,
        "only_task_numeric_change": True, "wheel_beta_reward_added_or_removed": False}


def validate_protocol(protocol):
    require(sha256(BASELINE_PROTOCOL) == BASELINE_PROTOCOL_SHA256, "Immutable baseline400Hz protocol SHA differs")
    baseline = json.loads(BASELINE_PROTOCOL.read_text())
    _ORIGINAL["validate_protocol"](protocol)
    expected = copy.deepcopy(baseline)
    expected.update(cohort_id=COHORT_ID, beta_ablation=ablation_identity())
    require(protocol == expected, "Beta-off protocol differs beyond the declared identity/one reward ablation")
    return protocol


def derive_inputs(nominal, profile, protocol):
    contract, task = _ORIGINAL["derive_inputs"](nominal, profile, protocol)
    require(task["reward_weights"]["sideslip"] == 2., "Baseline sideslip reward is not 2")
    task["reward_weights"]["sideslip"] = 0.
    contract.update(task_profile=copy.deepcopy(task), trained_task_profile=copy.deepcopy(task),
        beta_ablation=ablation_identity(), sideslip_reward_weight=0.)
    return contract, task


def validate_task_profile(profile, protocol):
    require(sha256(ROOT / "config/nominal_actuator_contract.json") == NOMINAL_SHA256
            and sha256(ROOT / "config/car_drift_height_smooth_task.json") == PROFILE_SHA256,
            "Pinned original contract/profile changed")
    _, expected = derive_inputs(json.loads((ROOT / "config/nominal_actuator_contract.json").read_text()),
        json.loads((ROOT / "config/car_drift_height_smooth_task.json").read_text()), protocol)
    require(profile == expected, "Beta-off profile differs beyond the single sideslip2->0 reward change")
    return profile


def prepare_declared_inputs(protocol_path, output_dir, *,
        nominal_path=ROOT / "config/nominal_actuator_contract.json",
        profile_path=ROOT / "config/car_drift_height_smooth_task.json"):
    return _CORE.prepare_declared_inputs(protocol_path, output_dir, nominal_path=nominal_path, profile_path=profile_path)


def prepare_plan(argv):
    plan = _ORIGINAL["prepare_plan"](argv)
    plan["wrapper_sha256"] = sha256(__file__)
    return plan


def _core_runtime_fields(values):
    # The installed Isaac SDK lists the zero-weight term in active_terms,
    # just as the custom GPU objective retains its zero-valued map key.
    # Preserve the original 15-listed-term guard; never replace zero by epsilon.
    _ORIGINAL["validate_runtime_fields"](values)
    return values


def validate_runtime_fields(values):
    _core_runtime_fields(values)
    origin = values.get("training_backend")
    require(origin in ("physx", "mjlab") and values.get("reward_term_count") == 15
            and values.get("listed_reward_term_count") == 15
            and values.get("nonzero_reward_term_count") == 14
            and values.get("declared_reward_term_count") == 15
            and values.get("sideslip_reward_weight") == 0.
            and values.get("sideslip_term_listed") is True
            and values.get("effective_sideslip_reward_enabled") is False
            and values.get("declared_reward_weights") == REWARD_WEIGHTS,
            "Beta-off runtime weight/active-term identity differs")
    return values


def runtime_audit(env, args, metadata, plan):
    result = _ORIGINAL["runtime_audit"](env, args, metadata, plan)
    if args.backend == "physx":
        weights = {name: float(env.reward_manager.get_term_cfg(name).weight) for name in env.reward_manager.active_terms}
        require("drift_sideslip" in weights and weights["drift_sideslip"] == 0.
                and float(env.cfg.rewards.drift_sideslip.weight) == 0.,
                "PhysX must truly disable the sideslip reward, not retain epsilon")
        declared = weights
    else:
        weights = dict(env.objective.weights)
        declared = weights
    fields = result["fields"]
    fields.update(training_backend=args.backend, sideslip_reward_weight=0., declared_reward_weights=declared,
        listed_reward_term_count=len(weights), nonzero_reward_term_count=sum(float(value) != 0. for value in weights.values()),
        declared_reward_term_count=15, sideslip_term_listed="drift_sideslip" in weights,
        effective_sideslip_reward_enabled=weights.get("drift_sideslip") != 0.)
    validate_runtime_fields(fields)
    result["beta_ablation"] = ablation_identity()
    return result


def before_learn(env, args, manifest, runner, plan):
    require(manifest["initial_policy_state_sha256"] == INITIAL_POLICY_SHA256,
            "Initial policy does not match the completed beta-on seed42 network")
    validate_task_profile(json.loads(args.task_profile.read_text()), plan["protocol"])
    _ORIGINAL["before_learn"](env, args, manifest, runner, plan)
    manifest["beta_ablation"] = ablation_identity()
    manifest["cohort"]["beta_ablation"] = ablation_identity()
    manifest["cohort"]["initial_policy_matches_beta_on"] = True
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "cohort_startup_audit.json", manifest["cohort"])


def finalize_contract(args, manifest, runner, exported, probe, expected, plan):
    validate_task_profile(json.loads(args.task_profile.read_text()), plan["protocol"])
    _ORIGINAL["finalize_contract"](args, manifest, runner, exported, probe, expected, plan)
    contract = json.loads((args.output_dir / "final_policy_contract.json").read_text())
    require(contract["beta_ablation"] == ablation_identity() and contract["sideslip_reward_weight"] == 0.,
            "Export lost beta-off identity")
    validate_task_profile(contract["trained_task_profile"], plan["protocol"])
    require(contract["task_profile"] == contract["trained_task_profile"], "Exported training profiles differ")


def complete(args, manifest, report, plan):
    contract = json.loads((args.output_dir / "final_policy_contract.json").read_text())
    validate_task_profile(contract["trained_task_profile"], plan["protocol"])
    require(contract["sideslip_reward_weight"] == 0. and manifest["initial_policy_state_sha256"] == INITIAL_POLICY_SHA256,
            "Cannot complete a beta-on or differently initialized policy as beta-off")
    report.update(beta_ablation=ablation_identity(), sideslip_reward_weight=0., initial_policy_matches_beta_on=True)
    _ORIGINAL["complete"](args, manifest, report, plan)
    path = args.output_dir / "cohort_completion_audit.json"
    audit = json.loads(path.read_text())
    audit.update(beta_ablation=ablation_identity(), sideslip_reward_weight=0., initial_policy_matches_beta_on=True,
        declared_reward_term_count=15, listed_reward_term_count=15, nonzero_reward_term_count=14,
        sideslip_term_listed=True, effective_sideslip_reward_enabled=False)
    write_json(path, audit)


def materialize_source(text):
    # Keep baseline source/15-listed-term SDK guard byte-for-byte identical.
    return _ORIGINAL["materialize_source"](text)


_CORE.SOURCE_FILES += ("scripts/train_ppo_backend_400hz_beta_off.py", "scripts/ppo_backend_400hz_beta_contract.py",
    "config/ppo_backend_400hz_beta_off_v1.json", "config/ppo_backend_400hz_comparison_v1.json")
for _name in ("validate_protocol", "derive_inputs", "prepare_plan", "runtime_audit", "before_learn",
              "finalize_contract", "complete", "materialize_source"):
    setattr(_CORE, _name, globals()[_name])
_CORE.validate_runtime_fields = _core_runtime_fields


def run(argv):
    return _CORE.run(argv)


if __name__ == "__main__":
    run(sys.argv[1:])
