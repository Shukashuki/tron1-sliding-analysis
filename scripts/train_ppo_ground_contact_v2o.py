"""O: K force reward plus bounded preceding-stance clipped-target change cost.

Other14 rewards, beta+2/std.12, grounded+.4, DR, physical clocks, actuators and
from-scratch seed42 budget remain fixed. Nominal dual-engine screens decide.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import ground_contact_reward_v2o as definition
from ground_contact_beta_contract import verify_beta_source, verify_beta_runtime

ROOT = Path(__file__).resolve().parents[1]
VERSION = "ground_contact_v2o_any_wheel"
EXPECTED_INITIAL_POLICY_SHA256 = "7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def source_ledger(base):
    definition.verify_base()
    paths = [Path(__file__), ROOT/"scripts/ground_contact_reward_v2.py",
             ROOT/"scripts/ground_contact_reward_v2b.py", ROOT/"scripts/ground_contact_reward_v2c.py",
             ROOT/"scripts/ground_contact_reward_v2d.py",
             ROOT/"scripts/ground_contact_reward_v2f.py",
             ROOT/"scripts/ground_contact_beta_contract.py",
             ROOT/"scripts/ground_contact_reward_v2g.py",
             ROOT/"scripts/ground_contact_reward_v2i.py",
             ROOT/"scripts/ground_contact_reward_v2k.py",
             ROOT/"scripts/ground_contact_reward_v2o.py",
             ROOT/"scripts/ppo_ground_contact_v2k.py",
             ROOT/"scripts/ppo_ground_contact_v2o.py"]
    paths += [base/"scripts"/name for name in (
        "train_ppo_backend_400hz.py", "train_ppo_backend.py", "physx_mjlab_recipe.py",
        "height_smoothness.py", "ppo_leg_torque_smoothness.py") if (base/"scripts"/name).is_file()]
    return {str(path.resolve()): sha256(path) for path in paths}


def load_base(snapshot):
    path = Path(snapshot).resolve()/"scripts/train_ppo_backend_400hz.py"
    require(path.is_file(), "Frozen baseline snapshot is missing")
    require(sha256(path) == "5234070be219e59c61e6c4e4435e0dd0b2ab71df60d9c23886e507ed73692390",
            "Frozen400Hz baseline wrapper differs")
    spec = importlib.util.spec_from_file_location("_contact_v2_base_cohort", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_source(cohort):
    source = cohort.materialize_source((cohort.ROOT/"scripts/train_ppo_backend.py").read_text())
    anchor = "            accumulator = install_drift_reward_sampling(env, task)\n"
    require(source.count(anchor) == 1, "Require one counted environment-install anchor")
    source = source.replace(anchor, anchor+
        "            if accumulator is not None:\n"
        "                raise ValueError('Contact V2 cannot stack another substep owner')\n"
        "            accumulator = _contact_v2_install(env, _cohort_plan)\n", 1)
    compile(source, "<ground_contact_v2o_400hz_trainer>", "exec")
    return source


def _run(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    own, inherited = parser.parse_known_args(argv)
    snapshot = own.base_snapshot.resolve()
    previous_path = list(sys.path)
    sys.path.insert(1, str(snapshot/"scripts"))
    cohort = load_base(snapshot)
    original_prepare = cohort.prepare_declared_inputs
    protocol_arg = inherited.index("--cohort-protocol")+1
    protocol_path = Path(inherited[protocol_arg]).resolve()
    protocol = json.loads(protocol_path.read_text())
    require(isinstance(protocol.get("ground_contact_reward_v2"), dict), "Explicit contact settings are required")
    settings = definition.validate_settings(protocol["ground_contact_reward_v2"])
    require(protocol.get("reward_definition_version") == VERSION,
            "Use a declared new contact-reward protocol, not an old cohort label")
    require(protocol["cohort_id"] == "tron1_ground_contact_v2o_any_wheel_400hz_v1", "Wrong V2 cohort identity")
    source_options = [i for i, value in enumerate(inherited) if value == "--tron1-source"]
    require(len(source_options) == 1 and source_options[0]+1 < len(inherited),
            "Require one explicit existing --tron1-source")
    beta_source = verify_beta_source(Path(inherited[source_options[0]+1])/"training/tron1_drift.py")
    own_ledger = source_ledger(snapshot)
    own_ledger[beta_source["path"]] = beta_source["sha256"]

    def prepared(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        task_path, contract_path = result["task_profile"], result["contract"]
        task = json.loads(task_path.read_text())
        task["ground_contact_reward_v2"] = copy.deepcopy(settings)
        task["description"] = "O retains K contact arithmetic; adds bounded target-change cost on the least-changing previously loaded leg. Sigma0.1rad perjoint RMS; actual clipped target changes; no wheel-torque cost. Other14 rewards, beta,400Hz DR and seed42 budget unchanged."
        contract = json.loads(contract_path.read_text())
        contract["task_profile"] = copy.deepcopy(task)
        contract["trained_task_profile"] = copy.deepcopy(task)
        contract["reward_definition_version"] = VERSION
        contract["ground_contact_reward_v2"] = copy.deepcopy(settings)
        write_json(task_path, task)
        write_json(contract_path, contract)
        result["derived_input_sha256"] = {"contract": sha256(contract_path), "task_profile": sha256(task_path)}
        derivation = task_path.parent/"derivation.json"
        value = json.loads(derivation.read_text())
        value.update(derived_inputs_sha256=result["derived_input_sha256"],
                     reward_definition_version=VERSION, contact_source_sha256=own_ledger)
        write_json(derivation, value)
        return result
    cohort.prepare_declared_inputs = prepared
    plan = cohort.prepare_plan(inherited)
    plan["contact_settings"] = settings
    plan["contact_source_sha256"] = own_ledger
    generated = materialize_source(cohort)
    plan["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()

    def before_start(args, current):
        require(args.backend == "physx", "This first V2 experiment is PhysX-trained; do not mislabel MJWarp")
        cohort.before_start(args, current)

    def configure(cfg, args, current, recipe):
        cohort.configure_physx(cfg, args, current, recipe)
        import ppo_ground_contact_v2o as contact
        contact.configure_sensors(cfg.scene)
        cfg.rewards.drift_grounded = contact.reward_term(.4)

    def install(env, current):
        import ppo_ground_contact_v2o as contact
        owner = contact.install(env, current["contact_settings"])
        current["contact_owner"] = owner
        return owner

    def before_learn(env, args, manifest, runner, current):
        cohort.before_learn(env, args, manifest, runner, current)
        require(manifest.get("initial_policy_state_sha256") == EXPECTED_INITIAL_POLICY_SHA256,
                "O must start from the identical from-scratch A/B/C/D seed42 network")
        owner = current["contact_owner"]
        audit = owner.audit(health=True)
        require(audit["support_gap_coefficient_current"] == 1.
                and audit["stance_command_cost_added"] is True
                and audit["stance_command_delta_std_rad"] == .1
                and audit["stance_command_penalty_weight"] == 1.
                and audit["immutable_v2k_definition_sha256"] == definition.BASE_K_DEFINITION_SHA256
                and audit["immutable_v2k_collector_sha256"] == "3fb2ce270ed5783101bc558cc615e311dd9f586c3096a28184e9174e3d595f2d"
                and audit["device_finite_stance_command_health"]
                and audit["command_unique_reward_windows"] == audit["unique_reward_windows"]
                and audit["preceding_force_uses_previous_completed_native_sample"]
                and audit["reset_clears_preceding_force_and_previous_command"]
                and audit["command_history_updated_before_underlying_step"]
                and audit["force_slew_cost_added"] is False,
                "O must retain K force arithmetic and use preceding-load-conditioned command cost")
        require(audit["device_finite_normal_force_health"] and audit["device_finite_reward_health"]
                and audit["steps_started"] == audit["steps_completed"]
                and audit["physics_samples"] == 8*audit["steps_completed"]
                and audit["unique_reward_windows"] == audit["steps_completed"],
                "Actual loaded sensor/clock/reset smoke did not pass")
        verify_beta_runtime(audit, expected_envs=args.num_envs)
        require(audit.get("beta_quality_source_path") == beta_source["path"],
                "Collector used another beta source")
        # At least one positive normal sample is required to verify sign/units.
        require(sum(audit["force_bearing_samples_L_R_any_both"][:2]) > 0,
                "Setup smoke never observed actual loaded wheel/floor contact")
        manifest["reward_definition_version"] = VERSION
        manifest["ground_contact_v2o"] = {"settings": settings, "runtime": audit,
            "source_sha256": own_ledger, "other14_reward_terms_changed": False,
            "single_wheel_support_allowed": True, "from_scratch": True}
        manifest["cohort"]["reward_definition_version"] = VERSION
        write_json(args.output_dir/"manifest.json", manifest)
        write_json(args.output_dir/"cohort_startup_audit.json", manifest["cohort"])
        write_json(args.output_dir/"ground_contact_v2o_startup_audit.json", manifest["ground_contact_v2o"])

    def complete(args, manifest, report, current):
        require(manifest.get("initial_policy_state_sha256") == EXPECTED_INITIAL_POLICY_SHA256,
                "O initial network identity differs from A/B/C/D")
        audit = current["contact_owner"].audit(health=True)
        require(audit["support_gap_coefficient_current"] == 1.
                and audit["stance_command_cost_added"] is True
                and audit["stance_command_delta_std_rad"] == .1
                and audit["stance_command_penalty_weight"] == 1.
                and audit["immutable_v2k_definition_sha256"] == definition.BASE_K_DEFINITION_SHA256
                and audit["immutable_v2k_collector_sha256"] == "3fb2ce270ed5783101bc558cc615e311dd9f586c3096a28184e9174e3d595f2d"
                and audit["device_finite_stance_command_health"]
                and audit["command_unique_reward_windows"] == audit["unique_reward_windows"]
                and audit["preceding_force_uses_previous_completed_native_sample"]
                and audit["reset_clears_preceding_force_and_previous_command"]
                and audit["command_history_updated_before_underlying_step"]
                and audit["force_slew_cost_added"] is False,
                "O must retain K force arithmetic and use preceding-load-conditioned command cost")
        require(audit["device_finite_normal_force_health"] and audit["device_finite_reward_health"]
                and audit["steps_started"] == audit["steps_completed"]
                and audit["physics_samples"] == 8*audit["steps_completed"]
                and audit["unique_reward_windows"] == audit["steps_completed"], "Final contact collector audit failed")
        verify_beta_runtime(audit, expected_envs=args.num_envs)
        require(audit.get("beta_quality_source_path") == beta_source["path"],
                "Collector beta source changed during training")
        require(all(sha256(path) == value for path, value in own_ledger.items()), "New contact sources changed during training")
        manifest["ground_contact_v2o"]["runtime_final"] = audit
        write_json(args.output_dir/"manifest.json", manifest)
        cohort.complete(args, manifest, report, current)
        smoke = current["run_kind"] == "smoke"
        result = {"status": "completed_verified", "reward_definition_version": VERSION,
            "cohort_id": protocol["cohort_id"], "protocol_sha256": sha256(protocol_path),
            "training_environment_steps": report["training_environment_steps"],
            "source_sha256": {**own_ledger, **{str((snapshot/name).resolve()): value
                                  for name, value in current["source_hashes"].items()}},
            "checkpoint_sha256": report["checkpoint_sha256"],
            "exported_policy_sha256": report["exported_policy_sha256"], "normalizer_in_actor": True,
            "manifest_sha256": sha256(args.output_dir/"manifest.json"),
            "policy_contract_sha256": sha256(args.output_dir/"policy_contract.json"),
            "initial_policy_state_sha256": manifest["initial_policy_state_sha256"],
            "from_scratch": True, "smoke_only": smoke, "trained_policy_claimed": not smoke,
            "contact_runtime_audit": audit, "settings": settings,
            "other14_reward_terms_changed": False, "new_reward_count": 15}
        write_json(args.output_dir/"ground_contact_v2o_training_audit.json", result)

    namespace = {"__name__": "_ground_contact_v2o_trainer", "__file__": str(plan["base"]),
        "_cohort_plan": plan, "_cohort_before_start": before_start,
        "_cohort_launch_app": cohort.launch_app, "_cohort_set_torch_device": cohort.set_torch_device,
        "_cohort_configure_physx": configure, "_contact_v2_install": install,
        "_cohort_before_learn": before_learn, "_cohort_finalize_contract": cohort.finalize_contract,
        "_cohort_complete": complete}
    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<ground_contact_v2o_400hz_trainer>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


def run(argv):
    """Restore import/CLI state even when preparation fails before app launch."""
    path, arguments = list(sys.path), sys.argv
    try:
        return _run(argv)
    finally:
        sys.path[:] = path
        sys.argv = arguments


if __name__ == "__main__":
    run(sys.argv[1:])
