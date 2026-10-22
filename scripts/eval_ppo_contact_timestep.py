"""Frozen-actor physics/explicit-PD update-rate sensitivity, policy fixed50Hz.

The legacy evaluator and clock helper stay immutable. Only its over-restrictive
clock CLI conjunct and one final-export profile compatibility call are adapted
in memory. This is not a contact-solver-only factor or a training comparison.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import eval_ppo_contact_damping as base
import eval_ppo_leg_target_filter as devices


KIND = "fixed_ppo_nominal_physics_and_explicit_pd_timestep_diagnostic"
BASE_ADAPTER_SHA256 = "5353f849d7d4f263c735d2cb000c8000403bf81280b51634065c9b721bb3b277"
DEVICE_HELPER_SHA256 = "3ec3dbc1cbbc88ea1a1e95c6d4411e346dd8e8dd074317a9db442bb96ea8cbc3"
CLOCK_HELPER_SHA256 = "bcd89d6a24652d4501baf6744b46944d369fd224a9ab4727fc8b6073a20cf44e"
DECIMATION = {.005: 4, .0025: 8}
AUTHORIZED_GPU_INDICES = (1, 3, 4, 5)


def prepare_plan(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--physics-dt-s", type=float, choices=tuple(DECIMATION), required=True)
    parser.add_argument("--physical-gpu-index", type=int, choices=AUTHORIZED_GPU_INDICES, default=1)
    options, inherited = parser.parse_known_args(list(argv))
    base._option(argv, "--physics-dt-s"); base._option(argv, "--physical-gpu-index")
    requested_device = f"cuda:{options.physical_gpu_index}"
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") is None and base._option(inherited, "--device") == requested_device and
        base._option(inherited, "--kit_args") == devices.kit_args_for(options.physical_gpu_index), "Unmasked selected device/Kit ordinals differ")
    for word in inherited:
        base.require(word.split("=", 1)[0] not in ("--alpha", "--leg-kd", "--leg-kp", "--solver-velocity-iterations", "--solver-position-iterations"),
            "No LPF, gain or solver-count intervention may be combined with timestep sensitivity")
    base.require(base.sha256(base.__file__) == BASE_ADAPTER_SHA256 and base.sha256(devices.__file__) == DEVICE_HELPER_SHA256,
        "Frozen provenance/device helper differs")
    # The old provenance preparer blocks physics overrides. Strip our own
    # declared argument for its checks, then forward only that exact clock.
    plan = base.prepare_plan(["--leg-kd", "10", *inherited]); plan.pop("leg_kd")
    helper = plan["legacy_evaluator"].parent.parent/"training/tron1_physics_clock.py"
    base.require(helper.is_file() and base.sha256(helper) == CLOCK_HELPER_SHA256, "Pinned numerical-clock helper differs")
    plan.update(physics_dt_s=options.physics_dt_s, decimation=DECIMATION[options.physics_dt_s],
        physical_gpu_index=options.physical_gpu_index, clock_helper=helper)
    plan["inherited_argv"] += ["--physics-dt-s", str(options.physics_dt_s)]
    audit = plan["audit"]
    for key in ("deployment_leg_kd", "trained_leg_kd"): audit.pop(key)
    audit.update(kind=KIND, physics_dt_s=options.physics_dt_s, evaluation_physics_dt_s=options.physics_dt_s,
        trained_physics_dt_s=.005, trained_decimation=4, evaluation_decimation=plan["decimation"],
        policy_dt_s=.02, unchanged_policy_fps=50., unchanged_leg_kp=500., unchanged_leg_kd=10.,
        low_level_explicit_pd_update_hz=1./options.physics_dt_s,
        configured_solver_position_iterations=8, configured_solver_velocity_iterations=4,
        physical_gpu_index_requested=options.physical_gpu_index, requested_device=requested_device,
        wrapper_sha256=base.sha256(__file__), provenance_adapter_sha256=BASE_ADAPTER_SHA256,
        device_helper_sha256=DEVICE_HELPER_SHA256, physics_clock_helper_sha256=CLOCK_HELPER_SHA256,
        only_changed_control_or_physics_fields=["physics_dt", "decimation", "explicit_PD_update_interval", "net_contact_sensor_update_period", "render_interval"],
        actor_observation_layout_modified=False, requested_action_history_modified=False, wheel_action_mapping_modified=False,
        leg_target_filter_installed=False, additional_torque_reward=False, additional_training_updates=0,
        interpretation="Numerical physics timestep AND explicit DC-motor PD update-rate sensitivity; same frozen actor and50Hz policy, not a contact-solver-only or training comparison")
    return plan


def apply_config(cfg, evaluation_profile, plan):
    props, legs = cfg.scene.robot.spawn.articulation_props, cfg.scene.robot.actuators["legs"]
    base.require((props.solver_position_iteration_count, props.solver_velocity_iteration_count) == (8, 4) and
        (legs.stiffness, legs.damping, cfg.sim.dt, cfg.decimation) == (500., 10., .005, 4) and
        evaluation_profile["contact_friction"] == .7, "Original solver counts, PD, trained clock or nominal material differs")
    # No writes here: immediately following legacy configure_physics_clock
    # applies its pinned dt/decimation/contact-sensor/render clock together.


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def refine_export_contract(contract, dt, *, trained_contract, clock_helper, plan):
    """Validate only approved material difference; never mutate runtime/profile."""
    base.require(dt == plan["physics_dt_s"] and dt in DECIMATION, "Export clock differs from declared diagnostic")
    trained_profile, evaluated_profile = trained_contract.get("task_profile"), contract.get("task_profile")
    base.require(isinstance(trained_profile, dict) and isinstance(evaluated_profile, dict) and
        contract.get("trained_task_profile") == trained_profile and trained_profile.get("contact_friction") == 1. and
        evaluated_profile.get("contact_friction") == .7, "Expected original trained1.0 and approved evaluated0.7 profiles")
    validation_profile = copy.deepcopy(evaluated_profile); validation_profile["contact_friction"] = 1.
    base.require(validation_profile == trained_profile, "Export profile differs beyond approved material coefficient")
    override = contract.get("evaluation_contact_friction_override", {})
    base.require(override.get("trained_contact_friction") == 1. and override.get("evaluation_contact_friction") == .7,
        "Actual material-override evidence is missing")
    view = copy.deepcopy(contract); view["task_profile"] = copy.deepcopy(trained_profile)
    # Calls the unchanged original strict helper; its clock/actor assertions
    # remain active. Only this deep-copy integration-validation view uses the
    # original profile, and the output restores the actual evaluated profile.
    out = clock_helper(view, dt, trained_contract=trained_contract)
    out["task_profile"] = copy.deepcopy(evaluated_profile)
    out["trained_task_profile"] = copy.deepcopy(trained_profile)
    out["evaluation_contact_friction_override"] = copy.deepcopy(override)
    plan["audit"]["final_export_profile_bridge"] = {"scope": "deep_copy_profile_validation_only_no_runtime_changes",
        "only_approved_profile_difference": {"contact_friction": {"trained": 1., "evaluated": .7}},
        "trained_task_profile_canonical_sha256": _canonical_sha(trained_profile),
        "evaluation_task_profile_canonical_sha256": _canonical_sha(evaluated_profile),
        "trained_profile_modified": False, "runtime_material_modified_by_bridge": False,
        "original_helper_all_clock_interface_checks_retained": True}
    out["contact_timestep_diagnostic"] = copy.deepcopy(plan["audit"])
    return out


def decorate_contract(contract, plan):
    base.require((contract["leg_kp"], contract["leg_kd"]) == (500., 10.) and contract["actor_dim"] == 51 and
        contract["action_dim"] == 8, "Original controller/actor layout differs")
    contract["contact_timestep_diagnostic"] = copy.deepcopy(plan["audit"])


def materialize_source(source, legacy_root):
    generated = base.materialize_source(source, legacy_root)
    guard = "            or args.physics_dt_s is not None or not args.record_physics"
    base.require(generated.count(guard) == 1, "Exact main diagnostic clock guard differs")
    generated = generated.replace(guard, "            or not args.record_physics", 1)
    final = '            contract = apply_physics_clock_to_contract(\n                contract, args.physics_dt_s, trained_contract=trained_manifest["contract"])'
    replacement = '            contract = _timestep_refine_export(\n                contract, args.physics_dt_s, trained_contract=trained_manifest["contract"],\n                clock_helper=apply_physics_clock_to_contract, plan=_contact_plan)'
    base.require(generated.count(final) == 1, "Exact final-export clock call differs")
    generated = generated.replace(final, replacement, 1)
    report = '            "contact_stability_diagnostic": copy.deepcopy(_contact_plan["audit"]),'
    base.require(generated.count(report) == 1, "Exact provenance report anchor differs")
    generated = generated.replace(report, '            "contact_timestep_diagnostic": copy.deepcopy(_contact_plan["audit"]),', 1)
    compile(generated, "<pinned_contact_timestep_evaluator>", "exec")
    return generated


def run(argv):
    plan = prepare_plan(argv)
    generated = materialize_source(plan["legacy_evaluator"].read_text(), plan["legacy_evaluator"].parent.parent)
    plan["audit"]["generated_evaluator_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    namespace = {"__name__": "_pinned_contact_timestep_evaluator", "__file__": str(Path(__file__).resolve()),
        "_contact_plan": plan, "_contact_apply_config": apply_config, "_contact_decorate_contract": decorate_contract,
        "_timestep_refine_export": refine_export_contract}
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<pinned_contact_timestep_evaluator>", "exec"), namespace); namespace["main"]()
    finally: sys.argv = previous


if __name__ == "__main__": run(sys.argv[1:])
