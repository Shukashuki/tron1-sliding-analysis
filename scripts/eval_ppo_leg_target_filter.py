"""Frozen-actor leg-target low-pass diagnostic; original observations preserved.

Only six processed leg position targets are filtered. The original actor,
requested action/history and wheel commands are unchanged. Final telemetry is
written after rollout, not copied from zero-count installation-time metadata.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

BASE_ADAPTER_SHA256 = "5353f849d7d4f263c735d2cb000c8000403bf81280b51634065c9b721bb3b277"
BASE_ADAPTER_PATH = Path(__file__).with_name("eval_ppo_contact_damping.py")
FILTER_MODULE_SHA256 = "a2951a577ab6be3e8864a696f81d321c937c6fc00d1716125b66c9a7e2b491d9"
KIND = "fixed_ppo_nominal_leg_target_filter_diagnostic"
CONTROL_FILE = "leg_target_filter_controls.npz"
PHYSICS_FILE = "leg_target_filter_physics.npz"
RUNTIME_FILE = "leg_target_filter_runtime_audit.json"
AUTHORIZED_GPU_INDICES = (1, 3, 4, 5)
CLOCK_TIME_TOLERANCE_S = 2e-6


def kit_args_for(index):
    base.require(index in AUTHORIZED_GPU_INDICES, "Unauthorized physical GPU index")
    return (f"--/renderer/activeGpu={index} --/renderer/multiGpu/enabled=false "
            f"--/physics/cudaDevice={index} --/physics/sceneMultiGPUMode=0")


def load_base_adapter(path=BASE_ADAPTER_PATH):
    path = Path(path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != BASE_ADAPTER_SHA256:
        raise ValueError("Frozen provenance adapter SHA mismatch")
    spec = importlib.util.spec_from_file_location("_leg_filter_provenance", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


base = load_base_adapter()


def load_filter_module():
    path = Path(__file__).with_name("ppo_leg_target_filter.py")
    base.require(path.is_file() and base.sha256(path) == FILTER_MODULE_SHA256, "Frozen target-filter source SHA mismatch")
    spec = importlib.util.spec_from_file_location("_diagnostic_leg_target_filter", path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def prepare_plan(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--alpha", type=float, choices=(1., .5), required=True)
    parser.add_argument("--physical-gpu-index", type=int, choices=AUTHORIZED_GPU_INDICES, default=1)
    options, inherited = parser.parse_known_args(list(argv))
    requested_device = f"cuda:{options.physical_gpu_index}"
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES") is None, "Require an unmasked explicit physical GPU")
    base.require(base._option(inherited, "--device") == requested_device, "Explicit device differs from selected physical GPU")
    base.require(base._option(inherited, "--kit_args") == kit_args_for(options.physical_gpu_index), "Renderer/PhysX Kit ordinals differ from selected physical GPU")
    for word in inherited:
        base.require(word.split("=", 1)[0] not in ("--leg-kd", "--leg-kp", "--solver-velocity-iterations", "--solver-position-iterations"), "Combined deployment overrides are blocked")
    plan = base.prepare_plan(["--leg-kd", "10", *inherited]); plan.pop("leg_kd")
    path = Path(__file__).with_name("ppo_leg_target_filter.py")
    base.require(path.is_file() and base.sha256(path) == FILTER_MODULE_SHA256, "Frozen target-filter source SHA mismatch")
    plan["alpha"] = options.alpha
    plan["physical_gpu_index"] = options.physical_gpu_index
    plan["requested_device"] = requested_device
    audit = plan["audit"]
    for key in ("deployment_leg_kd", "trained_leg_kd"):
        audit.pop(key)
    audit.update(kind=KIND, alpha=options.alpha, unchanged_leg_kd=10., configured_solver_position_iterations=8,
        physical_gpu_index_requested=options.physical_gpu_index, requested_device=requested_device,
        configured_solver_velocity_iterations=4, solver_counts_are_native_readback=False,
        only_changed_control_field="six_leg_processed_position_target_lowpass" if options.alpha < 1 else None,
        filter_module_sha256=FILTER_MODULE_SHA256, wrapper_sha256=base.sha256(__file__),
        provenance_adapter_sha256=BASE_ADAPTER_SHA256, actor_observation_layout_modified=False,
        requested_action_history_modified=False, wheel_requests_modified=False,
        processed_leg_position_targets_only=True, additional_physics_frames=0, additional_actor_calls=0,
        filtering_applied_during_training=False,
        within_case_wheel_request_invariance_not_cross_case_action_equality=True,
        previous_action_meaning="original requested clipped eight-action vector, not filtered target",
        filter_definition="per decision q_filtered=alpha*q_requested+(1-alpha)*q_filtered_previous; first history after reset uses measured leg q",
        actor_export_includes_filter=False,
        deployment_requires_same_external_filter=options.alpha < 1,
        interpretation="One frozen -0.5 PPO actor at nominal mu=.7; six leg position targets only; not retraining or a contact-stability certification.")
    return plan


def apply_config(cfg, evaluation_profile, plan):
    props = cfg.scene.robot.spawn.articulation_props; legs = cfg.scene.robot.actuators["legs"]
    base.require((props.solver_position_iteration_count, props.solver_velocity_iteration_count) == (8, 4), "Original solver cfg differs")
    base.require((legs.stiffness, legs.damping, cfg.sim.dt, cfg.decimation) == (500., 10., .005, 4), "Original PD/clock differs")
    base.require(evaluation_profile["contact_friction"] == .7, "Require original nominal material")
    # Deliberately no cfg/profile writes: the intervention is inside the action
    # term's processed six-leg target buffer, not the policy observation/actions.


def install_filter(env, plan):
    module = load_filter_module()
    return module.install_leg_target_filter(env, alpha=plan["alpha"])


def clock_diagnostics(control, physics):
    """Preserve measured SDK float times; integer clocks remain exact gates.

    Installed SimulationContext.current_time exposes callback-accumulated
    _current_time, not counter*Python cfg.dt. A float32 .005 callback step alone
    would accumulate ~1.56e-7 s residual in1395steps. This is source rationale,
    not a claim that this exact drift was measured in an earlier unsaved run.
    """
    import numpy as np
    components, values, arrays = {}, {}, {}
    specifications = (("control_time", control, "time_s", 349, "f"), ("physics_time", physics, "time_s", 1396, "f"),
        ("control_counter", control, "sim_step_counter", 349, "iu"), ("physics_counter", physics, "sim_step_counter", 1396, "iu"),
        ("control_native_index", control, "native_step_index", 349, "iu"), ("physics_native_index", physics, "native_step_index", 1396, "iu"))
    for label, source, name, length, kind in specifications:
        value = np.asarray(source.get(name, [])); arrays[label] = value
        passed = value.shape == (length,) and value.dtype.kind in kind and bool(np.isfinite(value).all())
        components[label+"_shape_type_finite"] = {"passed": bool(passed), "shape": list(value.shape), "dtype": str(value.dtype), "expected_samples": length}
        values[label] = {"available": bool(passed), "first": float(value[0]) if passed else None, "last": float(value[-1]) if passed else None}
    def usable(label):
        return values[label]["available"]
    for label, step in (("control_time", .02), ("physics_time", .005)):
        if not usable(label):
            continue
        value = arrays[label]; relative = value-value[0]; expected = np.arange(len(value))*step; error = relative-expected
        maximum_index = int(np.argmax(np.abs(error))); maximum = float(np.abs(error[maximum_index]))
        differences = np.diff(value)
        values[label].update(nominal_step_s=step, observed_step_range_s=[float(differences.min()), float(differences.max())],
            maximum_absolute_cumulative_drift_s=maximum, maximum_drift_sample_index=maximum_index,
            final_signed_cumulative_drift_s=float(error[-1]),
            max_abs_residual_against_float32_step_hypothesis_s=float(np.max(np.abs(relative-np.arange(len(value))*float(np.float32(step))))))
        components[label+"_relative_grid"] = {"passed": maximum <= CLOCK_TIME_TOLERANCE_S,
            "maximum_absolute_error_s": maximum, "sample_index": maximum_index, "tolerance_s": CLOCK_TIME_TOLERANCE_S}
    if usable("control_time") and usable("physics_time"):
        difference = float(arrays["physics_time"][0]-arrays["control_time"][0])
        components["control_physics_start_time_match"] = {"passed": abs(difference) <= CLOCK_TIME_TOLERANCE_S, "difference_s": difference, "tolerance_s": CLOCK_TIME_TOLERANCE_S}
    for label, origin_label, expected in (("control_counter", "control_counter", np.arange(349)*4),
        ("physics_counter", "control_counter", np.arange(1, 1397)),
        ("control_native_index", "control_native_index", np.arange(349)*4),
        ("physics_native_index", "control_native_index", np.arange(1396))):
        if not (usable(label) and usable(origin_label)):
            continue
        relative = arrays[label]-arrays[origin_label][0]; bad = np.flatnonzero(relative != expected)
        components[label+"_exact_progression"] = {"passed": len(bad) == 0, "integer_comparison_exact": True,
            "first_bad_sample_index": int(bad[0]) if len(bad) else None,
            "actual_relative_counter": int(relative[bad[0]]) if len(bad) else None,
            "expected_relative_counter": int(expected[bad[0]]) if len(bad) else None}
    return {"time_tolerance_s": CLOCK_TIME_TOLERANCE_S, "integer_counters_and_native_indices_exact": True,
        "all_components_passed": all(row["passed"] for row in components.values()), "components": components, "measured": values,
        "sdk_time_source": "SimulationContext.current_time returns callback-accumulated _current_time; installed source audit supplied by parent",
        "precision_rationale": "Use the same2e-6s bound as the pinned handler's native per-window clock checks; no rounding, resampling, alignment or replacement of recorded timestamps",
        "float32_step_hypothesis_is_runtime_precision_proof": False,
        "previous_unsaved_v2_drift_measured": False}


def verify_telemetry(control, physics, runtime, alpha, *, initial_joint_position=None, requested_device="cuda:1"):
    """Pure parent-verifiable target/clock/history proof, no simulator access."""
    import numpy as np
    required = {"control_steps": 349, "physics_apply_calls": 1396, "samples_last_control": 4, "failures": 0,
        "requested_action_history_unchanged": True, "wheel_requests_unchanged": True,
        "applied_targets_match_filtered": True, "device_health_finite": True, "closed": True,
        "physics_dt_s": .005, "policy_dt_s": .02, "decimation": 4, "module_sha256": FILTER_MODULE_SHA256}
    for key, value in required.items():
        base.require(runtime.get(key) == value, "Final filter runtime proof differs: "+key)
    base.require(requested_device in tuple(f"cuda:{index}" for index in AUTHORIZED_GPU_INDICES), "Unauthorized requested telemetry device")
    source = runtime.get("source_api_provenance", {})
    base.require(runtime.get("cpu_fixture_only") is False and runtime.get("num_envs") == 1 and runtime.get("device") == requested_device
        and runtime.get("device_health_checked") is True and source.get("cpu_fixture_only") is False
        and source.get("source_pins_checked") is True, "Actual GPU/native API source audit is missing")
    base.require(runtime.get("additional_physics_steps") == runtime.get("additional_reset_calls") == runtime.get("additional_training_updates") == 0
        and runtime.get("filter_update_rate_hz") == 50 and runtime.get("apply_rate_hz") == 200
        and runtime.get("wheel_outputs_filtered") is False, "Unexpected steps, reset calls, training or wheel filtering")
    if "alpha" in runtime:
        base.require(runtime["alpha"] == alpha, "Actual handler alpha differs")
    if alpha == 1.:
        base.require(runtime.get("alpha_one_exact_identity") is True, "Alpha1 exact-identity runtime proof missing")
    for arrays in (control, physics):
        base.require(all(np.isfinite(value).all() for value in arrays.values() if np.asarray(value).dtype.kind in "fiu"), "Nonfinite filter telemetry")
    requested = np.asarray(control["requested_actions_clipped"])
    target, filtered = np.asarray(control["requested_targets_rad"]), np.asarray(control["filtered_targets_rad"])
    base.require(requested.shape == (349, 8) and target.shape == filtered.shape == (349, 6)
        and np.max(np.abs(requested)) <= 1. and np.asarray(control["last_action_unchanged"]).shape == (349,)
        and np.asarray(control["last_action_unchanged"]).dtype.kind == "b" and np.asarray(control["last_action_unchanged"]).all(), "Invalid requested action/target/history telemetry")
    initialized = np.asarray(control["initialized_from_measured"])
    base.require(initialized.shape == (349,) and initialized.dtype.kind == "b" and initialized[0] and not initialized[1:].any(), "Require one original first-episode measured initialization, not reset-spliced filter history")
    clocks = clock_diagnostics(control, physics)
    for name, component in clocks["components"].items():
        base.require(component["passed"], "Original target clock component failed: "+name+" "+json.dumps(component, allow_nan=False))
    time, physical_time = np.asarray(control["time_s"]), np.asarray(physics["time_s"])
    expected_control, expected_substep = np.repeat(np.arange(349), 4), np.tile(np.arange(1, 5), 349)
    base.require(np.array_equal(physics["control_index"], expected_control) and np.array_equal(physics["substep_index"], expected_substep)
        and np.asarray(physics["applied_targets_rad"]).shape == np.asarray(physics["filtered_targets_rad"]).shape == (1396, 6)
        and np.array_equal(physics["filtered_targets_rad"], filtered[expected_control])
        and np.array_equal(physics["applied_targets_rad"], filtered[expected_control]), "Every one of four native applied targets must match its filtered control target")
    if alpha == 1.:
        base.require(np.array_equal(filtered, target), "Alpha1 must be exact requested-target identity, not a numerically approximated filter")
    else:
        base.require(np.allclose(filtered[1:], alpha*target[1:]+(1-alpha)*filtered[:-1], rtol=0, atol=5e-7), "Recorded target recurrence differs from alpha=.5")
        if initial_joint_position is not None:
            base.require(np.allclose(filtered[0], alpha*target[0]+(1-alpha)*np.asarray(initial_joint_position), rtol=0, atol=5e-7), "First filtered history was not measured initial leg q")
    return {"control_steps": 349, "physics_apply_calls": 1396, "all_four_substep_targets_verified": True,
        "requested_action_history_unchanged": True, "wheel_requests_unchanged": True,
        "alpha_one_exact_identity": alpha == 1., "control_native_time_origin_s": float(time[0]),
        "physics_pre_step_native_time_origin_s": float(physical_time[0]),
        "physics_telemetry_is_state_sample": False,
        "clock_diagnostics": clocks,
        "physics_telemetry_timing": "Immediately after original target setter, before physics advance; sim counter already incremented"}


def finalize_filter(env, handler, plan):
    import numpy as np
    base.require("final_runtime_evidence" not in plan["audit"], "Final runtime proof may be captured only once")
    control, physics = handler.control_arrays(), handler.physics_arrays()
    output = plan["paths"]["output_dir"]
    for name in (CONTROL_FILE, PHYSICS_FILE, RUNTIME_FILE):
        base.require(not (output/name).exists(), "Preserve prior filter evidence")
    np.savez_compressed(output/CONTROL_FILE, **control); np.savez_compressed(output/PHYSICS_FILE, **physics)
    final = {"schema_version": 1, "kind": KIND, "alpha": plan["alpha"], "finalized_after_rollout": True,
        "verification_status": "not_verified_yet", "filter_handler_audit": None, "telemetry_verification": None,
        "clock_diagnostics": clock_diagnostics(control, physics),
        "control_npz_sha256": base.sha256(output/CONTROL_FILE), "physics_npz_sha256": base.sha256(output/PHYSICS_FILE)}
    (output/RUNTIME_FILE).write_text(json.dumps(final, indent=2, allow_nan=False)+"\n")
    try:
        handler.close()
        runtime = handler.audit(check_device_health=True)
        final["filter_handler_audit"] = runtime
        (output/RUNTIME_FILE).write_text(json.dumps(final, indent=2, allow_nan=False)+"\n")
        proof = verify_telemetry(control, physics, runtime, plan["alpha"], requested_device=plan["requested_device"])
    except BaseException as error:
        final.update(verification_status="failed", verification_error={"type": type(error).__name__, "message": str(error)})
        (output/RUNTIME_FILE).write_text(json.dumps(final, indent=2, allow_nan=False)+"\n")
        raise
    final.update(verification_status="verified", telemetry_verification=proof)
    (output/RUNTIME_FILE).write_text(json.dumps(final, indent=2, allow_nan=False)+"\n")
    plan["audit"]["final_runtime_evidence"] = {"finalized_after_rollout": True, "control_steps": 349, "physics_apply_calls": 1396,
        "verification_status": "verified",
        "runtime_audit_file": RUNTIME_FILE, "runtime_audit_sha256": base.sha256(output/RUNTIME_FILE),
        "control_file": CONTROL_FILE, "control_npz_sha256": final["control_npz_sha256"],
        "physics_file": PHYSICS_FILE, "physics_npz_sha256": final["physics_npz_sha256"]}
    return final


def decorate_contract(contract, plan):
    base.require((contract["leg_kp"], contract["leg_kd"]) == (500., 10.), "Original exported controller differs")
    base.require(plan["audit"].get("final_runtime_evidence", {}).get("finalized_after_rollout") is True, "Final—not installation-time—runtime proof required before export")
    contract["leg_target_filter_diagnostic"] = copy.deepcopy(plan["audit"])
    contract["actor_export_includes_leg_target_filter"] = False
    contract["external_leg_target_filter_alpha"] = plan["alpha"]
    contract["previous_action_is_requested_not_filtered"] = True


def materialize_source(source, legacy_root):
    generated = base.materialize_source(source, legacy_root)
    replacements = [
        ('    wrapped = None', '    wrapped = None\n    _leg_filter_handler = None'),
        ('            "contact_stability_diagnostic": copy.deepcopy(_contact_plan["audit"]),', '            "leg_target_filter_diagnostic": copy.deepcopy(_contact_plan["audit"]),'),
        ('            contact_material_audit = audit_isaac_materials(env, evaluation_profile)',
         '            contact_material_audit = audit_isaac_materials(env, evaluation_profile)\n            _leg_filter_handler = _leg_filter_install(env, _contact_plan)'),
        ('        summary = summarize_episodes(arrays, outcomes, reference, args.contact_threshold_n)',
         '        summary = summarize_episodes(arrays, outcomes, reference, args.contact_threshold_n)\n        _leg_filter_finalize(env, _leg_filter_handler, _contact_plan)'),
        ('                if wrapped is not None:\n                    wrapped.close()',
         '                try:\n                    if _leg_filter_handler is not None:\n                        _leg_filter_handler.close()\n                finally:\n                    if wrapped is not None:\n                        wrapped.close()'),
    ]
    for old, new in replacements:
        base.require(generated.count(old) == 1, "Pinned target-filter insertion anchor differs")
        generated = generated.replace(old, new, 1)
    compile(generated, "<pinned_leg_target_filter_evaluator>", "exec")
    return generated


def run(argv):
    plan = prepare_plan(argv)
    generated = materialize_source(plan["legacy_evaluator"].read_text(), plan["legacy_evaluator"].parent.parent)
    plan["audit"]["generated_evaluator_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
    namespace = {"__name__": "_pinned_leg_target_filter_evaluator", "__file__": str(Path(__file__).resolve()),
        "_contact_plan": plan, "_contact_apply_config": apply_config, "_contact_decorate_contract": decorate_contract,
        "_leg_filter_install": install_filter, "_leg_filter_finalize": finalize_filter}
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<pinned_leg_target_filter_evaluator>", "exec"), namespace); namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    run(sys.argv[1:])
