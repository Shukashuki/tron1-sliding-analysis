"""From-scratch K PPO with a batched external nonlinear tire force law.

Reuses the immutable K trainer/reward and original startup random draws. All
native-wheel material zeros and external DR mapping are reported explicitly.
This is a new plant cohort, not an unchanged native-Coulomb reproduction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

PARENT_SHA = "65e6d41695219b676200f61d6d6941ca546fb8f1e95fbd6c60ca89347cf1d1e8"
OLD_ID = "tron1_ground_contact_v2k_any_wheel_400hz_v1"
COHORT_ID = "tron1_ppo_k_nonlinear_tire_400hz_v1"
MODEL_ID = "authored_cylinder_external_combined_slip_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def materialize_generated(source):
    old = 'else recipe_audit(env)'
    assert source.count(old) == 2, "Need honest startup AND final substituted material audits"
    source = source.replace(old, 'else _tire_audit(env)')
    old='        runner.save(str(args.output_dir/"model_final.pt"))\n'
    assert source.count(old)==1
    source=source.replace(old,'        _tire_pre_export_audit(args, manifest, _cohort_plan)\n'+old,1)
    old='            expected = runner.alg.policy.act_inference({"policy": probe.to(args.device)})\n'
    assert source.count(old)==1
    source=source.replace(old,'            expected = _tire_export_expected(runner.alg.policy, exported, probe, args, manifest, _cohort_plan)\n',1)
    compile(source, "<K_nonlinear_tire_trainer>", "exec")
    return source


def materialize_parent(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    pairs = [
        ('protocol["cohort_id"] == "'+OLD_ID+'"', 'protocol["cohort_id"] == "'+COHORT_ID+'"'),
        ('    own_ledger = source_ledger(snapshot)\n',
         '    own_ledger = source_ledger(snapshot)\n    own_ledger.update(_tire_sources)\n'),
        ('        contract["ground_contact_reward_v2"] = copy.deepcopy(settings)\n',
         '        contract["ground_contact_reward_v2"] = copy.deepcopy(settings)\n'
         '        contract["wheel_contact_model"] = copy.deepcopy(protocol["wheel_contact_model"])\n'),
        ('    generated = materialize_source(cohort)\n',
         '    generated = _tire_materialize(materialize_source(cohort))\n'),
        ('        cfg.rewards.drift_grounded = contact.reward_term(.4)\n',
         '        cfg.rewards.drift_grounded = contact.reward_term(.4)\n        _tire_configure(cfg)\n'),
        ('        current["contact_owner"] = owner\n',
         '        current["contact_owner"] = owner\n        _tire_install(env, owner, current)\n'),
        ('        cohort.before_learn(env, args, manifest, runner, current)\n',
         '        cohort.before_learn(env, args, manifest, runner, current)\n'
         '        _tire_before_learn(env, args, manifest, current)\n'),
        ('        manifest["ground_contact_v2k"]["runtime_final"] = audit\n',
         '        manifest["ground_contact_v2k"]["runtime_final"] = audit\n'
         '        _tire_before_complete(args, manifest, current)\n'),
        ('        write_json(args.output_dir/"ground_contact_v2k_training_audit.json", result)\n',
         '        write_json(args.output_dir/"ground_contact_v2k_training_audit.json", result)\n'
         '        _tire_seal(args, manifest, report, current)\n'),
        ('        "_cohort_plan": plan, "_cohort_before_start": before_start,\n',
         '        "_cohort_plan": plan, "_cohort_before_start": before_start,\n'
         '        "_tire_audit": _tire_recipe_audit,\n'
         '        "_tire_pre_export_audit": _tire_pre_export,\n'
         '        "_tire_export_expected": _tire_export,\n')]
    for old, new in pairs:
        assert source.count(old) == 1, "Missing/ambiguous nonlinear model hook: " + old[:100]
        source = source.replace(old, new, 1)
    compile(source, "<K_nonlinear_tire_parent>", "exec")
    return source


def native_geometry_audit(env, report_path):
    """All worlds retain original mass frames, shape count, joints and offsets."""
    import numpy as np
    reference = json.loads(Path(report_path).read_text())["contact_material_audit"]["native_plant_readback"]
    robot = env.scene["robot"]; view = robot.root_physx_view
    assert list(robot.body_names) == reference["body_names"] and list(robot.joint_names) == reference["joint_names"]
    assert len(robot.joint_names) == 8 and view.max_shapes == 9
    observed = {}
    for name in ("get_masses", "get_inertias", "get_coms", "get_dof_armatures", "get_dof_dampings"):
        value = getattr(view, name)().detach().cpu().numpy().copy()
        expected = np.asarray(reference["readbacks"][name]["values"], dtype=float)
        assert value.shape[0] == env.num_envs and np.isfinite(value).all()
        if name != "get_coms":
            np.testing.assert_allclose(value, np.broadcast_to(expected, value.shape), atol=1e-7, rtol=0)
        else:
            base = robot.body_names.index("base_Link"); other = [i for i in range(len(robot.body_names)) if i != base]
            np.testing.assert_allclose(value[:, other], np.broadcast_to(expected[:, other], value[:, other].shape), atol=1e-7, rtol=0)
            np.testing.assert_allclose(value[:, base, 3:], np.broadcast_to(expected[:, base, 3:], value[:, base, 3:].shape), atol=1e-7, rtol=0)
            assert np.all(np.abs(value[:, base, :3]-expected[:, base, :3]) <= np.array([.025, .05, .05])+1e-6)
        observed[name] = hashlib.sha256(value.tobytes()).hexdigest()
    from physx_mjlab_recipe import _shape_map
    slots = _shape_map(robot)
    wheel_slots = [i for name in ("wheel_L_Link", "wheel_R_Link") for i in slots[name]]
    assert len(wheel_slots) == 2
    for name, expected in (("get_contact_offsets", .001), ("get_rest_offsets", 0.)):
        value = getattr(view, name)().cpu().numpy()
        assert value.shape == (env.num_envs, 9)
        np.testing.assert_allclose(value[:, wheel_slots], expected, atol=1e-8, rtol=0)
    return dict(num_envs=env.num_envs, model_id=MODEL_ID, physical_wheels_checked=2*env.num_envs,
        total_shapes_per_robot=9, joint_count=8, additional_DOF=0,
        original_mass_inertia_and_nonbase_COM_retained=True, base_COM_original_DR_only=True,
        contact_offset_m=.001, rest_offset_m=0., native_property_sha256=observed,
        authored_report_sha256=sha(report_path))


def run(argv):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--k-snapshot", type=Path, required=True)
    p.add_argument("--authored-native-report", type=Path, required=True)
    own, inherited = p.parse_known_args(argv)
    protocol_path = Path(inherited[inherited.index("--cohort-protocol")+1]).resolve()
    protocol = json.loads(protocol_path.read_text())
    assert protocol["cohort_id"] == COHORT_ID and protocol["wheel_contact_model"]["model_id"] == MODEL_ID
    assert protocol["wheel_contact_model"]["force_law"] == "combined_slip_magic_formula"
    assert sha(own.authored_native_report) == "29499d653e272b74aadb491be2f41c940b5232674ffb7e5f1c3a4dbc333d9130"
    parent = own.k_snapshot.resolve()/"scripts/train_ppo_ground_contact_v2k.py"
    source = materialize_parent(parent.read_text())
    here = Path(__file__).resolve().parent
    paths = [Path(__file__), here/"physx_batched_tire.py", here/"nonlinear_tire_torch.py",
             here/"verify_ppo_k_nonlinear_tire.py",here/"policy_export_precision.py", protocol_path, own.authored_native_report]
    ledger = {str(path.resolve()):sha(path) for path in paths}

    def configure(cfg):
        from physx_batched_tire import configure_sensors
        configure_sensors(cfg.scene)

    def install(env, owner, current):
        from physx_batched_tire import BatchedTireRuntime
        current["nonlinear_tire"] = BatchedTireRuntime(env, owner=owner)

    def recipe_audit(env):
        return env._tron1_batched_tire_runtime.audit_materials(check_usd=True)

    def before_learn(env, args, manifest, current):
        tire = current["nonlinear_tire"]
        audit = tire.audit()
        assert audit["physical_writes"] == current["contact_owner"].total_samples == 64
        assert min(audit["loaded_wheel_samples_L_R"]) > 0
        payload = dict(cohort_id=COHORT_ID, settings=copy.deepcopy(protocol["wheel_contact_model"]),
            startup_runtime=audit, startup_geometry=native_geometry_audit(env, own.authored_native_report),
            materialized_parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
            source_sha256=ledger, from_scratch=True, other14_rewards_and_K_arithmetic_unchanged=True)
        manifest["nonlinear_tire_training"] = payload
        manifest["cohort"]["wheel_contact_model"] = copy.deepcopy(protocol["wheel_contact_model"])
        write(args.output_dir/"nonlinear_tire_startup_audit.json", payload)

    def before_complete(args, manifest, current):
        tire = current["nonlinear_tire"]; payload = manifest["nonlinear_tire_training"]
        payload["final_runtime"] = tire.audit()
        assert payload["final_runtime"]["physical_writes"] == current["contact_owner"].total_samples
        payload["final_geometry"] = native_geometry_audit(tire.env, own.authored_native_report)
        assert all(sha(path) == digest for path, digest in ledger.items())

    def pre_export(args,manifest,current):
        tire=current["nonlinear_tire"]
        # Preserve the actual live counters/health before any export can fail.
        pending=dict(kind="pre_export_live_nonlinear_tire_audit",
            physical_writes=tire.calls,invalid_measurement_count=int(tire.fault.sum().item()),
            maximum_capacity_excess_n=float(tire.maximum_capacity_excess),
            maximum_contact_power_w=float(tire.maximum_power),
            contact_owner=current["contact_owner"].audit(health=True))
        write(args.output_dir/"pre_export_runtime_audit.json",pending)
        pending["tire_runtime"]=tire.audit()
        assert tire.calls==current["contact_owner"].total_samples
        pending["status"]="live_native_health_verified_before_export"
        write(args.output_dir/"pre_export_runtime_audit.json",pending)

    def export_expected(policy,exported,probe,args,manifest,current):
        from policy_export_precision import verify_export
        reference,precision=verify_export(policy,exported,probe)
        current["export_precision"]=precision
        manifest["nonlinear_tire_training"]["export_precision"]=precision
        write(args.output_dir/"export_precision_audit.json",precision)
        return reference

    def seal(args, manifest, report, current):
        smoke = current["run_kind"] == "smoke"
        audit = dict(kind="model_specific_K_nonlinear_tire_PPO_training", status="completed_verified",
            cohort_id=COHORT_ID, model_id=MODEL_ID, protocol_sha256=sha(protocol_path),
            training_environment_steps=report["training_environment_steps"], smoke_only=smoke,
            trained_policy_claimed=not smoke, model_audit=manifest["nonlinear_tire_training"],
            initial_policy_state_sha256=manifest["initial_policy_state_sha256"],
            K_training_audit_sha256=sha(args.output_dir/"ground_contact_v2k_training_audit.json"),
            startup_audit_sha256=sha(args.output_dir/"nonlinear_tire_startup_audit.json"),
            manifest_sha256=sha(args.output_dir/"manifest.json"),
            policy_contract_sha256=sha(args.output_dir/"policy_contract.json"),
            actor_sha256=sha(args.output_dir/"policy.pt"), checkpoint_sha256=sha(args.output_dir/"model_final.pt"),
            pre_export_runtime_audit_sha256=sha(args.output_dir/"pre_export_runtime_audit.json"),
            export_precision_audit_sha256=sha(args.output_dir/"export_precision_audit.json"),
            task_success_claimed=False)
        write(args.output_dir/"nonlinear_tire_training_audit.json", audit)

    namespace = dict(__name__="_K_nonlinear_tire_parent", __file__=str(parent), _tire_sources=ledger,
        _tire_materialize=materialize_generated, _tire_configure=configure, _tire_install=install,
        _tire_recipe_audit=recipe_audit, _tire_before_learn=before_learn,
        _tire_before_complete=before_complete, _tire_seal=seal,
        _tire_pre_export=pre_export,_tire_export=export_expected)
    previous_path, previous_argv = list(sys.path), sys.argv
    try:
        sys.path.append(str(parent.parent))
        exec(compile(source, "<K_nonlinear_tire_parent>", "exec"), namespace)
        namespace["run"](inherited)
    finally:
        sys.path[:], sys.argv = previous_path, previous_argv


if __name__ == "__main__":
    run(sys.argv[1:])
