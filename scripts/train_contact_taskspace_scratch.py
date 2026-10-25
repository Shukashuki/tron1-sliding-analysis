"""Matched K/M task-space PPO from zero, without inherited static-leg policy.

Original K/M force arithmetic and task-space targets are reused unchanged.
No checkpoint, optimizer/normalizer resume, leg-velocity cost or action filter.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import car_taskspace_tracking as targets
import train_ppo_ground_contact_v2k as frozen
from ground_contact_beta_contract import verify_beta_source, verify_beta_runtime

COHORT_ID = "tron1_contact_taskspace_scratch_K_M_400hz_v1"
K_TRAINER_SHA = "65e6d41695219b676200f61d6d6941ca546fb8f1e95fbd6c60ca89347cf1d1e8"
RANDOM_STATE_SHA = "7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d"
RUNNER_SHA = "e51e4dd49664ccc8743a16b71d2971d392f40a6988136f10fe8c5f03ab6a83b5"
PPO_SHA = "deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c"
OPTIMIZER = dict(learning_rate=.001, schedule="adaptive", clip_param=.2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def validate_protocol(protocol, baseline):
    assert protocol["cohort_id"] == COHORT_ID and protocol["variants"] == ["K", "M"]
    for name in ("algorithm", "smoke", "training_randomization", "validation",
                 "nominal_acceptance_before_training", "physics_dt_s", "decimation",
                 "policy_dt_s", "policy_fps", "action_rate_weight", "grounded_reward_weight"):
        assert protocol[name] == baseline[name], name
    assert protocol["from_scratch"] is True and protocol["historical_checkpoint_used"] is False
    assert "warm_start" not in protocol and "refinement_optimizer" not in protocol
    assert protocol["optimizer"] == OPTIMIZER
    assert protocol["leg_command_execution"] == "original_position_target_hold"
    targets.settings(protocol["task_space_tracking"])
    for variant in ("K", "M"):
        d = importlib.import_module("ground_contact_reward_v2"+variant.lower())
        assert protocol["variant_contact_settings"][variant] == d.validate_settings()
    return protocol


def materialize(source, cohort):
    generated = cohort.materialize_source(source)
    pairs = [
        ("            accumulator = install_drift_reward_sampling(env, task)\n",
         "            accumulator = install_drift_reward_sampling(env, task)\n"
         "            if accumulator is not None:\n                raise ValueError('Do not stack contact owners')\n"
         "            accumulator = _scratch_install(env, _cohort_plan)\n"),
        ('        runner.save(str(args.output_dir/"model_final.pt"))\n',
         '        _scratch_pre_export(env, args, manifest, runner, _cohort_plan)\n'
         '        runner.save(str(args.output_dir/"model_final.pt"))\n'),
        ('            expected = runner.alg.policy.act_inference({"policy": probe.to(args.device)})\n',
         '            expected = _scratch_export(runner.alg.policy, exported, probe, args, manifest, _cohort_plan)\n'),
    ]
    for old, new in pairs:
        assert generated.count(old) == 1, old
        generated = generated.replace(old, new, 1)
    compile(generated, "<fresh_taskspace_contact_PPO>", "exec")
    return generated


def run(argv):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--base-snapshot", type=Path, required=True)
    p.add_argument("--baseline-protocol", type=Path, required=True)
    p.add_argument("--reward-variant", choices=["K", "M"], required=True)
    own, inherited = p.parse_known_args(argv)
    assert sha(frozen.__file__) == K_TRAINER_SHA
    assert sha(own.baseline_protocol) == "855ca5a08e2b228c5146c65fb76f55cf6c9152d70480dc43a32a875aa4ff17d9"
    previous = list(sys.path), sys.argv
    sys.path.insert(1, str(own.base_snapshot.resolve()/"scripts"))
    try:
        cohort = frozen.load_base(own.base_snapshot.resolve())
        plan = cohort.prepare_plan(inherited)
        protocol = validate_protocol(plan["protocol"], read(own.baseline_protocol))
        variant = own.reward_variant
        definition = importlib.import_module("ground_contact_reward_v2"+variant.lower())
        contact = importlib.import_module("ppo_ground_contact_v2"+variant.lower())
        settings = definition.validate_settings()
        version = "ground_contact_v2"+variant.lower()+"_any_wheel"
        here = Path(__file__).resolve().parent
        files = [Path(__file__), Path(targets.__file__), here/"verify_contact_taskspace_scratch.py",
                 here/"policy_export_precision.py", here/"ground_contact_reward_v2m.py",
                 here/"ppo_ground_contact_v2m.py", own.baseline_protocol, plan["protocol_path"]]
        ledger = {**frozen.source_ledger(own.base_snapshot.resolve()),
                  **{str(path.resolve()): sha(path) for path in files}}
        beta_source = verify_beta_source(Path(inherited[inherited.index("--tron1-source")+1])/"training/tron1_drift.py")
        ledger[beta_source["path"]] = beta_source["sha256"]
        prepare = cohort.prepare_declared_inputs

        def prepared(*args, **kwargs):
            result = prepare(*args, **kwargs)
            task, contract = read(result["task_profile"]), read(result["contract"])
            task.update(ground_contact_reward_v2=copy.deepcopy(settings), task_space_tracking=targets.settings(),
                description="Fresh task-space "+variant+" PPO: no static-leg checkpoint, original held targets, nonzero beta and original DR.")
            contract.update(task_profile=copy.deepcopy(task), trained_task_profile=copy.deepcopy(task),
                reward_definition_version=version, ground_contact_reward_v2=copy.deepcopy(settings),
                cohort_variant=variant, from_scratch=True, historical_checkpoint_used=False,
                leg_command_execution="original_position_target_hold")
            write(result["task_profile"], task); write(result["contract"], contract)
            result["derived_input_sha256"] = {"task_profile": sha(result["task_profile"]), "contract": sha(result["contract"])}
            derivation = result["task_profile"].parent/"derivation.json"
            value = read(derivation); value.update(variant=variant,
                derived_inputs_sha256=result["derived_input_sha256"], contact_source_sha256=ledger,
                initial_checkpoint=None, normalizer_resume=False, optimizer_resume=False)
            write(derivation, value)
            return result

        cohort.prepare_declared_inputs = prepared

        def before_start(args, current):
            assert args.backend == "physx"
            cohort.before_start(args, current)

        def configure(cfg, args, current, recipe):
            cohort.configure_physx(cfg, args, current, recipe)
            contact.configure_sensors(cfg.scene); cfg.rewards.drift_grounded = contact.reward_term(.4)
            current["taskspace_configured"] = targets.configure(cfg, protocol["task_space_tracking"])

        def install(env, current):
            owner = contact.install(env, settings); current["contact_owner"] = owner
            return owner

        def live(env, args, manifest, current):
            a = current["contact_owner"].audit(health=True)
            assert a["device_finite_normal_force_health"] and a["device_finite_reward_health"]
            assert a["steps_started"] == a["steps_completed"] == a["unique_reward_windows"]
            assert a["physics_samples"] == 8*a["steps_completed"] and not a["cpu_fixture_only"]
            verify_beta_runtime(a, expected_envs=args.num_envs)
            assert a["beta_quality_source_path"] == beta_source["path"]
            assert a["settings"] == settings
            assert all(sha(path) == digest for path, digest in ledger.items())
            term = env.action_manager.get_term("joint_pos")
            assert term.apply_actions.__func__ is type(term).apply_actions
            assert not hasattr(env, "_tron1_batched_leg_ramp")
            return dict(contact=a, runtime=cohort.runtime_audit(env, args, manifest["environment"], current),
                        task_space_tracking=targets.runtime_audit(env, current["taskspace_configured"]))

        def before_learn(env, args, manifest, runner, current):
            cohort.before_learn(env, args, manifest, runner, current)
            assert manifest["initial_policy_state_sha256"] == RANDOM_STATE_SHA
            assert manifest["algorithm_source_sha256"] == dict(runner=RUNNER_SHA, ppo=PPO_SHA)
            state = runner.alg.policy.state_dict()
            assert int(state["actor_obs_normalizer.count"]) == int(state["critic_obs_normalizer.count"]) == 0
            assert runner.current_learning_iteration == runner.tot_timesteps == 0
            assert not runner.alg.optimizer.state_dict()["state"]
            for key, value in OPTIMIZER.items(): assert getattr(runner.alg, key) == value
            audit = live(env, args, manifest, current)
            assert audit["contact"]["physics_samples"] == 64
            c = manifest["cohort"]
            c.update(variant=variant, live_startup=audit, source_sha256=ledger,
                normalizer_initial_count=0, optimizer_initial_state_empty=True, initial_checkpoint=None,
                optimizer=OPTIMIZER, task_space_tracking=targets.settings())
            manifest.update(reward_variant=variant, reward_definition_version=version, contact_settings=settings)
            write(args.output_dir/"manifest.json", manifest); write(args.output_dir/"cohort_startup_audit.json", c)

        def pre_export(env, args, manifest, runner, current):
            a = live(env, args, manifest, current)
            assert a["contact"]["steps_completed"] == 8+args.iterations*24
            assert runner.tot_timesteps == current["expected_transitions"]
            state = runner.alg.policy.state_dict()
            assert int(state["actor_obs_normalizer.count"]) == int(state["critic_obs_normalizer.count"]) == runner.tot_timesteps
            manifest["cohort"].update(live_final=a, normalizer_final_count=runner.tot_timesteps)
            write(args.output_dir/"pre_export_runtime_audit.json", a)

        def export(policy, exported, probe, args, manifest, current):
            from policy_export_precision import verify_export
            expected, audit = verify_export(policy, exported, probe)
            manifest["cohort"]["export_precision"] = audit
            write(args.output_dir/"export_precision_audit.json", audit)
            return expected

        def complete(args, manifest, report, current):
            assert manifest["cohort"]["from_scratch"] and not manifest["cohort"]["historical_checkpoint_used"]
            assert all(sha(path) == digest for path, digest in ledger.items())
            cohort.complete(args, manifest, report, current)
            path = args.output_dir/"cohort_completion_audit.json"
            a = read(path); a.update(variant=variant, from_scratch=True, cohort=manifest["cohort"],
                source_sha256=ledger, initial_checkpoint=None, task_success_claimed=False,
                pre_export_runtime_audit_sha256=sha(args.output_dir/"pre_export_runtime_audit.json"),
                export_precision_audit_sha256=sha(args.output_dir/"export_precision_audit.json"))
            write(path, a)

        generated = materialize((own.base_snapshot/"scripts/train_ppo_backend.py").read_text(), cohort)
        plan["generated_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
        namespace = dict(__name__="_taskspace_fresh_frozen_backend", __file__=str(plan["base"]), _cohort_plan=plan,
            _cohort_before_start=before_start, _cohort_launch_app=cohort.launch_app, _cohort_set_torch_device=cohort.set_torch_device,
            _cohort_configure_physx=configure, _scratch_install=install, _cohort_before_learn=before_learn,
            _scratch_pre_export=pre_export, _scratch_export=export, _cohort_finalize_contract=cohort.finalize_contract,
            _cohort_complete=complete)
        sys.argv = [str(Path(__file__).resolve()), *plan["inherited_argv"]]
        exec(compile(generated, "<fresh_taskspace_contact_PPO>", "exec"), namespace)
        namespace["main"]()
    finally:
        sys.path[:], sys.argv = previous


if __name__ == "__main__":
    run(sys.argv[1:])
