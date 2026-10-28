#!/usr/bin/env python3
"""Fresh PhysX hockey PPO, or deterministic full-horizon validation.

All runs use fresh output directories. The exported actor includes its own
normalizer. Validation records native forces and every 400 Hz physical state;
it neither resets on failure nor calls the training reward as a success test.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def runner_config(protocol):
    p = protocol["ppo"]
    return {
        "seed": protocol["seed"], "num_steps_per_env": p["steps_per_env"],
        "save_interval": 100, "logger": "tensorboard", "experiment_name": "hockey_stop",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {"class_name": "ActorCritic", "init_noise_std": p["init_noise_std"],
            "noise_std_type": "log", "actor_obs_normalization": True,
            "critic_obs_normalization": True, "actor_hidden_dims": [256, 128, 128],
            "critic_hidden_dims": [256, 128, 128], "activation": "elu"},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1., "use_clipped_value_loss": True,
            "clip_param": p["clip_param"], "entropy_coef": p["entropy_coef"],
            "num_learning_epochs": p["epochs"], "num_mini_batches": p["mini_batches"],
            "learning_rate": p["learning_rate"], "schedule": p["schedule"],
            "gamma": p["gamma"], "lam": p["lam"], "desired_kl": .01, "max_grad_norm": 1.}}


def state_sha(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate(env, actor_path, output, protocol, case=None):
    import numpy as np
    import torch
    from hockey_ppo_task import WHEELS, SENSORS
    from physx_reference_math import reduce_contacts
    from training.tron1_tracking import ACTION_JOINT_NAMES
    actor = torch.jit.load(str(actor_path), map_location=env.device).eval()
    robot, command = env.scene["robot"], env.command_manager.get_term("motion")
    body_names = list(robot.body_names)
    jid = [robot.joint_names.index(k) for k in ACTION_JOINT_NAMES]
    wid = [body_names.index(k) for k in WHEELS]
    records, raw, observations, actions = {}, [], [], []
    fault = torch.zeros(1, device=env.device, dtype=torch.bool)

    def record(contact=None):
        state = dict(root_pos=robot.data.root_pos_w-env.scene.env_origins,
            root_quat=robot.data.root_quat_w, root_com_vel=robot.data.root_com_lin_vel_w,
            root_omega=robot.data.root_com_ang_vel_w, joint_pos=robot.data.joint_pos[:, jid],
            joint_vel=robot.data.joint_vel[:, jid], body_pos=robot.data.body_pos_w-env.scene.env_origins[:, None],
            body_com_pos=robot.data.body_com_pos_w-env.scene.env_origins[:, None],
            body_quat=robot.data.body_quat_w, body_com_vel=robot.data.body_com_lin_vel_w,
            body_omega=robot.data.body_com_ang_vel_w)
        if contact is None:
            contact = {k: torch.zeros((1, 2, 3) if k == "friction_force_w" else (1, 2), device=env.device)
                       for k in ("normal_fz", "lateral_slip_rms", "utilization_proxy", "friction_anchor_power",
                                 "friction_force_w", "normal_points_count", "friction_anchors_count")}
        state.update(contact)
        for k, value in state.items():
            records.setdefault(k, []).append(value[0].detach().clone())

    with torch.inference_mode():
        env.reset()
        env.scene.write_data_to_sim()
        env.scene.update(.0025)
        if case is not None:
            from hockey_speed_study import perturbed_root
            pose = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), -1)[0].cpu().numpy()
            velocity = torch.cat((robot.data.root_com_lin_vel_w, robot.data.root_com_ang_vel_w), -1)[0].cpu().numpy()
            expected_pose, expected_velocity = perturbed_root(pose, velocity, case)
            # Nominal retains the exact historical reset sequence.
            if case["kind"] != "nominal":
                robot.write_root_pose_to_sim(torch.as_tensor(expected_pose[None], device=env.device, dtype=torch.float32))
                robot.write_root_com_velocity_to_sim(torch.as_tensor(expected_velocity[None], device=env.device, dtype=torch.float32))
                env.scene.write_data_to_sim()
                env.scene.update(.0025)
            actual_pose = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), -1)[0].cpu().numpy()
            actual_velocity = torch.cat((robot.data.root_com_lin_vel_w, robot.data.root_com_ang_vel_w), -1)[0].cpu().numpy()
            error = max(float(abs(actual_pose-expected_pose).max()), float(abs(actual_velocity-expected_velocity).max()))
            if error > 5e-5:
                raise ValueError("Reset perturbation readback mismatch")
            write(output/"case.json", dict(case=case, maximum_initial_state_error=error,
                initial_root_pose_w=actual_pose.tolist(), initial_COM_velocity_w=actual_velocity.tolist(),
                commands_transformed=False, height_and_joint_pose_changed=False))
        record()
        for frame in range(protocol["evaluation_steps"]):
            command.time_steps.fill_(frame)
            command._refresh_relative_targets()
            obs = env.observation_manager.compute()["policy"]
            if obs.shape != (1, protocol.get("actor_dim", 56)):
                raise ValueError("Wrong hockey actor input shape")
            action = actor(obs).clamp(-1, 1)
            env.action_manager.process_action(action)
            observations.append(obs[0].clone())
            actions.append(action[0].clone())
            for _ in range(8):
                env._sim_step_counter += 1
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(.0025)
                sides, buffers = [], {}
                for side, name in enumerate(SENSORS):
                    view = env.scene.sensors[name].contact_physx_view
                    normal = view.get_contact_data(.0025)
                    friction = view.get_friction_data(.0025)
                    for kind, values in (("normal", normal), ("friction", friction)):
                        for i, value in enumerate(values):
                            buffers[f"w{side}_{kind}_buffer_{i}"] = value.detach().clone()
                    b = wid[side]
                    data = reduce_contacts(normal, friction, robot.data.body_com_pos_w[:, b],
                        robot.data.body_com_lin_vel_w[:, b], robot.data.body_com_ang_vel_w[:, b],
                        robot.data.body_quat_w[:, b], protocol["evaluation_mu"])
                    fault.logical_or_(data.pop("fault"))
                    sides.append(data)
                raw.append(buffers)
                record({k: torch.stack([d[k] for d in sides], 1) for k in sides[0]})
        if bool(fault.any()):
            raise ValueError("Invalid native contact buffers")
    values = {k: torch.stack(v).cpu().numpy() for k, v in records.items()}
    values.update(actions=torch.stack(actions).cpu().numpy(), observations=torch.stack(observations).cpu().numpy(),
        joint_names=np.array(ACTION_JOINT_NAMES), body_names=np.array(body_names),
        environment_origin_w=env.scene.env_origins[0].cpu().numpy(), physics_dt_s=.0025, policy_dt_s=.02)
    np.savez_compressed(output/"final_rollout.npz", **values)
    np.savez_compressed(output/"native_contact_buffers.npz",
        **{k: torch.stack([v[k] for v in raw]).cpu().numpy() for k in raw[0]},
        physical_sample_indices=np.arange(1, len(raw)+1))
    files = ["final_rollout.npz", "native_contact_buffers.npz"]
    if case is not None:
        files.append("case.json")
    return {name: sha(output/name) for name in files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "motion", "asset", "tracking-root", "beyondmimic-root", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "formal", "eval", "eval-suite"), required=True)
    p.add_argument("--variant", choices=("tracking", "support"), required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--actor", type=Path)
    a = p.parse_args()
    cfg = json.loads(a.protocol.read_text())
    if a.output_dir.exists() or "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("Fresh output and explicit unmasked GPU required")
    evaluating = a.mode in ("eval", "eval-suite")
    if (a.actor is not None) != evaluating:
        raise ValueError("Only validation loads an actor; training starts from scratch")
    for path, key in ((a.motion, "reference"), (a.asset, "asset"),
                      (a.tracking_root/"training/tron1_tracking.py", "tracking")):
        if sha(path) != cfg[key+"_sha256"]:
            raise ValueError("Input hash changed: "+key)
    if importlib.metadata.version("rsl-rl-lib") != "3.0.1":
        raise ValueError("Require existing RSL-RL 3.0.1 runtime")
    if cfg["kind"] not in ("hockey_ppo_learning_v1", "hockey_ppo_speed_success_v1") or (cfg["physics_dt_s"], cfg["decimation"]) != (.0025, 8):
        raise ValueError("Wrong task or clock")
    method = cfg.get("method", "hockey")
    if method not in ("hockey", "beyondmimic"):
        raise ValueError("Unknown study method")
    if a.mode == "eval-suite" and not cfg.get("evaluation_cases"):
        raise ValueError("Evaluation suite requires predeclared cases")
    a.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(a.tracking_root))
    os.environ["TRON1_BEYONDMIMIC_ROOT"] = str(a.beyondmimic_root)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from isaaclab.app import AppLauncher
    ordinal = int(a.device.split(":")[1])
    app = AppLauncher(headless=True, device=a.device,
        kit_args=f"--/renderer/activeGpu={ordinal} --/renderer/multiGpu/enabled=false --/physics/cudaDevice={ordinal} --/physics/sceneMultiGPUMode=0").app
    env, started = None, time.monotonic()
    report = dict(kind=cfg["kind"], mode=a.mode, variant=a.variant, status="initializing",
        config_sha256=sha(a.protocol), motion_sha256=sha(a.motion), asset_sha256=sha(a.asset),
        source_sha256=sha(__file__), task_source_sha256=sha(Path(__file__).with_name("hockey_ppo_task.py")),
        algorithm="RSL-RL 3.0.1 PPO", task_success_claimed=False,
        method=method, seed=cfg["seed"], speed_scale=cfg.get("speed_scale", 1.))
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from training.tron1_tracking import make_env_cfg
        from hockey_ppo_task import configure, SupportWindow
        from physx_mjlab_recipe import audit_physx_mjlab_recipe
        torch.cuda.set_device(ordinal)
        torch.manual_seed(cfg["seed"])
        training = not evaluating
        n = cfg[a.mode]["num_envs"] if training else 1
        ecfg = make_env_cfg(a.motion, a.asset, n, a.device, eval_mode=not training)
        ecfg.seed = cfg["seed"]
        if method == "beyondmimic":
            from hockey_beyondmimic_task import configure as configure_original
            configure_original(ecfg, cfg, training=training)
            report["baseline_adapter_sha256"] = sha(Path(__file__).with_name("hockey_beyondmimic_task.py"))
        else:
            configure(ecfg, cfg, a.variant, training=training)
        env = ManagerBasedRLEnv(cfg=ecfg)
        collector = SupportWindow(env) if training else None
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        obs, _ = wrapped.reset()
        if tuple(obs["policy"].shape) != (n, cfg["actor_dim"]) or tuple(obs["critic"].shape) != (n, cfg["critic_dim"]):
            raise ValueError("Declared observation ABI changed")
        report["material_audit"] = audit_physx_mjlab_recipe(env, check_usd=True)
        report["reward_terms"] = {k: float(env.reward_manager.get_term_cfg(k).weight) for k in env.reward_manager.active_terms}
        report["termination_terms"] = list(env.termination_manager.active_terms)
        allowed = {"action_rate_l2", "joint_limit", "undesired_contacts", "hockey_vertical_acceleration"}
        allowed.update("hockey_"+k for k in ("xy", "velocity", "heading", "yaw_rate", "leg_pose", "wheel_axis", "beta", "stop_hold", "tilt", "fall", "support"))
        expected_terminations = {"time_out", "motion_end", "hockey_fall"}
        if method == "beyondmimic":
            allowed = {"motion_global_anchor_pos", "motion_global_anchor_ori", "motion_body_pos",
                "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel", "action_rate_l2",
                "joint_limit", "undesired_contacts", "height_vertical_acceleration"}
            if set(report["reward_terms"]) != allowed:
                raise ValueError("Historical tracking rewards changed")
            expected_terminations = {"time_out", "motion_end", "anchor_ori"}
        if not set(report["reward_terms"]) <= allowed:
            raise ValueError("Unexpected inherited reward survived task configuration")
        if set(report["termination_terms"]) != expected_terminations:
            raise ValueError("Unexpected inherited termination")
        report["worlds"] = n
        write(a.output_dir/"protocol.json", cfg)
        if a.mode == "eval-suite":
            report.update(status="validation_suite_running", actor_sha256=sha(a.actor),
                single_world_validation=True, steps=cfg["evaluation_steps"], external_forces=0, cases=[])
            report["case_math_sha256"] = sha(Path(__file__).with_name("hockey_speed_study.py"))
            for case in cfg["evaluation_cases"]:
                directory = a.output_dir/case["id"]
                directory.mkdir()
                artifacts = validate(env, a.actor, directory, cfg, case)
                row = dict(report)
                row.pop("cases")
                row.update(status="completed_requires_independent_qualification", mode="eval", case_id=case["id"], artifacts=artifacts)
                write(directory/"report.json", row)
                report["cases"].append(dict(id=case["id"], report_sha256=sha(directory/"report.json")))
                write(a.output_dir/"run_status.json", report)
                print("HOCKEY_CASE_COMPLETE", case["id"], flush=True)
            report["status"] = "validation_suite_completed_requires_independent_qualification"
        elif not training:
            report.update(status="completed_requires_independent_qualification", actor_sha256=sha(a.actor),
                single_world_validation=True, steps=cfg["evaluation_steps"], external_forces=0,
                artifacts=validate(env, a.actor, a.output_dir, cfg))
        else:
            for _ in range(8):
                obs, rew, _, _ = wrapped.step(torch.zeros((n, 8), device=a.device))
                if not (torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
                        and torch.isfinite(rew).all() and torch.isfinite(collector.force).all()):
                    raise ValueError("Nonfinite hockey smoke")
            wrapped.reset()
            config = runner_config(cfg)
            write(a.output_dir/"runner_config.json", config)
            # Same model initialization after environment/smoke random draws.
            torch.manual_seed(cfg["seed"])
            torch.cuda.manual_seed_all(cfg["seed"])
            runner = OnPolicyRunner(wrapped, copy.deepcopy(config), log_dir=str(a.output_dir), device=a.device)
            report.update(status="training", initial_policy_state_sha256=state_sha(runner.alg.policy))
            write(a.output_dir/"run_status.json", report)
            iterations = cfg[a.mode]["iterations"]
            runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
            runner.save(str(a.output_dir/"model_final.pt"))
            actor = torch.nn.Sequential(copy.deepcopy(runner.alg.policy.actor_obs_normalizer).cpu(),
                                        copy.deepcopy(runner.alg.policy.actor).cpu()).eval()
            exported = torch.jit.script(actor)
            exported.save(str(a.output_dir/"actor_normalized.pt"))
            probe = torch.linspace(-2., 2., cfg["actor_dim"]).repeat(2, 1)
            with torch.inference_mode():
                expected = runner.alg.policy.act_inference({"policy": probe.to(a.device)})
                if not torch.allclose(exported(probe), expected.cpu(), rtol=1e-5, atol=1e-6):
                    raise ValueError("Actor normalization export mismatch")
            transitions = n*cfg["ppo"]["steps_per_env"]*iterations
            if runner.tot_timesteps != transitions:
                raise ValueError("Transition accounting mismatch")
            report.update(status="training_completed_not_task_success", transitions=transitions,
                checkpoint_sha256=sha(a.output_dir/"model_final.pt"),
                actor_sha256=sha(a.output_dir/"actor_normalized.pt"), actor_includes_normalizer=True)
            if runner.writer is not None:
                runner.writer.flush()
                runner.writer.close()
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        write(a.output_dir/"report.json", report)
        write(a.output_dir/"run_status.json", report)
        if env is not None:
            env.close()
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    main()
