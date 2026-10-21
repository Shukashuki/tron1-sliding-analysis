"""Private Isaac Sim 5.1 vectorized physics worker for shooting MPC.

Only isolated candidate robots are reset. Actions pass through the original
Isaac Lab action terms and DC motors at 200 Hz; the shared NumPy objective is
evaluated at 50 Hz with a pre-action held reference. No root forces, policy,
action replay, or per-stage root prescription is used.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np


class PhysXCandidateWorker:
    def __init__(self, request):
        self.config = cfg = request["config"]
        self.mu = float(request["mu"])
        self.app = self.env = None
        self.completed_batches = 0
        started = time.perf_counter()
        if hashlib.sha256(Path(cfg["motion"]).read_bytes()).hexdigest() != request["expected_motion_sha256"]:
            raise ValueError("PhysX worker motion hash differs from the controller reference")
        from isaaclab.app import AppLauncher
        ordinal = int(cfg["device"].split(":")[1])
        kit_args = (f"--/renderer/activeGpu={ordinal} --/renderer/multiGpu/enabled=false "
                    f"--/physics/cudaDevice={ordinal} --/physics/sceneMultiGPUMode=0")
        args = argparse.Namespace(headless=True, device=cfg["device"], enable_cameras=False,
                                  kit_args=kit_args)
        self.app = AppLauncher(args).app
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        self.torch = torch
        source_root = Path(cfg["source_root"])
        for directory in (source_root, source_root/"scripts", Path(__file__).resolve().parent):
            sys.path.insert(0, str(directory))
        from training.tron1_tracking import make_env_cfg
        from training.tron1_drift import load_task_profile
        from training.tron1_runtime_device import validate_simulation_device
        from run_nmpc_tracking_isaac import configure_fixed_evaluation, audit_materials, audit_actuators
        from height_smoothness import install_height_smoothness
        from nmpc_tracking_objective import BODY_NAMES, JOINT_NAMES, HeightSmoothTrackingObjective
        from nmpc_batch_objective import BODY_FIELDS, BatchHeightSmoothTrackingObjective
        self.body_names, self.joint_names = BODY_NAMES, JOINT_NAMES
        self.body_fields = BODY_FIELDS
        self.profile = profile = load_task_profile(cfg["task_profile"])
        env_cfg = make_env_cfg(cfg["motion"], cfg["asset_path"], cfg["max_candidates"],
                               cfg["device"], eval_mode=True, task_profile=profile)
        env_cfg.seed = cfg["seed"]
        env_cfg.scene.env_spacing = cfg["environment_spacing_m"]
        if cfg["environment_spacing_m"] == 0. and not (
                env_cfg.scene.replicate_physics and env_cfg.scene.filter_collisions):
            raise ValueError("Overlapping candidate worlds require GPU replicated physics and environment collision filtering")
        install_height_smoothness(env_cfg, profile["height_regularization"])
        fixed = configure_fixed_evaluation(env_cfg, self.mu)
        if env_cfg.sim.dt != .005 or env_cfg.decimation != 4:
            raise ValueError("Expected the shared 200 Hz physics / 50 Hz control contract")
        self.env = env = ManagerBasedRLEnv(cfg=env_cfg)
        env.reset()
        if env.termination_manager.active_terms:
            raise AssertionError("Candidate environments retain an automatic-reset gate")
        self.robot = robot = env.scene["robot"]
        self.sensor = sensor = env.scene.sensors["contact_forces"]
        self.body_ids = [robot.body_names.index(name) for name in BODY_NAMES]
        self.joint_ids = [robot.joint_names.index(name) for name in JOINT_NAMES]
        self.force_ids = [sensor.body_names.index(name) for name in BODY_NAMES]
        if len(robot.joint_names) != 8:
            raise ValueError("Only the original eight actuated joints are supported")
        self.origins = self.cpu(env.scene.env_origins)
        if cfg["environment_spacing_m"] == 0. and np.any(self.origins):
            raise AssertionError("Coordinate-identical candidate worlds have nonzero origins")
        self.soft_limits = self.cpu(robot.data.soft_joint_pos_limits[0, self.joint_ids])
        expected = request["expected_soft_joint_limits"]
        if expected is not None and not np.allclose(self.soft_limits, expected, atol=2e-6, rtol=0):
            raise AssertionError("PhysX predictor soft joint limits differ from the configured objective limits")
        expected_legs = request.get("expected_leg_soft_limits")
        # XML decimal ranges and USD degree-to-radian float32 ranges differ by
        # up to 2.91e-6 rad in the archived asset. Score with the measured runtime
        # limits and report this import precision difference explicitly.
        leg_limit_error = (None if expected_legs is None else
                           float(np.max(np.abs(self.soft_limits[:6]-expected_legs))))
        if leg_limit_error is not None and leg_limit_error > 5e-6:
            raise AssertionError(f"PhysX/base penalized leg soft limits differ by {leg_limit_error} rad")
        scalar = HeightSmoothTrackingObjective(profile, dt=.02,
                    soft_joint_limits=self.soft_limits if expected is None else np.asarray(expected))
        if scalar.contract() != request["objective_contract"]:
            raise AssertionError("PhysX worker objective differs from the controller objective")
        self.objective = BatchHeightSmoothTrackingObjective.from_scalar(scalar)
        with np.load(cfg["motion"], allow_pickle=False) as archive:
            names = tuple(str(value) for value in archive["body_names"])
            indices = [names.index(name) for name in BODY_NAMES]
            self.reference = {name: archive[name][:, indices].copy() for name in BODY_FIELDS}
        self.reference["body_names"] = BODY_NAMES
        self.reference_frames = len(self.reference["body_pos_w"])
        legacy = getattr(env, "legacy_joint_friction_audit", None)
        if not legacy or not legacy.get("all_environments_verified_zero"):
            raise AssertionError("Candidate legacy joint friction lacks verified-zero runtime readback")
        runtime_paths = [Path(__file__), Path(__file__).with_name("physx_batch_predictor.py"),
                         Path(__file__).with_name("nmpc_batch_objective.py"),
                         Path(__file__).with_name("nmpc_tracking_objective.py"),
                         Path(__file__).with_name("run_nmpc_tracking_isaac.py"),
                         Path(cfg["task_profile"])]
        for relative in ("training/tron1_tracking.py", "training/tron1_drift.py",
                         "training/tron1_runtime_device.py", "training/tron1_domain_randomization.py"):
            runtime_paths.append(source_root/relative)
        for method in (type(env).step, type(env.scene).reset_to, type(robot).write_root_pose_to_sim,
                       type(robot).write_joint_state_to_sim, type(env.sim).forward):
            runtime_paths.append(Path(inspect.getsourcefile(method)))
        source_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in runtime_paths}
        self.metadata = {
            "engine": "Isaac/PhysX", "num_candidate_environments": env.num_envs,
            "environment_spacing_m": cfg["environment_spacing_m"],
            "cross_environment_collision_filtering": bool(env_cfg.scene.replicate_physics and env_cfg.scene.filter_collisions),
            "coordinate_identical_candidates": bool(np.all(self.origins == 0.)),
            "physics_dt_s": float(env.physics_dt), "control_dt_s": float(env.step_dt),
            "body_order": list(BODY_NAMES), "action_order": list(JOINT_NAMES),
            "device_audit": validate_simulation_device(env, requested_device=cfg["device"]),
            "app_launcher_kit_args": kit_args,
            "material_audit": audit_materials(env, self.mu), "actuator_audit": audit_actuators(robot),
            "legacy_joint_friction_audit": legacy, "fixed_evaluation": fixed,
            "runtime_soft_joint_pos_limits": self.soft_limits.tolist(),
            "base_predictor_leg_soft_limits_verified": expected_legs is not None,
            "base_predictor_leg_soft_limits_max_abs_error_rad": leg_limit_error,
            "leg_soft_limit_import_tolerance_rad": 5e-6,
            "objective_contract": scalar.contract(),
            "source_sha256": source_hashes,
            "motion_sha256": hashlib.sha256(Path(cfg["motion"]).read_bytes()).hexdigest(),
            "asset_sha256": hashlib.sha256(Path(cfg["asset_path"]).read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__), "cuda_device_name": torch.cuda.get_device_name(env.device),
            "initialization_elapsed_s": time.perf_counter()-started,
            "history": "own chronological oldest-to-newest history, sampled after every real 5 ms physics step",
            "candidate_reset": "write LINK pose, world COM velocity, ordered joint state; clear actuator/sensor buffers; refresh FK without physics stepping",
            "hidden_solver_state": "PhysX internal contact caches are not captured by Isaac Lab scene state APIs",
            "phase": "reference[min(start_index+h,last)] held against pre-action candidate body state",
            "padding": "all configured environments step; unused slots repeat the final active candidate and are excluded from returned costs",
            "manager_history": "manual physical substeps; command/reward/observation managers do not advance or score rollouts",
            "plant_access": False, "trained_policy_used": False, "root_forces_used": False,
        }

    @staticmethod
    def cpu(value):
        return value.detach().cpu().numpy().copy()

    def tensor(self, value):
        return self.torch.as_tensor(value, device=self.env.device, dtype=self.torch.float32)

    def sample(self, batch):
        d = self.robot.data
        return {"body_pos_w": self.cpu(d.body_pos_w[:batch, self.body_ids])-self.origins[:batch, None],
                "body_quat_w": self.cpu(d.body_quat_w[:batch, self.body_ids]),
                "body_lin_vel_w": self.cpu(d.body_lin_vel_w[:batch, self.body_ids]),
                "body_ang_vel_w": self.cpu(d.body_ang_vel_w[:batch, self.body_ids]),
                "joint_pos": self.cpu(d.joint_pos[:batch, self.joint_ids]),
                "body_net_forces_w": self.cpu(self.sensor.data.net_forces_w[:batch, self.force_ids])}

    def reset_candidates(self, measurement):
        n = self.env.num_envs
        self.env.scene.reset()
        self.env.action_manager.reset()
        pose = np.tile(np.r_[measurement["root_pos"], measurement["root_quat_wxyz"]], (n, 1))
        pose[:, :3] += self.origins
        velocity = np.tile(np.r_[measurement["root_com_lin_vel_w"], measurement["root_ang_vel_w"]], (n, 1))
        joints = np.tile(measurement["joint_pos"], (n, 1))
        joint_vel = np.tile(measurement["joint_vel"], (n, 1))
        self.robot.write_root_link_pose_to_sim(self.tensor(pose))
        self.robot.write_root_com_velocity_to_sim(self.tensor(velocity))
        self.robot.write_joint_state_to_sim(self.tensor(joints), self.tensor(joint_vel), joint_ids=self.joint_ids)
        self.robot.set_joint_position_target(self.tensor(joints), joint_ids=self.joint_ids)
        self.robot.set_joint_velocity_target(self.tensor(np.zeros_like(joints)), joint_ids=self.joint_ids)
        self.robot.set_joint_effort_target(self.tensor(np.zeros_like(joints)), joint_ids=self.joint_ids)
        # Explicit DC motors only; targets are overwritten by process/apply_action
        # before each step. Refresh FK without advancing time or making contacts.
        self.env.sim.forward()
        if self.env.sim.physics_sim_view is not None:
            self.env.sim.physics_sim_view.update_articulations_kinematic()
        self.env.scene.update(dt=0.)
        state = self.sample(n)
        root_pos_error = float(np.max(np.abs(state["body_pos_w"][:, 0]-measurement["root_pos"])))
        root_vel_error = float(np.max(np.abs(state["body_lin_vel_w"][:, 0]-measurement["root_com_lin_vel_w"])))
        joints_error = float(np.max(np.abs(state["joint_pos"]-measurement["joint_pos"])))
        actual_joint_vel = self.cpu(self.robot.data.joint_vel[:, self.joint_ids])
        joint_vel_error = float(np.max(np.abs(actual_joint_vel-measurement["joint_vel"])))
        quat = state["body_quat_w"][:, 0]
        target_quat = np.asarray(measurement["root_quat_wxyz"])
        root_quat_error = float(np.max(np.minimum(np.max(np.abs(quat-target_quat), axis=-1),
                                                    np.max(np.abs(quat+target_quat), axis=-1))))
        if max(root_pos_error, root_vel_error, joints_error, joint_vel_error, root_quat_error) > 5e-4:
            raise AssertionError(f"Candidate reset readback failed: position={root_pos_error}, COMvel={root_vel_error}, joints={joints_error}")
        return state, {
            "root_link_position_max_abs_m": root_pos_error,
            "root_quaternion_sign_invariant_max_abs": root_quat_error,
            "root_com_velocity_max_abs_m_s": root_vel_error,
            "joint_position_max_abs_rad": joints_error,
            "joint_velocity_max_abs_rad_s": joint_vel_error,
            "body_link_position_vs_source_max_abs_m": float(np.max(np.abs(state["body_pos_w"]-measurement["body_pos_w"]))),
            "body_com_velocity_vs_source_max_abs_m_s": float(np.max(np.abs(state["body_lin_vel_w"]-measurement["body_lin_vel_w"]))),
            "physics_steps_during_reset": 0,
        }

    def canonical(self, batch):
        data = self.robot.data
        return {
            "root_pos": self.cpu(data.body_pos_w[:batch, self.body_ids[0]])-self.origins[:batch],
            "root_quat_wxyz": self.cpu(data.body_quat_w[:batch, self.body_ids[0]]),
            "root_com_lin_vel_w": self.cpu(data.body_lin_vel_w[:batch, self.body_ids[0]]),
            "root_ang_vel_w": self.cpu(data.body_ang_vel_w[:batch, self.body_ids[0]]),
            "joint_pos": self.cpu(data.joint_pos[:batch, self.joint_ids]),
            "joint_vel": self.cpu(data.joint_vel[:batch, self.joint_ids]),
        }

    def rollout(self, request):
        from nmpc_batch_objective import select_reference_sequence
        controls = np.asarray(request["controls"], dtype=float)
        batch, horizon, width = controls.shape
        if width != 8 or not 0 < batch <= self.env.num_envs or horizon < 1:
            raise ValueError("Invalid candidate action shape")
        if not np.isfinite(controls).all() or np.max(np.abs(controls)) > 1.+1e-9:
            raise ValueError("Invalid bounded normalized actions")
        controls = controls.astype(np.float32).astype(float)
        measurement = request["measurement"]
        _, reset_audit = self.reset_candidates(measurement)
        history = np.tile(np.asarray(measurement["contact_force_history_w"])[None], (batch, 1, 1, 1))
        pre_rows, post_rows, torque_rows = [], [], []
        initial_measurement = self.canonical(batch) if request.get("return_trajectory") else None
        initial_counter = self.env._sim_step_counter
        for h in range(horizon):
            if not self.app.is_running():
                raise RuntimeError("Isaac application closed during candidate rollout")
            pre_rows.append(self.sample(batch))
            padded = np.repeat(controls[-1:, h], self.env.num_envs, axis=0)
            padded[:batch] = controls[:, h]
            self.env.action_manager.process_action(self.tensor(padded))
            step_torques = []
            for _ in range(4):
                self.env._sim_step_counter += 1
                self.env.action_manager.apply_action()
                self.env.scene.write_data_to_sim()
                self.env.sim.step(render=False)
                self.env.scene.update(dt=.005)
                forces = self.cpu(self.sensor.data.net_forces_w[:batch, self.force_ids])
                history = np.concatenate((history[:, 1:], forces[:, None]), axis=1)
                if request.get("return_trajectory"):
                    step_torques.append(self.cpu(self.robot.data.applied_torque[:batch, self.joint_ids]))
            post = self.sample(batch)
            post["contact_force_history_w"] = history.copy()
            post_rows.append(post)
            if request.get("return_trajectory"):
                torque_rows.append(np.stack(step_torques, axis=1))
        if self.env._sim_step_counter-initial_counter != 4*horizon:
            raise AssertionError("Candidate physical clock mismatch")
        pre = {name: np.stack([row[name] for row in pre_rows], axis=1) for name in self.body_fields}
        post = {name: np.stack([row[name] for row in post_rows], axis=1) for name in post_rows[0]}
        pre["body_names"] = post["body_names"] = self.body_names
        post["soft_joint_pos_limits"] = self.soft_limits
        post["soft_joint_limits_source"] = "isaac_runtime"
        reference, indices = select_reference_sequence(self.reference, request["start_index"], horizon)
        previous_action = np.broadcast_to(request["previous_action"], (batch, 8))
        previous = np.concatenate((previous_action[:, None], controls[:, :-1]), axis=1)
        vz = np.concatenate((np.full((batch, 1), request["previous_vz"]),
                             post["body_lin_vel_w"][:, :-1, 0, 2]), axis=1)
        reset = np.zeros((batch, horizon), dtype=bool)
        reset[:, 0] = measurement["first_reward_after_reset"]
        scored = self.objective.evaluate(post, self.objective.hold_reference(reference, pre),
                                         controls, previous, vz, reset_samples=reset)
        costs = np.sum(scored["stage_cost"], axis=1)
        if not np.isfinite(costs).all():
            raise FloatingPointError("Nonfinite PhysX candidate costs")
        self.completed_batches += 1
        result = {"kind": "rollout_result", "request_id": request["request_id"],
                  "candidate_costs": costs, "reset_audit": reset_audit,
                  "completed_rollout_batches": self.completed_batches,
                  "reference_indices": indices, "physical_steps_per_candidate": 4*horizon,
                  "padded_environment_count": self.env.num_envs-batch}
        if request.get("return_trajectory"):
            result["trajectory"] = {name: post[name] for name in (*self.body_fields, "joint_pos", "body_net_forces_w", "contact_force_history_w")}
            result["pre_trajectory"] = {name: pre[name] for name in self.body_fields}
            result["stage_cost"] = scored["stage_cost"]
            result["applied_joint_torque_nm"] = np.stack(torque_rows, axis=1)
            result["initial_measurement"] = initial_measurement
            result["terminal_measurement"] = self.canonical(batch)
            result["terminal_measurement"]["contact_force_history_w"] = history.copy()
            result["terminal_measurement"]["first_reward_after_reset"] = False
            result["native_physics_steps"] = self.env._sim_step_counter-initial_counter
        return result

    def close(self):
        if self.env is not None:
            self.env.close()
        if self.app is not None:
            self.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    key = bytes.fromhex(os.environ.pop("TRON1_PHYSX_WORKER_AUTH"))
    connection = Client(args.socket, family="AF_UNIX", authkey=key)
    worker = None
    try:
        request = connection.recv()
        if request.get("kind") != "initialize":
            raise ValueError("Expected worker initialization")
        worker = PhysXCandidateWorker(request)
        connection.send({"kind": "ready", "metadata": worker.metadata})
        while True:
            request = connection.recv()
            if request.get("kind") == "close":
                break
            if request.get("kind") != "rollout":
                raise ValueError("Unknown worker request")
            connection.send(worker.rollout(request))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        traceback.print_exc()
        try:
            connection.send({"kind": "error", "error_type": type(error).__name__, "message": str(error)[:4096]})
        except (OSError, EOFError):
            pass
        raise
    finally:
        if worker is not None:
            worker.close()
        connection.close()


if __name__ == "__main__":
    main()
