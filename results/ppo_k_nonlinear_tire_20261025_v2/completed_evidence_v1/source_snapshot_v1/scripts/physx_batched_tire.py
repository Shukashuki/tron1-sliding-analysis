"""Device-only physical-step tire substitution for the existing K PPO task.

Native PhysX normal contact is retained. The external tangential law uses the
preceding sample (2.5 ms delay). Native wheel friction is zero. The original
startup DR draw instead supplies each world's external mu, shared by wheels.
No step/update/reward owner is added; K continues to own reward sampling.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import torch

from nonlinear_tire_torch import tangent_wrench_torch

WHEELS = ("wheel_L_Link", "wheel_R_Link")
SENSORS = ("batched_tire_L_ground", "batched_tire_R_ground")
GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"
MARKER = "_tron1_batched_tire_runtime"


def configure_sensors(scene):
    from isaaclab.sensors import ContactSensorCfg
    for name, body in zip(SENSORS, WHEELS):
        assert getattr(scene, name, None) is None
        setattr(scene, name, ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + body, update_period=0., history_length=0,
            filter_prim_paths_expr=[GROUND], track_contact_points=True,
            max_contact_data_count_per_prim=128, debug_vis=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotate_y(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), dim=-1)


class BatchedTireRuntime:
    def __init__(self, env, *, owner, nonlinear=True):
        from physx_mjlab_recipe import _shape_map, audit_physx_mjlab_recipe
        assert not hasattr(env, MARKER)
        assert env.num_envs > 0 and env.physics_dt == .0025 and env.step_dt == .02
        assert env.cfg.scene.terrain.terrain_type == "plane"
        assert tuple(env.cfg.sim.gravity)[:2] == (0., 0.)
        assert torch.device(env.device).type == "cuda"
        assert owner.env is env and owner.scene is env.scene
        self.env, self.robot, self.owner = env, env.scene["robot"], owner
        self.nonlinear, self.n = nonlinear, env.num_envs
        self.before_material_audit = audit_physx_mjlab_recipe(env, check_usd=True)
        self.mapping = _shape_map(self.robot)
        self.ids = [self.robot.body_names.index(name) for name in WHEELS]
        self.body_ids = torch.tensor(self.ids, dtype=torch.long, device=env.device)
        self.sdk_source = Path(inspect.getsourcefile(type(self.robot))).resolve()
        assert sha(self.sdk_source) == "9cc03b85642c36c801ff9683e94b8ccc3fbef1178761338974b433dacc78ef75"
        # The pinned SDK setter calls bool(forces.any()), which synchronizes
        # CUDA on every substep. Fill exactly its audited buffers instead; the
        # original articulation writer still owns the native application.
        for name in ("_external_force_b", "_external_torque_b", "_external_wrench_positions_b"):
            buffer = getattr(self.robot, name)
            assert buffer.shape == (env.num_envs, self.robot.num_bodies, 3)
            assert buffer.device == torch.device(env.device) and buffer.dtype == torch.float32
            assert torch.count_nonzero(buffer) == 0, "Do not overwrite another wrench owner"
        self.robot.has_external_wrench = True
        self.robot.uses_external_wrench_positions = True
        self.robot._use_global_wrench_frame = True
        self.slots = [i for name in WHEELS for i in self.mapping[name]]
        self.sensors = tuple(env.scene.sensors[name] for name in SENSORS)
        self.original_materials = self.robot.root_physx_view.get_material_properties().clone()
        values = self.original_materials.clone()
        values[:, self.slots, :2] = 0.
        ids = torch.arange(self.n, dtype=torch.int32, device="cpu")
        self.robot.root_physx_view.set_material_properties(values, ids)
        torch.testing.assert_close(self.robot.root_physx_view.get_material_properties(), values, rtol=0, atol=0)
        self.mu = env.mjlab_recipe_mu.to(device=env.device, dtype=torch.float32)[:, None].clone()
        self.mu_original = env.mjlab_recipe_mu.clone()
        self.normal = torch.zeros((self.n, 2, 3), device=env.device)
        self.normal[..., 2] = 1.
        self.ready = torch.zeros(self.n, device=env.device, dtype=torch.bool)
        self.fault = torch.zeros((self.n, 2), device=env.device, dtype=torch.bool)
        self.minimum_power = torch.zeros((), device=env.device)
        self.maximum_power = torch.zeros((), device=env.device)
        self.maximum_capacity_excess = torch.zeros((), device=env.device)
        self.loaded_count = torch.zeros(2, device=env.device, dtype=torch.long)
        self.applied_count = torch.zeros(2, device=env.device, dtype=torch.long)
        self.maximum_force = torch.zeros((), device=env.device)
        self.calls, self.reset_calls, self.reset_worlds = 0, 0, 0
        self.last_counter = int(env._sim_step_counter)
        self.previous_write = env.scene.__dict__.get("write_data_to_sim")
        self.original_write = env.scene.write_data_to_sim
        self.previous_reset = env.scene.__dict__.get("reset")
        self.original_reset = env.scene.reset
        env.scene.write_data_to_sim = self.write
        env.scene.reset = self.reset
        setattr(env, MARKER, self)
        self.audit_materials(check_usd=True)

    def reset(self, env_ids=None):
        result = self.original_reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.ready[ids] = False
        # Native asset reset clears its external-wrench buffers. Clear only
        # our temporal validity; never write root/joint state or another world.
        self.reset_calls += 1
        self.reset_worlds += self.n if env_ids is None else len(env_ids)
        return result

    def write(self, *args, **kwargs):
        if not self.owner.inside_step:
            return self.original_write(*args, **kwargs)
        counter = int(self.env._sim_step_counter)
        if counter == self.last_counter:
            # A reset/forward flush is not another physical application.
            return self.original_write(*args, **kwargs)
        assert counter == self.last_counter + 1, "Missing or duplicated physical tire application"
        force = torch.stack([s.data.force_matrix_w[:, 0, 0] for s in self.sensors], dim=1)
        points = torch.stack([s.data.contact_pos_w[:, 0, 0] for s in self.sensors], dim=1)
        assert force.shape == points.shape == (self.n, 2, 3)
        assert force.device == points.device == self.normal.device
        measured_load = force[..., 2]
        self.fault.logical_or_((~torch.isfinite(measured_load)) | (measured_load < -1e-4))
        load = torch.where(self.ready[:, None], measured_load.clamp_min(0), 0.)
        data = self.robot.data
        pos = data.body_com_pos_w[:, self.ids]
        value = tangent_wrench_torch(data.body_com_lin_vel_w[:, self.ids],
            data.body_com_ang_vel_w[:, self.ids], points-pos,
            rotate_y(data.body_link_quat_w[:, self.ids]), self.normal, load, self.mu,
            nonlinear=self.nonlinear)
        self.fault.logical_or_(value["invalid_measurement"])
        F, T = value["force_w_n"], value["torque_about_COM_w_nm"]
        magnitude = torch.linalg.vector_norm(F, dim=-1)
        self.maximum_capacity_excess.copy_(torch.maximum(self.maximum_capacity_excess,
                                                         (magnitude-self.mu*load).amax()))
        self.maximum_force.copy_(torch.maximum(self.maximum_force, magnitude.amax()))
        power = value["contact_power_w"]
        self.minimum_power.copy_(torch.minimum(self.minimum_power, power.amin()))
        self.maximum_power.copy_(torch.maximum(self.maximum_power, power.amax()))
        self.loaded_count.add_((load > 0).sum(0))
        self.applied_count.add_((magnitude > 0).sum(0))
        self.robot._external_force_b.index_copy_(1, self.body_ids, F)
        self.robot._external_torque_b.index_copy_(1, self.body_ids, T)
        self.robot._external_wrench_positions_b.index_copy_(1, self.body_ids, pos)
        self.calls += 1
        self.last_counter = counter
        self.ready.fill_(True)
        return self.original_write(*args, **kwargs)

    def audit_materials(self, *, check_usd=True):
        """Read back the actual zero materials; never impersonate native mu DR."""
        from physx_mjlab_recipe import _audit_bound_materials
        values = self.robot.root_physx_view.get_material_properties().cpu()
        expected = self.original_materials.cpu().clone(); expected[:, self.slots, :2] = 0.
        torch.testing.assert_close(values, expected, rtol=0, atol=0)
        assert torch.equal(self.env.mjlab_recipe_mu, self.mu_original)
        torch.testing.assert_close(self.mu[:, 0].cpu(), self.mu_original, rtol=0, atol=0)
        assert torch.isfinite(values).all() and torch.isfinite(self.mu).all()
        manifest = copy.deepcopy(self.env.mjlab_recipe_manifest)
        lower, upper = manifest["wheel_contact_friction"]["range"]
        assert torch.all((self.mu >= lower-1e-6) & (self.mu <= upper+1e-6))
        action = self.env.action_manager.get_term("joint_pos")
        command = self.env.command_manager.get_term("motion")
        nominal = self.robot.data.default_joint_pos[:, command.leg_joint_indices]
        bias = self.env.mjlab_recipe_encoder_bias
        torch.testing.assert_close(action._offset, nominal-bias, rtol=0, atol=1e-7)
        assert torch.isfinite(bias).all() and torch.all(bias.abs() <= .010001)
        manifest["wheel_contact_friction"].update(
            implementation="external_combined_slip" if self.nonlinear else "external_regularized_coulomb",
            physical_native_static_dynamic_mu=0.,
            sampled_mu_applies_to="external_tangential_force_capacity_mu_times_preceding_Fz",
            native_material_DR_draw_retained_as_external_mu=True,
            native_static_adhesion_modeled=False)
        result = dict(recipe=manifest, wheel_mu_range=[float(self.mu.min()), float(self.mu.max())],
            wheel_shape_slots=self.slots, all_wheel_shapes_share_one_draw_per_environment=True,
            encoder_bias_range_rad=[float(bias.min()), float(bias.max())],
            position_target_bias_sign_verified=True, live_physx_materials_verified=True,
            ground_material_usd_verified=False, native_wheel_static_dynamic_mu=0.,
            external_mu_per_environment_shared_between_wheels=True,
            nonwheel_materials_and_restitution_unchanged=True)
        if check_usd:
            result.update(_audit_bound_materials(self.env))
            result["ground_material_usd_verified"] = True
        return result

    def audit(self):
        fault = bool(self.fault.any().item())
        result = dict(kind="batched_PhysX_external_tire_runtime", device=str(self.normal.device),
            num_envs=self.n, physical_writes=self.calls, environment_physical_writes=self.calls*self.n,
            reset_calls=self.reset_calls, reset_worlds=self.reset_worlds,
            loaded_wheel_samples_L_R=self.loaded_count.cpu().tolist(),
            nonzero_force_samples_L_R=self.applied_count.cpu().tolist(),
            maximum_force_n=float(self.maximum_force), contact_power_min_w=float(self.minimum_power),
            contact_power_max_w=float(self.maximum_power),
            capacity_excess_max_n=float(self.maximum_capacity_excess), invalid_measurement=fault,
            normal_load_delay_physical_steps=1, physics_dt_s=.0025,
            force_law="combined_slip_magic_formula" if self.nonlinear else "external_coulomb_regularized",
            parameters=dict(B=4.5, C=1.8, E=1., speed_regularization_m_s=.2,
                            coulomb_regularization_m_s=.02),
            per_physical_write_host_tensor_transfers=0,
            reset_first_physical_step_has_zero_external_force=True,
            original_step_update_reward_owner_retained=True,
            root_joint_state_or_action_writes=0,
            native_material_readback=self.audit_materials(),
            native_wrench_writer_source_sha256={str(self.sdk_source):sha(self.sdk_source)},
            SDK_tensor_any_host_synchronization_bypassed=True,
            source_sha256={str(Path(p).resolve()):sha(p)
                for p in (Path(__file__),Path(__file__).with_name("nonlinear_tire_torch.py"))})
        assert not fault, "Loaded tire measurement/axis/DR health failed"
        assert result["contact_power_max_w"] <= .001, "External tire injected contact energy"
        assert result["capacity_excess_max_n"] <= .001, "External force exceeded used-load Coulomb bound"
        return result
