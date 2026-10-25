"""Physical-step tangential tire-law substitution for the new PhysX study.

Native normal contact is retained. This is an explicit, one-sample-delayed
external wrench, not the built-in PhysX Vehicle SDK or an identified tire.
Native friction is read from contact patches, never inferred from articulated
body accelerations. No reward, action, pose, or observation is modified here.
"""
from __future__ import annotations

import numpy as np

from nonlinear_tire_contact import TireParameters, tangent_wrench


WHEELS = ("wheel_L_Link", "wheel_R_Link")
SENSORS = ("tire_L_ground", "tire_R_ground")
GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"


def configure_sensors(scene):
    from isaaclab.sensors import ContactSensorCfg
    for name, body in zip(SENSORS, WHEELS):
        assert getattr(scene, name, None) is None
        setattr(scene, name, ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/"+body,
            update_period=0., history_length=0, filter_prim_paths_expr=[GROUND],
            track_contact_points=True, max_contact_data_count_per_prim=128))


def numpy(value):
    return value.detach().cpu().numpy().copy()


def rotate_y(quaternion_wxyz):
    w,x,y,z=np.moveaxis(quaternion_wxyz,-1,0)
    return np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)


class TireRuntime:
    def __init__(self, env, *, law, mu):
        import torch
        from physx_mjlab_recipe import _shape_map
        assert env.num_envs==1 and env.physics_dt in (.0025,.00125) and env.step_dt==.02
        assert law in ("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula")
        self.env,self.robot,self.torch,self.law=env,env.scene["robot"],torch,law
        self.parameters=TireParameters(mu=mu)
        self.ids=[self.robot.body_names.index(n) for n in WHEELS]
        assert all(hasattr(self.robot.data,n) for n in
            ("body_com_pos_w","body_com_lin_vel_w","body_com_ang_vel_w","body_link_quat_w"))
        self.sensors=[env.scene.sensors[n] for n in SENSORS]
        self.mapping=_shape_map(self.robot)
        before=self.robot.root_physx_view.get_material_properties().clone()
        slots=[i for n in WHEELS for i in self.mapping[n]]
        assert torch.allclose(before[:,slots,:2],torch.full_like(before[:,slots,:2],mu),atol=1e-6,rtol=0)
        expected=before.clone()
        if law!="native_coulomb":
            expected[:,slots,:2]=0.
            self.robot.root_physx_view.set_material_properties(expected,torch.tensor([0],dtype=torch.int32))
        after=self.robot.root_physx_view.get_material_properties()
        assert torch.allclose(after,expected,atol=1e-6,rtol=0)
        self.material_audit={"before":numpy(before).tolist(),"after":numpy(after).tolist(),
            "wheel_shape_slots":slots,"nonwheel_materials_changed":False,
            "ground_material_changed":False,"native_wheel_tangential_friction_disabled":law!="native_coulomb"}
        self.rows=[]
        self.original_write=env.scene.write_data_to_sim
        env.scene.write_data_to_sim=self.write

    def write(self,*args,**kwargs):
        if int(self.env._sim_step_counter)==0:
            return self.original_write(*args,**kwargs)
        torch=self.torch;robot=self.robot
        pos=numpy(robot.data.body_com_pos_w[0,self.ids])
        vel=numpy(robot.data.body_com_lin_vel_w[0,self.ids])
        omega=numpy(robot.data.body_com_ang_vel_w[0,self.ids])
        q=numpy(robot.data.body_link_quat_w[0,self.ids])
        axle=rotate_y(q)
        Fn,points,frictions,counts,patch_norm_sums=[],[],[],[],[]
        for sensor in self.sensors:
            force=numpy(sensor.data.force_matrix_w)
            assert force.shape==(1,1,1,3) and np.isfinite(force).all()
            Fn.append(max(float(force[0,0,0,2]),0.))
            points.append(numpy(sensor.data.contact_pos_w)[0,0,0])
            f,p,c,s=sensor.contact_physx_view.get_friction_data(self.env.physics_dt)
            f,p,c,s=map(numpy,(f,p,c,s))
            assert c.shape==(1,1) and s.shape==(1,1)
            count,start=int(c[0,0]),int(s[0,0])
            assert 0<=count<=128 and 0<=start<=len(f) and start+count<=len(f)
            assert np.isfinite(f[start:start+count]).all()
            frictions.append(f[start:start+count].sum(axis=0))
            patch_norm_sums.append(np.linalg.norm(f[start:start+count,:2],axis=-1).sum())
            counts.append(count)
        Fn=np.asarray(Fn);points=np.asarray(points)
        F,T,slip,alpha,power=np.zeros((2,3)),np.zeros((2,3)),np.full((2,2),np.nan),np.full(2,np.nan),np.zeros(2)
        for k in range(2):
            if Fn[k]<=0:continue
            assert np.isfinite(points[k]).all(),"Loaded wheel lacks contact position"
            value=tangent_wrench(vel[k],omega[k],points[k]-pos[k],axle[k],[0,0,1],Fn[k],
                law=self.law if self.law!="native_coulomb" else "external_coulomb_regularized",parameters=self.parameters)
            slip[k]=value["normalized_contact_slip"];alpha[k]=value["hub_side_angle_rad"]
            if self.law!="native_coulomb":
                F[k],T[k],power[k]=value["force_w_n"],value["torque_about_COM_w_nm"],value["contact_power_w"]
        if self.law!="native_coulomb":
            native=np.asarray(frictions)
            # The first buffer precedes the material substitution and has not
            # yet seen any physical step with the new zero-friction material.
            if self.rows and np.linalg.norm(native)>=1e-3:
                import json
                evidence={"error":"native_tangent_force_after_substitution", "counter":int(self.env._sim_step_counter),
                    "friction_force_w_n":native.tolist(),"normal_load_n":Fn.tolist(),
                    "live_material_properties":numpy(robot.root_physx_view.get_material_properties()).tolist()}
                raise RuntimeError(json.dumps(evidence))
            robot.set_external_force_and_torque(torch.as_tensor(F[None],device=self.env.device,dtype=torch.float32),
                torch.as_tensor(T[None],device=self.env.device,dtype=torch.float32),body_ids=self.ids,
                positions=torch.as_tensor(pos[None],device=self.env.device,dtype=torch.float32),is_global=True)
        self.rows.append({"sim_counter":int(self.env._sim_step_counter),"COM_position_w_m":pos,
            "COM_velocity_w_m_s":vel,"angular_velocity_w_rad_s":omega,"wheel_quaternion_wxyz":q,
            "preceding_normal_load_n":Fn,"preceding_contact_position_w_m":points,
            "native_friction_force_w_n":np.asarray(frictions),"native_patch_count":np.asarray(counts),
            "native_sum_patch_tangent_force_norm_n":np.asarray(patch_norm_sums),
            "applied_tangent_force_w_n":F,"applied_tangent_moment_COM_w_nm":T,
            "normalized_contact_slip":slip,"hub_side_angle_rad":alpha,"external_contact_power_w":power})
        return self.original_write(*args,**kwargs)

    def save(self,path):
        expected=349*round(self.env.step_dt/self.env.physics_dt)
        assert len(self.rows)==expected,"Expected exactly all physical writes for349 actions"
        arrays={key:np.asarray([r[key] for r in self.rows]) for key in self.rows[0]}
        counters=arrays["sim_counter"]
        assert np.array_equal(counters,np.arange(counters[0],counters[0]+expected))
        arrays["time_before_step_s"]=np.arange(expected)*self.env.physics_dt
        np.savez_compressed(path,**arrays)
        return {"physical_steps":expected,"normal_load_delay_physical_steps":0 if self.law=="native_coulomb" else 1,
            "normal_forces_are_not_total_contact_forces":True,"native_friction_source":"RigidContactView.get_friction_data(dt), pair patch vectors summed in world frame",
            "native_patch_sum_is_not_pointwise_friction_cone_identification":True,
            "external_law_static_adhesion_modeled":False,"external_law_tire_parameters_identified":False,
            "material_audit":self.material_audit,"law":self.law,"mu":self.parameters.mu,
            "no_root_state_or_action_or_reward_mutation":True}

