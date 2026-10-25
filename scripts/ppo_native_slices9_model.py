"""Pre-clone geometry substitution and all-environment native model audits.

This module changes wheel collision representation only. Frozen K arithmetic,
motor/observation contracts, randomization and simulation clocks stay in the
parent trainer. Runtime checks inspect every physical wheel, not just env0.
"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from tron1_wheel_geometry import WHEELS,apply_wheel_geometry

COHORT_ID="tron1_ppo_k_native_slices9_400hz_v1"
MODEL_ID="uniform_axial_slices9_native_coulomb_offsets1mm_v1"
BASELINE_REPORT_SHA="29499d653e272b74aadb491be2f41c940b5232674ffb7e5f1c3a4dbc333d9130"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_spawn(cfg,context):
    from isaaclab.sim.utils import clone
    import carb
    from omni.physx.bindings._physx import SETTING_COLLISION_APPROXIMATE_CYLINDERS
    original=cfg.scene.robot.spawn.func
    # Installed clone uses functools.wraps; insert inside its single-prototype
    # spawn, before it copies USD/physics into the other environments.
    assert hasattr(original,"__wrapped__"),"Require the audited Isaac Lab clone decorator"
    bare=original.__wrapped__
    assert bare.__name__=="spawn_from_usd"
    assert carb.settings.get_settings().get(SETTING_COLLISION_APPROXIMATE_CYLINDERS) is False
    context["spawn_calls"]=0
    context["num_envs"]=cfg.scene.num_envs
    context["spawn_source_sha256"]={str(Path(inspect.getsourcefile(f)).resolve()):sha(inspect.getsourcefile(f))
                                      for f in (bare,clone)}

    @clone
    def spawn(path,spawn_cfg,*args,**kwargs):
        assert context["spawn_calls"]==0,"Expected one prototype before cloning"
        prim=bare(path,spawn_cfg,*args,**kwargs)
        context["prototype_geometry"]=apply_wheel_geometry(prim.GetStage(),str(prim.GetPath()),
            level="slices9",contact_offset_m=.001,rest_offset_m=0.)
        context["spawn_calls"]+=1
        context["geometry_applied_before_clone"]=True
        return prim
    cfg.scene.robot.spawn.func=spawn


def native_audit(env,context,baseline_report):
    from physx_mjlab_recipe import _shape_map,audit_physx_mjlab_recipe
    assert context["spawn_calls"]==1 and context["geometry_applied_before_clone"]
    assert all(sha(path)==digest for path,digest in context["spawn_source_sha256"].items()),"SDK spawn source changed"
    assert env.num_envs==context["num_envs"] and env.physics_dt==.0025 and env.step_dt==.02
    assert sha(baseline_report)==BASELINE_REPORT_SHA
    reference=json.loads(Path(baseline_report).read_text())["contact_material_audit"]["native_plant_readback"]
    robot=env.scene["robot"];view=robot.root_physx_view
    assert list(robot.body_names)==reference["body_names"] and list(robot.joint_names)==reference["joint_names"]
    assert len(robot.joint_names)==8 and view.max_shapes==25
    plain=lambda value:value.detach().cpu().numpy().copy()
    values={name:plain(getattr(view,name)()) for name in
            ("get_masses","get_inertias","get_coms","get_dof_armatures","get_dof_dampings")}
    for name in values:
        assert np.isfinite(values[name]).all()
        original=np.asarray(reference["readbacks"][name]["values"],dtype=float)
        assert values[name].shape[0]==env.num_envs
        if name!="get_coms":
            np.testing.assert_allclose(values[name],np.broadcast_to(original,values[name].shape),rtol=0,atol=1e-7)
    com=values["get_coms"];expected=np.asarray(reference["readbacks"]["get_coms"]["values"])
    base_id=robot.body_names.index("base_Link");other=[i for i in range(len(robot.body_names)) if i!=base_id]
    np.testing.assert_allclose(com[:,other],np.broadcast_to(expected[:,other],com[:,other].shape),rtol=0,atol=1e-7)
    np.testing.assert_allclose(com[:,base_id,3:],np.broadcast_to(expected[:,base_id,3:],com[:,base_id,3:].shape),rtol=0,atol=1e-7)
    delta=com[:,base_id,:3]-expected[:,base_id,:3]
    assert np.all(np.abs(delta)<=np.array([.025,.05,.05])+1e-6),"Base COM moved outside original DR"
    mapping=_shape_map(robot)
    assert all(len(mapping[name])==9 for name in WHEELS)
    link_paths=view.link_paths;assert len(link_paths)==env.num_envs
    ids=[robot.body_names.index(name) for name in WHEELS]
    counts=[];paths=[]
    for links in link_paths:
        assert len(links)==len(robot.body_names)
        row=[]
        for index in ids:
            single=robot._physics_sim_view.create_rigid_body_view(links[index])
            assert single.count==1 and single.max_shapes==9,"Every native wheel must have9 shapes"
            row.append(single.max_shapes);paths.append(links[index])
        counts.append(row)
    assert len(set(paths))==2*env.num_envs,"Cloned environments do not have unique wheel bodies"
    contact=plain(view.get_contact_offsets());rest=plain(view.get_rest_offsets())
    wheel_slots=[slot for name in WHEELS for slot in mapping[name]]
    assert contact.shape==rest.shape==(env.num_envs,25)
    np.testing.assert_allclose(contact[:,wheel_slots],.001,rtol=0,atol=1e-8)
    np.testing.assert_allclose(rest[:,wheel_slots],0.,rtol=0,atol=1e-8)
    material=audit_physx_mjlab_recipe(env,check_usd=True)
    assert material["recipe"]["training"] and material["wheel_mu_range"][0]>=.3-1e-6
    assert material["wheel_mu_range"][1]<=1.2+1e-6 and material["all_wheel_shapes_share_one_draw_per_environment"]
    assert len(material["wheel_shape_slots"])==18
    return dict(model_id=MODEL_ID,status="all_environments_native_model_verified",num_envs=env.num_envs,
        native_wheel_bodies_checked=len(paths),per_environment_wheel_shape_counts=counts,total_shapes_per_robot=25,
        geometry_applied_before_clone=True,prototype_spawn_calls=1,contact_offset_m=.001,rest_offset_m=0.,
        all_body_masses_inertias_armatures_dampings_match_authored=True,
        nonbase_COM_and_all_principal_frames_match_authored=True,base_COM_uses_original_DR_only=True,
        base_COM_delta_xyz_min=delta.min(axis=0).tolist(),base_COM_delta_xyz_max=delta.max(axis=0).tolist(),
        native_readback_sha256={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in values.items()},
        original_DR_material_audit=material,authored_reference_report_sha256=BASELINE_REPORT_SHA,
        prototype_geometry=context["prototype_geometry"],SDK_spawn_source_sha256=context["spawn_source_sha256"],
        extra_simulation_steps_or_rewards=False,original_asset_file_modified=False)
