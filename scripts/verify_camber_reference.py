"""Independent FK/material-velocity and frame audit for the camber artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import generate_camber_reference as reference


def run(a):
    import mujoco
    import imageio_ffmpeg
    root=a.result;target=root/a.output_name
    assert not target.exists()
    folder=root/"independent"
    proof=json.loads((folder/"generation.json").read_text())
    render=json.loads((root/"render/provenance.json").read_text())
    assert proof["generator_sha256"]==reference.sha(Path(__file__).parent/"independent_camber_reference.py")
    assert proof["forward_generator_sha256"]==reference.sha(reference.__file__)
    assert proof["config_sha256"]==reference.sha(a.config)
    assert proof["model_sha256"]==reference.sha(a.model)
    assert proof["reference_sha256"]==reference.sha(a.reference)
    assert render["generation_sha256"]==reference.sha(folder/"generation.json")
    assert render["renderer_sha256"]==reference.sha(Path(__file__).parent/"render_camber_reference.py")
    for n,h in render["artifacts"].items():assert reference.sha(root/"render"/n)==h
    with np.load(a.reference,allow_pickle=False) as f:original={k:f[k].copy() for k in f.files}
    m=mujoco.MjModel.from_xml_path(str(a.model));d=mujoco.MjData(m)
    wheels=[m.body(n).id for n in reference.WHEELS]
    geoms=[m.geom("wheel_"+s+"_collision").id for s in ("L","R")]
    rows=[]
    with np.load(root/"render/camera_plan.npz",allow_pickle=False) as camera:
        assert float(camera["distance_m"])==render["fixed_distance_m"]
        for row in proof["rows"]:
            path=folder/(row["id"]+".npz");assert reference.sha(path)==row["reference_sha256"]
            with np.load(path,allow_pickle=False) as f:out={k:f[k].copy() for k in f.files}
            q,v=out["qpos_mujoco"],out["qvel_mujoco"]
            np.testing.assert_array_equal(q[:,:2],original["qpos_mujoco"][:,:2])
            np.testing.assert_array_equal(q[:,3:7],original["qpos_mujoco"][:,3:7])
            np.testing.assert_allclose(v,reference.differentiate(m,q,.02),atol=1e-10,rtol=0)
            np.testing.assert_allclose(v[:,:2],original["qvel_mujoco"][:,:2],atol=1e-10,rtol=0)
            delta=camera[row["id"]+"__lookat"]-q[:,:3]
            np.testing.assert_allclose(delta,np.broadcast_to(delta[:1],delta.shape),atol=1e-12,rtol=0)
            body_names=out["body_names"].astype(str).tolist()
            leg_ids=[j for j,n in enumerate(out["joint_names"].astype(str)) if n not in reference.SPINS]
            assert len(leg_ids)==6
            errors=[];expected_cam=[]
            for i in range(len(q)):
                d.qpos[:]=q[i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
                for side,b in enumerate(wheels):
                    bi=body_names.index(reference.WHEELS[side]);g=geoms[side]
                    axis=d.xmat[b].reshape(3,3)[:,1]
                    expected_cam.append(abs(np.arcsin(axis[2])-out["camber_command_rad"][i,side]))
                    np.testing.assert_allclose(d.xpos[b],out["body_pos_w"][i,bi],atol=1e-10,rtol=0)
                    np.testing.assert_allclose(d.geom_xpos[g],out["wheel_center_w"][i,side],atol=1e-10,rtol=0)
                    yaw=float(original["root_yaw_rad"][i])
                    forward_xy=np.array([np.cos(yaw),np.sin(yaw)])
                    # Original collider centers differ from link origins only
                    # along localY,so their body-longitudinal coordinates agree.
                    reference_x=(original["body_pos_w"][i,bi,:2]-original["qpos_mujoco"][i,:2])@forward_xy
                    generated_x=(d.geom_xpos[g,:2]-q[i,:2])@forward_xy
                    assert abs(reference_x-generated_x)<2e-6
                    # Independent support-value equation,not the generator's point routine.
                    low=d.geom_xpos[g,2]-m.geom_size[g,0]*np.sqrt(1-axis[2]**2)-m.geom_size[g,1]*abs(axis[2])
                    assert abs(low)<1e-7
                    p=out["wheel_support_point_w"][i,side]
                    material=out["body_lin_vel_w"][i,bi]+np.cross(out["body_ang_vel_w"][i,bi],p-d.xipos[b])
                    errors.append(float(np.max(abs(material-out["wheel_point_velocity_w_m_s"][i,side]))))
                    lateral=axis-np.array([0.,0.,axis[2]]);lateral/=np.linalg.norm(lateral)
                    np.testing.assert_allclose(material@lateral,out["wheel_point_lateral_velocity_m_s"][i,side],atol=1e-10,rtol=0)
                assert d.time==0.
            assert max(errors)<1e-10 and max(expected_cam)<1e-10
            switches={}
            for side,name in enumerate(("left","right")):
                active=abs(out["wheel_camber_rad"][:,side])>1e-9
                signed=np.sign(out["wheel_camber_rad"][:,side])
                selected=(signed[1:]!=signed[:-1])&active[1:]&active[:-1]
                switches[name]=out["time_s"][1:][selected].tolist()
            rows.append(dict(id=row["id"],frames=349,max_COM_cross_product_point_velocity_error_m_s=max(errors),
                max_camber_FK_error_rad=max(expected_cam),both_geometric_supports_checked=True,
                max_abs_material_point_normal_velocity_m_s=float(abs(out["wheel_point_velocity_w_m_s"][:,:,2]).max()),
                leg_max_abs_finite_difference_acceleration_rad_s2=float(abs(np.gradient(out["joint_vel"][:,leg_ids],.02,axis=0)).max()),
                rigid_cylinder_rim_branch_switches_between_nonzero_camber_samples_s=switches,
                discontinuous_point_selection_is_not_measured_PhysX_contact_switch=True))
    stream=imageio_ffmpeg.read_frames(str(root/"render/camber_reference_4x1.mp4"),pix_fmt="rgb24")
    try:
        metadata=next(stream);assert metadata["size"]==(2560,820) and metadata["fps"]==50
        count=0
        for f in stream:assert len(f)==2560*820*3;count+=1
        assert count==349
    finally:stream.close()
    # The two fitted references must preserve separate left/right errors.
    for f in proof["fits"]:
        row=next(r for r in proof["rows"] if r["id"]==f["id"])
        assert len(f["core_target_RMSE_m_s_L_R"])==2 and not f["optimization_convergence_is_target_or_dynamic_success"]
    result=dict(status="verified_kinematic_reference_artifacts",rows=rows,frames_decoded=count,
        camera_translation_only_fixed_distance_verified=True,source_and_artifact_hashes_verified=True,
        generator_did_not_integrate_physics=True,contact_force_or_dynamic_feasibility_verified=False,
        source_sha256=reference.sha(__file__),generation_sha256=reference.sha(folder/"generation.json"),
        render_provenance_sha256=reference.sha(root/"render/provenance.json"),task_success_claimed=False)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("result","reference","model","config"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--output-name",choices=("verification_v1.json","verification_v2.json"),default="verification_v1.json")
    run(p.parse_args())
