"""Four separate fitted geometric snapshots,not an animated reference or rollout."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','osmesa')
import numpy as np
import generate_camber_reference as base


def run(a):
    import mujoco
    from PIL import Image,ImageDraw,ImageFont
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    fit=json.loads((a.fit/'analysis.json').read_text())
    assert base.sha(a.model)==fit['model_sha256']==base.MODEL_SHA
    model=mujoco.MjModel.from_xml_path(str(a.model));data=mujoco.MjData(model)
    option=mujoco.MjvOption();option.geomgroup[0]=0
    # Only the floor changes visualization group;all physical arrays stay intact.
    model.geom_group[model.geom('floor').id]=1
    renderer=mujoco.Renderer(model,height=360,width=640)
    font=ImageFont.load_default(size=14);canvas=Image.new('RGB',(2560,520),'white');camera_rows=[]
    try:
        for i,row in enumerate(fit['rows']):
            best=row['best'];q=np.asarray(best['qpos']);data.qpos[:]=q
            mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_camlight(model,data)
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth=135.;camera.elevation=-20.;camera.distance=1.9;camera.lookat[:]=q[:3]+[0.,0.,-.25]
            renderer.update_scene(data,camera=camera,scene_option=option);image=Image.fromarray(renderer.render().copy())
            canvas.paste(image,(i*640,42));draw=ImageDraw.Draw(canvas)
            draw.text((i*640+8,4),f"GEOMETRIC SNAPSHOT ONLY | t={row['time_s']:.2f}s | beta={row['root_beta_deg']:.1f}deg",font=font,fill='black')
            p=best['posture_parameters'];camber=best['world_camber_deg_L_R'];heading=best['wheel_heading_offset_deg_L_R']
            lines=[f"Root roll/pitch {p[0]:.1f}/{p[1]:.1f}deg; height {p[4]:.3f}m (search variable)",
                f"Wheel camber L/R {camber[0]:.1f}/{camber[1]:.1f}deg",
                f"Rolling heading/body yaw L/R {heading[0]:.1f}/{heading[1]:.1f}deg",
                f"Conditional normalized wrench residual {best['weighted_base_wrench_residual']:.4f}",
                'No measured forces,physics rollout,policy or stable-support claim']
            for j,text in enumerate(lines):draw.text((i*640+8,410+20*j),text,font=font,fill='black')
            camera_rows.append(dict(frame=row['frame'],distance_m=1.9,azimuth_deg=135.,elevation_deg=-20.,lookat_offset_m=[0.,0.,-.25]))
            assert data.time==0.
    finally:renderer.close()
    a.output_dir.mkdir(parents=True);canvas.save(a.output_dir/'snapshots_4x1.png')
    record=dict(kind='four_separate_fitted_geometry_snapshots_not_reference_replay',source_sha256=base.sha(__file__),
        fit_sha256=base.sha(a.fit/'analysis.json'),model_sha256=base.sha(a.model),camera=camera_rows,
        robot_collision_visuals_hidden=True,floor_visualization_group_only_override=True,physical_geometry_or_inertia_changed=False,
        image_sha256=base.sha(a.output_dir/'snapshots_4x1.png'),native_physics_or_policy_steps=0,task_success_claimed=False)
    (a.output_dir/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ('fit','model','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
