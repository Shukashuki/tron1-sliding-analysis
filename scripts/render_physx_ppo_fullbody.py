"""Render the recorded mu=.15 PhysX PPO episode as opaque TRON1 meshes.

MuJoCo is a graphics/FK renderer only. No controller inference, integrator,
contact solver, pose correction, retargeting, or new physics is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from controller_trace import native_trace
from nmpc_tracking_objective import quaternion_rotate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml"
DEFAULT_RECORDING = ROOT / "outputs/policy_primary_grid_raw_20261016_v1/PPO/mu0p150/isaac/physics_trajectory.npz"
DEFAULT_REPORT = DEFAULT_RECORDING.with_name("report.json")
DEFAULT_PUBLIC_SUMMARY = ROOT / "results/policy_primary_friction_dynamics_20261016_v1/summary.json"
CAMERAS = ({"name": "Oblique follow camera", "azimuth_deg": 135., "elevation_deg": -24., "distance_m": 2.15},
           {"name": "Side follow camera", "azimuth_deg": 225., "elevation_deg": -8., "distance_m": 2.15})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_recording(recording=DEFAULT_RECORDING, report=DEFAULT_REPORT,
                   public_summary=DEFAULT_PUBLIC_SUMMARY):
    """Bind the original single-environment episode to published SHA identities."""
    source = read_json(public_summary)
    run = source["runs"]["PPO_mu0p150"]
    evaluation = read_json(report)
    require(sha256(recording) == run["input_sha256"]["isaac_trace"], "Original PhysX recording SHA mismatch")
    require(sha256(report) == run["input_sha256"]["isaac_report"], "Original PhysX report SHA mismatch")
    require(evaluation["checkpoint_sha256"] == run["training"]["checkpoint_sha256"], "PPO checkpoint mismatch")
    require(evaluation["simulator"] == "IsaacLab/PhysX" and
            evaluation["evaluation_contact_friction_override"]["evaluation_contact_friction"] == .15,
            "Require original PhysX mu=.15 evaluation")
    with np.load(recording, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    require(raw["valid_mask"].shape == (1397, 1) and raw["valid_mask"].all(),
            "Require all 1397 original physical samples, without reset splicing")
    trace = native_trace(raw, "policy_isaac")
    require(trace["time_s"].shape == (1397,) and
            np.allclose(trace["time_s"], np.arange(1397)*.005, atol=1e-9, rtol=0),
            "Require unchanged native 0--6.98 s physical clock")
    compact_path = Path(public_summary).parent / "policy_dynamics_trace.npz"
    require(sha256(compact_path) == source["artifacts"]["policy_dynamics_trace.npz"]["sha256"],
            "Published reference/physical replay archive SHA mismatch")
    with np.load(compact_path, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in ("reference_time_s", "reference_position")}
        require(np.array_equal(trace["position"], archive["PPO_mu0p150_isaac_position"]),
                "Recorded world positions differ from published replay")
    provenance = {
        "recording_sha256": sha256(recording), "report_sha256": sha256(report),
        "public_summary_sha256": sha256(public_summary), "public_compact_sha256": sha256(compact_path),
        "model_sha256_expected": source["input_sha256"]["model"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "training_manifest_sha256": run["training"]["manifest_sha256"],
        "task_profile_sha256": run["training"]["task_profile_sha256"],
        "core_metrics": run["windows"]["core"]["isaac"],
        "posture_diagnostics": run["posture_diagnostics"]["isaac"],
        "reference": reference}
    return trace, provenance


def build_qpos(model, trace):
    """Map canonical recorded joint names to MuJoCo coordinates, never raw qpos."""
    require(model.nq == 15 and model.nv == 14, "Require original floating-base TRON1")
    names = tuple(trace["joint_names"])
    require(len(names) == 8 and len(set(names)) == 8, "Require eight unique named joints")
    ids = np.array([int(model.joint(name).qposadr[0]) for name in names], dtype=np.int64)
    require(np.array_equal(np.sort(ids), np.arange(7, 15)), "Unexpected model named-joint mapping")
    require(int(model.joint("root").qposadr[0]) == 0, "Unexpected free-root mapping")
    require(np.allclose(np.linalg.norm(trace["quaternion"], axis=-1), 1., atol=1e-5, rtol=0),
            "Recorded root rotations must be unit wxyz quaternions")
    qpos = np.empty((len(trace["time_s"]), model.nq), dtype=np.float64)
    qpos[:, :3] = trace["position"]
    qpos[:, 3:7] = trace["quaternion"]
    qpos[:, ids] = trace["joint_pos"]
    require(np.isfinite(qpos).all(), "Nonfinite recorded poses")
    return qpos, ids


def _pose(model, data, qpos):
    import mujoco
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)


def validate_fk(model, trace, qpos, tolerance_m=2e-6):
    """Check every recorded wheel origin without invoking any dynamics solver."""
    import mujoco
    require(np.isfinite(tolerance_m) and 0 < tolerance_m <= 2e-6, "FK tolerance must be <=2 micrometers")
    require(qpos.shape == (len(trace["time_s"]), 15), "Unexpected replay qpos shape")
    data = mujoco.MjData(model)
    wheels = [int(model.body(name).id) for name in ("wheel_L_Link", "wheel_R_Link")]
    maximum = 0.
    for index, pose in enumerate(qpos):
        _pose(model, data, pose)
        error = np.linalg.norm(data.xpos[wheels] - trace["wheel_pos"][index], axis=1)
        maximum = max(maximum, float(error.max()))
    require(maximum < tolerance_m, "Named-joint wheel FK differs from recorded PhysX wheel positions")
    return {"samples_checked": len(qpos), "max_wheel_origin_error_m": maximum,
            "tolerance_m": tolerance_m, "physics_stepping": False,
            "kinematic_functions": ["mj_kinematics", "mj_comPos", "mj_camlight"]}


def create_runtime(model_path=DEFAULT_MODEL, width=640, height=480):
    """Create a headless graphics renderer; XML and meshes remain untouched."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco
    from PIL import ImageFont
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = max(width, int(model.vis.global_.offwidth))
    model.vis.global_.offheight = max(height, int(model.vis.global_.offheight))
    floor = int(model.geom("floor").id)
    model.geom_group[floor] = 2
    mesh_mask = model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
    model.geom_rgba[mesh_mask, 3] = 1.
    if model.nmat:
        model.mat_rgba[:, 3] = 1.
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[1:3] = 1
    option.sitegroup[:] = 0
    cameras = []
    for spec in CAMERAS:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = spec["azimuth_deg"]; camera.elevation = spec["elevation_deg"]
        camera.distance = spec["distance_m"]
        cameras.append(camera)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    return SimpleNamespace(model=model, data=mujoco.MjData(model),
        renderer=mujoco.Renderer(model, height=height, width=width),
        option=option, cameras=cameras, width=width, height=height, font=font, small=small)


def _camera_image(runtime, qpos, index, camera_index):
    from PIL import Image
    _pose(runtime.model, runtime.data, qpos[index])
    camera = runtime.cameras[camera_index]
    camera.lookat[:] = qpos[index, :3] + np.array([0., 0., -.24])
    runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
    return Image.fromarray(runtime.renderer.render().copy())


def render_frame(runtime, trace, qpos, index, *, reference=None):
    """Two views of the same original state; cameras move but the robot does not."""
    from PIL import Image, ImageDraw
    w, h = runtime.width, runtime.height
    image = Image.new("RGB", (2*w, h+142), "#fafafa")
    draw = ImageDraw.Draw(image)
    for side, spec in enumerate(CAMERAS):
        view = _camera_image(runtime, qpos, index, side)
        image.paste(view, (side*w, 0))
        draw.rectangle((side*w, 0, (side+1)*w, 29), fill="#f5f5f5")
        draw.text((side*w+10, 4), spec["name"], font=runtime.font, fill="#222222")
    t = float(trace["time_s"][index])
    quaternion = trace["quaternion"][index]
    up = quaternion_rotate(quaternion, np.array([0., 0., 1.]))
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1., 1.))))
    speed = float(np.linalg.norm(trace["com_velocity"][index, :2]))
    xy_text = ""
    if reference is not None:
        target = np.array([np.interp(t, reference["reference_time_s"], reference["reference_position"][:, j]) for j in range(2)])
        xy_text = f" | XY error {np.linalg.norm(trace['position'][index, :2]-target):.3f} m"
    draw.text((12, h+7), f"PhysX-trained PPO / PhysX mu=0.15 | t={t:.2f} s | COM speed {speed:.2f} m/s" + xy_text,
              font=runtime.font, fill="#111111")
    draw.text((12, h+34), f"Original base height {trace['position'][index, 2]:.3f} m | tilt {tilt:.2f} deg | CORE 1.44-4.00 s",
              font=runtime.font, fill="#111111")
    for side, name in enumerate(("Left", "Right")):
        count = int(trace["contact_count"][index, side])
        fz = float(trace["contact_force"][index, side, 2])
        eligible = count > 0 and fz > 5.
        draw.text((12+side*510, h+62), f"{name} wheel: recorded contacts {count} | Fz {fz:.1f} N | gate {'ON' if eligible else 'OFF'}",
                  font=runtime.small, fill="#225522" if eligible else "#666666")
    draw.text((12, h+87), "PhysX recorded poses; MuJoCo graphics/FK only. No resimulation, root correction, IK or new policy inference.",
              font=runtime.small, fill="#333333")
    draw.text((12, h+111), "Contact gate: recorded count > 0 AND upward force > 5 N. Contact/height/tilt are diagnostics, not failure labels.",
              font=runtime.small, fill="#333333")
    return image


def mesh_provenance(model_path):
    root = ET.parse(model_path).getroot()
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", "."))
    if not meshdir.is_absolute():
        meshdir = Path(model_path).parent / meshdir
    meshes = [{"file": mesh.get("file"), "sha256": sha256(meshdir / mesh.get("file"))}
              for mesh in root.findall("./asset/mesh")]
    source_path = meshdir.parent / "SOURCE.json"
    source = read_json(source_path) if source_path.exists() else None
    if source:
        expected = {Path(item["destination"]).name: item["sha256"] for item in source["files"]}
        require(all(expected[item["file"]] == item["sha256"] for item in meshes), "Official mesh provenance differs")
    return {"meshes": meshes, "official_source": source, "meshes_copied_to_output": False}


def run(args):
    from PIL import Image, ImageDraw
    output = Path(args.output_dir)
    require(not output.exists(), "Preserve existing output: choose a fresh directory")
    trace, provenance = load_recording(args.recording, args.report, args.public_summary)
    require(sha256(args.model) == provenance["model_sha256_expected"], "Audited display model SHA mismatch")
    runtime = create_runtime(args.model)
    try:
        qpos, qids = build_qpos(runtime.model, trace)
        fk = validate_fk(runtime.model, trace, qpos)
        meshes = mesh_provenance(args.model)
        output.mkdir(parents=True)
        np.savez_compressed(output / "replay_states.npz", time_s=trace["time_s"], qpos=qpos,
            joint_names=np.array(trace["joint_names"]), named_joint_qpos_addresses=qids,
            wheel_ground_contact_count=trace["contact_count"], wheel_ground_contact_force_w_n=trace["contact_force"],
            original_root_position_m=trace["position"], original_root_quaternion_wxyz=trace["quaternion"])
        reference = provenance.pop("reference")
        snapshot_index = int(np.argmin(np.abs(trace["time_s"]-2.4)))
        render_frame(runtime, trace, qpos, snapshot_index, reference=reference).save(output / "fullbody.png")
        keytimes = [1.44, 2.4, 3.36, 4.]
        panel = Image.new("RGB", (4*512, 420), "white")
        draw = ImageDraw.Draw(panel)
        for column, t in enumerate(keytimes):
            index = int(np.argmin(np.abs(trace["time_s"]-t)))
            frame = _camera_image(runtime, qpos, index, 0).resize((512, 384))
            panel.paste(frame, (column*512, 30))
            draw.text((column*512+10, 5), f"Recorded PhysX t={trace['time_s'][index]:.2f} s", font=runtime.font, fill="black")
        panel.save(output / "keyframes.png")
        files = ["replay_states.npz", "fullbody.png", "keyframes.png"]
        if not args.skip_video:
            import imageio_ffmpeg
            stream = imageio_ffmpeg.write_frames(str(output / "fullbody.mp4"),
                (runtime.width*2, runtime.height+142), fps=50, codec="libx264", quality=7,
                macro_block_size=2, output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            stream.send(None)
            try:
                for index in range(0, len(qpos), 4):
                    stream.send(np.asarray(render_frame(runtime, trace, qpos, index, reference=reference)))
            finally:
                stream.close()
            files.append("fullbody.mp4")
        import mujoco
        summary = {"schema_version": 1, "kind": "recorded_physx_ppo_mu015_opaque_fullbody_mesh_replay",
            "training_engine": "PhysX", "evaluation_engine": "PhysX", "graphics_engine": "MuJoCo",
            "graphics_version": mujoco.__version__, "graphics_backend": os.environ.get("MUJOCO_GL"),
            "mu": .15, "reference_id": "drift_drl_traj4", **provenance,
            "model_sha256": sha256(args.model), "analysis_source_sha256": sha256(__file__),
            "named_joint_order": list(trace["joint_names"]), "named_joint_qpos_addresses": qids.tolist(),
            "raw_qpos_array_copied": False, "fk_validation": fk, "mesh_provenance": meshes,
            "camera_settings": list(CAMERAS), "camera_lookat_offset_from_recorded_root_m": [0., 0., -.24],
            "frame_size_pixels": [1280, 622], "video_fps": 50,
            "video_frames": 0 if args.skip_video else 350, "native_sample_stride": 4,
            "first_native_time_s": 0., "last_native_time_s": 6.98,
            "native_samples_preserved_in_states": 1397, "keyframe_times_s": keytimes,
            "robot_mesh_alpha": 1., "camera_follows_only": True, "root_height_corrected": False,
            "dynamics_stepping": False, "contact_solver_called": False,
            "new_policy_inference": False, "new_training": False, "retargeting": False,
            "height_or_flight_failure_gate": False, "tilt_failure_gate": False,
            "contact_source": "Original recorded PhysX ground-filtered counts and upward force; never renderer contact data",
            "limitations": ["Graphics/FK replay is not a new MuJoCo dynamics validation.",
                "One historical PhysX checkpoint/episode at mu=.15, not a statistical success rate.",
                "Absolute height, tilt and contact status are displayed without altering original poses or XY failure criteria."],
            "artifacts": {name: {"file": name, "sha256": sha256(output/name)} for name in files}}
        (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        return summary
    finally:
        runtime.renderer.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--public-summary", type=Path, default=DEFAULT_PUBLIC_SUMMARY)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--skip-video", action="store_true")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": "rendered_existing_physx_episode", "dynamics_stepping": False,
                      "max_wheel_origin_error_m": result["fk_validation"]["max_wheel_origin_error_m"]}))
