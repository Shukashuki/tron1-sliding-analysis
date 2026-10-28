"""Render four recorded PhysX motions with force-independent contact labels.

Produces a four-body comparison, an 8x slow physical-step excerpt, and the
reference/body/XY/contact layout for each motion. Uses mesh graphics and FK;
never steps physics, smooths poses, corrects height or changes camera distance.
"""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from evaluate_hockey_physx import load, sha
from render_physx_ppo_fullbody import build_qpos, create_runtime, _pose, mesh_provenance

W, H = 640, 720
COLORS = ((35, 105, 185), (205, 105, 30))  # RGB
CELLS = ("beyondmimic_s100_seed42", "hockey_s100_seed42", "hockey_s150_seed42", "hockey_s200_seed42")
LABELS = ("Earlier tracking PPO | 1x", "Hockey PPO | 1x", "Hockey PPO | 1.5x", "Hockey PPO | 2x")


def text(image, value, y, x=16, scale=.51, color=(35, 35, 35)):
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def blank():
    return np.full((H, W, 3), 249, np.uint8)


def writer(ffmpeg, path):
    return subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{4*W}x{H}", "-r", "50", "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def close_writer(process):
    process.stdin.close()
    if process.wait():
        raise RuntimeError("Video encoding failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--aligned-audit", type=Path,
                   help="Use independently audited pre-step contact geometry/velocity and matching poses")
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ffmpeg")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.ffmpeg is None:
        import imageio_ffmpeg
        a.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    report = json.loads((a.analysis / "report.json").read_text())
    if report["kind"] != "hockey_ground_contact_evaluation_v2":
        raise ValueError("Require contact-first analysis")
    if sha(Path(__file__).with_name("analyze_hockey_ground_contact.py")) != report["source_sha256"]:
        raise ValueError("Contact analyzer differs from recorded analysis")
    tolerance = report["config"]["contact_separation_tolerance_m"]
    archive = load(a.analysis / "contact_signals.npz")
    aligned_report, aligned_archive = None, None
    if a.aligned_audit is not None:
        aligned_report = json.loads((a.aligned_audit / "report.json").read_text())
        assert aligned_report["kind"] == "hockey_contact_timing_audit_v3"
        assert sha(a.aligned_audit / "aligned_signals.npz") == aligned_report["aligned_signals_sha256"]
        aligned_archive = load(a.aligned_audit / "aligned_signals.npz")
    runtime = create_runtime(a.model, W, 420)
    import mujoco
    camera = runtime.cameras[0]
    camera.distance, camera.azimuth, camera.elevation = 2.4, 135., -18.
    cases, xy = [], []
    for name, label in zip(CELLS, LABELS):
        row = next(r for r in report["cells"] if r["cell"] == name)
        cell = a.runs / name
        run = cell / "eval-suite/nominal"
        assert sha(run / "report.json") == row["provenance"]["run_report_sha256"]
        for filename, digest in row["provenance"]["recorded_artifacts"].items():
            assert sha(run / filename) == digest
        tr = load(run / "final_rollout.npz")
        native = load(run / "native_contact_buffers.npz")
        s = load(cell / "analyze/nominal.signals.npz")
        touch = {k.split("__", 1)[1]: v for k, v in archive.items() if k.startswith(name + "__")}
        refdir = a.references / name.split("_")[1]
        refmeta = json.loads((refdir / "report.json").read_text())
        runmeta = json.loads((run / "report.json").read_text())
        assert sha(a.model) == refmeta["model_sha256"]
        assert sha(refdir / "motion.npz") == runmeta["motion_sha256"]
        command = load(refdir / "command_diagnostics.npz")
        qpos, _ = build_qpos(runtime.model, dict(time_s=s["time_s"], joint_names=tr["joint_names"],
            position=tr["root_pos"], quaternion=tr["root_quat"], joint_pos=tr["joint_pos"]))
        ids = [tr["body_names"].astype(str).tolist().index("wheel_" + side + "_Link") for side in ("L", "R")]
        mjids = [int(runtime.model.body("wheel_" + side + "_Link").id) for side in ("L", "R")]
        maximum = 0.
        for i, pose in enumerate(qpos):
            _pose(runtime.model, runtime.data, pose)
            maximum = max(maximum, float(np.linalg.norm(runtime.data.xpos[mjids] - tr["body_pos"][i, ids], axis=-1).max()))
        if maximum > 2e-5:
            raise ValueError(f"Wheel FK differs for {name}: {maximum}")
        if aligned_archive is not None:
            audited = next(r for r in aligned_report["cells"] if r["cell"] == name)
            assert audited["provenance"] == row["provenance"]
            touch = {k.split("__", 1)[1]: v for k, v in aligned_archive.items() if k.startswith(name + "__")}
            touch["friction_utilization"] = touch["rho"]
            touch["sliding"] = touch["touching"] & (touch["tangent_rms_m_s"] > .05)
            touch["time_s"] = touch["time_start_s"]
            s = {key: np.concatenate((v[:1], v[:-1]), axis=0) for key, v in s.items()}
            s["time_s"] = touch["time_start_s"]
            qpos = np.concatenate((qpos[:1], qpos[:-1]), axis=0)
            tr["body_pos"] = np.concatenate((tr["body_pos"][:1], tr["body_pos"][:-1]), axis=0)
        cases.append(dict(name=name, label=label, tr=tr, raw=native, s=s, touch=touch, qpos=qpos,
                          command=command, refdir=refdir, ids=ids, row=row, fk_error=maximum))
        xy.extend([qpos[:, :2], command["qpos"][:, :2], tr["body_pos"][:, ids, :2].reshape(-1, 2)])
    xy = np.concatenate(xy)
    center = (xy.min(0) + xy.max(0)) / 2
    extent = max(float(np.ptp(xy, axis=0).max()*1.3), 2.)
    end = float(cases[0]["command"]["time_s"][-1])
    assert all(np.isclose(c["command"]["time_s"][-1], end) for c in cases)

    def project(points):
        v = (np.asarray(points) - center) * 450 / extent
        return np.stack((320+v[..., 0], 330-v[..., 1]), axis=-1).astype(int)

    def mesh(pose, c=None, i=0):
        _pose(runtime.model, runtime.data, pose)
        camera.lookat[:] = pose[:3] + [0., 0., -.24]
        runtime.renderer.update_scene(runtime.data, camera=camera, scene_option=runtime.option)
        if c is not None and i > 0:
            for side in (0, 1):
                raw = c["raw"]; row = i-1
                count = int(raw[f"w{side}_normal_buffer_4"][row].item())
                start = int(raw[f"w{side}_normal_buffer_5"][row].item())
                sl = slice(start, start+count)
                normals = raw[f"w{side}_normal_buffer_2"][row, sl]
                sep = raw[f"w{side}_normal_buffer_3"][row, sl, 0]
                valid = (np.linalg.norm(normals, axis=-1) >= 1e-8) & (sep <= tolerance)
                assert valid.sum() == c["touch"]["touching_point_count"][i, side]
                points = raw[f"w{side}_normal_buffer_1"][row, sl][valid] - c["tr"]["environment_origin_w"]
                for point in points:
                    scene = runtime.renderer.scene
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, .009), point,
                                       np.eye(3).ravel(), np.array([.1, .9, .25, 1.]))
                    scene.ngeom += 1
        return runtime.renderer.render().copy()

    for c in cases:
        timeline = blank()
        for side in (0, 1):
            r = c["touch"]; t = r["time_s"]
            for i in range(1, len(t)):
                if t[i] > end + 1e-9:
                    break
                color = (50, 160, 90) if r["touching"][i, side] else ((215, 160, 35) if r["contact_unknown"][i, side] else (205, 65, 65))
                begin, finish = ((r["time_start_s"][i], r["time_end_s"][i])
                                 if aligned_archive is not None else (t[i-1], t[i]))
                x0, x1 = int(45+max(begin, 0.)/end*575), int(45+min(finish, end)/end*575)
                cv2.rectangle(timeline, (x0, 604+side*16), (max(x0+1, x1), 614+side*16), color, -1)
        text(timeline, "L", 613); text(timeline, "R", 629)
        c["timeline"] = timeline

    def body_panel(c, i, slow=False):
        panel = blank(); now = c["s"]["time_s"][i]; r = c["touch"]
        panel[65:485] = mesh(c["qpos"][i], c, i)
        text(panel, c["label"], 27, scale=.65)
        text(panel, f"PhysX | mu=0.7 | seed42 | {'8x slow / 400 Hz samples' if slow else 'real time'}", 52, scale=.45)
        text(panel, f"t={now:.4f} s | speed={c['s']['body_speed_m_s'][i]:.2f} m/s | tilt={c['s']['root_tilt_deg'][i]:.1f} deg", 510)
        for side in (0, 1):
            state, value = "NO CONTACT", "v_t: n/a | rho: n/a"
            if i == 0:
                state = "INITIAL / no native sample"
            elif r["touching"][i, side]:
                state = "TOUCH + SLIP" if r["sliding"][i, side] else "TOUCH / no slip"
                if r["friction_corroborated_lateral_sliding"][i, side]:
                    state = "TOUCH + SLIP + FRICTION"
                rho = r["friction_utilization"][i, side]
                rhotext = f"{rho:.2f}" if np.isfinite(rho) else "n/a"
                value = f"v_t: {r['tangent_rms_m_s'][i, side]:.2f} m/s | rho: {rhotext}"
            elif r["contact_unknown"][i, side]:
                state = "CONTACT UNKNOWN"
            text(panel, f"{'Left' if side == 0 else 'Right'}: {state}", 537+side*44, color=COLORS[side], scale=.49)
            text(panel, value, 554+side*44, x=30, color=COLORS[side], scale=.43)
        panel[600:635] = c["timeline"][600:635]
        cursor = int(45+now/end*575)
        cv2.line(panel, (cursor, 601), (cursor, 633), (20, 20, 20), 2)
        text(panel, "Timeline: green touch | red off | amber unknown", 655, scale=.43)
        text(panel, "Green dots: valid touching points | gap <=0.1 mm", 677, scale=.43)
        text(panel, "Fixed 2.4 m follow camera | no zoom | no force gate", 700, scale=.43)
        return panel

    def reference_panel(c, j, rendered):
        panel = blank(); panel[65:485] = rendered
        text(panel, "GMR pose + designed stop reference", 28, scale=.6)
        text(panel, c["label"], 52, scale=.47)
        text(panel, f"t={j*.02:.2f} s | speed={c['command']['designed_root_link_speed_m_s'][j]:.2f} m/s", 520)
        text(panel, "Kinematic playback / no simulated contact force", 552, scale=.48)
        text(panel, "Braking command: 0.40-1.80 s", 584)
        text(panel, "Human-inspired posture; designed global path", 616, scale=.48)
        text(panel, "Wheel spin is a placeholder, excluded from imitation", 648, scale=.44)
        text(panel, "Same fixed 2.4 m camera / no zoom", 690, scale=.49)
        return panel

    def xy_panel(c, i, j):
        panel = blank()
        for axis in (0, 1):
            for v in np.arange(np.floor((center[axis]-extent/2)/.25)*.25, center[axis]+extent/2, .25):
                q = np.tile(center, (2, 1)); q[:, axis] = v; q[:, 1-axis] += [-extent/2, extent/2]
                pp = project(q); cv2.line(panel, tuple(pp[0]), tuple(pp[1]), (230, 230, 230), 1)
        cv2.polylines(panel, [project(c["command"]["qpos"][:, :2])], False, (30, 150, 95), 2, cv2.LINE_AA)
        cv2.polylines(panel, [project(c["qpos"][:, :2])], False, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.polylines(panel, [project(c["qpos"][:i+1, :2])], False, (185, 55, 65), 3, cv2.LINE_AA)
        for side, color in enumerate(COLORS):
            center_w = c["tr"]["body_pos"][i, c["ids"][side], :2]
            angle = np.radians(c["s"]["wheel_heading_deg"][i, side])
            d = .12*np.array([np.cos(angle), np.sin(angle)])
            pp = project([center_w-d, center_w+d]); cv2.line(panel, tuple(pp[0]), tuple(pp[1]), color, 4, cv2.LINE_AA)
        angle = np.radians(c["s"]["body_heading_deg"][i]); root = c["qpos"][i, :2]
        pp = project([root, root+.23*np.array([np.cos(angle), np.sin(angle)])])
        cv2.arrowedLine(panel, tuple(pp[0]), tuple(pp[1]), (175, 40, 55), 2, cv2.LINE_AA, tipLength=.3)
        text(panel, "XY trajectory + body / wheel headings", 28, scale=.58)
        text(panel, "Shared equal XY scale across all four motions", 55, scale=.46)
        text(panel, "Green: reference | Red: recorded PhysX path", 635, scale=.48)
        text(panel, "Blue / orange: left / right wheel plane", 665, scale=.49)
        text(panel, f"Grid 0.25 m | span {extent:.2f} m | t={c['s']['time_s'][i]:.3f} s", 695, scale=.47)
        return panel

    def contact_panel(c, i):
        panel = blank(); r = c["touch"]; t = r["time_s"]
        specs = [(r["min_separation_m"]*1000, "Valid contact separation [mm]", -.2, 20., .1),
                 (r["tangent_rms_m_s"], "Touching-point tangent speed [m/s]", 0., 3., .05),
                 (r["friction_utilization"], "Contact-only aggregate friction ratio", 0., 1.4, .9),
                 (np.where(r["touching"] & c["s"]["wheel_beta_valid"], c["s"]["wheel_axial_beta_deg"], np.nan),
                  "Touching wheel axial beta [deg]", -90., 90., 20.)]
        text(panel, "Native PhysX contact-first diagnostics", 28, scale=.58)
        text(panel, "No 5 N contact threshold; missing data stay blank", 53, scale=.45)
        for row, (values, label, lo, hi, threshold) in enumerate(specs):
            y0 = 100+row*133
            text(panel, label, y0-12, scale=.49)
            cv2.rectangle(panel, (60, y0), (615, y0+86), (180, 180, 180), 1)
            text(panel, f"{hi:g}", y0+5, x=8, scale=.4); text(panel, f"{lo:g}", y0+86, x=8, scale=.4)
            yth = int(y0+86*(hi-threshold)/(hi-lo)); cv2.line(panel, (60, yth), (615, yth), (180, 180, 180), 1)
            for side, color in enumerate(COLORS):
                mask = (t > 0) & (t <= end+1e-9) & np.isfinite(values[:, side])
                edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
                for start, stop in zip(edges[::2], edges[1::2]):
                    pp = np.stack((60+t[start:stop]/end*555,
                                   y0+86*(hi-np.clip(values[start:stop, side], lo, hi))/(hi-lo)), axis=-1).astype(int)
                    if len(pp)>1: cv2.polylines(panel, [pp], False, color, 1, cv2.LINE_AA)
                    elif len(pp): cv2.circle(panel, tuple(pp[0]), 1, color, -1)
            x = int(60+t[i]/end*555); cv2.line(panel, (x, y0), (x, y0+86), (20, 20, 20), 1)
        text(panel, "Contact gap <=0.1 mm | slip speed >0.05 m/s", 650, scale=.46)
        text(panel, "Friction evidence is wheel-level, not pointwise", 677, scale=.46)
        text(panel, "Fixed axes; chart clipping is visual only", 704, scale=.46)
        return panel

    a.output.mkdir(parents=True)
    overview = a.output / "hockey_contact_fullbody_4x1.mp4"
    individual = [a.output / (c["name"]+"_reference_physx_xy_contact_4x1.mp4") for c in cases]
    processes = [writer(a.ffmpeg, overview)] + [writer(a.ffmpeg, path) for path in individual]
    try:
        offset = int(aligned_archive is not None)
        for frame, i in enumerate(range(offset, round(end/.0025)+1+offset, 8)):
            body = [body_panel(c, i) for c in cases]
            image = np.concatenate(body, axis=1); processes[0].stdin.write(image.tobytes())
            refs = {}
            for k, c in enumerate(cases):
                key = str(c["refdir"])
                if key not in refs: refs[key] = mesh(c["command"]["qpos"][frame])
                full = np.concatenate((reference_panel(c, frame, refs[key]), body[k], xy_panel(c, i, frame), contact_panel(c, i)), axis=1)
                processes[k+1].stdin.write(full.tobytes())
            if frame in (28, 41, 134):
                cv2.imwrite(str(a.output / f"overview_{(i-offset)*.0025:.2f}s.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if frame % 35 == 0: print(f"Real-time frames: {frame+1}/139", flush=True)
    finally:
        for process in processes: close_writer(process)
    slow = a.output / "hockey_contact_transition_8x_slow_4x1.mp4"
    process = writer(a.ffmpeg, slow)
    slow_indices = np.arange(192+offset, 369+offset)  # 0.480--0.920 s; every native 400 Hz sample.
    try:
        for frame, i in enumerate(slow_indices):
            image = np.concatenate([body_panel(c, int(i), slow=True) for c in cases], axis=1)
            process.stdin.write(image.tobytes())
            if frame % 45 == 0: print(f"Slow-motion frames: {frame+1}/{len(slow_indices)}", flush=True)
    finally:
        close_writer(process)
        runtime.renderer.close()
    videos = [overview, *individual, slow]
    verification = []
    for path in videos:
        cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS); count = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            assert frame.shape == (H, W*4, 3)
            count += 1
        cap.release()
        expected = len(slow_indices) if path == slow else round(end/.02)+1
        assert count == expected and np.isclose(fps, 50.)
        verification.append(dict(path=path.name, sha256=sha(path), frames=count, fps=fps, width=4*W, height=H))
    gifs = []
    for path in (overview, slow):
        gif = path.with_suffix(".gif")
        subprocess.run([a.ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "2", "-i", str(path),
            "-filter_complex", "fps=12.5,scale=1600:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer",
            "-loop", "0", str(gif)], check=True)
        gifs.append(dict(path=gif.name, sha256=sha(gif)))
    provenance = dict(script_sha256=sha(__file__), contact_report_sha256=sha(a.analysis / "report.json"),
        contact_signals_sha256=sha(a.analysis / "contact_signals.npz"), model_sha256=sha(a.model),
        mesh_provenance=mesh_provenance(a.model), physics_steps_performed=0, pose_smoothing=False,
        height_correction=False, source_RGB_included=False, policy_inference=False,
        camera=dict(distance_m=2.4, azimuth_deg=135., elevation_deg=-18., lookat="current root + [0,0,-0.24]"),
        xy_center_m=center.tolist(), xy_extent_m=extent, all_panels_share_camera_and_xy_scale=True,
        physical_dt_s=.0025, realtime_frame_stride=8,
        contact_timing=("pre-step geometry + pre-step velocity + pre-step display pose; force over following interval"
                        if aligned_archive is not None else "legacy post-step pose/velocity with pre-step contact geometry"),
        aligned_audit_sha256=sha(a.aligned_audit / "report.json") if a.aligned_audit is not None else None,
        display_pose_index_offset_from_native_row_end=-1 if aligned_archive is not None else 0,
        slow_motion=dict(first_physical_index=int(slow_indices[0]), last_physical_index=int(slow_indices[-1]),
                         input_time_interval_s=[.48,.92], physical_frame_stride=1, slowdown_factor=8.),
        contact_markers="Valid native normal points with separation <=0.1 mm, no normal-force gate; display sphere radius 9 mm",
        friction_label="Touch + slip + friction: contact lateral slip, near-limit aggregate ratio and dissipative anchor evidence",
        no_overall_task_success_labels=True, videos=verification, gifs=gifs,
        cases=[dict(cell=c["name"], maximum_wheel_FK_error_m=c["fk_error"], physical_poses_checked=len(c["qpos"]),
                    recorded_provenance=c["row"]["provenance"],
                    command_sha256=sha(c["refdir"] / "command_diagnostics.npz")) for c in cases])
    (a.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(output=str(a.output), videos=len(videos), gifs=len(gifs))))


if __name__ == "__main__":
    main()
