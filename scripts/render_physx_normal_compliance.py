"""Four actual normal-contact transfers: opaque poses and fixed relative camera."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np

import analyze_physx_normal_compliance as analysis
import render_ppo_ground_contact_v2 as graphics

GRAPHICS_SHA = '0289c7da1764758cb79c184a7fbf9f5f0e9afd4adab23ef1ea700411be997efa'


def run(args):
    assert not args.output_dir.exists() and os.environ['MUJOCO_GL'] == 'osmesa'
    assert analysis.sha(graphics.__file__) == GRAPHICS_SHA
    summary = analysis.read(args.analysis)
    assert summary['status'] == 'completed_analysis' and summary['analyzer_sha256'] == analysis.sha(analysis.__file__)
    assert summary['kind'] == 'PhysX_native_normal_contact_sensitivity_frozen_K_actor'
    reference = graphics.metrics.reference_data(args.reference)
    assert analysis.sha(args.reference) == summary['reference_sha256']
    assert analysis.verify_collection(args.collection) == summary['collection_manifest_sha256']
    rows, traces, _ = analysis.load_cases(args.collection, reference)
    assert rows == summary['rows']
    kinematics, fk_source = graphics.metrics.load_fk(args.model, args.legacy_evaluator)
    cases = {}
    for row, title in zip(rows, analysis.TITLES, strict=True):
        trace = traces[row['id']]
        fk = kinematics.reconstruct(trace)
        trace.update(signals=fk['signals'], wheel_quaternion_wxyz=fk['wheel_quaternion_wxyz'],
            wheel_center_velocity_w=fk['wheel_center_velocity_w'])
        trace['body_heading_geometry_valid'], trace['wheel_heading_geometry_valid'] = graphics.metrics.heading_geometry_masks(trace)
        cases[row['id']] = dict(variant='normal_contact_sensitivity', engine='PhysX', analysis=row['metrics'],
            title=title, new_nominal_screen=row['nominal_screen'])
    runtime = graphics.create_runtime(args.model, width=640, height=360)
    try:
        qposes, fk_audits = {}, {}
        for key in analysis.ORDER:
            qposes[key], mapping = graphics.build_qpos(runtime.model, traces[key])
            fk_audits[key] = dict(graphics.replay_fk_audit(runtime.model, traces[key], qposes[key]),
                named_joint_qpos_addresses=mapping.tolist())
        camera = graphics.prepare_camera(runtime, traces, qposes, list(analysis.ORDER), graphics.native_indices())
        for key in analysis.ORDER:
            delta = camera['lookat_by_case'][key]-traces[key]['position'][graphics.native_indices()]
            np.testing.assert_allclose(delta, np.broadcast_to(delta[:1], delta.shape), rtol=0, atol=1e-12)
        bounds = [v.tolist() for v in graphics.world_bounds(traces, reference)]
        args.output_dir.mkdir(parents=True)
        from PIL import ImageDraw
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(str(args.output_dir/'fullbody_comparison.mp4'), (2560, 850), fps=50,
            codec='libx264', quality=7, macro_block_size=2, output_params=['-movflags', '+faststart'])
        writer.send(None)
        try:
            for frame in range(350):
                canvas = graphics.render_frame(runtime, traces, qposes, reference, cases,
                    list(analysis.ORDER), camera, bounds, frame)
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 2560, 29), fill='#fafafa')
                draw.text((12, 4), f'Same frozen K PPO | native normal-contact sensitivity | PhysX mu=.7 | beta ON | t={frame*.02:.2f}s', font=runtime.small, fill='#111111')
                draw.rectangle((0, 763, 2560, 786), fill='#fafafa')
                draw.text((12, 765), 'Four actual completed arms | all pass XY, speed, beta and HF checks; none achieves continuous any-wheel support | task completion not claimed', font=runtime.small, fill='#8f2929')
                writer.send(np.asarray(canvas))
                if frame in (112, 150, 200, 348): canvas.save(args.output_dir/f'frame_{frame:03d}.png')
                if frame == 150: canvas.save(args.output_dir/'comparison.png')
                if frame % 100 == 0: print(f'Normal-contact replay frame={frame}/350', flush=True)
        finally:
            writer.close()
        arrays = dict(native_indices=graphics.native_indices(), fixed_camera_distance_m=np.asarray(camera['distance_m']))
        for key in analysis.ORDER:
            arrays[key+'__qpos'] = qposes[key]
            arrays[key+'__lookat'] = camera['lookat_by_case'][key]
            arrays[key+'__floor_normal_force_w'] = traces[key]['contact_force']
        np.savez_compressed(args.output_dir/'replay_states.npz', **arrays)
        names = [p.name for p in args.output_dir.iterdir() if p.is_file()]
        provenance = dict(kind='four_actual_normal_contact_transfer_recorded_replay', case_order=list(analysis.ORDER),
            analysis_sha256=analysis.sha(args.analysis), analyzer_sha256=analysis.sha(analysis.__file__),
            renderer_sha256=analysis.sha(__file__), graphics_sha256=GRAPHICS_SHA,
            video_frames=350, video_fps=50, video_size_pixels=[2560, 850], native_states_per_case=2793,
            FK_source=fk_source, FK_readback_audits=fk_audits, source_model_sha256=analysis.sha(args.model),
            reference_sha256=analysis.sha(args.reference), actor_sha256=analysis.prior.previous.ACTOR,
            fixed_camera_distance_m=camera['distance_m'], camera_same_distance_all_cases_frames=True,
            camera_relative_translation_verified=True, camera_yaw_follow=False, camera_zoom=False,
            robot_mesh_alpha=1., smoothing_interpolation_alignment_height_correction=False,
            new_physics_policy_inference_training=False, normal_contact_parameters_physically_identified=False,
            visual_review_complete=False, task_success_claimed=False,
            artifacts={name: analysis.sha(args.output_dir/name) for name in names})
        (args.output_dir/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
    finally:
        runtime.renderer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('collection', 'analysis', 'output-dir'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--reference', type=Path, default=graphics.assets.paired.DEFAULT_REFERENCE)
    p.add_argument('--model', type=Path, default=graphics.assets.paired.DEFAULT_MODEL)
    p.add_argument('--legacy-evaluator', type=Path, default=graphics.assets.paired.DEFAULT_LEGACY)
    run(p.parse_args())
