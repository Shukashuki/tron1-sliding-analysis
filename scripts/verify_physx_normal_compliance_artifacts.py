"""Verify the completed local replay against native evidence and decoded media."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

import analyze_physx_normal_compliance as audit


def run(args):
    root = args.result
    out = root/'artifact_verification.json'
    assert not out.exists()
    provenance = audit.read(root/'render/provenance.json')
    analysis = audit.read(root/'analysis/analysis.json')
    assert analysis['analyzer_sha256'] == provenance['analyzer_sha256'] == audit.sha(audit.__file__)
    assert provenance['analysis_sha256'] == audit.sha(root/'analysis/analysis.json')
    assert provenance['renderer_sha256'] == audit.sha(audit.ROOT/'scripts/render_physx_normal_compliance.py')
    assert provenance['graphics_sha256'] == audit.sha(audit.ROOT/'scripts/render_ppo_ground_contact_v2.py')
    assert analysis['collection_manifest_sha256'] == audit.verify_collection(root/'evidence')
    for name, digest in provenance['artifacts'].items():
        assert audit.sha(root/'render'/name) == digest
    arrays = audit.npz(root/'render/replay_states.npz')
    assert len(arrays['native_indices']) == 350
    for key in audit.ORDER:
        relative = arrays[key+'__lookat']-arrays[key+'__qpos'][arrays['native_indices'], :3]
        np.testing.assert_allclose(relative, np.broadcast_to(relative[:1], relative.shape), rtol=0, atol=1e-12)
    stream = imageio_ffmpeg.read_frames(str(root/'render/fullbody_comparison.mp4'), pix_fmt='rgb24')
    try:
        metadata = next(stream)
        assert metadata['size'] == (2560, 850) and metadata['fps'] == 50.
        count = 0
        for frame in stream:
            assert len(frame) == 2560*850*3
            count += 1
        assert count == 350
    finally:
        stream.close()
    for index in (112, 150, 200, 348):
        with Image.open(root/'render'/f'frame_{index:03d}.png') as im:
            assert im.size == (2560, 850)
            im.verify()
    result = dict(status='verified', video_all_frames_decoded=count, video_fps=50,
        video_size_pixels=[2560, 850], whole_collection_hashes_verified=True,
        source_and_artifact_hashes_verified=True, camera_translation_relative_to_recorded_root_constant=True,
        fixed_camera_distance_m=provenance['fixed_camera_distance_m'],
        native_states_per_arm=2793, samples_per_model=1,
        task_success_claimed=False, goal_acceptance_thresholds_changed=False,
        artifact_verifier_sha256=audit.sha(__file__),
        analysis_sha256=audit.sha(root/'analysis/analysis.json'), provenance_sha256=audit.sha(root/'render/provenance.json'))
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--result', type=Path, required=True)
    run(p.parse_args())
