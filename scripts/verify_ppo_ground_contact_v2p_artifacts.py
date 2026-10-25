"""Decode every K/P video frame and bind native replay/camera/source artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from PIL import Image
import analyze_ppo_ground_contact_v2p as audit


def run(args):
    root = args.result
    target = root/'artifact_verification.json'
    assert not target.exists()
    summary = audit.read(root/'analysis/analysis.json')
    provenance = audit.read(root/'render/provenance.json')
    assert summary['analyzer_sha256'] == provenance['analyzer_sha256'] == audit.sha(audit.__file__)
    assert provenance['analysis_sha256'] == audit.sha(root/'analysis/analysis.json')
    assert provenance['renderer_sha256'] == audit.sha(audit.ROOT/'scripts/render_ppo_ground_contact_v2p.py')
    assert provenance['graphics_sha256'] == audit.sha(audit.ROOT/'scripts/render_ppo_ground_contact_v2.py')
    assert provenance['case_order'] == list(audit.ORDER)
    audit.verify_collection(root/'evidence')
    for name, digest in provenance['artifacts'].items(): assert audit.sha(root/'render'/name) == digest
    states = audit.npz(root/'render/replay_states.npz')
    assert len(states['native_indices']) == 350
    for key in audit.ORDER:
        delta = states[key+'__lookat']-states[key+'__qpos'][states['native_indices'], :3]
        np.testing.assert_allclose(delta, np.broadcast_to(delta[:1], delta.shape), rtol=0, atol=1e-12)
    stream = imageio_ffmpeg.read_frames(str(root/'render/fullbody_comparison.mp4'), pix_fmt='rgb24')
    try:
        metadata = next(stream)
        assert metadata['size'] == (2560, 850) and metadata['fps'] == 50.
        count = 0
        for frame in stream:
            assert len(frame) == 2560*850*3
            count += 1
        assert count == 350
    finally: stream.close()
    for index in (112, 150, 200, 348):
        with Image.open(root/'render'/f'frame_{index:03d}.png') as im:
            assert im.size == (2560, 850); im.verify()
    result = dict(status='verified', frames_decoded=350, fps=50, pixels=[2560, 850],
        native_states_per_arm=2793, case_order=list(audit.ORDER),
        original_support_tracking_speed_beta_HF_acceptance_unchanged=True,
        fixed_relative_camera_verified=True, visual_motion_review_complete=False,
        task_success_claimed=False, source_sha256=audit.sha(__file__),
        analysis_sha256=audit.sha(root/'analysis/analysis.json'),
        provenance_sha256=audit.sha(root/'render/provenance.json'))
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--result', type=Path, required=True)
    run(p.parse_args())
