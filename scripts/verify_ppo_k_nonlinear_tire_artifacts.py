"""Verify native numerical replay, fixed relative cameras and every video frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

import analyze_ppo_k_nonlinear_tire as analysis


def run(root):
    root=Path(root);target=root/"artifact_verification_v1.json"
    assert not target.exists()
    summary=analysis.read(root/"analysis/analysis.json");provenance=analysis.read(root/"render/provenance.json")
    assert summary["analyzer_sha256"]==analysis.sha(analysis.__file__)
    assert provenance["analysis_sha256"]==analysis.sha(root/"analysis/analysis.json")
    assert provenance["renderer_sha256"]==analysis.sha(analysis.ROOT/"scripts/render_ppo_k_nonlinear_tire.py")
    assert provenance["graphics_sha256"]==analysis.sha(analysis.ROOT/"scripts/render_ppo_ground_contact_v2.py")
    assert provenance["case_order"]==list(analysis.ORDER)
    reference=analysis.metrics.metrics.reference_data(root/"reference.npz")
    assert summary["reference_sha256"]==analysis.sha(root/"reference.npz")
    rows,traces,_,proof,_=analysis.load_cases(root/"completed_evidence_v1",reference)
    assert rows==summary["rows"] and proof["actor_sha256"]==provenance["new_actor_sha256"]
    for name,digest in provenance["artifacts"].items():assert analysis.sha(root/"render"/name)==digest
    states=analysis.npz(root/"render/replay_states.npz")
    np.testing.assert_array_equal(states["native_indices"],np.arange(350)*8)
    assert float(states["fixed_camera_distance_m"])==provenance["fixed_camera_distance_m"]
    for key in analysis.ORDER:
        delta=states[key+"__lookat"]-states[key+"__qpos"][states["native_indices"],:3]
        np.testing.assert_allclose(delta,np.broadcast_to(delta[:1],delta.shape),atol=1e-12,rtol=0)
        np.testing.assert_array_equal(states[key+"__floor_normal_force_w"],traces[key]["floor_normal_force"])
    stream=imageio_ffmpeg.read_frames(str(root/"render/fullbody_comparison.mp4"),pix_fmt="rgb24")
    try:
        metadata=next(stream);assert metadata["size"]==(2560,850) and metadata["fps"]==50.
        count=0
        for frame in stream:
            assert len(frame)==2560*850*3;count+=1
        assert count==350
    finally:stream.close()
    for index in (112,150,200,348):
        with Image.open(root/"render"/(f"frame_{index:03d}.png")) as image:
            assert image.size==(2560,850);image.verify()
    result=dict(status="verified",frames_decoded=350,fps=50,pixels=[2560,850],native_states_per_case=2793,
        case_order=list(analysis.ORDER),fixed_relative_camera_verified=True,
        native_numerical_recomputation_matches=True,formal_policy_identity_verified=True,
        source_sha256=analysis.sha(__file__),analysis_sha256=analysis.sha(root/"analysis/analysis.json"),
        render_provenance_sha256=analysis.sha(root/"render/provenance.json"),
        manual_still_review_separate=True,full_manual_video_playback_claimed=False,task_success_claimed=False)
    target.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True);run(p.parse_args().result)
