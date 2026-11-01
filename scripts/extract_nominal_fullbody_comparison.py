"""Extract the nominal row from an audited, already-rendered PPO comparison.

This changes video layout only. It performs no simulation, policy inference,
state filtering, retiming, or pose correction. Existing evidence is preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results/ppo_smoothness_action_rate_0p5_fullbody_20261021_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_source(folder):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    require(summary["kind"] == "physx_ppo_action_rate_fullbody_recorded_comparison"
            and summary["candidate_action_rate_weight"] == -.5
            and summary["baseline_action_rate_weight"] == -.1,
            "Require the original completed -0.5 comparison")
    require(summary["row_mu"] == [.7, .15]
            and summary["column_order"] == ["baseline", "candidate"]
            and summary["video_frames"] == 350 and summary["video_fps"] == 50
            and summary["dynamics_stepping"] is False
            and summary["root_height_corrected"] is False,
            "Source layout, clock or replay semantics differ")
    for name, artifact in summary["artifacts"].items():
        require(Path(name).name == name and artifact["file"] == name,
                "Source artifacts must have local basenames")
        require(sha256(folder / name) == artifact["sha256"], "Source artifact SHA differs: " + name)
    return summary


def run(source, output):
    import imageio_ffmpeg
    from PIL import Image

    source, output = Path(source), Path(output)
    require(not output.exists(), "Preserve existing outputs: choose a fresh directory")
    original = audit_source(source)
    video = source / "fullbodycomparison.mp4"
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    reader = imageio_ffmpeg.read_frames(str(video))
    try:
        metadata = next(reader)
    finally:
        reader.close()
    require(metadata["size"] == (1280, 970) and metadata["fps"] == 50
            and frames == 350 and abs(seconds - 7.) < .001,
            "Source media dimensions or complete timing differ")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "fullbody_mu0p7_comparison.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-n",
               "-i", str(video), "-vf", "crop=1280:490:0:0", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    subprocess.run(command, check=True)
    with Image.open(source / "comparison.png") as image:
        require(image.size == (1280, 970), "Source snapshot layout differs")
        image.crop((0, 0, 1280, 490)).save(output / "comparison.png")
    output_frames, output_seconds = imageio_ffmpeg.count_frames_and_secs(str(target))
    require(output_frames == frames and abs(output_seconds - seconds) < .001,
            "Extraction must preserve the full original video clock")
    summary = {
        "kind": "nominal_row_extracted_from_existing_physx_ppo_fullbody_render",
        "mu": .7, "column_action_rate_weights": [-.1, -.5],
        "candidate_checkpoint_sha256": original["candidate_checkpoint_sha256"],
        "source_summary_sha256": sha256(source / "summary.json"),
        "source_video_sha256": sha256(video),
        "source_renderer": "scripts/render_ppo_smoothness_comparison.py",
        "extraction_source_sha256": sha256(__file__),
        "crop_xywh": [0, 0, 1280, 490], "fps": 50,
        "frames": output_frames, "encoded_duration_s": output_seconds,
        "physical_recording_interval_s": [0., 6.98],
        "training_engine": "PhysX", "evaluation_engine": "PhysX",
        "original_graphics": "MuJoCo forward kinematics only, replaying recorded PhysX states",
        "new_training": False, "new_physics": False, "new_inference": False,
        "state_filtering": False, "retiming": False, "pose_correction": False,
        "camera_changed": False, "video_reencoded": True,
        "artifacts": {name: {"file": name, "sha256": sha256(output / name)}
                      for name in (target.name, "comparison.png")},
    }
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_dir, args.output_dir)
    print(json.dumps({"mu": result["mu"], "frames": result["frames"],
                      "duration_s": result["encoded_duration_s"], "new_physics": False}))
