#!/usr/bin/env python3
"""Recreate the visually reviewed, uncut VVYG hockey-stop candidate episode."""
import argparse
import hashlib
from pathlib import Path
import subprocess

SOURCE_SHA = "ae90bba60de82e0a0e8b0b6e65fc7e9f58b5460587f69e2c5342b93e7530897c"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ffmpeg", help="Use an explicit FFmpeg executable instead of imageio-ffmpeg")
    args = p.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Source differs from the visually reviewed video")
    if args.output.exists():
        raise FileExistsError("Preserving previous episode; choose a new output path")
    if args.ffmpeg is None:
        import imageio_ffmpeg
        args.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(args.source),
                    "-vf", "trim=start_frame=40:end_frame=105,setpts=PTS-STARTPTS,fps=30", "-an",
                    "-c:v", "libx264rgb", "-crf", "0", "-preset", "fast",
                    "-pix_fmt", "rgb24", str(args.output)], check=True)


if __name__ == "__main__":
    main()
