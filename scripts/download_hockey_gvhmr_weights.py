#!/usr/bin/env python3
"""Download hash-pinned GVHMR demo weights, never licensed body models.

The mirror is also used by the demo linked from upstream GVHMR's README:
https://huggingface.co/spaces/LittleFrog/GVHMR/blob/main/app.py
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


WEIGHTS = {
    "gvhmr/gvhmr_siga24_release.ckpt": (
        163508011, "4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91"
    ),
    "hmr2/epoch=10-step=25000.ckpt": (
        2709494041, "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e"
    ),
    "vitpose/vitpose-h-multi-coco.pth": (
        2549075546, "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc"
    ),
    "yolo/yolov8x.pt": (
        136867539, "c4d5a3f000d771762f03fc8b57ebd0aae324aeaefdd6e68492a9c4470f2d1e8b"
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(root, name, size, expected):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/camenduru/GVHMR/resolve/main/" + quote(name)
    if path.exists():
        if path.stat().st_size != size or sha256(path) != expected:
            raise ValueError(f"Existing checkpoint does not match pinned SHA256: {path}")
    else:
        temp = path.with_name(path.name + ".part")
        print(f"Downloading {name} ({size / 1e6:.1f} MB)", flush=True)
        with urlopen(Request(url, headers={"User-Agent": "tron1-hockey-research"}), timeout=60) as src:
            with temp.open("wb") as dst:
                total = 0
                for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
                    total += len(block)
                    if total > size:
                        raise ValueError(f"Download exceeds pinned size: {name}")
                    dst.write(block)
        if temp.stat().st_size != size or sha256(temp) != expected:
            raise ValueError(f"Downloaded checkpoint does not match pinned SHA256: {name}")
        temp.replace(path)
    print(f"Verified {name}", flush=True)
    return {"file": name, "bytes": size, "sha256": expected, "source_url": url}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for name, (size, expected) in WEIGHTS.items():
        records.append(download(args.output, name, size, expected))
        (args.output / "download_inventory.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
