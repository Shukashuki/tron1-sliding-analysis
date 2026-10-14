"""Download skeleton-standardized HDM05 BVH clips from the Holden data archive.

HTTP range reads retrieve only the ZIP directory and selected members, not the
entire 5.65 GB archive. These BVH files are a preprocessed derivative: the
archive's retarget_hdm05.py uses human-to-human Jacobian IK to standardize the
skeleton. They must not be called raw ASF/AMC or a TRON1 GMR result.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


ARCHIVE_URL = "https://theorangeduck.com/media/uploads/other_stuff/motionsynth_data.zip"
SOURCE_PAGE = "https://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing"
ACTIONS = ("walkLeft2Steps", "walkLeft3Steps", "walkRightCrossFront2Steps",
           "walkRightCrossFront3Steps", "walkLeftCircle4StepsRstart",
           "walkRightCircle4StepsRstart", "turnLeft", "turnRight",
           "shuffle2StepsLStart", "shuffle2StepsRStart")


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
            self.last_modified = response.headers.get("Last-Modified")
        self.position = 0
        self.downloaded_bytes = 0
        self.cache = OrderedDict()
        self.chunk_size = 512 * 1024

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size, max(0, self.size - self.position))
        if not size:
            return b""
        if size > 32_000_000:
            raise ValueError("Unexpectedly large range request")
        parts = []
        while size:
            start = self.position // self.chunk_size * self.chunk_size
            if start not in self.cache:
                end = min(start + self.chunk_size, self.size) - 1
                self.cache[start] = self.fetch_range(start, end)
                if len(self.cache) > 16:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            offset = self.position - start
            piece = self.cache[start][offset:offset + size]
            parts.append(piece)
            self.position += len(piece)
            size -= len(piece)
        return b"".join(parts)

    def fetch_range(self, start, end):
        size = end - start + 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Match"] = self.etag
        request = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise ValueError("Server did not honor the bounded range request")
            payload = response.read(size + 1)
        if len(payload) != size:
            raise ValueError("Incomplete ZIP range")
        self.downloaded_bytes += size
        return payload


def choose_members(infos):
    return [info for info in infos
            if "/hdm05/" in ("/" + info.filename).lower()
            and info.filename.lower().endswith(".bvh")
            and any(action.lower() in Path(info.filename).name.lower() for action in ACTIONS)]


def inspect_bvh(payload):
    lines = payload.decode("utf-8").splitlines()
    frame_count = next(int(line.split(":", 1)[1]) for line in lines if line.startswith("Frames:"))
    frame_line = next(i for i, line in enumerate(lines) if line.startswith("Frame Time:"))
    dt = float(lines[frame_line].split(":", 1)[1])
    channel_counts = [int(line.strip().split()[1]) for line in lines if line.strip().startswith("CHANNELS ")]
    root_channels = next(line.strip().split()[2:] for line in lines if line.strip().startswith("CHANNELS "))
    if root_channels[:3] != ["Xposition", "Yposition", "Zposition"]:
        raise ValueError("Expected preserved BVH root translation channels")
    frames = [list(map(float, line.split())) for line in lines[frame_line + 1:] if line.strip()]
    if len(frames) != frame_count or any(len(frame) != sum(channel_counts) for frame in frames):
        raise ValueError("BVH frame/channel count mismatch")
    span = [max(frame[i] for frame in frames) - min(frame[i] for frame in frames) for i in range(3)]
    joints = [line.strip().split()[1] for line in lines if line.strip().startswith(("ROOT ", "JOINT "))]
    return {"frames": frame_count, "frame_time_s": dt, "fps": 1 / dt,
            "last_minus_first_time_s": (frame_count - 1) * dt,
            "root_translation_span_raw_units": span, "joints": joints,
            "channel_count": sum(channel_counts), "root_channels": root_channels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    with RangeReader(ARCHIVE_URL) as remote, zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        selected = choose_members(infos)
        if not selected:
            print(json.dumps({"selected": [], "hdm05_paths": [info.filename for info in infos if "hdm" in info.filename.lower()][:40]}))
            raise ValueError("No requested HDM05 motions found")
        if args.list_only:
            print(json.dumps({"selected": [{"path": info.filename, "bytes": info.file_size} for info in selected], "range_bytes": remote.downloaded_bytes}))
            return
        processor_path = "motionsynth_data/data/processed/retarget_hdm05.py"
        processor = archive.read(processor_path)
        if b"JacobianInverseKinematics" not in processor or b"anim.positions[:,0] = targets[:,0]" not in processor:
            raise ValueError("Unexpected archive processing provenance")
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        names = [Path(info.filename).name for info in selected]
        if len(names) != len(set(names)):
            raise ValueError("Archive member filename collision")
        rows = []
        for info in selected:
            if info.file_size > 10_000_000:
                raise ValueError("Unexpectedly large motion member")
            target = output / Path(info.filename).name
            # zipfile verifies each member's CRC on retrieval.
            payload = archive.read(info)
            checks = inspect_bvh(payload)
            if target.exists():
                if target.read_bytes() != payload:
                    raise ValueError("Preserving conflicting existing file: " + str(target))
            else:
                with target.open("xb") as handle:
                    handle.write(payload)
            rows.append({"file": target.name, "archive_member": info.filename,
                         "bytes": len(payload), "zip_crc32": f"{info.CRC:08x}",
                         "sha256": hashlib.sha256(payload).hexdigest(), **checks})
            print(json.dumps({"downloaded": target.name, "frames": checks["frames"]}), flush=True)
        audit = {"dataset": "HDM05", "source_page": SOURCE_PAGE, "archive_url": ARCHIVE_URL,
                 "archive_etag": remote.etag, "archive_last_modified": remote.last_modified,
                 "archive_size_bytes": remote.size, "range_download_bytes": remote.downloaded_bytes,
                 "processor_archive_member": processor_path,
                 "processor_sha256": hashlib.sha256(processor).hexdigest(),
                 "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "files": rows,
                 "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
                 "processing": "Holden archive processed/hdm05: standardized human skeleton with Jacobian IK in retarget_hdm05.py; global root translation channels retained.",
                 "limitations": "Preprocessed human-to-human IK derivative, not original ASF/AMC and not a TRON1 GMR result. Foot heading preservation must be checked against originals before scientific offset claims. Units and axes require FK checks. No measured contact or tire slip. No policy training performed."}
        with (output / ("download_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")).open("x") as handle:
            handle.write(json.dumps(audit, indent=2) + "\n")
        print(json.dumps({"output_dir": str(output), "motion_clips": len(rows),
                          "motion_bytes": sum(row["bytes"] for row in rows),
                          "range_download_bytes": remote.downloaded_bytes}))


if __name__ == "__main__":
    main()
