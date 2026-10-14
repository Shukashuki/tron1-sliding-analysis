"""Fetch hash-locked raw HDM05 ASF/AMC files from a public research mirror."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


def validate_payload(payload, row):
    expected_size = row["bytes"]
    digest = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    if len(payload) != expected_size or digest != row["git_blob_sha1"]:
        raise ValueError("Size or Git blob hash mismatch: " + row["path"])
    # Original files contain German author names in Latin-1 comments.
    # Decode only for inspection; the original bytes are saved unchanged.
    lines = payload.decode("latin-1").splitlines()
    if row["path"].endswith(".asf"):
        if not any(line.strip() == ":bonedata" for line in lines):
            raise ValueError("Missing ASF skeleton")
        return {"format": "ASF"}
    if not row["path"].endswith(".amc"):
        raise ValueError("Expected ASF/AMC data")
    roots = [list(map(float, line.split()[1:])) for line in lines if line.startswith("root ")]
    frames = [int(line.strip()) for line in lines if line.strip().isdigit()]
    if not roots or len(roots) != len(frames) or any(len(root) != 6 for root in roots):
        raise ValueError("Incomplete AMC root translation/rotation channels")
    if frames != list(range(frames[0], frames[0] + len(frames))):
        raise ValueError("Noncontiguous AMC frames")
    for name in ("lfoot", "rfoot", "ltoes", "rtoes"):
        if sum(line.startswith(name + " ") for line in lines) != len(frames):
            raise ValueError("Missing bilateral foot/toe rotation channels: " + name)
    span = [max(root[i] for root in roots) - min(root[i] for root in roots) for i in range(3)]
    if max(span) <= 0:
        raise ValueError("No changing global root translation")
    return {"format": "AMC", "frames": len(frames), "fps_from_filename": 120,
            "last_minus_first_time_s": (len(frames) - 1) / 120,
            "root_translation_span_raw_units": span,
            "global_root_translation_present": True,
            "bilateral_foot_toe_rotation_channels_present": True}


def download_one(row, manifest, output):
    name = Path(row["path"]).name
    if name != row["path"].split("/")[-1] or name in ("", ".", ".."):
        raise ValueError("Invalid filename")
    url = ("https://raw.githubusercontent.com/" + manifest["repository"] + "/" +
           manifest["commit"] + "/" + row["path"])
    target = output / name
    if target.exists():
        payload = target.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "TRON1-HDM05-source-audit/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read(row["bytes"] + 1)
    checks = validate_payload(payload, row)
    if not target.exists():
        # Exclusive creation preserves user-owned files and interrupted download history.
        with target.open("xb") as handle:
            handle.write(payload)
    return {**row, "file": name, "url": url,
            "sha256": hashlib.sha256(payload).hexdigest(), **checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "config/hdm05_raw_subset.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = [Path(row["path"]).name for row in manifest["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Manifest filename collision")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda row: download_one(row, manifest, output), manifest["files"]))
    audit = {"dataset": manifest["dataset"], "repository": manifest["repository"],
             "commit": manifest["commit"], "selection": manifest["selection"],
             "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
             "license_url": manifest["license_url"], "files": rows,
             "limitations": "Raw mirror verified against pinned Git blobs, not inaccessible official archives. Units, axes and skeleton FK still require verification before GMR. No policy training."}
    audit_path = output / "download_audit.json"
    if not audit_path.exists():
        with audit_path.open("x") as handle:
            handle.write(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "files": len(rows),
                      "motion_clips": sum(row["format"] == "AMC" for row in rows),
                      "bytes": sum(row["bytes"] for row in rows),
                      "total_motion_frames": sum(row.get("frames", 0) for row in rows)}))


if __name__ == "__main__":
    main()
