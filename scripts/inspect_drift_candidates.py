"""Bounded source acquisition and data-capability audit; no IK or policy training.

Raw third-party files stay in ignored outputs/. The public JSON contains hashes,
source headers and source-space measurements only, not a claim of robot drifting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "sfu_traveling_turns": {
        "url": "https://mocap.cs.sfu.ca/nusmocap/0018_DanceTurns002.bvh",
        "filename": "0018_DanceTurns002.bvh", "max_bytes": 2_000_000,
        "source_page": "https://mocap.cs.sfu.ca/",
        "description": "Chinese dance turns (vertical circle), not declared a drift",
        "license": "SFU site: free for research purposes; raw file not redistributed",
    },
    "sfu_on_spot_control": {
        "url": "https://mocap.cs.sfu.ca/nusmocap/0018_DanceTurns001.bvh",
        "filename": "0018_DanceTurns001.bvh", "max_bytes": 2_000_000,
        "source_page": "https://mocap.cs.sfu.ca/",
        "description": "On-the-spot dance spins; negative control for translation",
        "license": "SFU site: free for research purposes; raw file not redistributed",
    },
    "alpine_gnss_imu": {
        "url": "https://zenodo.org/api/records/18468439/files/Biomechanics_AllRuns.xlsx/content",
        "filename": "Biomechanics_AllRuns.xlsx", "max_bytes": 14_200_000,
        "expected_md5": "afde48109aeec15daef3c090199cfc6a",
        "source_page": "https://zenodo.org/records/18468439",
        "description": "Giant-slalom biomechanical supplement; audit released columns before assuming full pose",
        "license": "CC BY 4.0, Zenodo API metadata",
    },
}


def acquire(spec, directory):
    path = directory / spec["filename"]
    if path.exists():
        payload = path.read_bytes()
    else:
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "TRON1-research-source-audit/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read(spec["max_bytes"] + 1)
    if len(payload) > spec["max_bytes"]:
        raise ValueError("Source exceeds bounded download")
    if spec.get("expected_md5") and hashlib.md5(payload).hexdigest() != spec["expected_md5"]:
        raise ValueError("Zenodo checksum mismatch")
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(payload)
    return path, payload


def xlsx_headers(payload):
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(si.itertext()) for si in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        relationships = {r.attrib["Id"]: r.attrib["Target"] for r in
                         ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        sheets = []
        for sheet in ET.fromstring(archive.read("xl/workbook.xml")).find("s:sheets", ns):
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = relationships[rid]
            member = target.lstrip("/") if target.startswith("/") else "xl/" + target
            rows = []
            com_xyz = []
            dimension = None
            with archive.open(member) as stream:
                for _, node in ET.iterparse(stream, events=("end",)):
                    tag = node.tag.rsplit("}", 1)[-1]
                    if tag == "dimension":
                        dimension = node.attrib.get("ref")
                    if tag != "row":
                        continue
                    cells = {}
                    for cell in node:
                        value = cell.find("s:v", ns)
                        text = value.text if value is not None else "".join(cell.find("s:is", ns).itertext()) if cell.find("s:is", ns) is not None else ""
                        if cell.attrib.get("t") == "s":
                            text = shared[int(text)]
                        cells[cell.attrib["r"]] = text
                    if len(rows) < 4:
                        rows.append(cells)
                    if sheet.attrib["name"].endswith("_COM"):
                        values = list(cells.values())
                        if len(values) == 3:
                            try:
                                com_xyz.append([float(v) for v in values])
                            except ValueError:
                                pass  # XYZ header, not a numeric sample
                    node.clear()
                    if len(rows) == 4 and not sheet.attrib["name"].endswith("_COM"):
                        break
            result = {"sheet": sheet.attrib["name"], "dimension": dimension, "first_four_rows": rows}
            if len(com_xyz) >= 3:
                xyz = np.asarray(com_xyz)
                # The article describes GNSS resampling to the 240 Hz IMU rate.
                # The XLSX itself has no timestamp column: retain this assumption.
                velocity = np.gradient(xyz[:, :2], 1 / 240., axis=0)
                speed = np.linalg.norm(velocity, axis=1)
                result["source_com_kinematics"] = {
                    "samples": len(xyz), "assumed_hz": 240,
                    "clock_basis": "Article methods: GNSS resampled to 240 Hz; no timestamp column in workbook",
                    "rate_source": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0347192",
                    "span_s_under_rate_assumption": (len(xyz) - 1) / 240.,
                    "source_xy_speed_m_s_p50_p95_max_under_rate_assumption": np.quantile(speed, [.5, .95, 1]).tolist(),
                    "global_xy_span_m": np.ptp(xyz[:, :2], axis=0).tolist(),
                    "global_z_span_m": float(np.ptp(xyz[:, 2])),
                    "caution": "Measured/processed HUMAN CoM coordinates, not robot-scale speed; no individual foot heading or contact state."
                }
            sheets.append(result)
    return sheets


def inspect_bvh(path):
    # Reuse the hash-locked GMR vendor BVH reader for FK only. No retarget call.
    from hdm05_gmr import load_gmr, read_source, axial
    settings = json.loads((ROOT / "config/hdm05_gmr_experiment.json").read_text())
    settings.update(bvh_hip_nodes=["LeftUpLeg", "RightUpLeg"],
                    bvh_knee_nodes=["LeftLeg", "RightLeg"],
                    bvh_ankle_nodes=["LeftFoot", "RightFoot"],
                    bvh_toe_base_nodes=["LeftToeBase", "RightToeBase"])
    _, _, extract, utils = load_gmr(settings)
    source = read_source(path, extract, utils, settings)
    xyz, index, t = source["positions_native"], source["index"], source["time_s"]
    lengths = [np.median(np.linalg.norm(xyz[:, index[knee]] - xyz[:, index[hip]], axis=1)
                        + np.linalg.norm(xyz[:, index[ankle]] - xyz[:, index[knee]], axis=1))
               for hip, knee, ankle in zip(settings["bvh_hip_nodes"], settings["bvh_knee_nodes"], settings["bvh_ankle_nodes"])]
    historical = ROOT / "results/hdm05_gmr_bd/HDM_bd_walkLeft2Steps_001_120/report.json"
    robot_leg = np.mean(json.loads(historical.read_text())["robot_leg_length_m_L_R"])
    scale = float(robot_leg / np.mean(lengths))
    xyz = scale * (xyz - xyz[:1, :1])
    root = xyz[:, 0]
    velocity = np.gradient(root[:, :2], t, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    course = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0]))
    ankle = xyz[:, [index[n] for n in settings["bvh_ankle_nodes"]]]
    toe = xyz[:, [index[n] for n in settings["bvh_toe_base_nodes"]]]
    direction = toe[:, :, :2] - ankle[:, :, :2]
    valid_axis = np.linalg.norm(direction, axis=-1) > .02
    heading = np.degrees(np.arctan2(direction[:, :, 1], direction[:, :, 0]))
    beta = axial(course[:, None] - heading)
    low = np.minimum(ankle[:, :, 2], toe[:, :, 2])
    near = low <= np.quantile(low, .05) + settings["human_near_ground_height_m"]
    stats = []
    for threshold in [.8, 1.5, 2.]:
        mask = (speed[:, None] >= threshold) & near & valid_axis & (np.abs(beta) >= 20)
        stats.append({"speed_threshold_robot_scale_m_s": threshold,
                      "samples_L_R": mask.sum(axis=0).tolist(),
                      "longest_s_L_R": [longest_run(t, mask[:, side]) for side in range(2)]})
    return {"frames": len(t), "duration_s": float(t[-1]), "dt_s": source["dt"],
            "scale_m_per_native_unit": scale, "robot_leg_normalization_m": float(robot_leg),
            "root_xy_span_robot_scale_m": np.ptp(root[:, :2], axis=0).tolist(),
            "root_speed_robot_scale_m_s_p50_p95_max": np.quantile(speed, [.5, .95, 1]).tolist(),
            "high_offset_near_ground_windows": stats,
            "limitations": ["Source FK only: NOT a GMR retarget result, not robot CoM speed.",
                            "Units normalized by leg length, not anatomical human SI speed.",
                            "Ankle-to-toe-base axis and pooled low-height proxy, not measured contact or tire slip.",
                            "These windows do not yet require course change or speed retention: not drift passes."]}


def longest_run(t, mask):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return max((float(t[b - 1] - t[a]) for a, b in edges.reshape(-1, 2)), default=0.)


def inspect_hdm_index():
    from download_hdm05_walking import RangeReader, ARCHIVE_URL
    with RangeReader(ARCHIVE_URL) as remote, zipfile.ZipFile(remote) as archive:
        paths = [item.filename for item in archive.infolist()
                 if "/hdm05/" in item.filename.lower() and item.filename.lower().endswith(".bvh")]
        classes = Counter("_".join(Path(p).name.split("_")[2:-2]) for p in paths)
        return {"archive_url": ARCHIVE_URL, "archive_etag": remote.etag,
                "archive_size_bytes": remote.size, "range_bytes_read": remote.downloaded_bytes,
                "bvh_members": len(paths), "named_class_counts": dict(sorted(classes.items())),
                "scope": "Available processed/hdm05 ZIP members, NOT all raw/full-take HDM05 recordings. An empty class denotes a nonstandard filename.",
                "limitations": "File/class inventory only; class labels do not establish drift or motion absence in inaccessible raw full takes."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/data/drift_candidates")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit", type=Path)
    parser.add_argument("--include-hdm-index", action="store_true")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    expected = json.loads(args.expected_audit.read_text())["sources"] if args.expected_audit else {}
    rows = {}
    for key, spec in SOURCES.items():
        path, payload = acquire(spec, args.data_dir)
        digest = hashlib.sha256(payload).hexdigest()
        if key in expected and digest != expected[key]["sha256"]:
            raise ValueError("Source changed since public audit: " + key)
        result = {**spec, "bytes": len(payload), "sha256": digest}
        if path.suffix == ".bvh":
            result["source_fk_audit"] = inspect_bvh(path)
        else:
            result["released_workbook_schema"] = xlsx_headers(payload)
        rows[key] = result
        print(key, len(payload), digest, flush=True)
    report = {"scope": "Source discovery only; excludes lateral/crossover steps, running and skating. Skiing is an optional scope-expansion candidate, not selected or retargeted.",
              "sources": rows, "policy_trained": False, "physics_validated": False}
    if args.include_hdm_index:
        report["hdm05_available_archive_inventory"] = inspect_hdm_index()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
