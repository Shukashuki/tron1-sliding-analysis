"""Publish actual completed O PPO dual-engine evidence; no overall success claim.

Requires the same normalized actor, full native episodes, frozen metric screens,
actual four-panel replay and completed smoke/formal/continuation proofs.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import tarfile

import numpy as np

import analyze_ppo_ground_contact_v2o as audit
import render_ppo_ground_contact_v2o as renderer
from package_ground_contact_v2_partial import copy_bound, source_archive
from ppo_backend_400hz_contract import read_json, require, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT/"outputs/ground_contact_v2o_20261025_v1"
REPLAY = '''"""Read-only four-case O PPO native metric replay; no physical assets or actors."""
import argparse,copy,json,pathlib,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(a.repository.resolve()/"scripts"))
import analyze_ppo_ground_contact_v2o as audit
from ppo_backend_400hz_contract import read_json,require,sha256
manifest=read_json(root/"manifest.json")
for name,row in manifest["files"].items():
    require(sha256(root/name)==row["sha256"],"Changed published artifact: "+name)
for name,digest in manifest["source_sha256"].items():
    require(sha256(a.repository/name)==digest,"Restore frozen source in a fresh checkout: "+name)
with np.load(root/"native_states.npz",allow_pickle=False) as archive:
    actual=audit.recompute_compact({name:archive[name].copy() for name in archive.files})
summary=read_json(root/"analysis.json");protocol=read_json(root/"protocol.json");screens={}
for engine,row in zip(audit.BACKENDS,summary["rows"],strict=True):
    for variant,field in (("new","new_V2"),("baseline","original_grounded_baseline")):
        expected=copy.deepcopy(row[field]);expected.pop("support_force_source",None)
        require(actual[variant+"_"+engine]==expected,"Four-case native metrics differ: "+variant+engine)
    screen=audit.nominal_screen(row["new_V2"],row["original_grounded_baseline"],protocol,engine)
    require(screen==row["nominal_screen"],"Frozen nominal screen differs: "+engine)
    screens[engine]={"pass":screen["numeric_nominal_episode_pass"],"failed_checks":screen["failed_checks"]}
print(json.dumps({"four_case_native_metrics_exact":True,"screens":screens,
"physics_policy_inference_or_training_run":False,"overall_task_complete_claimed":False}))
'''


def source_ledger():
    todo = [Path(__file__).resolve(), Path(audit.__file__).resolve(), Path(renderer.__file__).resolve()]
    seen = set()
    while todo:
        path = todo.pop()
        if path in seen: continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                candidate = ROOT/"scripts"/((name or "").split(".")[0]+".py")
                if candidate.is_file() and candidate not in seen: todo.append(candidate)
    seen.add(audit.DEFAULT_PROTOCOL)
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(seen)}


def run(args):
    root, output = args.input_root.resolve(), args.output_dir.resolve()
    require(not output.exists(), "Fresh result directory required")
    training = root/"ppo_training_proof_v1"
    analysis, render = root/"ppo_dual_engine_analysis_v1", root/"ppo_render_v1"
    summary, provenance = read_json(analysis/"summary.json"), read_json(render/"provenance.json")
    protocol = audit.load_protocol()
    require(summary["kind"] == "PPO_ground_contact_V2o_nominal_offline_audit"
            and summary["status"] == "completed_analysis"
            and summary["analyzer_sha256"] == sha256(audit.__file__)
            and summary["protocol_sha256"] == audit.PROTOCOL_SHA256
            and summary["both_PPO_validation_backends_complete"] is True
            and summary["completed_validation_engines"] == ["physx", "mujoco"],
            "Completed actual O dual-engine analysis required")
    options = renderer.parser().parse_args(["--training-audit", str(training/"ground_contact_v2o_training_audit.json"),
        "--input-root", str(root/"ppo_nominal_raw_v1"), "--output-dir", str(output)])
    reference = renderer.graphics.metrics.reference_data(options.reference)
    _, cases, evidence, order, training_evidence = renderer.load_cases(options, reference)
    require(provenance["input_evidence"] == evidence and provenance["case_order"] == order
            and provenance["cases"] == cases and provenance["explicit_training_audit"] == training_evidence
            and provenance["renderer_sha256"] == sha256(renderer.__file__),
            "Actual four-case render/proofs differ")
    require(provenance["full_four_panel_comparison"] and provenance["explicit_subset"] is None
            and provenance["video_frames"] == 350 and provenance["video_size_pixels"] == [2560, 850]
            and provenance["camera_same_distance_all_cases_frames"]
            and not provenance["camera_yaw_follow"] and not provenance["camera_zoom"]
            and provenance["robot_mesh_alpha"] == 1.
            and not provenance["smoothing_interpolation_alignment_height_correction"],
            "Opaque original poses and fixed relative camera required")
    with np.load(analysis/"native_states.npz", allow_pickle=False) as archive_file:
        compact = audit.recompute_compact({name: archive_file[name].copy() for name in archive_file.files})
    for engine, row in zip(audit.BACKENDS, summary["rows"], strict=True):
        for variant, field in (("new", "new_V2"), ("baseline", "original_grounded_baseline")):
            expected = copy.deepcopy(row[field]); expected.pop("support_force_source", None)
            require(compact[variant+"_"+engine] == expected, "Compact metric mismatch")
        require(row["input_evidence"] == evidence["v2o_"+engine]
                and row["nominal_screen"] == cases["v2o_"+engine]["new_nominal_screen"],
                "Numerical result differs from actual render input")
    collection = read_json(root/"nominal_collection_v1.manifest.json")
    require(collection["kind"] == "completed_O_PPO_dual_engine_nominal_allowlisted_22_file_collection"
            and collection["formal_training_transitions"] == 12288000
            and collection["models_weights_credentials_native_binaries_included"] is False,
            "Execution-host strict completed collection required")
    for name, row in collection["files"].items():
        require(sha256(root/name) == row["sha256"], "Changed collected proof/recording: "+name)
    formal = read_json(training/"physx_formal_v1_launch.json")
    for name in ("continuation_status.json", "final_audit.json"):
        state = read_json(training/name)
        require(state["status"] == "nominal_dual_engine_validation_completed_requires_task_review"
                and state["training_pid"] == formal["pid"]
                and state["training_launch_record_sha256"] == sha256(training/"physx_formal_v1_launch.json")
                and len(state["children"]) == 2, "Incomplete continuation proof")
        for child in state["children"]:
            require(child["status"] == "completed_verified" and child["returncode"] == 0
                    and child["verification"] == evidence["v2o_"+child["backend"]]["evaluation_proof"],
                    "Completed child proof differs")
    review = read_json(root/"ppo_video_review_v1/decode_verification.json")
    require(review["frame_count"] == 350 and review["video_sha256"] == sha256(render/"fullbody_comparison.mp4"),
            "All frames must decode before publication")
    output.mkdir(parents=True)
    copied = {}
    def save(source, target, expected=None):
        destination = output/target; destination.parent.mkdir(exist_ok=True, parents=True)
        copied[target] = copy_bound(source, destination, expected)
    save(analysis/"summary.json", "analysis.json", sha256(analysis/"summary.json"))
    for name, digest in summary["artifact_sha256"].items(): save(analysis/name, name, digest)
    for name, digest in provenance["artifacts"].items(): save(render/name, name, digest)
    save(render/"provenance.json", "render_provenance.json")
    save(audit.DEFAULT_PROTOCOL, "protocol.json", audit.PROTOCOL_SHA256)
    save(root/"nominal_collection_v1.manifest.json", "execution_collection_manifest.json")
    save(root/"ppo_video_review_v1/decode_verification.json", "decode_verification.json")
    for name, row in collection["files"].items():
        if name.startswith("ppo_training_proof_v1/"):
            save(root/name, "training_proof/"+Path(name).name, row["sha256"])
        elif Path(name).suffix == ".json":
            save(root/name, name.removeprefix("ppo_nominal_raw_v1/"), row["sha256"])
    for stem in ("source_snapshot_v1", "continuation_snapshot_v1"):
        snapshot = read_json(root/(stem+".manifest.json"))
        require(snapshot["archive"]["sha256"] == sha256(root/(stem+".tar.gz"))
                and snapshot["models_weights_credentials_native_binaries_included"] is False,
                "Source-only executed snapshot bytes differ")
        if stem == "source_snapshot_v1":
            require(sha256(root/(stem+".tar.gz")) == collection["source_archive_sha256"],
                    "Training source archive differs from execution-host verification")
        members = {row["path"]: row for row in snapshot["files"]}
        with tarfile.open(root/(stem+".tar.gz")) as archive_file:
            entries = archive_file.getmembers()
            require(len(entries) == len(members) and {entry.name for entry in entries} == members.keys(),
                    "Executed snapshot member set differs")
            for entry in entries:
                path = Path(entry.name)
                require(entry.isfile() and not path.is_absolute() and ".." not in path.parts
                        and path.suffix in (".py", ".json")
                        and entry.size == members[entry.name]["bytes"], "Invalid executed source member")
                import hashlib
                require(hashlib.sha256(archive_file.extractfile(entry).read()).hexdigest() == members[entry.name]["sha256"],
                        "Executed source member bytes differ")
        for suffix in (".tar.gz", ".manifest.json"): save(root/(stem+suffix), "executed_snapshots/"+stem+suffix)
    source = source_ledger(); archive = source_archive(output, source)
    copied[archive["file"]] = {"sha256": archive["sha256"], "bytes": (output/archive["file"]).stat().st_size}
    (output/"replay.py").write_text(REPLAY)
    copied["replay.py"] = {"sha256": sha256(output/"replay.py"), "bytes": (output/"replay.py").stat().st_size}
    rows = []
    for engine, row in zip(audit.BACKENDS, summary["rows"], strict=True):
        current = row["new_V2"]; core = current["windows"]["core"]["tracking"]
        support = current["any_wheel_ground_support"]["full_contact"]
        rows.append({"engine": engine, "core_xy_rmse_m": core["xy_rmse_m"],
            "whole_xy_rmse_m": current["windows"]["whole_reference_overlap"]["tracking"]["xy_rmse_m"],
            "speed_retention": core["speed_retention"], "beta_rmse_deg": core["beta_rmse_deg"],
            "support_fraction": support["any_wheel_support_sample_fraction"],
            "longest_no_support_s": support["longest_no_support_observed_s"],
            "numeric_pass": row["nominal_screen"]["numeric_nominal_episode_pass"],
            "failed_checks": row["nominal_screen"]["failed_checks"]})
    status = {"status": ("ACTUAL_O_PPO_COMPLETED_DUAL_ENGINE_NUMERICAL_CRITERIA_FAILED"
                         if not any(row["numeric_pass"] for row in rows)
                         else "ACTUAL_O_PPO_DUAL_ENGINE_NOMINAL_COMPLETED_REQUIRES_TASK_REVIEW"), "rows": rows,
        "same_normalized_actor_sha256": collection["actor_sha256"], "initial_policy_sha256": collection["initial_policy_sha256"],
        "training_transitions": 12288000, "sample_size": 1, "mu": .7, "beta_on": True,
        "single_wheel_support_allowed": True, "free_absolute_height": True, "goal_complete": False,
        "full_manual_video_review_claimed": False, "new_training_or_evaluation_during_packaging": False}
    write_json(output/"status.json", status)
    copied["status.json"] = {"sha256": sha256(output/"status.json"), "bytes": (output/"status.json").stat().st_size}
    page = '<!doctype html><meta charset="utf-8"><title>O contact reward: PPO comparison</title><style>body{font:17px system-ui;max-width:1200px;margin:36px auto;line-height:1.5}video{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}</style><h1>Stance-command continuity penalty: PPO dual-engine comparison</h1><p>PhysX training from scratch,12,288,000 transitions, seed42. Same normalized actor at μ=0.7 in both validation engines. β ON,400 Hz physics,50 Hz policy, action-rate−0.5, original DR and actuators retained. K remains; a bounded pre-load-conditioned stance-leg command-jump cost is added with0.1 rad scale and unchanged external +0.4 weight. Wheel torque actions are excluded.</p><video controls preload="metadata" poster="comparison.png" src="fullbody_comparison.mp4"></video><p>Original / O in PhysX; Original / O in native CPU MuJoCo. Fixed relative camera distance and world angles, original recorded poses, opaque meshes. One wheel may lift, absolute body height is free.</p><table><tr><th>Engine</th><th>Core XY RMSE(m)</th><th>Whole XY RMSE(m)</th><th>Speed ratio</th><th>β RMSE(°)</th><th>Support fraction</th><th>Longest gap(s)</th><th>Numerical pass</th></tr>'
    for row in rows:
        page += '<tr>'+''.join('<td>'+html.escape(str(row[key]) if isinstance(row[key], (str,bool)) else format(row[key], '.6g'))+'</td>' for key in ("engine","core_xy_rmse_m","whole_xy_rmse_m","speed_retention","beta_rmse_deg","support_fraction","longest_no_support_s","numeric_pass"))+'</tr>'
    page += '</table><p>Numerical screens retain all original tracking, speed, reference-β, support and angular-HF criteria. N=1 per engine. This bundle does not claim population robustness, hardware readiness or completion of the overall task.</p><p><a href="status.json">Status</a> · <a href="analysis.json">Full metrics</a> · <a href="replay.py">Four-case read-only replay</a> · <a href="../ground_contact_v2o_cem_nominal_20261025_v1/report.html">Same-reward CEM negative case</a></p><pre>python replay.py --repository /path/to/tron1-sliding-analysis</pre><p>The replay recomputes all four recorded cases without physical models, actors, native libraries, inference or dynamics. Restore executed_source.tar.gz only into a fresh isolated checkout if source pins differ. The original training, smoke and successful continuation launch records are preserved in training_proof.</p>'
    (output/"report.html").write_text(page)
    copied["report.html"] = {"sha256": sha256(output/"report.html"), "bytes": (output/"report.html").stat().st_size}
    write_json(output/"manifest.json", {"kind": "actual_O_PPO_dual_engine_nominal_snapshot", "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": copied, "source_sha256": source, "source_archive": archive,
        "models_weights_credentials_native_libraries_included": False, "goal_complete": False})
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(p.parse_args())))
