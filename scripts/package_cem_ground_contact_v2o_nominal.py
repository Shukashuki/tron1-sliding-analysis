"""Publish the completed negative O CEM case without claiming a PPO outcome.

Recorded metrics and all349 actual15-term stage rewards are recomputed before
publication. The small replay needs numeric reference data and source code;
it never loads a physical model, actor, shared library or credentials.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np

import analyze_cem_ground_contact_v2o as audit
import render_cem_ground_contact_v2o as renderer
from package_ground_contact_v2_partial import copy_bound, source_archive
from ppo_backend_400hz_contract import read_json, require, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT/"outputs/ground_contact_v2o_20261025_v1"

REPLAY = '''"""Read-only replay of O metrics and all349 recorded stage rewards."""
import argparse,json,pathlib,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(a.repository.resolve()/"scripts"))
import analyze_cem_ground_contact_v2o as audit
from ppo_backend_400hz_contract import read_json,require,sha256
manifest=read_json(root/"manifest.json")
for name,row in manifest["files"].items():
    require(sha256(root/name)==row["sha256"],"Changed published artifact: "+name)
for name,digest in manifest["source_sha256"].items():
    require(sha256(a.repository/name)==digest,"Restore frozen source in a fresh checkout: "+name)
with np.load(root/"actual/rollout.npz",allow_pickle=False) as f:
    raw={name:f[name].copy() for name in f.files}
summary=read_json(root/"analysis.json");report=read_json(root/"actual/report.json")
decisions=json.loads((root/"actual/solver_log.json").read_text())
reference=audit.base.metrics.reference_data(root/"reference.npz")
actual=audit.base.recorded_metrics(raw,reference)
require(actual==summary["actual_V2o"],"Native O metrics do not reproduce")
reward=audit.runner.recompute_actual_rewards(raw,report,decisions,reference_file=root/"reference.npz")
require(reward==summary["actual_O_reward_audit"],"Recorded15-term rewards do not reproduce")
screen=audit.base.nominal_screen(actual,summary["original_grounded_baseline"],read_json(root/"protocol.json"))
require(screen==summary["nominal_screen"],"Frozen nominal screen does not reproduce")
print(json.dumps({"current_O_metrics_exact":True,"recorded_stage_rewards_exact":True,"stages":349,
"numeric_nominal_episode_pass":screen["numeric_nominal_episode_pass"],"failed_checks":screen["failed_checks"],
"physics_policy_inference_or_training_run":False,"overall_task_complete_claimed":False}))
'''


def source_ledger():
    pending = [Path(__file__).resolve(), Path(audit.__file__).resolve(), Path(renderer.__file__).resolve()]
    paths = {ROOT/name for name in audit.SOURCE_PINS if not name.endswith(".py")}
    pending += [ROOT/name for name in audit.SOURCE_PINS if name.endswith(".py")]
    while pending:
        path = pending.pop()
        if path in paths: continue
        paths.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                candidate = ROOT/"scripts"/((name or "").split(".")[0]+".py")
                if candidate.is_file() and candidate not in paths: pending.append(candidate)
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def run(args):
    root, output = args.input_root.resolve(), args.output_dir.resolve()
    require(not output.exists(), "Fresh result directory required")
    analysis = root/"cem_actual_analysis_v1"; render = root/"cem_render_v1"
    summary = read_json(analysis/"summary.json")
    require(summary["nominal_screen"]["numeric_nominal_episode_pass"] is False,
            "This publication records the completed negative O case")
    provenance = read_json(render/"provenance.json")
    reference_path = audit.runner.paired.DEFAULT_REFERENCE
    reference = audit.base.metrics.reference_data(reference_path)
    options = renderer.parser().parse_args([
        "--analysis-summary", str(analysis/"summary.json"),
        "--analysis-sha256", sha256(analysis/"summary.json"),
        "--v2o-dir", str(root/"cem_actual_nominal_v1"), "--output-dir", str(output)])
    _, cases, evidence, order, binding = renderer.load_cases(options, reference)
    require(provenance["input_evidence"] == evidence and provenance["case_order"] == order
            and provenance["frozen_actual_analysis"] == binding
            and provenance["cases"] == cases, "Actual render and analysis inputs differ")
    require(provenance["renderer_sha256"] == sha256(renderer.__file__)
            and provenance["camera_same_distance_all_cases_frames"]
            and not provenance["camera_yaw_follow"] and not provenance["camera_zoom"]
            and provenance["robot_mesh_alpha"] == 1.
            and not provenance["smoothing_interpolation_alignment_height_correction"],
            "Fixed camera and original recorded poses required")
    for name, digest in provenance["artifacts"].items():
        require(sha256(render/name) == digest, "Changed rendered artifact: "+name)
    review = read_json(root/"cem_video_review_v1/decode_verification.json")
    require(review["frame_count"] == provenance["video_frames"] == 350
            and review["video_sha256"] == sha256(render/"fullbody_comparison.mp4"),
            "Complete video decode evidence required")
    output.mkdir(parents=True)
    copied = {}
    def save(source, target, expected=None):
        destination = output/target; destination.parent.mkdir(exist_ok=True, parents=True)
        copied[target] = copy_bound(source, destination, expected)
    save(analysis/"summary.json", "analysis.json", sha256(analysis/"summary.json"))
    for name in ("contact_diagnostics.json", "native_contact_context.npz"):
        save(analysis/name, name)
    for name, digest in provenance["artifacts"].items(): save(render/name, name, digest)
    save(render/"provenance.json", "render_provenance.json")
    for name, digest in summary["input_sha256"].items():
        save(root/"cem_actual_nominal_v1"/name, "actual/"+name, digest)
    save(audit.DEFAULT_PROTOCOL, "protocol.json", audit.PROTOCOL_SHA256)
    save(reference_path, "reference.npz", provenance["reference_sha256"])
    save(root/"cem_video_review_v1/decode_verification.json", "decode_verification.json")
    save(root/"cem_nominal_launch.json", "cem_launch.json")
    save(root/"cem_render_launch_v2.json", "render_launch.json")
    save(root/"cem_render_launch_v1.json", "render_initial_attempt_launch.json")
    save(root/"cem_render_attempt_v1_status.json", "render_initial_attempt_status.json")
    source = source_ledger(); archive = source_archive(output, source)
    copied[archive["file"]] = {"sha256": archive["sha256"], "bytes": (output/archive["file"]).stat().st_size}
    (output/"replay.py").write_text(REPLAY)
    copied["replay.py"] = {"sha256": sha256(output/"replay.py"), "bytes": (output/"replay.py").stat().st_size}
    actual = summary["actual_V2o"]; core = actual["windows"]["core"]["tracking"]
    support = actual["any_wheel_ground_support"]["full_contact"]
    status = {"status": "COMPLETED_CEM_O_NEGATIVE_NOMINAL_CASE", "mu": .7,
        "engine": "native CPU MuJoCo", "sample_size": 1, "beta_reward_on": True,
        "core_xy_rmse_m": core["xy_rmse_m"], "whole_xy_rmse_m": actual["windows"]["whole_reference_overlap"]["tracking"]["xy_rmse_m"],
        "core_speed_retention": core["speed_retention"], "core_beta_rmse_deg": core["beta_rmse_deg"],
        "any_wheel_support_fraction": support["any_wheel_support_sample_fraction"],
        "longest_no_support_s": support["longest_no_support_observed_s"],
        "failed_checks": summary["nominal_screen"]["failed_checks"],
        "numeric_nominal_episode_pass": summary["nominal_screen"]["numeric_nominal_episode_pass"],
        "PPO_results_included": False, "goal_complete": False,
        "qualitative_review": "Selected stills112/150/200/348 inspected; O remains upright in these stills, but loses support and overshoots the reference endpoint. All350 frames decoded; no full manual playback claim.",
        "new_absolute_height_or_tilt_failure_gate": False}
    write_json(output/"status.json", status)
    copied["status.json"] = {"sha256": sha256(output/"status.json"), "bytes": (output/"status.json").stat().st_size}
    page = '<!doctype html><meta charset="utf-8"><title>O contact reward: negative CEM case</title><style>body{font:17px system-ui;max-width:1200px;margin:36px auto;line-height:1.5}video,img{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}</style><h1>Stance-command continuity penalty: negative CEM result</h1><p>Native CPU MuJoCo, μ=0.7, β ON, fixed CEM budget, N=1. At least one wheel may support; the other may lift. The new reward retains K and subtracts a bounded stance-leg command-jump cost: 1-exp(-min_loaded mean3(delta clipped leg action squared)/0.1 squared). Load is measured before the new command; wheel torque actions are excluded. External contact reward weight remains +0.4. Other14 rewards and the nominal criteria are unchanged.</p><video controls preload="metadata" poster="comparison.png" src="fullbody_comparison.mp4"></video><p>Four horizontal panels: Original / A / K / O. Original poses, opaque meshes, fixed relative camera distance and world angles, translation only. Current wheel and body headings and native force-support samples are shown.</p><table><tr><th>Metric</th><th>O result</th></tr>'
    for name in ("core_xy_rmse_m", "whole_xy_rmse_m", "core_speed_retention", "core_beta_rmse_deg", "any_wheel_support_fraction", "longest_no_support_s"):
        page += '<tr><td>'+html.escape(name)+'</td><td>'+format(status[name], '.6g')+'</td></tr>'
    page += '</table><p><b>Task failed.</b> Core tracking, speed and intended nonzero beta pass, but whole-path tracking and continuous support fail. The longest unsupported interval is180 ms; endpoint tracking error reaches about0.67 m in the selected final still. No topple is inferred from the metric failure. Height and tilt are diagnostic, not new failure criteria. This is an offline CEM result, not a PPO result or a20 ms real-time controller.</p><p><a href="status.json">Status</a> · <a href="analysis.json">Full metrics</a> · <a href="replay.py">Read-only metric and reward replay</a></p><pre>python replay.py --repository /path/to/tron1-sliding-analysis</pre><p>Restore executed_source.tar.gz only into a fresh isolated checkout if source pins differ. The bundle includes numeric reference and trajectory data; physical models, actors, shared libraries and credentials are omitted. Original/A/K recordings remain bound in the render provenance. The replay independently recomputes the current O episode and all349 recorded15-term rewards.</p>'
    (output/"report.html").write_text(page)
    copied["report.html"] = {"sha256": sha256(output/"report.html"), "bytes": (output/"report.html").stat().st_size}
    write_json(output/"manifest.json", {"kind": "actual_O_CEM_negative_nominal_snapshot", "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": copied, "source_sha256": source, "models_weights_credentials_native_libraries_included": False,
        "source_archive": archive, "current_O_replay_scope": "all recorded metrics, frozen screen and349x15 stage rewards; no physics, policy or prior-controller replay", "goal_complete": False})
    return status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(p.parse_args())))
