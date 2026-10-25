"""Publish recorded native force/command diagnostics, without training or actors."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np

import analyze_native_contact_dynamics as analysis
from package_ground_contact_v2_partial import copy_bound, source_archive
from ppo_backend_400hz_contract import read_json, require, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT/"outputs/native_contact_dynamics_20261025_v1"
REPLAY = '''"""Recompute both recorded contact analyses, without actors or new simulation."""
import argparse,json,pathlib,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True)
p.add_argument("--output-dir",type=pathlib.Path,required=True)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(a.repository.resolve()/"scripts"))
import analyze_native_contact_dynamics as analysis
from ppo_backend_400hz_contract import read_json,require,sha256
manifest=read_json(root/"manifest.json")
for name,row in manifest["files"].items():
    require(sha256(root/name)==row["sha256"],"Changed published artifact: "+name)
for name,digest in manifest["source_sha256"].items():
    require(sha256(a.repository/name)==digest,"Restore frozen sources in a fresh checkout: "+name)
analysis.BASELINES={tag:root/(tag+"_native_v1/physics_trajectory.npz") for tag in ("k","n")}
actual=analysis.run(argparse.Namespace(collection=root,output_dir=a.output_dir))
expected=read_json(root/"analysis.json")
require(actual["rows"]==expected["rows"],"Contact, command or event metrics do not reproduce")
with np.load(root/"native_contact_channels.npz",allow_pickle=False) as x, \\
     np.load(a.output_dir/"native_contact_channels.npz",allow_pickle=False) as y:
    require(x.files==y.files,"Different native signal names")
    for name in x.files:
        require(np.array_equal(x[name],y[name],equal_nan=True),"Native signal differs: "+name)
print(json.dumps({"both_recorded_contact_analyses_exact":True,"all_native_channels_exact":True,
"new_physics_training_or_inference":False,"causality_claimed":False,"goal_complete":False}))
'''


def source_ledger():
    todo = [Path(__file__).resolve(), Path(analysis.__file__).resolve(),
            ROOT/"scripts/eval_physx_native_contact_diagnostic.py"]
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
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(seen)}


def run(args):
    root, output = args.input_root.resolve(), args.output_dir.resolve()
    require(not output.exists(), "Fresh publication directory required")
    folder = root/"analysis_v2"; summary = read_json(folder/"analysis.json")
    require(summary["kind"] == "native_PhysX_contact_and_command_dynamics_readonly_diagnostics"
            and summary["analyzer_sha256"] == sha256(analysis.__file__)
            and len(summary["rows"]) == 2 and summary["task_success_claimed"] is False,
            "Actual frozen two-case analysis required")
    collection = read_json(root/"native_contact_collection_v1.manifest.json")
    require(collection["independent_policy_seeds_added"] == 0
            and collection["models_weights_credentials_native_binaries_included"] is False,
            "Observer collection scope differs")
    for name, row in collection["files"].items():
        require(sha256(root/"raw_v1"/name) == row["sha256"], "Changed collected input: "+name)
    for tag in ("k", "n"):
        metadata = read_json(root/"raw_v1"/(tag+"_native_v1/native_contact_diagnostic.json"))
        require(metadata["baseline_recording_byte_identical"] is True
                and sha256(analysis.BASELINES[tag]) == metadata["baseline_recording_sha256"],
                "Baseline byte equality is not verified")
    output.mkdir(parents=True)
    copied = {}
    def save(source, name, expected=None):
        target = output/name; target.parent.mkdir(exist_ok=True, parents=True)
        copied[name] = copy_bound(source, target, expected)
    save(folder/"analysis.json", "analysis.json", sha256(folder/"analysis.json"))
    for name, digest in summary["artifacts"].items(): save(folder/name, name, digest)
    for name, row in collection["files"].items(): save(root/"raw_v1"/name, name, row["sha256"])
    for tag in ("k", "n"):
        save(analysis.BASELINES[tag], tag+"_native_v1/physics_trajectory.npz", summary["evidence"][tag]["baseline"]["sha256"])
    save(root/"native_contact_collection_v1.manifest.json", "collection_manifest.json")
    for suffix in (".tar.gz", ".manifest.json"): save(root/("source_snapshot_v1"+suffix), "observer_source_snapshot_v1"+suffix)
    source = source_ledger(); archive = source_archive(output, source)
    copied[archive["file"]] = {"sha256": archive["sha256"], "bytes": (output/archive["file"]).stat().st_size}
    (output/"replay.py").write_text(REPLAY)
    copied["replay.py"] = {"sha256": sha256(output/"replay.py"), "bytes": (output/"replay.py").stat().st_size}
    page = '<!doctype html><meta charset="utf-8"><title>Native contact and command timing</title><style>body{font:17px system-ui;max-width:1200px;margin:36px auto;line-height:1.5}img{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}</style><h1>Native PhysX: support loss and command timing</h1><p>K and N frozen policies at μ=0.7. Extra native contact readback reproduced each existing trajectory byte for byte. No tire-law, material, actuator, reward or policy update was applied. These are additional observations of existing same-seed cases.</p><p>Support loss clusters immediately after control-command updates and is associated with larger stance-leg target changes. Body β and contact-point sliding differ. The following relationships are descriptive; they do not establish a causal mechanism.</p><img src="contact_and_command_timing.png"><p>Both columns use the same axes. Blue/orange: left/right wheel. Red control-update samples lose support in the first2.5 ms; green samples remain supported. Stance jump is the minimum RMS leg-target change among previously loaded wheels; the other leg may move freely.</p><table><tr><th>Policy</th><th>Core gap onsets</th><th>At first substep</th><th>Median body |β|(°)</th><th>Median wheel |α|(°)</th><th>Lateral slip + near friction limit</th></tr>'
    for row in summary["rows"]:
        command = row["command_transitions"]
        values = [row["cohort"].upper(), str(command["core_gap_onsets"]), str(command["gap_onset_physics_phase_0_through7"][1]),
                  format(row["core_loaded_abs_body_beta_deg_quantiles_50_95_100"][0], '.4g'),
                  format(row["core_loaded_abs_wheel_hub_angle_deg_quantiles_50_95_100"][0], '.4g'),
                  format(100*row["core_loaded_fraction_joint_lateral_slip_and_near_limit"], '.4g')+'%']
        page += '<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in values)+'</tr>'
    page += '</table><p>The last column uses loaded wheel samples (normal force&gt;5 N), contact lateral speed≥0.1 m/s and aggregate |Ft|/(μ Fn)≥0.9. It identifies neither a pointwise friction cone nor drift from force saturation alone. The native impulse/contact readback is aligned to the preceding state; the write counter names the upcoming step. Force absence is not inferred to be geometric flight.</p><p><a href="analysis.json">Full event and command statistics</a> · <a href="replay.py">Recorded-data reproduction</a> · <a href="../ground_contact_v2n_ppo_nominal_20261025_v1/report.html">N PPO negative comparison</a></p><pre>python replay.py --repository /path/to/tron1-sliding-analysis --output-dir /tmp/new-contact-replay</pre><p>The bundle contains numerical trajectories, contact data and source only. No actor/checkpoint weights, physical models, credentials or compiled rollout binaries are included. Replay reads the two recordings, recomputes their statistics and regenerates the plot; it performs no new physics or training. Overall task completion is not claimed.</p>'
    (output/"report.html").write_text(page)
    copied["report.html"] = {"sha256": sha256(output/"report.html"), "bytes": (output/"report.html").stat().st_size}
    write_json(output/"manifest.json", {"kind": "recorded_native_PhysX_contact_command_timing", "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": copied, "source_sha256": source, "source_archive": archive, "numpy_version": np.__version__,
        "models_weights_credentials_native_binaries_included": False, "goal_complete": False})
    return {"files": len(copied), "both_baseline_recordings_byte_identical": True, "goal_complete": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(p.parse_args())))
