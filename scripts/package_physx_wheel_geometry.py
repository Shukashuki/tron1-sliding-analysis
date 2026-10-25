"""Publish native wheel diagnostics, source snapshots and read-only replay."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import tarfile

import analyze_physx_wheel_geometry as analysis

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/"outputs/physx_wheel_geometry_20261025_v1"

REPLAY = '''"""Native metric and tire-wrench replay; no Isaac, actors or physical models."""
import argparse,hashlib,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((root/"manifest.json").read_text())
for name,row in m["files"].items():assert sha(root/name)==row["sha256"],name
for name,digest in m["source_sha256"].items():assert sha(a.repository/name)==digest,name
sys.path.insert(0,str(a.repository/"scripts"))
import analyze_physx_wheel_geometry as audit
sources={v["path"]:v for v in json.loads((root/"evidence/source_snapshot_v5.manifest.json").read_text())["files"]}
summary=json.loads((root/"analysis.json").read_text())
for expected in summary["rows"]:
    actual,_=audit.analyze_case(root/"evidence/smooth_geometry_force_factorial_v5"/expected["id"],source_rows=sources)
    assert dict(id=expected["id"],**actual)==expected,expected["id"]
print(json.dumps(dict(native_cases_recomputed=6,external_wrenches_recomputed=True,xy_impulse_balance_checked=True,new_physics_or_training_run=False,task_success_claimed=False)))
'''


def run(a):
    source, output = a.input_root.resolve(), a.output_dir.resolve()
    assert not output.exists()
    collection = json.loads((source/"diagnostic_collection_v1.manifest.json").read_text())
    assert collection["kind"] == "six_completed_single_wheel_cases_with_preserved_setup_failures"
    assert collection["models_weights_credentials_native_binaries_included"] is False
    assert len(collection["files"]) == 97
    summary = json.loads((source/"analysis_v3/analysis.json").read_text())
    assert summary["status"] == "completed_analysis" and summary["analyzer_sha256"] == analysis.sha(analysis.__file__)
    src_manifest = json.loads((source/"collected_v1/source_snapshot_v5.manifest.json").read_text())
    rows = {r["path"]:r for r in src_manifest["files"]}
    assert all(analysis.sha(ROOT/n) == r["sha256"] for n,r in rows.items())
    for expected in summary["rows"]:
        actual,_ = analysis.analyze_case(source/"collected_v1/smooth_geometry_force_factorial_v5"/expected["id"],source_rows=rows)
        assert dict(id=expected["id"],**actual) == expected
    output.mkdir(parents=True)
    copied = {}
    def save(path, name, digest=None):
        path, relative = Path(path), Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts and path.is_file() and not path.is_symlink()
        data = path.read_bytes(); value = hashlib.sha256(data).hexdigest()
        assert digest is None or value == digest
        dest = output/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("xb") as stream:stream.write(data)
        assert analysis.sha(dest) == value
        copied[name] = dict(sha256=value,bytes=len(data))
    for name,row in collection["files"].items():
        save(source/"collected_v1"/name,"evidence/"+name,row["sha256"])
    save(source/"diagnostic_collection_v1.manifest.json","execution_collection_manifest.json")
    for name in ("analysis.json","comparison.png"):save(source/"analysis_v3"/name,name)
    sources = {n:r["sha256"] for n,r in rows.items()}
    for module in (Path(__file__).resolve(),Path(analysis.__file__).resolve()):
        sources[str(module.relative_to(ROOT))] = analysis.sha(module)
    archive = output/"executed_source.tar.gz"
    with tarfile.open(archive,"x:gz") as tar:
        for name in sorted(sources):tar.add(ROOT/name,arcname=name,recursive=False)
    copied[archive.name] = dict(sha256=analysis.sha(archive),bytes=archive.stat().st_size)
    (output/"replay.py").write_text(REPLAY)
    copied["replay.py"] = dict(sha256=analysis.sha(output/"replay.py"),bytes=(output/"replay.py").stat().st_size)
    page = '''<!doctype html><meta charset="utf-8"><title>PhysX wheel geometry and tire laws</title>
<style>body{font:17px system-ui;max-width:1200px;margin:32px auto;line-height:1.5}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}</style>
<h1>Single versus nine-cylinder wheels: six PhysX diagnostics</h1>
<p>Completed: one smooth cylinder / nine uniform axial smooth cylinders, each with native Coulomb, external regularized Coulomb and nonlinear combined-slip tire forces. μ=0.7, 400 Hz, initial hub side angle20°, speed1 m/s,2 seconds. These are single-wheel axle fixtures; TRON1 policy transfer and model-specific training have not run.</p>
<p>Each representation has one rigid body, equal0.127 m radius,0.05 m total width,1.08 kg mass, COM and explicitly matched principal inertia. Rotation about world X/Z is locked in this fixture; translation and wheel spin remain free. The original rendering cylinder has its collision APIs removed. Runtime shape counts are exactly1/9, respectively.</p>
<img src="comparison.png" alt="Lateral displacement and normal forces under the three force laws">
<table><tr><th>Force law</th><th>Slices</th><th>Peak normal force(N)</th><th>Samples above5N</th><th>Max patches</th><th>XY balance peak residual(N)</th></tr>'''
    for r in summary["rows"]:
        page += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (
            r["law"],r["slices"],round(r["normal_force_peak_n"],3),
            format(r["support_fraction_above5N"],".2%"),r["native_patch_count_max"],
            format(r["xy_momentum_balance_residual_peak_n"],".3g")))+'</tr>'
    page += '''</table><p>In these six traces, slicing reduces the largest normal-force peaks for native/nonlinear arms, but does not consistently improve support continuity. It increases the number of contact patches. This is one initial condition per arm, not a general stability finding or a sustained high-speed drift.</p>
<p>The measured native normal-force vector has nonzero XY components, despite the authored flat triangle floor. Horizontal impulse balance includes that whole normal vector, summed native friction patches and the applied external wrench. Its peak residual is below0.001 N in every case. XY force must not all be labeled tire friction.</p>
<p>External arms disable native tangential friction, retain native normal contact, and use the preceding physical sample's load/contact point. The nonlinear curve must therefore be compared first with the external Coulomb control with identical coupling. It is a dissipative combined-slip sensitivity law, without identified real-tire parameters or static adhesion; global-XY tire force uses global-Z load. Contact positions and force readings aggregate patches, so pointwise friction cones and individual slice loads are not identified.</p>
<p>Four earlier setup revisions are retained under evidence. In the final v4 same-source probes, the convex-approximation setting fails the support preflight, while smooth mode passes. This is an installed-configuration compatibility observation. Cooked geometry type has not been independently identified; it is not evidence that all PhysX convex wheels fail. The three proposed convex force-law arms have no usable task result.</p>
<p>USD cylinders and convex approximations are different representations; the installed setting and source are recorded. See <a href="https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html">NVIDIA collider documentation</a>. Explicit mass, COM and inertia follow <a href="https://openusd.org/dev/api/usd_physics_page_front.html">USD physics mass properties</a>. Uniform rigid slicing introduces no tire deformation. Authored contact/rest offsets remain automatic; effective cooked offsets were not read back, so these data do not isolate only manifold changes. Match explicit offsets or verify their runtime values before the full-robot comparison.</p>
<p>Current controller comparison is frozen separately: <a href="../ground_contact_v2o_ppo_nominal_20261025_v1/report.html">O PPO, two engines</a> and <a href="../ground_contact_v2o_cem_nominal_20261025_v1/report.html">O CEM</a>. Next: tilt/rolling and seam checks, then the declared frozen K actor across wheel-model arms at μ=0.7, followed by matched model-specific training. No new reward sweep or task-success claim is made here.</p>
<p><a href="analysis.json">Measurements</a> · <a href="execution_collection_manifest.json">Execution proofs</a> · <a href="replay.py">Read-only replay</a></p><pre>python replay.py --repository /path/to/tron1-sliding-analysis</pre>
<p>Replay verifies all published bytes, recomputes the six native metrics, all external tire wrenches and horizontal impulse balance. It needs NumPy and repository sources, without Isaac, actors, physical assets or native compilation. Source archives are pure code/configuration; restore only into a fresh checkout.</p>'''
    (output/"report.html").write_text(page)
    copied["report.html"] = dict(sha256=analysis.sha(output/"report.html"),bytes=(output/"report.html").stat().st_size)
    value=dict(kind="PhysX_single_wheel_geometry_force_diagnostic_publication",files=copied,
        source_sha256=sources,models_weights_credentials_native_binaries_included=False,
        completed_single_wheel_episodes=6,robot_policy_transfer_executed=False,
        model_specific_training_executed=False,task_success_claimed=False)
    (output/"manifest.json").write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    return dict(status="six_single_wheel_diagnostics_published",files=len(copied),output=str(output),task_success_claimed=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--input-root",type=Path,default=DEFAULT)
    p.add_argument("--output-dir",type=Path,required=True)
    print(json.dumps(run(p.parse_args())))
