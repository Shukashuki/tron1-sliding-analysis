"""English report of native point/load/gap distinctions; no task-success claim."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import analyze_native_contact_manifold as audit


def run(args):
    root = args.result
    data = audit.read(root/'analysis/analysis.json')
    assert data['status'] == 'completed_analysis' and data['analyzer_sha256'] == audit.sha(audit.__file__)
    assert not (root/'report.html').exists()
    table = []
    for row in data['rows']:
        values = (row['engine'], row['samples_below_anywheel_support'],
            row['below_support_with_any_native_points'], row['below_support_without_native_points'],
            f"{row['below_support_authored_min_gap_mm']['median']:.3f}",
            f"{row['below_support_authored_min_gap_mm']['max']:.3f}",
            f"{row['loaded_authored_min_gap_mm']['median']:.6f}",
            f"{row['point_force_to_parent_max_vector_difference_n']:.6g}")
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    text = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>TRON1: native manifold points versus load-bearing support</title><style>body{font:16px/1.55 system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#202830}img{max-width:100%}td,th{border:1px solid #cbd3da;padding:9px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf3f7}table{border-collapse:collapse;width:100%;font-size:14px}pre{white-space:pre-wrap;background:#f1f4f6;padding:15px}a{color:#0669a5}</style>
<h1>TRON1: native manifold points versus load-bearing support</h1>
<p>The K policy's support interruptions are not explained by a missing per-point force sum. Detailed native point forces recover the original force traces and5N masks. Reported point presence differs substantially across engines even when both wheels carry at most5N. Current-pose authored cylinder gaps provide a separate continuous diagnostic; they do not replace force-based acceptance.</p>
<p><a href="analysis/analysis.json">Native metrics and SHA-256 proofs</a> · <a href="analysis/signals.npz">Plot signals</a> · <a href="../ground_contact_v2p_ppo_nominal_20261025_v1/render/fullbody_comparison.mp4">Existing full-body video: K is columns1 and3</a></p>
<p>Same frozen normalized actor in both engines, nominal mu0.7,400Hz physics/50Hz actions. Each rerun reproduces its original K raw physical recording byte for byte. No model, material, action, reward or recorded pose is changed. N=1 per engine. The common analyzed interval0.02–6.96s has2777 actual samples in each engine.</p>
<table><tr><th>Engine</th><th>Samples with max wheel Fz≤5N</th><th>Of these: native point present</th><th>Of these: no native point</th><th>Median closest authored gap(mm), below support</th><th>Max closest authored gap(mm), below support</th><th>Median closest authored gap(mm), loaded</th><th>Max point-sum versus original force error(N)</th></tr>TABLE</table>
<p>The signed authored gap is positive above the plane and negative for geometric penetration. The closest wheel is selected independently at each sample, allowing one or alternating supporting wheels. Loaded means at least one wheel exceeds5N; below-support includes small positive loads up to5N and is not a claim of zero total force or geometric flight.</p>
<img src="analysis/comparison.png" alt="Native load, authored and native gaps, lowest-point normal velocity, native reported point counts in both engines">
<p>The figure shows detail1.8–3.5s; the table uses the full0.02–6.96s interval. Red shading denotes both wheel Fz≤5N. Missing native separation values are plotted as missing, never zero. Native report times are left unchanged; no lag fitting, time alignment, trajectory smoothing or state correction is performed.</p>
<h2>What was measured</h2>
<p>PhysX uses the installed107.3 tensor API <a href="https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidContactView.get_contact_data">get_contact_data(dt)</a> with the actual2.5ms conversion interval, the original two filtered wheel/ground views and their128-point capacities. The active count/start ranges select scalar normal forces, positions, normals and separations. Inactive buffer slots remain NaN. No view reaches capacity; at most2 reported points occur per wheel. The installed API file SHA is recorded; its SDK source is not copied into this repository.</p>
<p>MuJoCo records the actual <a href="https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjcontact">data.contact</a> structures and <a href="https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce">mj_contactForce</a> at the existing recorder hook. The original evaluator already calls mj_step followed by mj_forward before recording. The observer adds no forward/dynamics/state write. MuJoCo stores2793 point states including the final state; PhysX stores2792 pre-write states plus the unchanged2793-state parent record. Both cover the complete common analysis interval.</p>
<p>Point presence is a collision-report property, not identical to force-bearing support. Native contact manifolds, separation reporting and force timing remain backend-specific. This experiment does not identify which solver/CCD/contact-cache setting causes the reported point-persistence difference, and does not assert one simulator is more accurate against hardware.</p>
<h2>Authored cylinder geometry and normal motion</h2>
<p>For cylinder center c, unit axle a, unit ground normal n, radius r and half-width h, the support extent along n is r*sqrt(1-(a·n)^2)+h*abs(a·n). The plane gap is c·n minus that extent. The implementation retains the original cylinder center offsets and frame transforms; it uses the link frame for geometry and the actual COM frame for velocity transport. Lowest-point normal velocity comes from v_COM+omega×(p−p_COM), not finite differences across contact-point switches.</p>
<p>The authored current-pose surface gap is separate from native manifold separation. It is not a cooked-shape identity proof, a force estimate or a real-tire measurement. In this recipe, loaded authored penetration statistics differ between engines; those statistics alone do not isolate the physical cause. No absolute body-height target, bilateral support requirement, zero-slip objective or geometry-based failure gate is introduced.</p>
<h2>Implication for the next contact-reward experiment</h2>
<p>Keep the K force-based support criterion and task acceptance. When measured force vanishes, the force deficit saturates while the geometric gap and normal motion still vary. A continuous approach-to-ground signal is therefore a candidate for shaping credit toward recovery. Its runtime, reset/termination behavior and learned outcome still require a separate experiment. Native point count is unsuitable as a shared binary support reward here. No new controller or contact-reward policy was trained in this diagnostic.</p>
<pre>python -B scripts/analyze_native_contact_manifold.py --result results/native_contact_manifold_20261025_v1
python -B scripts/report_native_contact_manifold.py --result results/native_contact_manifold_20261025_v1</pre>
<p>Commands require fresh analysis/report output paths. The execution manifests and launch argv are under evidence/physx and evidence/mujoco. They reuse the existing environments, actor and physical assets; source snapshots contain only code/configuration/tests, with no neural weights, model assets, credentials or native binaries. Overall stable-contact task completion remains unachieved.</p></html>'''
    (root/'report.html').write_text(text.replace('TABLE', '\n'.join(table)))
    print(root/'report.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--result', type=Path, required=True)
    run(p.parse_args())
