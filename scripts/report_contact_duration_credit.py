"""Render the measured historical support-cost audit;no training claim."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np


def run(root):
    root=Path(root);a=json.loads((root/"analysis.json").read_text())
    assert a["kind"]=="offline_support_cost_temporal_credit_audit"
    assert a["physics_steps"]==a["policy_updates"]==0
    assert not a["trained_S_or_T_policy_claimed"] and not a["task_success_claimed"]
    table=[]
    with np.load(root/"costs.npz",allow_pickle=False) as arrays:
        for row in a["rows"]:
            key=row["id"];counts=arrays[key+"__unloaded_counts"][72:200]
            hist=np.bincount(counts,minlength=9)
            np.testing.assert_array_equal(hist,row["core_gap_count_histogram_0_to8"])
            assert int((counts>=2).sum())==row["core_multi_unloaded_windows_with_zero_K_cost_gain_for_repairing_one_substep"]
            values=[key,"128",str(int((counts>0).sum())),str(int((counts>=2).sum())),str(int(hist[8]))]
            table.append("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in values)+"</tr>")
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Support duration credit audit</title>
<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:auto;padding:24px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}pre{white-space:pre-wrap}.status{background:#fff1d2;padding:14px}</style>
<h1>Does the support cost distinguish shorter gaps?</h1>
<p class="status">An offline audit of six existing native episodes identifies a limitation in the worst-substep cost. It does not establish a PPO gradient failure or show that the new S/T rewards solve continuous support. The matched native learning test is recorded separately.</p>
<img src="comparison.png" alt="Historical unsupported-substep counts and prospective cost comparison">
<table><tr><th>Historical episode</th><th>Core policy windows</th><th>At least one missing-support sample</th><th>Two or more missing samples</th><th>All eight missing</th></tr>TABLE</table>
<p>Every source recording is SHA-checked against its completed analysis. Each 20 ms control window contains eight actual 2.5 ms force samples. Window starts are 1.44–3.98 s and their physical samples run 1.4425–4.00 s. This diagnostic has 128 windows;the nominal acceptance core separately includes 1,025 samples at 1.44–4.00 s. A missing-support sample means neither wheel has summed upward floor-normal force greater than 5 N. Individual contact-point counts are not substituted for wheel-summed force.</p>
<p>The 3D-axis PhysX actor fell in its original episode. Its post-fall support counts must not be interpreted as task success. The independent-camber actor remained upright but failed continuous support in both engines.</p>
<h2>One temporal change at matched cost scale</h2>
<pre>F_any,j = max(Fz_left,j,Fz_right,j,0)
d_j = clip((0.5*m*g - F_any,j)/(0.5*m*g - 5 N),0,1)
K cost = max_j d_j                 historical scale: [0,1]
S cost = max_j d_j + mean_j d_j    matched scale: [0,2]
T cost = 2*max_j d_j               matched scale: [0,2]
raw contact reward = unchanged gated positive G bonus - cost
external contribution = 0.4 * 0.02 s * raw contact reward</pre>
<p>S and T agree for every uniform-force window,including partial loading. If a window has at least two samples at or below 5 N,repairing just one to 0.5 mg while another remains leaves K and T cost unchanged. S cost improves by 1/8,corresponding to +0.001 in this dt-scaled reward contribution. The counterfactual holds positive G at zero to isolate the ungated cost;it changes recorded forces arithmetically and is not an integrated trajectory. The positive support bonus and all other rewards may still change in actual learning. A 0.5 mg shaping margin is not a new acceptance threshold:the original strict 5 N support test is retained.</p>
<p>The causal comparison is S versus T. Comparing S against historical K alone would also change the maximum penalty scale. Both prospective arms retain the same positive G term,held-reference-beta gate,14 other reward terms,DR,initial full K checkpoint,fresh Adam,learning rate 0.0001,clip 0.1 and 2,457,600 refinement transitions. Both use the same independent signed left/right camber reference,3D axle objective,400 Hz native physics and 50 Hz actions. Neither absolute-height constraints nor a bilateral support requirement are introduced.</p>
<p>The intended next native test is mu=0.7 in PhysX and CPU MuJoCo before any new friction sweep. All original nine nominal gates and the separate whole-body review remain. Reduced jitter after stopping or support after falling cannot qualify the task.</p>
<p><a href="analysis.json">Source hashes,window statistics and assumptions</a> · <a href="../contact_duration_pair_20261025_v1/CPU_checks_v3.json">23 tests in the extracted complete snapshot</a> · <a href="../../config/contact_duration_pair_400hz_v1.json">Frozen S/T protocol</a> · <a href="../../scripts/audit_contact_duration_credit.py">Audit reproduction</a> · <a href="../../scripts/ground_contact_duration_cost.py">Reward arithmetic</a> · <a href="../contact_duration_pair_20261025_v1/progress_v1.json">Time-stamped native execution observation</a>.</p></html>'''
    (root/"report.html").write_text(page.replace("TABLE","".join(table)))
    review=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),histograms_recomputed_from_saved_counts=True,comparison_plot_review_separate=True,task_success_claimed=False)
    (root/"report_verification_v1.json").write_text(json.dumps(review,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True);run(p.parse_args().result)
