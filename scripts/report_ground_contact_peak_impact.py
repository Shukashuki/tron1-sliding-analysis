"""Report force-only diagnostic and its limits; training results remain separate."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import numpy as np
import diagnose_ground_contact_peak_impact as diagnostic


def run(args):
    data = diagnostic.native.read(args.result/'analysis.json')
    assert data['diagnostic_source_sha256'] == diagnostic.native.sha(diagnostic.__file__)
    assert not (args.result/'report.html').exists()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = data['rows']
    x = np.arange(4)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), layout='constrained')
    ax[0].bar(x-.18, [r['over3mg_windows'] for r in rows], .36, label='over3mg')
    ax[0].bar(x+.18, [r['over3mg_and_gap_windows'] for r in rows], .36, label='over3mg and any support gap')
    ax[0].set(title='Held policy windows with high load', ylabel='Count /349 windows'); ax[0].legend(fontsize=8)
    ax[1].bar(x, [100*(1-r['positive_impact_gate_median_in_over3mg_windows']) for r in rows])
    ax[1].set(title='Old positive bonus reduction', ylabel='Median reduction at over3mg windows (%)')
    ax[2].bar(x, [r['new_ungated_cost_median_in_over3mg_windows'] for r in rows])
    ax[2].set(title='Proposed additional negative cost', ylabel='Median raw cost at over3mg windows')
    for a in ax:
        a.set_xticks(x, ['rigid', '1mm d1', '3mm d0', '3mm d1']); a.grid(axis='y', alpha=.2)
    fig.suptitle('Same frozen K actor: offline native-force arithmetic only, not a P policy result')
    fig.savefig(args.result/'comparison.png', dpi=150); plt.close(fig)
    body = '<!doctype html><meta charset="utf-8"><title>Peak-impact reward diagnostic</title><style>body{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto}img{width:100%}td,th{padding:9px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}</style><h1>Independent peak-impact cost: diagnostic before PPO training</h1><p>The current K actor has31 over3mg policy windows in the rigid PhysX turn,19 also containing a support interruption. The existing impact factor only multiplies the positive support bonus. Its median reduction at those31 windows is1.40%; it produces no independent cost when the positive bonus is zero. These observations motivate an experiment, not a predicted improvement.</p><img src="comparison.png" alt="Force window counts and isolated reward components"><table><tr><th>Model</th><th>Over3mg windows</th><th>Also gap</th><th>Full-recording peak (N)</th><th>Peak time (s)</th><th>Controlled/core peak (N)</th></tr>'
    for row in rows:
        body += '<tr>'+''.join('<td>'+html.escape(str(row[k]) if isinstance(row[k], str) else format(row[k], '.6g'))+'</td>' for k in ('id', 'over3mg_windows', 'over3mg_and_gap_windows', 'whole_recording_peak_force_n', 'whole_recording_peak_time_s', 'core_peak_force_n'))+'</tr>'
    body += '</table><p>The zero-damping9719 N peak is the t=0 initial readout before the first applied policy action, not a turn impact. Its controlled/core peak is1217.51 N. Reward windows use2792 applied physical samples grouped into349 held actions; the initial readout is excluded.</p><p>P preserves the entire K reward and adds a bounded ungated cost with the existing3mg threshold, coefficient1 and unchanged external+0.4 weight:</p><pre>excess=max(Fz/(3mg)-1,0)\ncost=1-exp(-max_over8_substeps_and2_wheels(excess**2))\nraw_P=raw_K-cost</pre><p>It does not impose absolute height, bilateral support, zero sideslip, force slew, a new velocity target, or action filtering. The other14 rewards, beta+2/std0.12,400Hz physics/50Hz policy, original DR, initial network and from-scratch budget remain fixed. A complete Isaac smoke is required before full training and nominal dual-engine validation.</p><p>Tracking, speed and beta qualities are set to1 only to isolate force arithmetic in this report. The actual training rewards were not recorded or reconstructed here. There is no causal claim that shock removal will restore support. The saved poses/actions/physical models remain unchanged.</p><p><a href="analysis.json">Metrics and source hashes</a> · <a href="force_windows.npz">All native force windows and isolated costs</a> · <a href="../physx_normal_compliance_20261025_v1/report.html">Completed contact-model transfer study</a></p><pre>python -B scripts/diagnose_ground_contact_peak_impact.py --collection results/physx_normal_compliance_20261025_v1/evidence --output-dir /tmp/peak-impact-diagnostic\npython -B scripts/report_ground_contact_peak_impact.py --result /tmp/peak-impact-diagnostic</pre><p>No trained P outcome or overall task completion is claimed by this diagnostic.</p>'
    (args.result/'report.html').write_text(body)
    print(args.result/'report.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--result', type=Path, required=True)
    run(p.parse_args())
