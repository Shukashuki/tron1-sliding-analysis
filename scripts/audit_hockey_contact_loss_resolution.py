"""Compare mean and mean-plus-worst contact costs on frozen rollouts.

This is counterfactual reward scoring, not retraining or a new success gate.
The training contact rule includes native pre-step contact and authored
post-step wheel clearance. Its missing time is reported separately from the
native-contact-only qualification, whose result is never replaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_timing import previous_state_trace
from audit_hockey_reference_contact_gap import wheel_geometry
from evaluate_hockey_physx import load, sha


def contact_scores(both, clearance, weights):
    if both.ndim != 2 or both.shape[1] != 8 or both.dtype != np.bool_:
        raise ValueError('Require eight bilateral Boolean samples per control window')
    if clearance.shape != (len(both),) or not np.isfinite(clearance).all():
        raise ValueError('Require finite per-window clearance')
    fraction = both.mean(-1)
    mean_loss = 1 - fraction
    proposed_loss = .5 * (mean_loss + (~both).any(-1))
    common = weights['contact_fraction'] * fraction + weights['clearance'] * clearance
    return dict(fraction=fraction, mean_loss=mean_loss, proposed_loss=proposed_loss,
                current_contact_score=common + weights['contact_loss'] * mean_loss,
                proposed_contact_score=common + weights['contact_loss'] * proposed_loss)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    read = lambda path: json.loads(path.read_text())
    sources = read(a.cases_json)
    if len(sources) != 4:
        raise ValueError('Use the established four-column comparison format')
    rows, arrays, plots = [], {}, []
    for number, source in enumerate(sources):
        bundle = (a.cases_json.parent / source['bundle']).resolve()
        evidence = (a.cases_json.parent / source['evidence']).resolve()
        cell, stage = source['case'].split(':')
        if stage != 'nominal':
            raise ValueError('Completed nominal qualification required')
        manifest = read(bundle / 'manifest.json')
        for item in manifest['files']:
            if sha(bundle / item['path']) != item['sha256']:
                raise ValueError('Frozen source changed')
        entry = next(c for c in read(bundle / 'plan.json')['cells'] if c['id'] == cell)
        protocol = bundle / entry['protocol']; cfg = read(protocol)
        run_dir = evidence / source['run_name'] / cell / stage
        qpath = run_dir.parent / 'nominal_qualification.json'; checked = read(qpath)
        run = read(run_dir / 'report.json')
        if (checked['protocol_sha256'] != sha(protocol)
                or checked['source_manifest_sha256'] != sha(bundle / 'manifest.json')
                or checked['run_report_sha256'] != sha(run_dir / 'report.json')):
            raise ValueError('Qualification identity mismatch')
        for name, digest in run['artifacts'].items():
            if sha(run_dir / name) != digest:
                raise ValueError('Raw recording changed')
        trace = load(run_dir / 'final_rollout.npz')
        raw = load(run_dir / 'native_contact_buffers.npz')
        dt = float(trace['physics_dt_s'])
        if len(trace['root_pos']) != 1113 or not np.isclose(dt, .0025, rtol=0, atol=1e-12):
            raise ValueError('Full 400 Hz recording required')
        b = cfg['bilateral_sliding']; tolerance = b['contact_tolerance_m']
        native = contact_kinematics(previous_state_trace(trace), raw, tolerance)
        pre_touch = native['touching'][1:].reshape(-1, 8, 2)
        _, _, gap = wheel_geometry(trace, 'body_pos', 'body_quat', b['geometry'])
        post_gap = gap[1:].reshape(-1, 8, 2)
        both = pre_touch.all(-1) & (post_gap <= tolerance).all(-1)
        excess = np.clip((post_gap - tolerance) / b['gap_scale_m'], 0, 1).max(-1)
        clearance = .5 * (excess.mean(-1) + excess.max(-1))
        scores = contact_scores(both, clearance, b['weights'])
        ends = np.arange(1, len(both) + 1) * .02; starts = ends - .02
        horizon = checked['qualification']['primary_contact_gate']['window_s'][1]
        inside = (starts >= -1e-9) & (ends <= horizon + 1e-9)
        loss = inside & ~both.all(-1)
        affected = []
        for i in np.flatnonzero(loss):
            affected.append(dict(window_s=[float(starts[i]), float(ends[i])],
                missing_substeps=int((~both[i]).sum()), clearance_cost=float(clearance[i]),
                current_contact_score=float(scores['current_contact_score'][i]),
                proposed_contact_score=float(scores['proposed_contact_score'][i])))
        rows.append(dict(cell=cell, qualifier_sha256=sha(qpath),
            source_manifest_sha256=sha(bundle / 'manifest.json'), raw_sha256=run['artifacts'],
            weights=b['weights'], control_windows=int(inside.sum()),
            native_pre_contact_missing_time_s=float((~pre_touch[inside].all(-1)).sum() * dt),
            training_rule_missing_time_s=float((~both[inside]).sum() * dt),
            affected_control_windows=int(loss.sum()),
            positive_current_contact_score_with_missing_substep_windows=int((loss & (scores['current_contact_score'] > 0)).sum()),
            positive_proposed_contact_score_with_missing_substep_windows=int((loss & (scores['proposed_contact_score'] > 0)).sum()),
            current_loss_integral=float((scores['mean_loss'][inside] * b['weights']['contact_loss'] * .02).sum()),
            proposed_loss_integral=float((scores['proposed_loss'][inside] * b['weights']['contact_loss'] * .02).sum()),
            current_contact_score_integral=float((scores['current_contact_score'][inside] * .02).sum()),
            proposed_contact_score_integral=float((scores['proposed_contact_score'][inside] * .02).sum()),
            task_success=checked['qualification']['task_success_claimed'], affected_windows=affected))
        signals = dict(control_start_s=starts, control_end_s=ends, within_reference=inside,
                       bilateral_substeps=both, clearance=clearance, **scores)
        arrays.update({f'case{number}_{key}': value for key, value in signals.items()})
        plots.append((source.get('label', cell), starts[inside], scores['current_contact_score'][inside],
                      scores['proposed_contact_score'][inside], loss[inside]))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'signals.npz', **arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharex=True, sharey=True)
    for ax, (label, time, old, new, missing) in zip(axes, plots):
        ax.step(time, old, where='post', label='Current: mean loss', color='#3267a8', linewidth=1.5)
        ax.step(time, new, where='post', label='Candidate: mean + worst', color='#d15828', linewidth=1.2)
        for t in time[missing]: ax.axvspan(t, t + .02, color='#ce4f4f', alpha=.1)
        ax.axhline(0, color='gray', linewidth=.6); ax.set_title(label, fontsize=10)
        ax.set_xlabel('Reference time (s)'); ax.grid(alpha=.2); ax.set_xlim(0, horizon)
    axes[0].set_ylabel('Contact terms before control-dt scaling')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Frozen validation trajectories: counterfactual contact-cost replay (no retraining)')
    fig.tight_layout(); fig.savefig(a.output / 'contact_loss_resolution_4x1.png', dpi=150); plt.close(fig)
    report = dict(kind='hockey_contact_loss_resolution_v1', script_sha256=sha(__file__),
        selection_sha256=sha(a.cases_json), signals_sha256=sha(a.output / 'signals.npz'),
        figure_sha256=sha(a.output / 'contact_loss_resolution_4x1.png'),
        current_formula='mean(missing) over eight physical intervals',
        proposed_formula='0.5 * (mean(missing) + max(missing)) over eight physical intervals',
        candidate_training_run=False, physics_steps=0, training_changed=False, criterion_changed=False,
        learned_result_predicted=False, causal_failure_mechanism_proven=False,
        note='Contact-only reward subset, not total task reward. Native pre-contact and training pre/post-geometry rules have different missing durations. Endpoint qualification remains authoritative.',
        cases=rows)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps([{k: v for k, v in row.items() if k != 'affected_windows'} for row in rows]))


if __name__ == '__main__':
    main()
