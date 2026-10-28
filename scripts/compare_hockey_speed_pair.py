"""Compare two audited recipes at the same speed, training seed and reset draws.

Only read existing evidence; do not rerun physics or change success criteria.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'command-dir', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--speed-scale', type=float, default=1.)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    reference = load(a.command_dir/'motion.npz')
    rt = np.arange(len(reference['body_pos_w']))*.02
    methods = ('beyondmimic', 'hockey')
    labels = ('Earlier BM-style PPO', 'Hockey PPO')
    colors = ('#2874b2', '#d77325')
    rows, reports, cases, traces, signals, hashes = [], [], [], [], [], {}
    for method in methods:
        cell = a.runs/f'{method}_s{round(a.speed_scale*100):03d}_seed{a.seed}'
        path = cell/'analyze/report.json'
        report = json.loads(path.read_text())
        if (report['method'], report['speed_scale'], report['seed'], report['perturbation_trials']) != (method, a.speed_scale, a.seed, 32):
            raise ValueError('Different or incomplete study cell')
        hashes[str(path)] = sha(path)
        reports.append(report)
        group = []
        for entry in report['cases']:
            path = cell/'analyze'/(entry['id']+'.json')
            if sha(path) != entry['report_sha256']:
                raise ValueError('Case qualification changed')
            case = json.loads(path.read_text())
            group.append(case)
            q = case['qualification']
            rows.append(dict(method=method, case_id=entry['id'], kind=entry['kind'],
                success=q['task_success_claimed'], xy_rmse_m=q['xy_RMSE_m'], xy_peak_m=q['xy_peak_m'],
                entry_speed_m_s=q['entry_mean_speed_last_0p1s_m_s'], approach_speed_ratio=q['approach_COM_speed_ratio'],
                support_fraction=q['support_fraction'], longest_support_gap_s=q['longest_support_gap_s'],
                terminal_stop_hold_s=q['terminal_stop_hold_s'], first_45deg_tilt_s=case['first_45deg_tilt_s'],
                body_heading_rmse_deg=q['braking_body_heading_RMSE_deg'],
                loaded_wheel_heading_rmse_deg=q['loaded_braking_wheel_heading_RMSE_deg'],
                high_beta_slide_left_s=case['contact_events'][0]['pre_fall_braking_high_beta_saturated_dissipative_slip']['longest_duration_s'],
                high_beta_slide_right_s=case['contact_events'][1]['pre_fall_braking_high_beta_saturated_dissipative_slip']['longest_duration_s'],
                failed_gates=';'.join(q['failed_gates'])))
        cases.append(group)
        run = cell/'eval-suite/nominal'
        meta = json.loads((run/'report.json').read_text())
        if meta['motion_sha256'] != sha(a.command_dir/'motion.npz'):
            raise ValueError('Different reference')
        for name in ('final_rollout.npz', 'native_contact_buffers.npz'):
            if sha(run/name) != group[0]['input_sha256'][name]:
                raise ValueError('Changed nominal recording')
        traces.append(load(run/'final_rollout.npz'))
        path = cell/'analyze/nominal.signals.npz'
        signals.append(load(path)); hashes[str(path)] = sha(path)
    if len(cases[0]) != 33 or [c['case'] for c in cases[0]] != [c['case'] for c in cases[1]]:
        raise ValueError('Unpaired reset conditions')
    if reports[0]['source_manifest_sha256'] != reports[1]['source_manifest_sha256']:
        raise ValueError('Different frozen cohort')
    summaries = []
    for method, report, sig in zip(methods, reports, signals):
        selected = [row for row in rows if row['method'] == method]
        values = selected[1:]
        task = sig['time_s'] <= rt[-1]+1e-9
        nominal = dict(selected[0], maximum_tilt_deg=float(sig['root_tilt_deg'][task].max()))
        summaries.append(dict(method=method, nominal=nominal,
            perturbed_successes=report['perturbation_successes'], perturbed_trials=32,
            conditional_Wilson_95_interval=report['conditional_Wilson_95_interval'],
            failure_counts_nonexclusive=report['failure_gate_counts_nonexclusive'],
            perturbed_median_min_max={key: [float(f([r[key] for r in values])) for f in (np.median, np.min, np.max)]
                for key in ('xy_rmse_m', 'xy_peak_m', 'support_fraction', 'longest_support_gap_s',
                            'high_beta_slide_left_s', 'high_beta_slide_right_s')}))
    paired = dict(both_success=0, original_only=0, hockey_only=0, neither_success=0)
    for left, right in zip(cases[0][1:], cases[1][1:]):
        old, new = [x['qualification']['task_success_claimed'] for x in (left, right)]
        paired['both_success' if old and new else 'original_only' if old else 'hockey_only' if new else 'neither_success'] += 1
    result = dict(seed=a.seed, speed_scale=a.speed_scale, evaluated_friction=.7, summaries=summaries,
        paired_perturbation_outcomes=paired, input_report_and_signal_sha256=hashes,
        motion_sha256=sha(a.command_dir/'motion.npz'), script_sha256=sha(__file__),
        interpretation='One training seed per recipe; conditional reset success is not training reliability. Complete recipes differ, so no single-reward causal attribution.',
        physics_steps_performed=0, success_criteria_changed=False, pointwise_friction_cone_certified=False)
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (a.output/'paired_trials.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8), constrained_layout=True)
    axes[0].plot(reference['body_pos_w'][:, 0, 0], reference['body_pos_w'][:, 0, 1], '--', color='#299650', label='Reference')
    axes[1].plot(rt, np.linalg.norm(reference['body_lin_vel_w'][:, 0, :2], axis=1), '--', color='#299650', label='Reference COM speed')
    for side, (trace, sig, label, color) in enumerate(zip(traces, signals, labels, colors)):
        t = sig['time_s']; mask = t <= rt[-1]+1e-9
        axes[0].plot(trace['root_pos'][mask, 0], trace['root_pos'][mask, 1], color=color, label=label)
        axes[0].scatter(trace['root_pos'][np.flatnonzero(mask)[-1], 0], trace['root_pos'][np.flatnonzero(mask)[-1], 1], color=color, marker='x')
        axes[1].plot(t[mask], sig['body_speed_m_s'][mask], color=color, label=label)
        axes[2].plot(t[mask], sig['root_tilt_deg'][mask], color=color, label=label)
        for wheel in range(2):
            events = cases[side][0]['contact_events'][wheel]['pre_fall_braking_high_beta_saturated_dissipative_slip']['events']
            axes[3].broken_barh([(e['start_s'], e['duration_s']) for e in events], (3-side*2-wheel-.28, .56), facecolors=color)
    axes[0].set(title='Nominal XY tracking', xlabel='X [m]', ylabel='Y [m]'); axes[0].axis('equal')
    axes[1].set(title='Retained entry speed and braking', xlabel='Time [s]', ylabel='COM speed [m/s]')
    axes[2].axhline(45, ls='--', color='#a33', lw=1, label='45-degree gate')
    axes[2].set(title='Whole-task body tilt', xlabel='Time [s]', ylabel='Tilt [deg]')
    axes[3].set(title='Pre-fall loaded high-beta sliding', xlabel='Time [s]', xlim=(0, rt[-1]), ylim=(-.5, 3.5))
    axes[3].set_yticks([3,2,1,0], ['BM left', 'BM right', 'Hockey left', 'Hockey right'])
    for axis in axes[1:]:
        axis.axvline(.4, color='gray', lw=.7, ls=':'); axis.axvline(1.8, color='gray', lw=.7, ls=':')
    for axis in axes:
        axis.grid(alpha=.2)
    for axis in axes[:3]:
        axis.legend(fontsize=8)
    fig.suptitle(f'Matched PhysX hockey-stop recipes | speed x{a.speed_scale:g}, seed {a.seed}, mu=0.7 | sliding uses wheel-aggregate friction evidence')
    fig.savefig(a.output/'nominal_physics_comparison_4x1.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(output=str(a.output), paired=paired)))


if __name__ == '__main__':
    main()
