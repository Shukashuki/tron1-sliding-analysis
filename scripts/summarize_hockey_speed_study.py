"""Summarize learning-seed success and conditional rollout success separately."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import sha
from hockey_speed_study import wilson_interval


def summarize(plan, reports):
    by_id = {row['id']: row for row in plan['cells']}
    if len(by_id) != len(plan['cells']) or set(reports)-set(by_id):
        raise ValueError('Unexpected or duplicate study cells')
    for identity, report in reports.items():
        cell = by_id[identity]
        if any(report[key] != cell[key] for key in ('method', 'seed', 'speed_scale')):
            raise ValueError('Study cell mislabeled')
        if report['perturbation_trials'] != 32 or len(report['cases']) != 33:
            raise ValueError('Incomplete trial set cannot enter a success denominator')
        if [r['id'] for r in report['cases']] != [c['id'] for c in plan['cases']]:
            raise ValueError('Unpaired evaluation cases')
        k = sum(r['success'] for r in report['cases'] if r['kind'] == 'perturbed')
        if (k != report['perturbation_successes'] or not np.isclose(report['perturbation_success_rate'], k/32)
                or report['cases'][0]['kind'] != 'nominal'
                or report['nominal_success'] != report['cases'][0]['success']):
            raise ValueError('Success count differs from audited cases')
    groups = []
    for ref in plan['references']:
        for method in plan['methods']:
            cells = [c for c in plan['cells'] if c['method'] == method and c['speed_scale'] == ref['scale']]
            found = [reports[c['id']] for c in cells if c['id'] in reports]
            rates = [r['perturbation_success_rate'] for r in found]
            k = sum(r['nominal_success'] for r in found)
            failures = {}
            for r in found:
                for name, count in r['failure_gate_counts_nonexclusive'].items():
                    failures[name] = failures.get(name, 0)+count
            groups.append(dict(method=method, speed_scale=ref['scale'], reference_speed_m_s=ref['root_link_entry_speed_m_s'],
                expected_training_seeds=plan['seeds'], completed_training_seeds=[r['seed'] for r in found],
                nominal_seed_successes=k, completed_models=len(found),
                nominal_seed_success_rate=k/len(found) if found else None,
                seed_success_Wilson_95_interval=wilson_interval(k, len(found)) if found else None,
                conditional_rollout_rate_mean=float(np.mean(rates)) if found else None,
                conditional_rollout_rate_min_max=[min(rates), max(rates)] if found else None,
                per_seed_rollout_rates=[dict(seed=r['seed'], successes=r['perturbation_successes'], trials=32,
                    rate=r['perturbation_success_rate'], Wilson_95=r['conditional_Wilson_95_interval']) for r in found],
                failure_counts_nonexclusive=failures,
                missing_cells=[c['id'] for c in cells if c['id'] not in reports]))
    paired = []
    for ref in plan['references']:
        for seed in plan['seeds']:
            cells = [c for c in plan['cells'] if c['speed_scale'] == ref['scale'] and c['seed'] == seed]
            selected = {c['method']: reports[c['id']] for c in cells if c['id'] in reports}
            if set(selected) != {'beyondmimic', 'hockey'}:
                continue
            counts = dict(both_succeed=0, original_only=0, hockey_only=0, neither_succeeds=0)
            for old, new in zip(selected['beyondmimic']['cases'][1:], selected['hockey']['cases'][1:]):
                if old['id'] != new['id']:
                    raise ValueError('Unpaired cases')
                key = ('both_succeed' if old['success'] and new['success'] else 'original_only' if old['success']
                       else 'hockey_only' if new['success'] else 'neither_succeeds')
                counts[key] += 1
            paired.append(dict(speed_scale=ref['scale'], seed=seed, trials=32, counts=counts))
    return dict(status='complete' if len(reports) == len(by_id) else 'partial_or_not_started',
        cells_complete=len(reports), cells_planned=len(by_id), groups=groups, paired_reset_outcomes=paired,
        pooled_rollout_binomial_confidence_claimed=False,
        interpretation='Per-policy rollout intervals condition on that policy. Across-seed min/max is variability, not a confidence interval. Three training seeds give a preliminary learning-reliability estimate.',
        missing_cells=[identity for identity in by_id if identity not in reports])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    plan = json.loads(a.plan.read_text())
    reports = {}
    hashes = {}
    for cell in plan['cells']:
        path = a.runs/cell['id']/'analyze/report.json'
        if path.exists():
            reports[cell['id']] = json.loads(path.read_text())
            hashes[cell['id']] = sha(path)
    result = summarize(plan, reports)
    result.update(plan_sha256=sha(a.plan), source_sha256=sha(__file__), cell_report_sha256=hashes)
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (a.output/'summary.csv').open('w', newline='') as f:
        fields = ['method','speed_scale','reference_speed_m_s','nominal_seed_successes','completed_models',
                  'nominal_seed_success_rate','conditional_rollout_rate_mean']
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n');writer.writeheader()
        writer.writerows({key:g[key] for key in fields} for g in result['groups'])
    if reports:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)
        colors = {'beyondmimic':'#2874b2', 'hockey':'#d77325'}
        for method in plan['methods']:
            groups = [g for g in result['groups'] if g['method']==method and g['completed_models']]
            xs = np.array([g['speed_scale'] for g in groups])
            ys = np.array([g['nominal_seed_success_rate'] for g in groups])
            ci = np.array([g['seed_success_Wilson_95_interval'] for g in groups])
            if len(xs):
                shift = -.02 if method=='beyondmimic' else .02
                axes[0].errorbar(xs+shift, ys, yerr=np.stack((ys-ci[:,0], ci[:,1]-ys)),
                    marker='o', capsize=3, color=colors[method], label=method)
                for x,y,g in zip(xs,ys,groups):
                    axes[0].annotate(f"{g['nominal_seed_successes']}/{g['completed_models']}", (x+shift,y),
                        xytext=(2,7),textcoords='offset points',fontsize=8)
                means = np.array([g['conditional_rollout_rate_mean'] for g in groups])
                limits = np.array([g['conditional_rollout_rate_min_max'] for g in groups])
                axes[1].errorbar(xs+shift, means, yerr=np.stack((means-limits[:,0], limits[:,1]-means)),
                    marker='o', capsize=3, color=colors[method],label=method)
        boxes, labels, box_colors, failure_rows = [], [], [], []
        categories = [('XY', {'xy_rmse_le_0p15','xy_peak_le_0p3'}),
            ('Support', {'anywheel_continuously_supported'}), ('Tilt', {'tilt_below_45deg_during_task'}),
            ('Stop', {'upright_supported_terminal_hold_ge_0p5'}),
            ('Slide', {'loaded_lateral_sliding_during_braking_ge_0p05'})]
        for g in result['groups']:
            subset = [r for r in reports.values() if r['method']==g['method'] and r['speed_scale']==g['speed_scale']]
            trials = [c for r in subset for c in r['cases'] if c['kind']=='perturbed']
            if not trials:
                continue
            boxes.append([c['xy_rmse_m'] for c in trials])
            labels.append(('BM' if g['method']=='beyondmimic' else 'Hockey')+f" {g['speed_scale']:g}x")
            box_colors.append(colors[g['method']])
            failure_rows.append([np.mean([bool(set(c['failed_gates'])&keys) for c in trials]) for _,keys in categories])
        bp=axes[2].boxplot(boxes, tick_labels=labels, patch_artist=True, showfliers=True)
        for box,color in zip(bp['boxes'],box_colors):
            box.set_facecolor(color);box.set_alpha(.5)
        axes[2].axhline(.15,color='black',ls='--',lw=.8)
        axes[2].set_yscale('symlog',linthresh=.03)
        axes[2].tick_params(axis='x',rotation=55,labelsize=8)
        axes[3].imshow(failure_rows, vmin=0, vmax=1, cmap='Reds',aspect='auto')
        axes[3].set_xticks(range(len(categories)),[c[0] for c in categories],rotation=35)
        axes[3].set_yticks(range(len(labels)),labels,fontsize=8)
        for i,row in enumerate(failure_rows):
            for j,value in enumerate(row):
                axes[3].text(j,i,f'{value:.0%}',ha='center',va='center',fontsize=8,
                             color='white' if value>.65 else 'black')
        axes[0].set(title='Nominal success across training seeds',xlabel='Reference speed multiplier',
                    ylabel='Success rate; bars = Wilson 95%',ylim=(-.05,1.15))
        axes[1].set(title='Perturbed-reset success per policy',xlabel='Reference speed multiplier',
                    ylabel='Seed mean; bars = min/max, not CI',ylim=(-.05,1.05))
        axes[2].set(title='All perturbed trials, including failures',ylabel='XY RMSE [m]')
        axes[3].set(title='Selected failure modes (nonexclusive)')
        for ax in axes[:2]:
            ax.set_xticks([1.,1.5,2.]);ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle(f"Hockey stop: original tracking recipe vs direct hockey PPO | {result['cells_complete']}/18 cells complete | mu=0.7")
        fig.savefig(a.output/'learning_and_rollout_success_4x1.png',dpi=160)
        plt.close(fig)
    print(json.dumps(dict(status=result['status'], completed=result['cells_complete'], planned=result['cells_planned'])))


if __name__ == '__main__':
    main()
