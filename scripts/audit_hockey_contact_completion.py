"""Replay the frozen 33-case hockey gates from public recordings, without weights.

Actor execution was independently replayed on the remote host; this local
audit verifies that report and replays contact, friction, task, reset and
observation calculations. It never steps a simulator or changes a criterion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(a, b, path='root'):
    """Compare every recorded result, allowing only floating arithmetic noise."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), path)
        for key in a:
            compare(a[key], b[key], path+'.'+key)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path+f'[{i}]')
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        require(np.isfinite(a) and np.isfinite(b) and np.isclose(a, b, rtol=1e-6, atol=2e-6), path)
    else:
        require(a == b, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'evidence', 'nominal-evidence', 'reference', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh audit directory required')
    read = lambda p: json.loads(p.read_text())
    bundle = args.bundle.resolve()
    for entry in read(bundle/'manifest.json')['files']:
        path = (bundle/entry['path']).resolve()
        require(path.is_relative_to(bundle) and sha(path) == entry['sha256'], 'Frozen source changed')
    plan = read(bundle/'plan.json')
    require(len(plan['cells']) == 1, 'Require the selected continuation cell')
    cell = plan['cells'][0]
    protocol = bundle/cell['protocol']
    cfg = read(protocol)
    evidence = args.evidence
    status = read(evidence/'status.json')
    suite = read(evidence/'eval_suite/report.json')
    summary = read(evidence/'qualification/report.json')
    nominal = args.nominal_evidence/'run_v8'/cell['id']
    training_path = nominal/'formal/report.json'
    training = read(training_path)
    require(status['status'] == 'completed_all_cases_pass_requires_goal_audit', 'Suite not completed')
    require(status['current'] is None and all(x['returncode'] == 0 for x in status['completed']), 'Unfinished stage')
    require(status['training_jobs_started'] == 0, 'Policy retrained during validation')
    require(sha(evidence/'qualification/report.json') == status['qualification_sha256'], 'Qualification changed')
    identities = {
        'source_manifest_sha256': sha(bundle/'manifest.json'),
        'protocol_sha256': sha(protocol),
        'training_report_sha256': sha(training_path),
        'suite_report_sha256': sha(evidence/'eval_suite/report.json'),
        'screen_qualification_sha256': sha(nominal/'nominal_qualification.json'),
        'nominal_reset_anchor_sha256': sha(nominal/'nominal/final_rollout.npz'),
        'analyzer_sha256': sha(bundle/'analyze_hockey_contact_robustness.py'),
    }
    for key, digest in identities.items():
        require(summary[key] == digest, 'Provenance mismatch: '+key)
    actor_hash = training['actor_sha256']
    require(summary['actor_sha256'] == suite['actor_sha256'] == status['actor_sha256'] == actor_hash, 'Policy mismatch')
    require(training['status'] == 'training_completed_not_task_success', 'Training incomplete')
    budget = cfg['formal']['num_envs'] * cfg['formal']['iterations'] * cfg['ppo']['steps_per_env']
    require(training['transitions'] == budget == 9216000, 'Continuation budget incomplete')
    require((cfg['physics_dt_s'], cfg['decimation'], cfg['evaluation_mu']) == (.0025, 8, .7), 'Changed physics')
    require(sha(args.reference/'motion.npz') == cfg['reference_sha256'], 'Reference changed')
    sys.path.insert(0, str(bundle))
    from analyze_hockey_bilateral import qualify
    from analyze_hockey_ppo import audit_observations
    from analyze_hockey_contact_robustness import check_case_list, check_reset
    from evaluate_hockey_physx import load
    from hockey_speed_study import perturbed_root
    import torch
    torch.set_num_threads(2)
    for name in ('analyze_hockey_bilateral', 'analyze_hockey_ppo', 'analyze_hockey_contact_robustness',
                 'evaluate_hockey_physx', 'hockey_speed_study'):
        require(Path(sys.modules[name].__file__).resolve().parent == bundle, 'Import escaped snapshot')
    check_case_list(cfg['evaluation_cases'], suite['cases'])
    check_case_list(cfg['evaluation_cases'], summary['cases'])
    reference = load(args.reference/'motion.npz')
    command = read(args.reference/'report.json')
    anchor = load(nominal/'nominal/final_rollout.npz')
    pose = np.r_[anchor['root_pos'][0]+anchor['environment_origin_w'], anchor['root_quat'][0]]
    velocity = np.r_[anchor['root_com_vel'][0], anchor['root_omega'][0]]
    rows = []
    case_hashes = {}
    for case, entry, qualified in zip(cfg['evaluation_cases'], suite['cases'], summary['cases']):
        name = case['id']; directory = evidence/'eval_suite'/name
        run = read(directory/'report.json')
        archived_path = evidence/'qualification'/(name+'.json')
        archived = read(archived_path)
        require(sha(directory/'report.json') == entry['report_sha256'] == archived['run_report_sha256'], 'Run changed')
        require(sha(archived_path) == qualified['report_sha256'], 'Case qualification changed')
        require(run['actor_sha256'] == actor_hash and run['case_id'] == name, 'Case policy changed')
        require((run['mode'],run['worlds'],run['steps'],run['external_forces']) == ('eval',1,139,0), 'Invalid case')
        require(run['config_sha256'] == sha(protocol) and run['motion_sha256'] == cfg['reference_sha256'], 'Input changed')
        require(run['asset_sha256'] == cfg['asset_sha256'], 'Robot asset changed')
        for key in ('source_sha256','task_source_sha256','bilateral_source_sha256','reward_search_source_sha256',
                    'reward_refinement_source_sha256','reward_refinement_math_sha256'):
            require(run[key] == suite[key] == training[key], 'Case source changed: '+key)
        require(set(run['artifacts']) == {'case.json','final_rollout.npz','native_contact_buffers.npz'}, 'Missing artifacts')
        require(archived['input_sha256'] == run['artifacts'], 'Input hashes differ')
        for file, digest in run['artifacts'].items():
            require(sha(directory/file) == digest, 'Recording changed: '+name+'/'+file)
        material = run['material_audit']; geometry = run['geometry_audit']
        require(material['live_physx_materials_verified'] and material['ground_material_usd_verified'], 'Material unverified')
        for key in ('wheel_mu_range','wheel_mu_static_range'):
            np.testing.assert_allclose(material[key], [.7,.7], rtol=0, atol=1e-6)
        require(all(x['friction'] == 1 and x['combine'] == 'multiply' for x in material['ground_colliders']), 'Ground changed')
        require(geometry['model'] == cfg['contact_model']['geometry'] and geometry['per_wheel_shapes'] == 1
                and geometry['native_bodies_checked'] == 2 and geometry['external_forces'] == 0, 'Geometry changed')
        trace = load(directory/'final_rollout.npz'); raw = load(directory/'native_contact_buffers.npz')
        require(trace['root_pos'].shape == (1113,3) and trace['actions'].shape == (139,8), 'Truncated recording')
        reset = check_reset(trace, read(directory/'case.json'), case, pose, velocity, anchor['joint_pos'][0], perturbed_root)
        compare(reset, archived['initial_state_audit'], name+'.reset')
        obs = audit_observations(trace, reference, cfg)
        compare(obs, archived['observation_audit'], name+'.observations')
        require(0 <= archived['action_replay_max_error'] <= 5e-5, 'Remote actor replay failed')
        q = qualify(trace, raw, reference, command, mu=.7, mu_static=.7)
        compare(q, archived['qualification'], name+'.qualification')
        require(q['task_success_claimed'] and len(q['gates']) == 11 and all(q['gates'].values()), 'Task failed: '+name)
        gate = q['primary_contact_gate']
        require(gate['checked_intervals'] == 1104 and gate['terminal_contact_checked'], 'Incomplete contact coverage')
        rows.append(dict(id=name, xy_rmse_m=q['xy_RMSE_m'], xy_peak_m=q['xy_peak_m'],
            bilateral_interruption_ms=1000*gate['bilateral_interruption']['total_s'],
            entry_speed_m_s=q['entry_mean_speed_m_s'], approach_speed_ratio=q['approach_speed_ratio'],
            stop_hold_s=q['terminal_stop_hold_s'], wheel_heading_rmse_deg=q['touching_braking_wheel_heading_RMSE_deg'],
            slide_left_total_s=q['corroborated_slide_total_s'][0], slide_right_total_s=q['corroborated_slide_total_s'][1],
            slide_left_longest_s=q['corroborated_slide_longest_s'][0], slide_right_longest_s=q['corroborated_slide_longest_s'][1],
            rho_left_max=q['wheel_aggregate_capacity']['rho_dynamic'][0][2],
            rho_right_max=q['wheel_aggregate_capacity']['rho_dynamic'][1][2],
            angular_acceleration_rms_rad_s2=q['body_angular_acceleration_RMS_rad_s2'],
            remote_actor_replay_max_error=archived['action_replay_max_error']))
        case_hashes[name] = dict(report=sha(directory/'report.json'), qualification=sha(archived_path), **run['artifacts'])
        print('REPLAYED', name, flush=True)
    args.output.mkdir(parents=True)
    with (args.output/'cases.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    ranges = {key:dict(min=float(min(r[key] for r in rows)), max=float(max(r[key] for r in rows)),
                       mean=float(np.mean([r[key] for r in rows]))) for key in rows[0] if key != 'id'}
    report = dict(kind='hockey_contact_completion_audit_v1', task_complete_under_frozen_scope=True,
        nominal_success=True, perturbation_successes=32, perturbation_trials=32,
        conditional_Wilson_95_interval=summary['conditional_Wilson_95_interval'],
        all_33_qualifications_reproduced=True, all_11_gates_pass=True,
        both_wheels_all_400Hz_samples_and_endpoints=True, missing_contact_samples=0,
        physics_steps_performed=0, criteria_changed=False, actor_replayed_locally=False,
        actor_replayed_remotely=True, actor_sha256=actor_hash, weights_published=False,
        continuation_transitions=budget, cumulative_same_recipe_transitions=12288000,
        different_recipe_parent_transitions=12288000, source_sha256=sha(__file__),
        identities=identities, cases=rows, case_hashes=case_hashes, ranges=ranges,
        limitations=['One adaptively selected seed and one reference at mu_s=mu_d=0.7.',
            '32 prespecified initial-state perturbations; no new friction/model/speed generalization claim.',
            '400 Hz sampled geometric contact, not a continuous-time contact proof.',
            'Wheel-resultant friction evidence, not a per-anchor cone or observed static/dynamic solver flag.',
            'Frozen stop gate is planar; terminal XYZ motion is separately reported.',
            'Public replay needs NumPy and Torch plus the separately obtained locked reference; actor replay needs private weights.'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,4,figsize=(22,4.5),constrained_layout=True)
    x = np.arange(33)
    for ax,key,title in zip(axs, ('xy_rmse_m','bilateral_interruption_ms','slide_left_longest_s','rho_left_max'),
                            ('XY tracking','Bilateral contact','Longest corroborated slide','Maximum aggregate utilization')):
        ax.plot(x,[r[key] for r in rows],'.-',label='Left' if key in ('slide_left_longest_s','rho_left_max') else None)
        ax.set(title=title,xlabel='Case (0 = nominal; 1–32 = perturbations)'); ax.grid(alpha=.2)
    axs[0].axhline(.15,color='red',ls='--',label='Frozen limit'); axs[0].set_ylabel('RMSE [m]'); axs[0].legend()
    axs[1].set(ylabel='Interruption [ms]',ylim=(-1,1))
    axs[2].plot(x,[r['slide_right_longest_s'] for r in rows],'.-',label='Right')
    axs[2].axhline(.05,color='red',ls='--',label='Required on at least one wheel'); axs[2].set_ylabel('Duration [s]'); axs[2].legend(fontsize=8)
    axs[3].plot(x,[r['rho_right_max'] for r in rows],'.-',label='Right')
    axs[3].axhline(1,color='black',ls='--'); axs[3].set(ylabel='|sum Ft| / (0.7 sum Fn)',ylim=(.99,1.01)); axs[3].legend()
    fig.suptitle('Frozen v8 policy: 33/33 pass | mu_s = mu_d = 0.7 | 400 Hz native PhysX recordings')
    fig.savefig(args.output/'robustness_4x1.png',dpi=140); plt.close(fig)
    report['artifact_sha256'] = {p.name:sha(p) for p in args.output.iterdir()}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(successes=33,ranges=ranges)))


if __name__ == '__main__':
    main()
