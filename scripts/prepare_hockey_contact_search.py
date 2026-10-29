"""Build immutable code/config reward screens, reusing authorized remote assets."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import tarfile

from evaluate_hockey_physx import sha

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def protocols(spec, parent, bilateral):
    cells = []
    for reward in spec['rewards']:
        for model in spec['models']:
            cfg = copy.deepcopy(parent)
            cfg.update(kind=spec['kind'], initialization='warm_start_stable_hockey_PPO_fresh_optimizer',
                warm_start=spec['warm_start'], contact_model=model, contact_search=spec,
                phase_clock='integer_reference_frames', adaptive_sampling='fall_only',
                rsi_full_clip_start_fraction=.5, static_dynamic_friction_ratio=model['static_dynamic_ratio'])
            cfg['formal'] = dict(num_envs=spec['num_envs'], iterations=spec['screen_iterations'])
            cfg['ppo'].update(learning_rate=spec['learning_rate'], init_noise_std=spec['exploration_std'],
                              entropy_coef=spec['entropy_coef'])
            cfg['reward_weights'].update(fall=-spec['fall_terminal_cost']/cfg['policy_dt_s'], leg_pose=reward['leg_pose_weight'])
            cfg['bilateral_sliding'] = copy.deepcopy(bilateral)
            cfg['bilateral_sliding'].update(weights=reward['weights'], gap_scale_m=reward['gap_scale_m'],
                gate_positive_tracking_on_bilateral_window=False, leg_pose_weight=reward['leg_pose_weight'],
                rationale='Dense task feedback plus independent contact shaping; contact remains first in external qualification')
            if reward.get('reward_refinement') is not None:
                from hockey_contact_refinement import validate_settings
                cfg['reward_refinement'] = validate_settings(reward['reward_refinement'])
            cells.append((model['id']+'_'+reward['id'], cfg))
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--spec', type=Path, default=ROOT/'config/hockey_contact_reward_search_v1.json')
    a = p.parse_args()
    if a.output.exists() or a.archive.exists():
        raise FileExistsError('Fresh snapshot required')
    spec = json.loads(a.spec.read_text())
    parent = ROOT/'results/hockey_speed_success_20261028_v1/source_snapshot_v1'
    cfg = json.loads((parent/'protocols/hockey_s100_seed42.json').read_text())
    bilateral = json.loads((ROOT/'config/hockey_bilateral_training_v1.json').read_text())
    a.output.mkdir(parents=True)
    # Pin the existing collector and qualifier, plus every transitive local import.
    old_manifest = ROOT/'results/hockey_split_friction_20261028_v1/source_snapshot_v1/manifest.json'
    files = {r['path'] for r in json.loads(old_manifest.read_text())['files'] if r['path'].endswith('.py')}
    files.update(('train_hockey_contact_search.py','hockey_contact_search.py', 'prepare_hockey_contact_search.py',
        'dispatch_hockey_contact_search.py','qualify_hockey_contact_search.py', 'dispatch_hockey_bilateral_capacity.py',
        'tron1_wheel_geometry.py', 'hockey_beyondmimic_task.py', 'evaluate_hockey_physx.py', 'analyze_hockey_ppo.py',
        'analyze_hockey_bilateral.py','analyze_hockey_ground_contact.py','analyze_hockey_support_failure.py',
        'audit_hockey_contact_timing.py','audit_physx_contact_float32.py','wheel_friction_capacity.py',
        'wheel_ground_geometry.py','evaluate_hockey_contact_priority.py'))
    tests = ['test_hockey_contact_search.py','test_hockey_bilateral_training.py']
    if any(reward.get('reward_refinement') is not None for reward in spec['rewards']):
        files.update(('hockey_contact_refinement.py','hockey_contact_refinement_math.py',
                      'analyze_hockey_contact_robustness.py'))
        tests.extend(('test_hockey_contact_refinement_math.py','test_hockey_contact_refinement_adapter.py',
                      'test_hockey_contact_robustness.py'))
    for file in sorted(files):
        shutil.copy2(ROOT/'scripts'/file, a.output/file)
    for name in tests:
        (a.output/'tests').mkdir(exist_ok=True)
        shutil.copy2(ROOT/'tests'/name,a.output/'tests'/name)
    shutil.copy2(parent/'upstream_provenance.json', a.output/'upstream_provenance.json')
    shutil.copy2(ROOT/'config/hockey_contact_priority_v4.json', a.output/'contact_priority.json')
    cells = []
    for name, protocol in protocols(spec, cfg, bilateral):
        rel = 'protocols/'+name+'.json'
        write(a.output/rel, protocol)
        cells.append(dict(id=name, protocol=rel, model=protocol['contact_model']['id'], reward=name.split('_')[-1]))
    reference = parent/'references/s100'
    plan = dict(kind=spec['kind'], cells=cells,
        screen_transitions_per_cell=spec['screen_transitions_per_cell'], maximum_screen_cells=len(cells),
        maximum_formal_transitions=len(cells)*spec['screen_transitions_per_cell'],
        reference_directory='/data/zengxiong/workspaces/tron1-hockey-speed-success-20261028-v1/source_snapshot_v1/references/s100',
        reference_sha256={f:sha(reference/f) for f in ('motion.npz','command_diagnostics.npz','report.json')},
        dependencies=[dict(path='/data/zengxiong/workspaces/tron1-hockey-speed-success-20261028-v1/runs_v1/status.json',
            manifest_sha256='a61d76d6a7003d874be48cb3304cb551a0e05c66afbcd51f1eed2b2d4e548197')],
        warm_start=spec['warm_start'], idle_wait_limit_s=21600, stop_after_nominal_pass=True,
        next_step_on_nominal_pass='33-case robustness qualification; nominal pass alone does not complete the research goal',
        old_training_remains_paused=True, external_nonlinear_not_in_this_native_batch=True)
    if spec.get('required_completed_predecessor') is not None:
        plan['required_completed_predecessor'] = spec['required_completed_predecessor']
    write(a.output/'plan.json', plan)
    rows = [dict(path=str(f.relative_to(a.output)), bytes=f.stat().st_size, sha256=sha(f))
            for f in sorted(a.output.rglob('*')) if f.is_file()]
    write(a.output/'manifest.json', dict(files=rows, file_count=len(rows), total_bytes=sum(r['bytes'] for r in rows),
        exclusions=['assets','policy weights','credentials','reference arrays reused remotely','videos']))
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.archive, 'w:gz') as tar:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():
                tar.add(f, arcname=str(f.relative_to(a.output)), recursive=False)
    print(json.dumps(dict(file_count=len(rows), bytes=sum(r['bytes'] for r in rows), archive_sha256=sha(a.archive))))


if __name__ == '__main__':
    main()
