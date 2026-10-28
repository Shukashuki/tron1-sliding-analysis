"""Package a two-arm reward ablation from the immutable hockey speed study."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tarfile

from prepare_hockey_speed_study import ROOT, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.archive.exists():
        raise FileExistsError('Use a fresh snapshot')
    source = ROOT/'results/hockey_speed_success_20261028_v1/source_snapshot_v1'
    spec = json.loads((ROOT/'config/hockey_friction_boundary_v1.json').read_text())
    base = json.loads((source/'protocols/hockey_s150_seed42.json').read_text())
    a.output.mkdir(parents=True)
    files = ['train_hockey_ppo.py','hockey_ppo_task.py','hockey_friction_boundary.py','hockey_speed_study.py',
        'height_smoothness.py','physx_mjlab_recipe.py','physx_reference_math.py','evaluate_hockey_physx.py',
        'analyze_hockey_ppo.py','analyze_hockey_speed_suite.py','dispatch_hockey_speed_study.py',
        'defer_hockey_boundary_study.py','prepare_hockey_boundary_study.py','prepare_hockey_speed_study.py']
    # Retain original task/material/reference/evaluator bytes. Only explicitly
    # listed integration scripts are taken from the new working implementation.
    new = {'train_hockey_ppo.py','hockey_friction_boundary.py','analyze_hockey_speed_suite.py',
           'dispatch_hockey_speed_study.py','defer_hockey_boundary_study.py','prepare_hockey_boundary_study.py'}
    for name in files:
        shutil.copy2((ROOT/'scripts' if name in new else source)/name, a.output/name)
    shutil.copytree(source/'references/s150', a.output/'references/s150')
    shutil.copy2(source/'upstream_provenance.json', a.output/'upstream_provenance.json')
    cells = []
    for arm, weight in spec['arms'].items():
        cfg = copy.deepcopy(base)
        cfg['friction_boundary'] = dict(spec['friction_boundary'], weight=weight)
        cfg['friction_saturation_is_evaluation_only'] = False
        cfg['status'] = 'prepared_boundary_ablation_not_trained'
        name = f'{arm}_s150_seed42'
        path = f'protocols/{name}.json'
        write(a.output/path, cfg)
        cells.append(dict(id=name,method='hockey',seed=42,speed_scale=1.5,protocol=path,
                          reference='references/s150',variant='tracking'))
    plan = dict(kind=spec['kind'], cells=cells, formal_training_runs=2,
        transitions_per_run=12288000,total_formal_transitions=24576000,total_evaluation_episodes=66,
        smoke_per_cell=dict(environments=32,iterations=2), formal_per_cell=dict(environments=512,iterations=1000),
        evaluation_mu=.7, paired_reset_cases=base['evaluation_cases'],
        source_protocol_sha256=sha(source/'protocols/hockey_s150_seed42.json'),
        frozen_success_evaluator_sha256=sha(source/'evaluate_hockey_physx.py'),
        difference='only friction-boundary reward weight 0 vs 0.5; identical native instrumentation',
        observation_abi_unchanged=True, all_existing_rewards_unchanged=True,
        support_deficit_weight=0., free_height=True, contact_sampling_hz=400, policy_hz=50,
        cold_start=True, parent_study_must_complete_first=True,
        scheduling='one job at a time, only a completely idle RTX; stop on any infrastructure failure')
    write(a.output/'plan.json',plan)
    manifest = [dict(path=str(f.relative_to(a.output)),bytes=f.stat().st_size,sha256=sha(f))
                for f in sorted(a.output.rglob('*')) if f.is_file()]
    write(a.output/'manifest.json', dict(kind=spec['kind'],files=manifest,file_count=len(manifest),
        total_bytes=sum(x['bytes'] for x in manifest),exclusions=['weights','robot assets','credentials','videos','SMPL']))
    a.archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.archive,'w:gz') as tar:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():
                tar.add(f,arcname=str(f.relative_to(a.output)),recursive=False)
    print(json.dumps(dict(files=len(manifest),bytes=sum(x['bytes'] for x in manifest),
        archive_sha256=sha(a.archive),manifest_sha256=sha(a.output/'manifest.json'))))


if __name__ == '__main__':
    main()
