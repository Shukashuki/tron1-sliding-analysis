"""Prepare one unchanged-reward continuation after a completed failed screen.

The resulting dispatcher refuses to start until the entire source sweep finishes
without a nominal pass. This writes local code/configuration and starts no job.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

from hockey_training_continuation import matching_recipe

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/data/zengxiong/workspaces/tron1-hockey-contact-reward-search-20261028-v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(bundle, evidence, version, case, iterations, output, archive):
    if not re.fullmatch(r'v[1-9][0-9]*', version) or not re.fullmatch(r'[a-zA-Z0-9_]+', case):
        raise ValueError('Invalid source identifiers')
    if bundle.name != 'source_snapshot_'+version or type(iterations) is not int or iterations <= 0:
        raise ValueError('Invalid snapshot or additional iteration budget')
    if output.exists() or archive.exists():
        raise FileExistsError('Fresh continuation snapshot and archive required')
    manifest = read(bundle/'manifest.json')
    for row in manifest['files']:
        path = (bundle/row['path']).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Source snapshot changed')
    source_plan = read(bundle/'plan.json')
    cell = next(c for c in source_plan['cells'] if c['id'] == case)
    protocol_path = bundle/cell['protocol']; source = read(protocol_path)
    report_path = evidence/('run_'+version)/case/'formal/report.json'; report = read(report_path)
    qualification = read(report_path.parent.parent/'nominal_qualification.json')
    source_transitions = source['formal']['iterations']*source['formal']['num_envs']*source['ppo']['steps_per_env']
    if (report['status'] != 'training_completed_not_task_success' or report['transitions'] != source_transitions
            or report['config_sha256'] != sha(protocol_path) or report['bilateral_audit']['fault_seen']
            or qualification['source_manifest_sha256'] != sha(bundle/'manifest.json')
            or qualification['protocol_sha256'] != sha(protocol_path)
            or qualification['actor_sha256'] != report['actor_sha256']
            or qualification['qualification']['task_success_claimed']):
        raise ValueError('Require a completed failed nominal source with matching provenance')
    remote_run = REMOTE/('run_'+version)/case/'formal'
    warm_start = dict(checkpoint=str(remote_run/'model_final.pt'), checkpoint_sha256=report['checkpoint_sha256'],
                      actor=str(remote_run/'actor_normalized.pt'), actor_sha256=report['actor_sha256'])
    cfg = copy.deepcopy(source)
    cfg['formal']['iterations'] = iterations
    cfg.update(initialization='continue_PPO_optimizer_state_fresh_environment', warm_start=warm_start,
        training_continuation=dict(kind='hockey_optimizer_continuation_v1',
            source_report=str(remote_run/'report.json'), source_report_sha256=sha(report_path),
            source_protocol=str(REMOTE/bundle.name/cell['protocol']), source_protocol_sha256=sha(protocol_path)))
    matching_recipe(source, cfg)
    transitions = iterations*cfg['formal']['num_envs']*cfg['ppo']['steps_per_env']
    plan = copy.deepcopy(source_plan)
    plan.update(cells=[cell], screen_transitions_per_cell=transitions, maximum_screen_cells=1,
        maximum_formal_transitions=transitions, warm_start=warm_start,
        required_completed_predecessor=dict(status_path=str(REMOTE/('run_'+version)/'status.json'),
            manifest_sha256=sha(bundle/'manifest.json'), completed_screens=len(source_plan['cells'])),
        continuation_scope='Same reward/model/DR/optimizer; fresh environment and local counters; not bitwise resume',
        source_segment_transitions=source_transitions, source_training_report_sha256=sha(report_path))
    output.mkdir(parents=True)
    for row in manifest['files']:
        name=row['path']
        if name in ('plan.json', cell['protocol']) or name.startswith('protocols/'):
            continue
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(bundle/name,target)
    for name in ('train_hockey_contact_search.py','hockey_training_continuation.py',
                 'prepare_hockey_training_continuation.py'):
        shutil.copy2(ROOT/'scripts'/name,output/name)
    (output/'tests').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'tests/test_hockey_training_continuation.py',output/'tests/test_hockey_training_continuation.py')
    write(output/cell['protocol'],cfg);write(output/'plan.json',plan)
    rows=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha(p))
          for p in sorted(output.rglob('*')) if p.is_file()]
    write(output/'manifest.json',dict(files=rows,file_count=len(rows),total_bytes=sum(r['bytes'] for r in rows),
        source_manifest_sha256=sha(bundle/'manifest.json'),
        exclusions=['assets','weights','credentials','source videos','reference arrays reused remotely']))
    archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'w:gz') as stream:
        for path in sorted(output.rglob('*')):
            if path.is_file():stream.add(path,arcname=str(path.relative_to(output)),recursive=False)
    return dict(file_count=len(rows),total_bytes=sum(r['bytes'] for r in rows),
                source_segment_transitions=source_transitions,additional_transitions=transitions,
                source_manifest_sha256=sha(bundle/'manifest.json'),manifest_sha256=sha(output/'manifest.json'),
                archive_sha256=sha(archive),remote_files_written=False,jobs_started=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','evidence','output','archive'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('version','case'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--iterations',type=int,required=True,help='Additional iterations, excluding source training')
    args=parser.parse_args()
    print(json.dumps(prepare(args.bundle,args.evidence,args.version,args.case,args.iterations,args.output,args.archive)))


if __name__=='__main__':main()
