"""Build a code/config-only snapshot; reuse the already uploaded original reference."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tarfile

from evaluate_hockey_physx import sha

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.archive.exists():raise FileExistsError('Fresh snapshot required')
    old=ROOT/'results/hockey_speed_success_20261028_v1/source_snapshot_v1'
    spec=json.loads((ROOT/'config/hockey_bilateral_training_v1.json').read_text())
    cfg=copy.deepcopy(json.loads((old/'protocols/hockey_s100_seed42.json').read_text()))
    cfg.update(kind=spec['kind'],bilateral_sliding=spec,status='prepared_not_trained',
        phase_clock='integer_reference_frames',adaptive_sampling='fall_only',
        rsi_full_clip_start_fraction=spec['full_clip_start_fraction'],friction_saturation_is_evaluation_only=False)
    cfg['reward_weights']['leg_pose']=spec['leg_pose_weight']
    a.output.mkdir(parents=True)
    files=['train_hockey_bilateral.py','hockey_bilateral_task.py','hockey_ppo_task.py','hockey_friction_boundary.py',
           'hockey_ppo_sampling.py','hockey_speed_study.py','height_smoothness.py','physx_mjlab_recipe.py',
           'physx_reference_math.py','evaluate_hockey_physx.py','analyze_hockey_ppo.py','analyze_hockey_bilateral.py',
           'analyze_hockey_ground_contact.py','analyze_hockey_support_failure.py','audit_hockey_contact_timing.py',
           'wheel_ground_geometry.py','evaluate_hockey_contact_priority.py','dispatch_hockey_bilateral_training.py',
           'prepare_hockey_bilateral_training.py']
    for file in files:shutil.copy2(ROOT/'scripts'/file,a.output/file)
    shutil.copy2(old/'upstream_provenance.json',a.output/'upstream_provenance.json')
    shutil.copy2(ROOT/'config/hockey_contact_priority_v4.json',a.output/'contact_priority.json')
    write(a.output/'protocol.json',cfg)
    reference={f:sha(old/'references/s100'/f) for f in ('motion.npz','command_diagnostics.npz','report.json')}
    write(a.output/'plan.json',dict(kind=spec['kind'],formal_training_runs=1,transitions=12288000,
        smoke=dict(environments=32,iterations=2),formal=dict(environments=512,iterations=1000),
        evaluation_episodes=33,seed=42,speed_scale=1.,mu=.7,
        reference_directory='/data/zengxiong/workspaces/tron1-hockey-speed-success-20261028-v1/source_snapshot_v1/references/s100',
        reused_reference_sha256=reference,cold_start=True,training_DR_retained=True,
        dependencies=[
          dict(path='/data/zengxiong/workspaces/tron1-hockey-speed-success-20261028-v1/runs_v1/status.json',
               expected_status='completed_all_18_cells',expected_stages=72,
               manifest_sha256='a61d76d6a7003d874be48cb3304cb551a0e05c66afbcd51f1eed2b2d4e548197'),
          dict(path='/data/zengxiong/workspaces/tron1-hockey-friction-boundary-20261028-v1/runs_v1/status.json',
               expected_status='completed_all_2_cells',expected_stages=8,
               manifest_sha256=sha(ROOT/'results/hockey_friction_boundary_20261028_v1/source_snapshot_v1/manifest.json'))],
        boundary_wrapper_status='/data/zengxiong/workspaces/tron1-hockey-friction-boundary-20261028-v1/deferred_status.json',
        dependency_wait_limit_s=86400,idle_wait_limit_s=21600,
        scheduling='One job at a time, after the already approved studies, completely idle RTX only; no process termination or automatic retries',
        changes=['bilateral geometry and corrected pre-step contact timing','positive rewards require a continuous bilateral control window',
                 'bounded contact penalties and useful sliding bonus','relaxed leg pose weight 1 to 0.25',
                 'bounded body angular acceleration penalty','full-clip RSI start probability 0.2 to 0.5',
                 'integer phase clock and fall-only adaptive sampling'],
        success='Independent native geometry plus full-horizon bilateral contact, then task and sliding qualification'))
    rows=[dict(path=str(f.relative_to(a.output)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(a.output.rglob('*')) if f.is_file()]
    write(a.output/'manifest.json',dict(kind=spec['kind'],files=rows,file_count=len(rows),total_bytes=sum(r['bytes'] for r in rows),
                                      exclusions=['robot assets','policy weights','credentials','videos','SMPL','reference NPZ (reused remotely)']))
    a.archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.archive,'w:gz') as tar:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():tar.add(f,arcname=str(f.relative_to(a.output)),recursive=False)
    print(json.dumps(dict(file_count=len(rows),bytes=sum(r['bytes'] for r in rows),archive_sha256=sha(a.archive),manifest_sha256=sha(a.output/'manifest.json'))))


if __name__=='__main__':main()
