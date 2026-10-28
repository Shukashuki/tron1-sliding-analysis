"""Prepare a bounded hockey-search snapshot and audit its initial residual fit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
from hockey_mppi_math import stop_basis


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    old=root/'results/hockey_stop_conditioning_physx_20261027_v1/evidence'
    source=old/'source_snapshot_v1';run=old/'conditioning_probes_v1/reference_dq'
    report=json.loads((run/'report.json').read_text())
    if sha(run/'final_rollout.npz')!=report['artifacts']['final_rollout.npz']:raise ValueError('Initialization trace changed')
    with np.load(run/'final_rollout.npz',allow_pickle=False) as z:actions=z['actions']
    with np.load(source/'motion.npz',allow_pickle=False) as z:q=z['joint_pos']
    b=stop_basis(139,12)
    desired=q[:,:6]-q[0,:6]
    if np.max(abs(desired))>1:raise ValueError('Reference exceeds the unchanged action range')
    residual=desired-actions[:,:6]
    mean=np.zeros((12,8))
    mean[:,:6]=np.linalg.solve(b.T@b+.001*np.eye(12),b.T@residual)
    np.savez_compressed(a.output/'initial_mean.npz',latent_mean=mean)
    config=json.loads((source/'config.json').read_text())
    config.update(kind='hockey_native_physx_mppi_v1',samples=63,iterations=6,knots=12,temperature=.5,
        noise_std=[.20]*6+[.35]*2,
        description='Hockey target preserved; reference-dq conditioned PPO plus bounded-prior MPPI; initial residual fitted on failed archived states, not a feasible demonstration',
        conditioning_table_sha256=sha(source/'nominal_conditioning.npz'),initial_mean_sha256=sha(a.output/'initial_mean.npz'),
        planning_weights=dict(xy=1.,velocity=1.,body_heading=.5,wheel_heading=.25,
            leg_pose=.5,body_up=.5,camber=.25,beta=.2,support_gap=10.,tilt_excess=10.,
            terminal_speed=2.,action_rate=.5,sliding_deficit=2.))
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for name in ('motion.npz','nominal_conditioning.npz','command_report.json'):
        shutil.copyfile(source/name,a.output/name)
    for name in ('generate_hockey_physx_mppi.py','hockey_mppi_math.py','dispatch_hockey_physx_mppi.py'):
        shutil.copyfile(root/'scripts'/name,a.output/name)
    frozen=root/'results/physx_policy_mppi_reference_20261025_v1/evidence/source_snapshot_v2/scripts'
    for name in ('physx_reference_math.py','mppi_shooting.py','physx_mjlab_recipe.py'):
        shutil.copyfile(frozen/name,a.output/name)
    fit=np.clip(actions[:,:6]+b@mean[:,:6],-1,1)
    audit=dict(physics_run=False,task_success_claimed=False,initialization='Least-squares leg residual fit on archived reference_dq states; wheel residual zero',
        source_trace_sha256=sha(run/'final_rollout.npz'),motion_sha256=sha(source/'motion.npz'),
        fixed_state_action_target_RMSE_before_rad=float(np.sqrt(np.mean((actions[:,:6]-desired)**2))),
        fixed_state_action_target_RMSE_after_rad=float(np.sqrt(np.mean((fit-desired)**2))),
        reference_action_max_abs=float(abs(desired).max()),initial_mean_max_abs=float(abs(mean).max()),
        terminal_residual_authority=float(b[-1].sum()),initial_residual_authority=float(b[0].sum()),
        first_action_unchanged=True,actual_closed_loop_behavior_unverified=True)
    (a.output/'initialization_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    files=[dict(path=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(a.output.iterdir()) if f.is_file()]
    manifest=dict(kind='hockey_native_physx_mppi_snapshot_v1',files=files,total_bytes=sum(f['bytes'] for f in files),
        contains_models_weights_credentials=False,contains_original_RGB=False,
        destination='zengxiong@14.103.233.29:/data/zengxiong/workspaces/tron1-hockey-mppi-20261027-v1/source_snapshot_v1/',
        scope='Serial single-world 16-step smoke, 64-world 139-step search with 6 MPPI updates, single-world 139-step final-mean validation; freshly idle RTX before each job; no training or other process changes',
        simulator_rollouts=dict(smoke=1,search_passes=8,worlds_per_search_pass=64,final_validation=1),
        mu=.7,physics_dt_s=.0025,policy_dt_s=.02,qualification_thresholds_unchanged=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=a.output.with_suffix('.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for path in sorted(a.output.iterdir()):
            info=tar.gettarinfo(str(path),arcname=path.name)
            info.uid=info.gid=info.mtime=0;info.uname=info.gname=''
            with path.open('rb') as f:tar.addfile(info,f)
    print(json.dumps(dict(archive=str(archive),archive_sha256=sha(archive),payload_bytes=manifest['total_bytes'],files=len(files),audit=audit),indent=2))


if __name__=='__main__':main()
