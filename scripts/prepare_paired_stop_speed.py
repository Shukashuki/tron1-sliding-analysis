"""Freeze a fixed-distance, paired PPO speed ladder from existing GMR references."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import mujoco
import numpy as np

from direct_brake_reset import instrument_trainer
from hockey_speed_study import evaluation_cases
from paired_stop_speed_reference import reference_pair, scaled_reset

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/data/zengxiong/workspaces/tron1-paired-stop-speed-20261103-v1'
MODEL=Path('/home/roy/tron1-learning-from-human/assets/robots/WF_TRON1A/mujoco/robot.xml')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def analysis_dependencies():
    todo=['score_paired_stop_speed'];seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        path=ROOT/'scripts'/(name+'.py')
        if not path.exists():continue
        seen.add(name)
        def visit(node):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):return
            candidates=([a.name for a in node.names] if isinstance(node,ast.Import) else
                        [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for value in candidates:
                module=value.split('.')[0]
                if (ROOT/'scripts'/(module+'.py')).exists():todo.append(module)
            for child in ast.iter_child_nodes(node):visit(child)
        visit(ast.parse(path.read_text()))
    return sorted(seen)


def build(output,handoff_sha):
    if output.exists():raise FileExistsError('Never overwrite a frozen study')
    source=output/'source_snapshot_v1';source.mkdir(parents=True)
    parent=ROOT/'results/hockey_speed_success_20261028_v1/source_snapshot_v1'
    contact=ROOT/'results/hockey_contact_reward_search_20261028_v1/source_snapshot_v8'
    provenance=[]
    for label,old,trainer in [('parent',parent,'train_hockey_ppo.py'),('contact',contact,'train_hockey_contact_search.py')]:
        dest=source/label;dest.mkdir()
        for path in sorted(old.glob('*.py')):
            text=instrument_trainer(path.read_text()) if path.name==trainer else path.read_text()
            (dest/path.name).write_text(text)
            provenance.append(dict(path=f'{label}/{path.name}',source=str(path.relative_to(ROOT)),sha256=sha(path),evaluation_only_patch=path.name==trainer))
        shutil.copyfile(ROOT/'scripts/direct_brake_reset.py',dest/'direct_brake_reset.py')
    (source/'analysis').mkdir()
    for name in analysis_dependencies():
        shutil.copyfile(ROOT/'scripts'/(name+'.py'),source/'analysis'/(name+'.py'))
    shutil.copyfile(ROOT/'scripts/dispatch_direct_brake_training.py',source/'runtime_helpers.py')
    shutil.copyfile(ROOT/'scripts/dispatch_paired_stop_speed.py',source/'dispatch_paired_stop_speed.py')
    shutil.copyfile(contact/'upstream_provenance.json',source/'upstream_provenance.json')
    oldref=ROOT/'results/hockey_stop_evaluation_audit_20261026_v1/physx_command_v2'
    oldreport=json.loads((oldref/'report.json').read_text())
    if sha(MODEL)!=oldreport['model_sha256']:raise ValueError('Frozen model changed')
    original=dict(np.load(oldref/'motion.npz',allow_pickle=False))
    diagnostic=dict(np.load(oldref/'command_diagnostics.npz',allow_pickle=False))
    shared=ROOT/'results/hockey_equal_distance_20261102_v1'
    reset=dict(np.load(shared/'common_evaluation_initial_state.npz',allow_pickle=False))
    budget=json.loads((shared/'report.json').read_text())['budget']
    model=mujoco.MjModel.from_xml_path(str(MODEL))
    pbase=json.loads((parent/'protocols/hockey_s100_seed42.json').read_text())
    cbase=json.loads((ROOT/'results/hockey_contact_reward_search_20261028_v1/source_snapshot_v7/protocols/native_equal_smooth_stop_xyz_huber_continuity.json').read_text())
    cases=evaluation_cases();levels=[]
    for factor in (1.5,2.,2.5,3.):
        label=f's{round(100*factor)}';folder=source/'references'/label;folder.mkdir(parents=True)
        pair,diagnostics,metadata=reference_pair(model,original,diagnostic,factor)
        common=scaled_reset(reset,factor)
        np.savez_compressed(folder/'common_initial_state.npz',**common)
        metadata.update(budget=budget,model_sha256=sha(MODEL),source_motion_sha256=sha(oldref/'motion.npz'),
            initial_physical_COM_planar_speed_m_s=float(np.linalg.norm(common['root_com_vel'][:2])),
            fixed_forward_distance_m=budget['total_distance_m'],
            same_XY_at_every_policy_frame=True,initial_pose_unchanged=True,
            initial_velocities_scaled_together=True,reference_is_not_a_dynamics_prediction=True)
        for behavior,motion in pair.items():
            np.savez_compressed(folder/(behavior+'_motion.npz'),**motion)
            np.savez_compressed(folder/(behavior+'_kinematics.npz'),**diagnostics[behavior])
            for stage,original_cfg in [('parent',pbase),('contact',cbase)]:
                cfg=copy.deepcopy(original_cfg)
                for key in ('warm_start','training_continuation'):cfg.pop(key,None)
                cfg.update(seed=43,speed_scale=factor,reference_sha256=sha(folder/(behavior+'_motion.npz')),
                    brake_start_s=metadata['brake_start_s'],brake_end_s=metadata['brake_end_s'],
                    evaluation_steps=metadata['frames'],evaluation_cases=cases,
                    status='frozen_paired_speed_v1',initialization='fresh_own_behavior_and_seed_full_lineage',
                    common_evaluation_initial_state=dict(path=REMOTE+'/source_snapshot_v1/references/'+label+'/common_initial_state.npz',
                        sha256=sha(folder/'common_initial_state.npz')),
                    speed_comparison=dict(kind='fixed_distance_speed_ladder_v1',behavior=behavior,level=label,
                        full_distance_m=budget['total_distance_m'],horizon_s=metadata['endpoint_s'],paired_training_seeds=True))
                if behavior=='direct':cfg['reward_weights']['beta']=0.
                if 'bilateral_sliding' in cfg:
                    cfg['bilateral_sliding'].update(seed=43,speed_scale=factor)
                    if behavior=='direct':cfg['bilateral_sliding']['weights']['sliding']=0.
                write(source/f'protocols/{label}_{behavior}_{stage}.json',cfg)
        write(folder/'report.json',metadata)
        levels.append(dict(id=label,scale=factor,initial_speed_m_s=metadata['initial_physical_COM_planar_speed_m_s'],
            reference_report=f'references/{label}/report.json',reference_dir=f'references/{label}',frames=metadata['frames']))
    previous=json.loads((ROOT/'results/direct_brake_training_20261102_v1/source_snapshot_v1/plan.json').read_text())
    plan=dict(kind='paired_fixed_distance_stop_speed_v1',levels=levels,pilot_seed=43,confirmation_seeds=[44,45],
        behaviors=['direct','hockey'],cases=33,minimum_passes_before_next_speed=30,
        boundary_rule='After BOTH policies finish: first level with either common task score <30/33; otherwise highest declared level.',
        confirmation_rule='Train both behaviors from scratch with seeds 44 and 45 at the selected level. Retain every result.',
        no_preferred_winner=True,upper_speed_scale=3.,nominal_mu_static=.7,nominal_mu_dynamic=.7,
        formal_transitions_per_policy=24576000,maximum_formal_transitions=(4+2)*2*24576000,
        parent_iterations=1000,contact_iterations=250,continuation_iterations=750,
        formal_num_envs=512,physics_dt_s=.0025,policy_dt_s=.02,
        evaluation='Same common stopping/contact/distance gates; lateral sliding is diagnostic, not required for either behavior.',
        matched_condition='Same full physical state at t=0 and same forward-distance budget; actual brake-entry speed is reported, not clamped.',
        effect_scope='Behavior/reward recipes; not an isolated causal beta ablation.',
        runtime=previous['runtime'],expected_versions=previous['expected_versions'],
        idle_wait_limit_s=86400,poll_seconds=30,one_GPU_job_at_a_time=True,
        predecessor=dict(root='/data/zengxiong/workspaces/tron1-direct-brake-20261102-v1',
            handoff_file='handoff_to_speed_20261103_v1.json',handoff_sha256=handoff_sha),
        deferred_low_speed_seeds='Preserved; current worker finishes; no additional low-speed job launches during speed comparison.')
    write(source/'plan.json',plan);write(source/'source_provenance.json',provenance)
    files=[dict(path=str(p.relative_to(source)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(source.rglob('*')) if p.is_file()]
    write(source/'manifest.json',dict(kind='paired_stop_speed_snapshot_v1',files=files,
        includes_models_weights_credentials_or_video=False))
    archive=output/'source_snapshot_v1.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(source.rglob('*')):
            if path.is_file():tar.add(path,arcname='source_snapshot_v1/'+str(path.relative_to(source)),recursive=False)
    write(output/'payload.json',dict(files=len(files),bytes=sum(r['bytes'] for r in files),
        archive_sha256=sha(archive),manifest_sha256=sha(source/'manifest.json'),remote_root=REMOTE))
    print((output/'payload.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--handoff-sha256',required=True)
    a=p.parse_args();build(a.output,a.handoff_sha256)
