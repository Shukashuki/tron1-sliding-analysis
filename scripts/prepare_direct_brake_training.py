"""Freeze direct-braking full-lineage PPO; never initialize from a hockey actor."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from direct_brake_reset import instrument_trainer
from hockey_speed_study import evaluation_cases

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/data/zengxiong/workspaces/tron1-direct-brake-20261102-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def direct_protocol(source, seed, reference_sha, reset_sha):
    cfg = copy.deepcopy(source)
    cfg.update(seed=seed, reference_sha256=reference_sha, status='frozen_direct_brake',
               initialization='own_seed_direct_brake_full_lineage_no_hockey_actor')
    cfg['reward_weights']['beta'] = 0.
    cfg['evaluation_cases'] = evaluation_cases()
    for key in ('warm_start', 'training_continuation'):
        cfg.pop(key, None)
    if 'bilateral_sliding' in cfg:
        cfg['bilateral_sliding']['seed'] = seed
        cfg['bilateral_sliding']['weights']['sliding'] = 0.
    cfg['direct_brake'] = dict(kind='equal_distance_direct_brake_v1',
        deliberate_lateral_sliding_reward=False, beta_reward=False,
        common_evaluation_initial_state=dict(path=REMOTE+'/source_snapshot_v1/common_evaluation_initial_state.npz', sha256=reset_sha),
        shared_physical_reset_is_evaluation_only=True,
        soft_straight_pose_reference=True, wheels_actuated=True, wheel_spin_target=False,
        absolute_body_height_constraint=False, task_success_from_training_reward=False)
    if cfg.get('friction_boundary') is not None:
        raise ValueError('Direct braking must not include a friction-boundary incentive')
    return cfg


def build(output):
    if output.exists():
        raise FileExistsError('Never overwrite a frozen experiment')
    bundle = output/'source_snapshot_v1'
    bundle.mkdir(parents=True)
    origin = ROOT/'results/hockey_equal_distance_20261102_v1'
    for source, target in [(origin/'direct_brake/motion.npz', 'motion.npz'),
                           (origin/'common_evaluation_initial_state.npz', 'common_evaluation_initial_state.npz')]:
        shutil.copyfile(source, bundle/target)
    parents = ROOT/'results/hockey_speed_success_20261028_v1/source_snapshot_v1'
    contact = ROOT/'results/hockey_contact_reward_search_20261028_v1/source_snapshot_v8'
    provenance = []
    for label, directory, trainer in [('parent', parents, 'train_hockey_ppo.py'),
                                      ('contact', contact, 'train_hockey_contact_search.py')]:
        dest = bundle/label
        dest.mkdir()
        for path in sorted(directory.glob('*.py')):
            text = path.read_text()
            patched = path.name == trainer
            if patched:
                text = instrument_trainer(text)
            (dest/path.name).write_text(text)
            provenance.append(dict(path=f'{label}/{path.name}', source=str(path.relative_to(ROOT)),
                                   source_sha256=sha(path), evaluation_telemetry_patch=patched))
        shutil.copyfile(ROOT/'scripts/direct_brake_reset.py', dest/'direct_brake_reset.py')
    shutil.copyfile(contact/'upstream_provenance.json', bundle/'upstream_provenance.json')
    for name in ('dispatch_direct_brake_training.py',):
        shutil.copyfile(ROOT/'scripts'/name, bundle/name)
    base = json.loads((parents/'protocols/hockey_s100_seed42.json').read_text())
    old7 = ROOT/'results/hockey_contact_reward_search_20261028_v1/source_snapshot_v7/protocols'
    seeds = [43, 44, 45, 46, 47]
    for seed in seeds:
        kwargs = dict(seed=seed, reference_sha=sha(bundle/'motion.npz'), reset_sha=sha(bundle/'common_evaluation_initial_state.npz'))
        write(bundle/f'protocols/seed{seed}_parent.json', direct_protocol(base, **kwargs))
        for arm in ('equal', 'split'):
            original = json.loads((old7/f'native_{arm}_smooth_stop_xyz_huber_continuity.json').read_text())
            write(bundle/f'protocols/seed{seed}_{arm}.json', direct_protocol(original, **kwargs))
    objective = json.loads((origin/'prospective_protocol.json').read_text())
    objective.update(status='frozen_before_training', distance_and_lateral_tolerances=dict(
        forward_target_absolute_error_m=.05, forward_overshoot_m=.05, lateral_excursion_m=.10,
        rationale='Prospective engineering tolerances; not tuned to new policy outcomes or claimed as real-world calibration.'),
        runtime_physics_smoke_required=True, remote_launch_performed=False,
        no_hockey_specific_sliding_or_wheel_heading_gate=True,
        contact_criterion='both wheels at every physical sample, native geometry tolerance 0.1 mm, unknown cannot pass',
        no_success_claim_until_independent_qualification=True)
    write(bundle/'objective.json', objective)
    plan = dict(kind='direct_brake_full_lineage_v1', seeds=seeds, material_arms=['equal', 'split'],
        physics_hz=400, policy_hz=50, parallel_training_envs=512,
        stage_transitions=dict(parent=12288000, contact=3072000, continuation=9216000),
        unique_formal_transitions_per_seed=36864000, unique_formal_transitions_total=184320000,
        lineage_transitions_per_final_policy=24576000,
        evaluation_cases_per_material=33, cross_material_nominal=[[.7,.7],[.9,.7],[.9,.9]],
        primary_task='stable stopping at equal initial speed and equal forward stopping distance',
        stop_after_task_pass=False, stop_on_infrastructure_error=True, one_GPU_job_at_a_time=True,
        no_preexisting_policy_loaded=True, idle_wait_limit_s=86400, poll_seconds=30,
        runtime=dict(python='/data/zengxiong/.conda/envs/regrind/bin/python',
            asset='/data/zengxiong/workspaces/tron1-beyondmimic-20261002/outputs/training-asset/WF_TRON1A.usda',
            tracking_root='/data/zengxiong/workspaces/tron1-ppo-backend-comparison-20261022/source_snapshot_v2/upstream/tron1-learning-from-human',
            beyondmimic_root='/data/zengxiong/workspaces/tron1-beyondmimic-20261002/third_party/whole_body_tracking'),
        expected_versions={'rsl-rl-lib':'3.0.1','torch':'2.7.0+cu128','numpy':'1.26.0',
                           'scipy':'1.15.3','isaaclab':'0.47.2','isaacsim':'5.1.0.0'},
        historical_baseline='seed42 selected v8/v9 is descriptive; prospective matched hockey seeds must be trained separately',
        success_evaluation='Raw 400 Hz states, native contact buffers and shared-reset readback archived; task success requires separate qualification.')
    write(bundle/'plan.json', plan)
    write(bundle/'source_provenance.json', provenance)
    files = [dict(path=str(p.relative_to(bundle)), bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(bundle.rglob('*')) if p.is_file()]
    write(bundle/'manifest.json', dict(kind='direct_brake_training_snapshot_v1', files=files,
        contains_models_weights_credentials_or_raw_video=False))
    archive = output/'source_snapshot_v1.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                tar.add(p, arcname='source_snapshot_v1/'+str(p.relative_to(bundle)))
    write(output/'payload.json', dict(files=len(files), uncompressed_bytes=sum(v['bytes'] for v in files),
        archive_sha256=sha(archive), remote_root=REMOTE, manifest_sha256=sha(bundle/'manifest.json')))
    print((output/'payload.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
