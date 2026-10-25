"""Strict K/Q comparison from completed Q training and native dual-engine records.

The execution host verifies checkpoint/export bytes. This offline analyzer
requires those source-bound completion proofs and recomputes every native
metric; no weights, assets, physics, inference or FK are loaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_ground_contact_v2 as metrics
import ground_contact_reward_v2q as definition
import ppo_backend_400hz_contract as recording

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('K_physx', 'Q_physx', 'K_mujoco', 'Q_mujoco')
INITIAL = '7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d'
K_ACTOR = 'fafdccb45571b99da53a02455f2402a8cd8b960f16809ba8e6e0fe6996dce912'
Q_PROTOCOL = ROOT/'config/ppo_ground_contact_v2q_400hz_v1.json'
K_PROTOCOL = ROOT/'config/ppo_ground_contact_v2k_400hz_v1.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def verify_protocol():
    p, k = read(Q_PROTOCOL), read(K_PROTOCOL)
    assert p['cohort_id'] == 'tron1_ground_contact_v2q_any_wheel_400hz_v1'
    assert p['reward_definition_version'] == 'ground_contact_v2q_any_wheel'
    definition.validate_settings(p['ground_contact_reward_v2'])
    assert set(p) == set(k)
    allowed = {'cohort_id', 'reward_definition_version', 'ground_contact_reward_v2', 'experiment_design'}
    assert all(p[name] == k[name] for name in set(p)-allowed)
    old, new = k['ground_contact_reward_v2'], p['ground_contact_reward_v2']
    assert all(new[name] == value for name, value in old.items() if name != 'candidate')
    assert set(new)-set(old) == {'unsupported_geometry_gap_scale_m', 'unsupported_geometry_gap_weight', 'unsupported_geometry_gap_aggregation'}
    assert p['nominal_acceptance_before_training'] == k['nominal_acceptance_before_training']
    return p


def verify_collection(folder):
    folder = Path(folder)
    manifest = read(folder/'collection_manifest.json')
    assert manifest['kind'] == 'completed_Q_dual_engine_native_evidence'
    assert manifest['models_weights_credentials_binaries_included'] is False
    for name, row in manifest['files'].items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert sha(folder/path) == row['sha256'] and (folder/path).stat().st_size == row['bytes']
    return manifest


def verify_training(collection):
    collection = Path(collection)
    root = collection
    state = read(root/'driver_status.json')
    assert state['status'] == 'smoke_and_formal_completed_require_nominal_validation'
    assert [c['mode'] for c in state['children']] == ['smoke', 'formal']
    assert all(c['status'] == 'completed_verified' and c['returncode'] == 0 for c in state['children'])
    proof = read(root/'formal_v1_strict_verification.json')
    assert proof['mode'] == 'formal' and proof['status'] == 'completed_verified'
    assert proof['transitions'] == 12288000 and proof['initial_policy_state_sha256'] == INITIAL
    folder = root/'formal_v1'
    audit = read(folder/'ground_contact_v2q_training_audit.json')
    completion = read(folder/'cohort_completion_audit.json')
    manifest = read(folder/'manifest.json')
    contract = read(folder/'policy_contract.json')
    assert sha(folder/'ground_contact_v2q_training_audit.json') == proof['training_audit_sha256']
    assert audit['status'] == completion['status'] == 'completed_verified'
    assert audit['training_environment_steps'] == completion['training_environment_steps'] == 12288000
    assert audit['initial_policy_state_sha256'] == completion['initial_policy_state_sha256'] == INITIAL
    assert audit['exported_policy_sha256'] == completion['exported_policy_sha256'] == proof['normalized_actor_sha256']
    assert audit['checkpoint_sha256'] == completion['checkpoint_sha256']
    assert audit['smoke_only'] is False and audit['from_scratch'] and audit['trained_policy_claimed']
    assert audit['protocol_sha256'] == completion['protocol_sha256'] == sha(Q_PROTOCOL)
    assert contract['training_cohort_protocol_sha256'] == sha(Q_PROTOCOL)
    assert contract['ground_contact_reward_v2'] == definition.validate_settings()
    assert contract['training_environment_transitions'] == 12288000
    assert contract['policy_training_backend'] == 'physx' and contract['actor_includes_observation_normalizer']
    assert manifest['num_envs'] == 512 and manifest['iterations'] == 1000 and manifest['seed'] == 42
    assert audit['manifest_sha256'] == completion['manifest_sha256'] == sha(folder/'manifest.json')
    assert audit['policy_contract_sha256'] == completion['policy_contract_sha256'] == sha(folder/'policy_contract.json')
    assert (folder/'policy_contract.json').read_bytes() == (folder/'final_policy_contract.json').read_bytes()
    for remote, digest in audit['source_sha256'].items():
        remote = Path(remote)
        if remote.name == 'tron1_drift.py':
            path = ROOT/'upstream/tron1-learning-from-human/training/tron1_drift.py'
        else:
            path = ROOT/'scripts'/remote.name
        assert path.is_file() and sha(path) == digest, 'Executed training source differs: '+str(remote)
    runtime = audit['contact_runtime_audit']
    assert runtime == proof['contact_runtime_audit']
    assert runtime['reward_definition_version'] == 'ground_contact_v2q_any_wheel'
    assert runtime['steps_started'] == runtime['steps_completed'] == runtime['unique_reward_windows'] == 24008
    assert runtime['physics_samples'] == 192064 and runtime['sampled_environment_substeps'] == 98336768
    assert runtime['device_finite_normal_force_health'] and runtime['device_finite_reward_health']
    definition.verify_geometry_runtime(runtime,expected_envs=512,expected_windows=24008)
    assert not runtime['cpu_fixture_only']
    assert 0 <= runtime['mean_unsupported_geometry_gap_cost'] <= 1.
    assert 0 <= runtime['positive_geometry_cost_environment_windows'] <= 24008*512

    return proof, audit, manifest


def verify_episode(folder, engine, variant, training=None):
    folder = Path(folder)
    name = 'physics_trajectory.npz' if engine == 'physx' else 'rollout.npz'
    proof = read(folder/('strict_verification.json' if variant == 'Q' else 'remote_verification.json'))
    report = read(folder/'report.json')
    assert proof['status'] == 'verified' and proof['evaluation_mu'] == .7
    assert proof['report_sha256'] == sha(folder/'report.json') and proof['recording_sha256'] == sha(folder/name)
    label = 'PhysX' if engine == 'physx' else 'native CPU MuJoCo'
    assert proof['validation_backend'] == label
    actual = recording.verify_recording(folder/name, label)
    for field, value in actual.items(): assert proof[field] == value
    source = report['ground_contact_v2'+variant.lower()+'_evaluation']
    assert source['status'] == 'verified' and source['initial_policy_state_sha256'] == INITIAL
    assert source['training_backend'] == 'physx' and source['smoke_only'] is False
    assert source['normalized_actor_sha256'] == (K_ACTOR if variant == 'K' else training[0]['normalized_actor_sha256'])
    assert source['wrapper_sha256'] == sha(ROOT/'scripts'/('eval_ppo_ground_contact_v2'+variant.lower()+'.py'))
    assert source['recording_sha256'] == proof['recording_sha256'] and source['checkpoint_sha256'] == proof['checkpoint_sha256']
    assert source['protocol_sha256'] == sha(K_PROTOCOL if variant == 'K' else Q_PROTOCOL)
    if variant == 'Q':
        for key, digest in source['cohort_input_sha256'].items():
            if key.endswith('.json'):
                assert sha(Path(training[2]['_collection_root'])/'formal_v1'/key) == digest
    if engine == 'physx':
        assert report['status'] == 'evaluated' and report['domain_randomization_enabled'] is False
        assert report['observation_noise_enabled'] is False and report['deterministic_start_frame'] == 0
        assert report['runtime_device_audit']['status'] == 'verified'
        assert report['terminal_states_captured_before_auto_reset']
    else:
        assert report['status'] == 'completed' and report['completed_full_reference']
        assert report['root_state_writes'] == 1 and report['hidden_resets'] == 0
        assert report['physics_steps'] == 2792 and report['policy_steps'] == 349
        assert report['domain_parameters_resampled_during_episode'] is False
    return npz(folder/name), report, proof


def force_diagnostics(trace):
    f = trace['floor_normal_force'][:, :, 2]
    t = trace['time_s']
    assert f.shape == (2793, 2) and np.isfinite(f).all() and f.min() >= -1e-4
    index = np.unravel_index(f.argmax(), f.shape)
    core = (t >= 1.44-1e-9) & (t <= 4.+1e-9)
    return dict(whole_recording_peak_n=float(f.max()), whole_recording_peak_time_s=float(t[index[0]]),
        whole_recording_peak_wheel=int(index[1]), policy_controlled_peak_n=float(f[1:].max()),
        core_peak_n=float(f[core].max()), initial_readout_excluded_from_controlled_peak=True)


def load_cases(collection, k_raw_root, reference):
    protocol = verify_protocol()
    verify_collection(collection)
    training = verify_training(collection)
    training[2]['_collection_root'] = str(Path(collection).resolve())
    state = read(Path(collection)/'nominal_v1/status.json')
    assert state['status'] == 'both_nominal_engines_completed_require_analysis'
    assert [c['engine'] for c in state['children']] == ['mujoco', 'physx']
    assert all(c['status'] == 'completed_verified' and c['returncode'] == 0 for c in state['children'])
    original = read(ROOT/'results/ground_contact_v2k_ppo_nominal_20261024_v1/analysis.json')
    rows, traces = [], {}
    for key in ORDER:
        variant, engine = key.split('_')
        folder = Path(k_raw_root)/engine if variant == 'K' else Path(collection)/'nominal_v1'/engine
        raw, report, proof = verify_episode(folder, engine, variant, training)
        values, trace = metrics.recorded_metrics(raw, engine, reference)
        expected = original['rows'][0 if engine == 'physx' else 1]
        if variant == 'K': assert values == expected['new_V2']
        trace['contact_force'] = trace['floor_normal_force'].copy()
        screen = metrics.nominal_screen(values, expected['original_grounded_baseline'], protocol, engine)
        rows.append(dict(id=key, variant=variant, engine=engine, metrics=values, nominal_screen=screen,
            force_diagnostics=force_diagnostics(trace), native_proof=proof,
            normalized_actor_sha256=report['ground_contact_v2'+variant.lower()+'_evaluation']['normalized_actor_sha256']))
        traces[key] = trace
    for engine in ('physx', 'mujoco'):
        for field in ('position', 'quaternion', 'com_velocity', 'angular_velocity', 'joint_pos', 'joint_vel'):
            np.testing.assert_array_equal(traces['K_'+engine][field][0], traces['Q_'+engine][field][0])
    assert rows[1]['normalized_actor_sha256'] == rows[3]['normalized_actor_sha256']
    return rows, traces, training[0]


def run(args):
    assert not args.output_dir.exists()
    reference = metrics.metrics.reference_data(args.reference)
    rows, traces, training = load_cases(args.collection, args.k_raw_root, reference)
    result = dict(kind='PPO_K_vs_Q_unsupported_geometry_gap_dual_engine', status='completed_analysis', rows=rows,
        reference_sha256=sha(args.reference), analyzer_sha256=sha(__file__), protocol_sha256=sha(Q_PROTOCOL),
        completed_training_proof=training, sample_size_per_case=1, native_states_per_case=2793,
        physics_dt_s=.0025, policy_dt_s=.02, acceptance_thresholds_unchanged=True,
        same_initial_recorded_mechanical_state_within_each_engine=True,
        qualitative_whole_body_review_complete=False, overall_task_goal_complete_claimed=False)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    arrays = {}
    for key, trace in traces.items():
        arrays.update({key+'__'+name: trace[name] for name in ('time_s', 'position', 'quaternion', 'com_velocity', 'angular_velocity', 'joint_pos', 'joint_vel', 'wheel_pos', 'floor_normal_force')})
    np.savez_compressed(args.output_dir/'native_states.npz', **arrays)
    print(json.dumps([dict(id=r['id'], core_XY=r['metrics']['windows']['core']['tracking']['xy_rmse_m'],
        support=r['metrics']['any_wheel_ground_support']['full_contact']['any_wheel_support_sample_fraction'],
        gap_s=r['metrics']['any_wheel_ground_support']['full_contact']['longest_no_support_observed_s'],
        HF_ratio=r['nominal_screen']['core_HF_ratio_to_original_same_engine_PPO'], failed=r['nominal_screen']['failed_checks']) for r in rows]))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--k-raw-root', type=Path, default=ROOT/'outputs/ground_contact_v2k_20261024_v1/ppo_nominal_raw_v1')
    p.add_argument('--reference', type=Path, default=metrics.DEFAULT_REFERENCE)
    p.add_argument('--output-dir', type=Path, required=True)
    run(p.parse_args())
