"""Recompute native bench and frozen-actor normal-contact sensitivity evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_ppo_k_native_slices9 as prior
import physx_normal_compliance as normal
import ppo_backend_400hz_contract as recording

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/data/zengxiong/workspaces/tron1-normal-compliance-20261025')
ORDER = ('rigid', 'compliant_1mm_d1', 'compliant_3mm_d0', 'compliant_3mm_d1')
TITLES = ('Rigid cylinder', 'Compliant: 1mm, zeta=1', 'Compliant: 3mm, zeta=0', 'Compliant: 3mm, zeta=1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def verify_sources(ledger, collection):
    """Resolve immutable executed snapshots, including earlier producer versions."""
    for remote, digest in ledger.items():
        remote = Path(remote)
        if remote.is_relative_to(REMOTE):
            p = Path(collection)/remote.relative_to(REMOTE)
        elif remote.name == 'report.json':
            p = ROOT/'results/physx_tron1_wheel_models_20261025_v1/evidence/matched_models_v1/authored__native_coulomb/report.json'
        else:
            p = ROOT/'scripts'/remote.name
        assert p.is_file() and sha(p) == digest, 'Executed source differs: '+str(remote)


def verify_collection(collection):
    manifest = read(Path(collection)/'collection_v1.manifest.json')
    for name, item in manifest['files'].items():
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        p = Path(collection)/p
        assert p.is_file() and p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
    return sha(Path(collection)/'collection_v1.manifest.json')


def vertical_impulse_balance(data, mass, dt=.0025):
    """World-Z momentum uses normal plus tangential contact components."""
    return (mass*np.diff(data['root_state_w'][:, 9])/dt
        -(data['ground_normal_force_w_n'][1:, 2]+data['native_tangent_force_w_n'][1:, 2])
        +mass*9.81+data['applied_extra_downward_load_n'])


def bench_analysis(collection):
    collection = Path(collection)
    assert read(collection/'bench_v3/status.json')['status'] == 'four_native_bench_cases_completed_require_analysis'
    rows, traces = [], {}
    for key in ORDER:
        folder = collection/'bench_v3'/key
        status = read(folder/'status.json')
        assert status['kind'] == 'PhysX_loaded_wheel_native_normal_compliance_bench'
        assert status['status'] == 'completed_requires_analysis'
        assert status['native_trace_sha256'] == sha(folder/'native_trace.npz')
        assert status['protocol_sha256'] == sha(ROOT/'config/physx_normal_compliance_v1.json')
        verify_sources(status['source_sha256'], collection)
        data = npz(folder/'native_trace.npz')
        assert data['root_state_w'].shape == (1201, 13)
        assert all(np.isfinite(v).all() for v in data.values())
        np.testing.assert_allclose(data['time_s'], np.arange(1201)*.0025, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(np.diff(data['native_step_counter']), 1)
        mass = status['fixture_mass_kg']
        expected = normal.profile(key, mass)
        assert status['profile'] == expected
        assert mass == status['robot_reference_mass_kg']/2
        authored = status['authored_material']
        for name in ('stiffness_n_m', 'damping_n_s_m'):
            np.testing.assert_allclose(authored[name], expected[name], rtol=2e-7, atol=1e-7)
        assert authored['acceleration_spring'] is False
        assert status['geometry']['slices'] == 1 and status['geometry']['mass_kg'] == mass
        np.testing.assert_allclose(status['geometry']['copied_collision_offsets_m']['ContactOffset'], .001, rtol=0, atol=1e-9)
        assert status['geometry']['copied_collision_offsets_m']['RestOffset'] == 0
        load = np.zeros(1200); load[400:800] = mass*9.81
        np.testing.assert_array_equal(data['applied_extra_downward_load_n'], load)
        native = np.asarray(status['native_material_readback'])
        np.testing.assert_allclose(native[..., :2], .7, rtol=0, atol=1e-7)
        np.testing.assert_allclose(native[..., 2], -expected['stiffness_n_m'], rtol=2e-7, atol=1e-7)
        normal_z, tangent_z = data['ground_normal_force_w_n'][:, 2], data['native_tangent_force_w_n'][:, 2]
        residual = vertical_impulse_balance(data, mass)
        np.testing.assert_allclose(residual, data['vertical_impulse_balance_error_n'], rtol=0, atol=1e-4)
        assert np.abs(residual).max() < .1 and normal_z.min() >= -1e-4
        windows = []
        for start, stop in ((.75, 1.), (1.75, 2.), (2.75, 3.)):
            mask = (data['time_s'] >= start) & (data['time_s'] < stop)
            windows.append(dict(interval_s=[start, stop], mean_center_z_m=float(data['root_state_w'][mask, 2].mean()),
                vertical_velocity_rms_m_s=float(np.sqrt(np.mean(data['root_state_w'][mask, 9]**2))),
                mean_Fz_n=float(normal_z[mask].mean())))
        rows.append(dict(id=key, profile=expected, fixture_mass_kg=mass, windows=windows,
            peak_normal_force_n=float(normal_z.max()), minimum_normal_force_n=float(normal_z.min()),
            peak_impulse_balance_error_n=float(np.abs(residual).max()), source_sha256=status['source_sha256']))
        traces[key] = data
    for key in ORDER[1:]:
        np.testing.assert_array_equal(traces[key]['root_state_w'][0], traces['rigid']['root_state_w'][0])
    for row in rows:
        for window, rigid in zip(row['windows'], rows[0]['windows'], strict=True):
            window['mean_deflection_relative_to_rigid_m'] = rigid['mean_center_z_m']-window['mean_center_z_m']
    roundtrip = npz(collection/'material_roundtrip_v4/native_trace.npz')
    round_status = read(collection/'material_roundtrip_v4/status.json')
    verify_sources(round_status['source_sha256'], collection)
    assert sha(collection/'material_roundtrip_v4/native_trace.npz') == round_status['native_trace_sha256']
    assert np.abs(vertical_impulse_balance(roundtrip, round_status['fixture_mass_kg'])).max() < .1
    np.testing.assert_array_equal(roundtrip['root_state_w'][0], traces['compliant_3mm_d1']['root_state_w'][0])
    differences = {k: float(np.abs(roundtrip[k]-traces['compliant_3mm_d1'][k]).max()) for k in
        ('root_state_w', 'ground_normal_force_w_n', 'native_tangent_force_w_n')}
    assert differences['root_state_w'] > .3
    return dict(rows=rows, material_three_column_roundtrip_response_difference=differences,
        roundtrip_preserves_full_compliant_response=False, damping_direct_native_readback_available=False,
        per_contact_parameters_are_not_whole_manifold_stiffness=True), traces


def load_cases(collection, reference):
    collection = Path(collection)
    driver = read(collection/'turn_models_v2/status.json')
    assert driver['status'] == 'four_normal_contact_turn_cases_completed_require_analysis'
    criteria = read(ROOT/'config/ppo_ground_contact_v2k_400hz_v1.json')
    original = read(ROOT/'results/physx_tron1_wheel_models_20261025_v1/analysis.json')['original_baseline_metrics']
    rows, traces, raws = [], {}, {}
    for key in ORDER:
        folder = collection/'turn_models_v2'/key
        meta = read(folder/'normal_compliance_status.json')
        report = read(folder/'report.json')
        assert meta == read(folder/'wheel_contact_model_status.json') == report['wheel_contact_model_transfer']
        assert meta == report['native_normal_compliance_transfer']
        assert meta['kind'] == 'frozen_K_PhysX_native_normal_contact_model_transfer'
        assert meta['normalized_actor_sha256'] == prior.previous.ACTOR
        assert meta['additional_training_updates'] == 0 and meta['actor_and_command_execution_changed'] is False
        assert meta['mu'] == .7 and meta['geometry'] == 'authored' and meta['law'] == 'native_coulomb'
        assert meta['native_states'] == 2793 and meta['policy_decisions'] == 349
        assert meta['normal_protocol_sha256'] == sha(ROOT/'config/physx_normal_compliance_v1.json')
        assert meta['native_audit']['masses_COM_inertias_armature_damping_match_baseline']
        assert report['status'] == 'evaluated' and report['deterministic_start_frame'] == 0
        assert report['domain_randomization_enabled'] is False and report['observation_noise_enabled'] is False
        assert report['terminal_states_captured_before_auto_reset'] is True
        assert report['runtime_device_audit']['status'] == 'verified'
        verify_sources(meta['source_sha256'], collection)
        for name in ('physics_trajectory', 'tire_trajectory'):
            assert sha(folder/(name+'.npz')) == meta[name+'_sha256']
        intervention = meta['normal_contact_model_intervention']
        assert intervention['profile'] == normal.profile(key, intervention['robot_native_mass_kg']/2)
        assert not intervention['native_shape_material_tensor_setter_used']
        assert intervention['friction_and_nonwheel_materials_unchanged'] and intervention['no_additional_physical_steps']
        np.testing.assert_array_equal(intervention['native_material_before'], intervention['native_material_after'])
        native = np.asarray(intervention['native_material_after'])
        np.testing.assert_allclose(native[..., :2], .7, rtol=0, atol=1e-7)
        expected = np.zeros((1, 9)); expected[:, intervention['native_shape_slots']] = -intervention['profile']['stiffness_n_m']
        np.testing.assert_allclose(native[..., 2], expected, rtol=2e-7, atol=1e-7)
        verify_sources(intervention['bench_source_sha256'], collection)
        recording.verify_recording(folder/'physics_trajectory.npz', 'PhysX')
        raw = npz(folder/'physics_trajectory.npz')
        values, trace = prior.metrics.recorded_metrics(raw, 'physx', reference)
        assert np.min(trace['floor_normal_force'][:, :, 2]) >= -1e-4
        trace['contact_force'] = trace['floor_normal_force'].copy()
        rows.append(dict(id=key, profile=intervention['profile'], metrics=values,
            nominal_screen=prior.metrics.nominal_screen(values, original, criteria, 'physx'),
            support_gap_diagnostics=prior.support_gap_diagnostics(raw, trace),
            peak_normal_force_n=float(trace['floor_normal_force'][:, :, 2].max()),
            baseline_byte_identical=meta['read_only_baseline_byte_identical'],
            metadata=meta, inputs_sha256={p.name: sha(p) for p in folder.iterdir() if p.suffix in ('.json', '.npz')}))
        traces[key], raws[key] = trace, raw
    for key in ORDER[1:]:
        for name in ('position', 'quaternion', 'com_velocity', 'angular_velocity', 'joint_pos', 'joint_vel'):
            np.testing.assert_array_equal(traces[key][name][0], traces['rigid'][name][0])
    assert rows[0]['baseline_byte_identical']
    return rows, traces, raws


def run(args):
    assert not args.output_dir.exists()
    manifest_sha = verify_collection(args.collection)
    reference = prior.metrics.metrics.reference_data(args.reference)
    bench, bench_traces = bench_analysis(args.collection)
    rows, traces, _ = load_cases(args.collection, reference)
    result = dict(kind='PhysX_native_normal_contact_sensitivity_frozen_K_actor', status='completed_analysis',
        rows=rows, bench=bench, reference_sha256=sha(args.reference), analyzer_sha256=sha(__file__),
        collection_manifest_sha256=manifest_sha, samples_per_arm=1, additional_training_updates=0,
        rigid_reproduces_original_native_recording_byte_for_byte=True, same_initial_mechanical_state=True,
        actor_reference_reward_actuator_and_friction_unchanged=True, physical_tire_parameters_identified=False,
        nonlinear_tangential_tire_law_or_sliced_geometry_added_here=False,
        cross_engine_qualification_complete=False, task_success_claimed=False)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#222222', '#0072b2', '#d55e00', '#009e73')
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), layout='constrained')
    for key, title, color in zip(ORDER, TITLES, colors, strict=True):
        b, t = bench_traces[key], traces[key]
        axes[0, 0].plot(b['time_s'], b['root_state_w'][:, 2], label=title, color=color, lw=.8)
        axes[0, 1].plot(b['time_s'], b['ground_normal_force_w_n'][:, 2], label=title, color=color, lw=.8)
        axes[0, 2].plot(t['position'][:, 0], t['position'][:, 1], label=title, color=color)
        axes[1, 0].plot(t['time_s'], t['floor_normal_force'][:, :, 2].max(-1), label=title, color=color, lw=.7)
        axes[1, 1].plot(t['time_s'], np.linalg.norm(t['angular_velocity'][:, :2], axis=-1), label=title, color=color, lw=.7)
    axes[0, 2].plot(reference['position'][:, 0], reference['position'][:, 1], '--', color='gray', label='reference')
    axes[0, 0].set(title='Loaded fixture: free vertical response', xlabel='Time (s)', ylabel='Center Z (m)')
    axes[0, 1].set(title='Loaded fixture: native normal force', xlabel='Time (s)', ylabel='Fz (N)')
    axes[0, 2].set(title='Whole robot: recorded XY', xlabel='World X (m)', ylabel='World Y (m)', aspect='equal')
    axes[1, 0].set(title='Whole robot: actual upward normal load', xlabel='Time (s)', ylabel='Max wheel Fz (N)', yscale='symlog')
    axes[1, 0].axhline(5., color='gray', ls='--')
    axes[1, 1].set(title='Whole robot: native angular motion', xlabel='Time (s)', ylabel='Root angular XY norm (rad/s)')
    axes[1, 2].bar(np.arange(4), [r['metrics']['any_wheel_ground_support']['full_contact']['any_wheel_support_sample_fraction']*100 for r in rows], color=colors)
    axes[1, 2].set(title='Whole robot: any-wheel support', ylabel='Native samples with Fz >5N (%)', xticks=np.arange(4), xticklabels=['rigid', '1mm d1', '3mm d0', '3mm d1'], ylim=[0, 100])
    for ax in axes.ravel():
        ax.grid(alpha=.2)
        if ax is not axes[1, 2]: ax.legend(fontsize=7)
    fig.suptitle('Native PhysX contact sensitivity | same frozen K actor | mu=.7 | beta ON | N=1')
    fig.savefig(args.output_dir/'comparison.png', dpi=150); plt.close(fig)
    print(json.dumps([dict(id=r['id'], core_XY=r['metrics']['windows']['core']['tracking']['xy_rmse_m'],
        support=r['metrics']['any_wheel_ground_support']['full_contact']['any_wheel_support_sample_fraction'],
        gap_s=r['metrics']['any_wheel_ground_support']['full_contact']['longest_no_support_observed_s'],
        HF_ratio=r['nominal_screen']['core_HF_ratio_to_original_same_engine_PPO'], failed=r['nominal_screen']['failed_checks']) for r in rows]))
    return result


if __name__ == '__main__':
    import run_cem_ground_contact_v2 as assets
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=assets.paired.DEFAULT_REFERENCE)
    p.add_argument('--output-dir', type=Path, required=True)
    run(p.parse_args())
