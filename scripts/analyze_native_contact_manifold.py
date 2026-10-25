"""Native point/gap/load diagnostics of byte-identical K episodes in both engines."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import ppo_backend_400hz_contract as recording

ROOT = Path(__file__).resolve().parents[1]
BASELINES = dict(physx='cbd7d2b249737919bc21f2bccb05b3d82e8b869fbbe3d0a91c45e6b8eda4a67a',
    mujoco='af790839128d5d02a8233a7a7b2cd8b23eb508e9056cd4d1c619b8d4173d14ef')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}


def quantiles(value):
    value = np.asarray(value)
    assert np.isfinite(value).all()
    return dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'), np.percentile(value, [0,25,50,75,95,100]).tolist())) if value.size else None


def load_case(root, engine):
    collected = root/'evidence'/engine
    manifest = read(collected/'collection_manifest.json')
    assert manifest['models_weights_credentials_binaries_included'] is False
    for name, item in manifest['files'].items():
        p = Path(name); assert not p.is_absolute() and '..' not in p.parts
        assert sha(collected/p) == item['sha256'] and (collected/p).stat().st_size == item['bytes']
    folder = collected/(engine+'_v1')
    metadata = read(folder/'manifold_status.json')
    assert metadata['kind'] == ('K_readonly_PhysX_native_contact_manifold_probe' if engine == 'physx' else 'K_readonly_MuJoCo_native_contact_manifold_probe')
    assert metadata['normalized_actor_sha256'] == 'fafdccb45571b99da53a02455f2402a8cd8b960f16809ba8e6e0fe6996dce912'
    assert metadata['physics_trajectory_sha256'] == BASELINES[engine] and metadata['read_only_baseline_byte_identical']
    for remote, digest in metadata['source_sha256'].items():
        name = Path(remote).name
        assert (ROOT/'scripts'/name).is_file() and sha(ROOT/'scripts'/name) == digest
    audit = metadata['manifold_probe']
    assert sha(folder/'manifold_trajectory.npz') == audit['native_trace_sha256']
    for remote, digest in audit.get('source_sha256', {}).items():
        name = Path(remote).name
        if name == 'api.py':
            assert 'omni.physics.tensors-' in remote and len(digest) == 64
        else: assert sha(ROOT/'scripts'/name) == digest
    file = folder/('physics_trajectory.npz' if engine == 'physx' else 'rollout.npz')
    assert sha(file) == BASELINES[engine]
    recording.verify_recording(file, 'PhysX' if engine == 'physx' else 'native CPU MuJoCo')
    raw, data = npz(file), npz(folder/'manifold_trajectory.npz')
    if engine == 'physx':
        time = data['time_before_step_s']; expected = raw['wheel_ground_contact_force_w_n'][:-1, 0]
        np.testing.assert_array_equal(data['native_counter_before']-data['native_counter_before'][0], raw['simulation_step_counter'][:-1])
    else:
        time = data['time_s']; expected = raw['wheel_ground_contact_force_w_n']
    np.testing.assert_allclose(time, np.arange(len(time))*.0025, rtol=0, atol=1e-10)
    np.testing.assert_allclose(data['native_point_force_sum_w_n'], expected, rtol=1e-5, atol=.001)
    counts = data['native_point_count']; active = np.arange(128)[None,None,:] < counts[:,:,None]
    assert counts.shape == (len(time), 2) and (counts >= 0).all() and (counts < 128).all()
    for name in ('native_point_force_n', 'native_point_separation_m', 'native_point_position_w_m', 'native_point_normal_w', 'native_point_velocity_w_m_s'):
        assert np.isfinite(data[name][active]).all() and np.isnan(data[name][~active]).all()
    force = data['native_point_force_sum_w_n'][:, :, 2]
    assert force.min() >= -1e-4
    selected = (time >= .02-1e-9) & (time <= 6.96+1e-9)
    loss = selected & (force.max(-1) <= 5.)
    native_present = counts.sum(-1) > 0
    signed = data['authored_signed_gap_m']; closest = signed.argmin(-1)
    gap = signed.min(-1)
    min_sep = np.where(active, data['native_point_separation_m'], np.inf).min((1, 2))
    min_sep = np.where(native_present, min_sep, np.nan)
    nearest_vz = data['authored_lowest_point_velocity_w_m_s'][np.arange(len(time)), closest, 2]
    assert np.isfinite(gap).all() and np.isfinite(nearest_vz).all()
    row = dict(engine=engine, baseline_byte_identical=True, native_states=len(raw['time_s']), point_states=len(time),
        analyzed_interval_s=[.02, 6.96], analyzed_samples=int(selected.sum()), support_threshold_n_strict=5.,
        samples_below_anywheel_support=int(loss.sum()), anywheel_support_fraction=float(1-loss.sum()/selected.sum()),
        below_support_with_any_native_points=int(np.sum(loss & native_present)),
        below_support_without_native_points=int(np.sum(loss & ~native_present)),
        max_native_points_per_wheel=counts.max(0).tolist(),
        below_support_authored_min_gap_mm=quantiles(gap[loss]*1000),
        loaded_authored_min_gap_mm=quantiles(gap[selected & ~loss]*1000),
        below_support_native_min_separation_mm_only_when_reported=quantiles(min_sep[loss & native_present]*1000),
        below_support_author_gap_gt1mm=int(np.sum(loss & (gap > .001))),
        below_support_author_gap_gt10mm=int(np.sum(loss & (gap > .01))),
        below_support_closest_authored_point_vz_m_s=quantiles(nearest_vz[loss]),
        native_point_report_timing_is_engine_specific=True, native_distance_missing_not_zero=True,
        geometry_height_or_point_count_used_as_failure_gate=False,
        point_force_to_parent_max_vector_difference_n=float(np.abs(data['native_point_force_sum_w_n']-expected).max()),
        force_mask_disagreement_to_parent_samples=int(np.any((force > 5.) != (expected[:, :, 2] > 5.), axis=-1).sum()),
        raw_recording_sha256=sha(file), manifold_recording_sha256=sha(folder/'manifold_trajectory.npz'))
    return row, dict(time_s=time, max_Fz_n=force.max(-1), anywheel_below_support=loss,
        authored_min_gap_mm=gap*1000, native_min_separation_mm=min_sep*1000,
        native_point_counts=counts.sum(-1), closest_authored_point_vz_m_s=nearest_vz)


def run(args):
    target = args.result/'analysis'; assert not target.exists()
    rows, series = [], {}
    for engine in ('physx', 'mujoco'):
        row, data = load_case(args.result, engine); rows.append(row); series[engine] = data
    target.mkdir()
    result = dict(kind='native_manifold_vs_authored_geometry_K_same_actor_dual_engine', status='completed_analysis',
        rows=rows, analyzer_sha256=sha(__file__), sample_size_per_engine=1,
        extra_training_updates=0, policies_actions_rewards_states_physics_unchanged=True,
        point_report_presence_is_not_force_support=True, native_points_not_reconstructed_by_FK=True,
        authored_gap_not_cooked_shape_identity_or_hardware_measurement=True,
        task_success_claimed=False, native_contact_model_equivalence_claimed=False)
    (target/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(target/'signals.npz', **{e+'__'+n:v for e, d in series.items() for n,v in d.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(22, 8), layout='constrained')
    visible = {e:(a['time_s'] >= 1.8) & (a['time_s'] <= 3.5) for e,a in series.items()}
    gap_values = np.concatenate([a[n][visible[e]] for e,a in series.items()
        for n in ('authored_min_gap_mm', 'native_min_separation_mm')])
    gap_values = gap_values[np.isfinite(gap_values)]
    gap_padding = .05*max(1., np.ptp(gap_values))
    velocity_values = np.concatenate([a['closest_authored_point_vz_m_s'][visible[e]] for e,a in series.items()])
    velocity_padding = .05*max(1., np.ptp(velocity_values))
    force_upper = max(a['max_Fz_n'][visible[e]].max() for e,a in series.items())*1.1
    for i, engine in enumerate(('physx', 'mujoco')):
        a = series[engine]; t = a['time_s']; lost = a['anywheel_below_support']
        axes[i, 0].plot(t, a['max_Fz_n'], lw=.8); axes[i, 0].axhline(5, color='black', ls='--')
        axes[i, 0].set(ylabel='Max wheel Fz (N)', yscale='symlog', title=engine+': measured bearing load')
        axes[i, 1].plot(t, a['authored_min_gap_mm'], label='current-pose authored gap', lw=1.)
        axes[i, 1].plot(t, a['native_min_separation_mm'], label='reported native min separation', lw=.7)
        axes[i, 1].axhline(0, color='black', ls='--', lw=.7); axes[i, 1].legend(fontsize=7)
        axes[i, 1].set(ylabel='Gap/separation (mm)', title='Different measurement operators; no time alignment')
        axes[i, 2].plot(t, a['closest_authored_point_vz_m_s'], lw=.8)
        axes[i, 2].axhline(0, color='black', ls='--', lw=.7)
        axes[i, 2].set(ylabel='Normal velocity (m/s)', title='Current-pose lowest-point rigid velocity')
        axes[i, 3].step(t, a['native_point_counts'], where='post', lw=.8)
        axes[i, 3].set(ylabel='Native reported points (both wheels)', title='Points may exist without load')
        for ax in axes[i]:
            ax.fill_between(t, 0, 1, where=lost, color='#dc7b7b', alpha=.2, transform=ax.get_xaxis_transform())
            ax.set(xlim=(1.8, 3.5), xlabel='Native time (s)'); ax.grid(alpha=.2)
        axes[i, 0].set_ylim(-.1, force_upper)
        axes[i, 1].set_ylim(gap_values.min()-gap_padding, gap_values.max()+gap_padding)
        axes[i, 2].set_ylim(velocity_values.min()-velocity_padding, velocity_values.max()+velocity_padding)
        axes[i, 3].set_ylim(-.1, 4.2)
    fig.suptitle('Same K actor, mu=.7 | raw native traces reproduced byte-for-byte | red = both wheels Fz<=5N | plotted detail1.8–3.5s; metrics0.02–6.96s')
    fig.savefig(target/'comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(rows))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--result', type=Path, required=True)
    run(p.parse_args())
