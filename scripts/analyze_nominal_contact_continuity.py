"""Contact-envelope continuity for the four completed matched PPO nominal cells.

No policy inference, simulation, time alignment or gap smoothing. Geometry is
an XML cylinder proxy; force-thresholded records remain separate diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_ppo_backend_400hz_matrix as shared
from analyze_matched_contact_geometry import cylinder_gaps, descriptors
from replay_physx_actions_mujoco import load_evaluator


def interval_statistics(time, active):
    """Left-held sampled intervals; the final sample has no assigned duration."""
    time = np.asarray(time, dtype=float)
    active = np.asarray(active)
    if time.ndim != 1 or active.shape != time.shape or active.dtype.kind != 'b':
        raise ValueError('Expected time and a boolean mask of equal length')
    if len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Expected finite strictly increasing time')
    dt = np.diff(time)
    held = active[:-1]
    edges = np.diff(np.r_[False, held, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    durations = time[ends] - time[starts]
    return dict(duration_s=float(dt[held].sum()),
                time_fraction=float(dt[held].sum() / dt.sum()),
                longest_interval_s=float(durations.max()) if len(durations) else 0.,
                interval_count=len(starts),
                all_sampled_intervals_active=bool(held.all()),
                final_endpoint_active=bool(active[-1]),
                first_interval_start_s=float(time[starts[0]]) if len(starts) else None)


def gap_metrics(time, gap, native_count, native_slip):
    result = dict(gap_min_max_m=[float(gap.min()), float(gap.max())], tolerances={})
    for tol in (.0001, .001, .002):
        near = gap <= tol
        deep = gap < -.005
        result['tolerances'][str(tol)] = dict(
            bilateral_envelope_near_ground=interval_statistics(time, near.all(axis=1)),
            any_wheel_separated=interval_statistics(time, (~near).any(axis=1)),
            left_separated=interval_statistics(time, ~near[:, 0]),
            right_separated=interval_statistics(time, ~near[:, 1]),
            bilateral_in_gap_band=interval_statistics(time, (near & ~deep).all(axis=1)))
    result['any_envelope_penetration_beyond_5mm'] = interval_statistics(time, (gap < -.005).any(axis=1))
    eligible = native_count > 0
    result['native_recorded_contact_eligibility'] = dict(
        both=interval_statistics(time, eligible.all(axis=1)),
        neither=interval_statistics(time, (~eligible).all(axis=1)),
        lateral_speed_above_0p05_with_eligible_record=interval_statistics(
            time, (eligible & (np.abs(native_slip) > .05)).any(axis=1)))
    return result


def run(a):
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    summary = json.loads(a.summary.read_text())
    reference = shared.reference_data(a.reference)
    if shared.sha256(a.model) != shared.INPUT_PINS['model']:
        raise ValueError('Not the frozen cohort model')
    loader = load_evaluator(a.legacy_evaluator)
    model, removed = loader.load_model(a.model, no_visual_mesh=True)
    fk = shared.WheelKinematics(model)
    fk_proof = dict(model_sha256=shared.sha256(a.model),
                    loader_sha256=shared.sha256(a.legacy_evaluator),
                    removed_noncolliding_visual_meshes=removed,
                    pinned_CPU_dependency_validation=True,
                    isolated_FK_only=True, dynamics_stepping=False)
    geometry = descriptors(a.model)
    cases, series = [], {}
    for origin in ('physx', 'mjlab'):
        for engine in ('physx', 'mujoco'):
            row = next(r for r in summary['rows'] if r['training_backend'] == origin
                       and r['engine_tag'] == engine and r['mu'] == .7)
            folder = a.trace_root / (origin + '_trained' + ('_mujoco' if engine == 'mujoco' else ''))
            source = folder / ('physics_trajectory.npz' if engine == 'physx' else 'rollout.npz')
            if shared.sha256(source) != row['input_hashes']['recording']:
                raise ValueError('Trajectory differs from completed cohort')
            if shared.sha256(folder / 'report.json') != row['input_hashes']['report']:
                raise ValueError('Deployment report differs from completed cohort')
            with np.load(source, allow_pickle=False) as archive:
                trace, _ = shared.analyze_case(dict(archive), engine, reference, fk)
            gap = cylinder_gaps(trace, geometry, shared.quaternion_rotate)
            time = trace['time_s']
            windows = {}
            for name, (start, end) in dict(full=(0., 6.98), core=(1.44, 4.)).items():
                mask = (time >= start - 1e-9) & (time <= end + 1e-9)
                windows[name] = gap_metrics(time[mask], gap[mask], trace['contact_count'][mask],
                                            trace['contact_slip'][mask])
            key = origin + '_' + engine
            series[key + '_time_s'] = time
            series[key + '_gap_m'] = gap
            series[key + '_native_contact_count'] = trace['contact_count']
            series[key + '_native_lateral_contact_speed_m_s'] = trace['contact_slip']
            cases.append(dict(id=key, training_backend=origin, evaluation_backend=engine,
                              input_hashes=row['input_hashes'], windows=windows,
                              core_xy_rmse_m=row['windows']['core']['tracking']['xy_rmse_m']))
    a.output_dir.mkdir(parents=True)
    np.savez_compressed(a.output_dir / 'derived_continuity.npz', **series)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    max_gap = max(series[c['id'] + '_gap_m'].max() for c in cases) * 1000.
    min_gap = min(series[c['id'] + '_gap_m'].min() for c in cases) * 1000.
    for ax, case in zip(axs, cases):
        key = case['id']
        time, gap = series[key + '_time_s'], series[key + '_gap_m'] * 1000.
        for wheel, label, color in ((0, 'Left', '#d62728'), (1, 'Right', '#9467bd')):
            ax.plot(time, gap[:, wheel], color=color, lw=.65, label=label)
        ax.axhline(1., color='black', lw=.8, ls='--', label='1 mm envelope tolerance')
        ax.axhline(0., color='gray', lw=.5)
        ax.axvspan(1.44, 4., color='#15956c', alpha=.08, label='Turn core')
        core = case['windows']['core']['tolerances']['0.001']
        duty = core['bilateral_in_gap_band']['time_fraction']
        off = core['any_wheel_separated']['longest_interval_s']
        origin = 'PhysX' if case['training_backend'] == 'physx' else 'MJWarp'
        engine = 'PhysX' if case['evaluation_backend'] == 'physx' else 'CPU MuJoCo'
        ax.set(title=f'{origin}-trained → {engine}\nCore both in band: {duty:.1%}; max separation: {off:.3f} s',
               xlabel='Time (s)', ylabel='XML wheel-envelope gap (mm)', xlim=(0., 6.98),
               ylim=(min(-6., min_gap * 1.1), max(5., max_gap * 1.1)))
        ax.grid(alpha=.2)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    fig.suptitle('400 Hz PPO | mu=0.7 | identical axes, unsmoothed physical samples\n'
                 'XML envelope proxy, not native cooked-contact truth; band is −5 to +1 mm')
    fig.tight_layout(rect=(0., .10, 1., .87))
    fig.savefig(a.output_dir / 'nominal_contact_4x1.png', dpi=150)
    plt.close(fig)
    report = dict(status='completed_verified', cases=cases, geometry=geometry, fk_provenance=fk_proof,
                  source_sha256=shared.sha256(Path(__file__)), summary_sha256=shared.sha256(a.summary),
                  derived_sha256=shared.sha256(a.output_dir / 'derived_continuity.npz'),
                  limitations=[
                      'XML cylinder envelope includes collider center and axis; PhysX cooked collision shape is not verified.',
                      'Native contact counts and speed are retained as originally recorded; eligibility is not contact truth.',
                      'Continuous means every recorded interval; between-sample contact is not certified.',
                      'Gap tolerance is a sensitivity parameter. Deep penetration is separately reported, not successful support.',
                      'Native per-point force normals are absent here; friction-cone utilization and saturated sliding cannot be certified.',
                      'No states, controller actions, training settings or clocks are changed. One episode per cell.'])
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps([dict(id=c['id'], core=c['windows']['core']['tolerances']['0.001']) for c in cases], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('summary', 'trace-root', 'reference', 'model', 'legacy-evaluator', 'output-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    run(parser.parse_args())
