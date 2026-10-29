"""Offline liftoff and vertical-smoothness diagnostics for qualified recordings.

Native contact remains the definition of touching. Authored lowest-point normal
velocity and clearance rate diagnose separation; neither is lateral slip. This
audit changes no reward, training job, reference or qualification threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_timing import previous_state_trace
from audit_hockey_reference_contact_gap import wheel_geometry
from evaluate_hockey_physx import load, sha
from wheel_ground_geometry import cylinder_support, point_velocity


def vertical_windows(vz, dt, decimation=8):
    """Complete windows only; interval mean equals the endpoint difference."""
    vz = np.asarray(vz, dtype=float)
    if vz.ndim != 1 or not np.isfinite(vz).all() or dt <= 0:
        raise ValueError('Invalid vertical velocity or clock')
    if len(vz) < decimation + 1 or (len(vz) - 1) % decimation:
        raise ValueError('Incomplete control interval')
    acceleration = np.diff(vz).reshape(-1, decimation) / dt
    mean = np.diff(vz[::decimation]) / (dt * decimation)
    np.testing.assert_allclose(acceleration.mean(-1), mean, rtol=1e-10, atol=1e-10)
    return acceleration, mean


def loss_onsets(touching):
    touching = np.asarray(touching, dtype=bool)
    if touching.ndim != 1:
        raise ValueError('Expected one wheel over time')
    return np.flatnonzero(touching[:-1] & ~touching[1:]) + 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases-json', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    read = lambda path: json.loads(path.read_text())
    sources = read(a.cases_json)
    if len(sources) != 4:
        raise ValueError('Require four comparison panels')
    reference = load(a.reference / 'motion.npz')
    horizon = (len(reference['joint_pos']) - 1) / float(reference['fps'])
    report = dict(kind='hockey_contact_liftoff_audit_v1', script_sha256=sha(Path(__file__)),
        cases_selection_sha256=sha(a.cases_json), reference_sha256=sha(a.reference / 'motion.npz'),
        dependencies={name: sha(Path(__file__).with_name(name)) for name in
                      ('analyze_hockey_ground_contact.py', 'audit_hockey_contact_timing.py',
                       'audit_hockey_reference_contact_gap.py', 'evaluate_hockey_physx.py',
                       'wheel_ground_geometry.py')},
        scope='Selected completed nominal validation recordings; not archived training reward',
        normal_velocity_is_lateral_slip=False, reference_or_training_changed=False,
        contact_force_threshold_N=None, physics_steps=0, cases=[])
    arrays = {}; event_rows = []; plotted = []
    for number, source in enumerate(sources):
        paths = {k: (a.cases_json.parent / source[k]).resolve() for k in ('bundle', 'evidence', 'audit')}
        plan = read(paths['bundle'] / 'plan.json')
        cell, stage = source['case'].split(':')
        if stage != 'nominal':
            raise ValueError('Require independently qualified nominal recordings')
        entry = next(c for c in plan['cells'] if c['id'] == cell)
        protocol = paths['bundle'] / entry['protocol']; cfg = read(protocol)
        run_path = paths['evidence'] / source['run_name'] / cell / stage
        qpath = run_path.parent / 'nominal_qualification.json'; q = read(qpath)
        run = read(run_path / 'report.json')
        if (q['run_report_sha256'] != sha(run_path / 'report.json')
                or q['protocol_sha256'] != sha(protocol)
                or q['source_manifest_sha256'] != sha(paths['bundle'] / 'manifest.json')
                or run['motion_sha256'] != report['reference_sha256']):
            raise ValueError('Qualification/source identity mismatch')
        for name, digest in run['artifacts'].items():
            if sha(run_path / name) != digest:
                raise ValueError('Recorded trajectory changed')
        trace = load(run_path / 'final_rollout.npz'); raw = load(run_path / 'native_contact_buffers.npz')
        dt = float(trace['physics_dt_s']); count = round(horizon / dt) + 1
        if not np.isclose(dt, .0025, atol=1e-12, rtol=0):
            raise ValueError('Expected 400 Hz')
        contacts = contact_kinematics(previous_state_trace(trace), raw, .0001)
        # Native row i is reconstructed at index i+1 and describes state i.
        touch = contacts['touching'][1:count+1]; unknown = contacts['contact_unknown'][1:count+1]
        if touch.shape != (count, 2) or unknown.any():
            raise ValueError('Incomplete or unknown native geometry')
        geometry = cfg['bilateral_sliding']['geometry']
        center, axle, gap = wheel_geometry(trace, 'body_pos', 'body_quat', geometry)
        support = cylinder_support(center, axle, geometry['radius_m'], geometry['half_width_m'],
                                   plane_offset_m=geometry['floor_z_m'])
        ids = [trace['body_names'].astype(str).tolist().index('wheel_' + side + '_Link') for side in ('L', 'R')]
        velocity = point_velocity(support['authored_lowest_point_w_m'], trace['body_com_pos'][:, ids],
                                  trace['body_com_vel'][:, ids], trace['body_omega'][:, ids])[..., 2]
        ambiguous = (abs(axle[..., 2]) < 1e-5) | (abs(axle[..., 2]) > 1-1e-6)
        gap = gap[:count]; velocity = velocity[:count]; ambiguous = ambiguous[:count]
        native = contacts['min_separation_m'][1:count+1]
        valid = np.isfinite(native)
        error = float(abs(native[valid] - gap[valid]).max())
        if error > 1e-6:
            raise ValueError('Native geometry does not align to state timestamps')
        expected = q['qualification']['primary_contact_gate']['both_touch_fraction']
        if not np.isclose(touch[:-1].all(-1).mean(), expected, atol=1e-12, rtol=0):
            raise ValueError('Contact fraction differs from frozen qualification')
        vz = trace['root_com_vel'][:count, 2].astype(float)
        root_id = trace['body_names'].astype(str).tolist().index('base_Link')
        np.testing.assert_allclose(trace['body_com_vel'][:count, root_id],
                                   trace['root_com_vel'][:count], rtol=0, atol=1e-6)
        com_z = trace['body_com_pos'][:count, root_id, 2].astype(float)
        hold_start = round((horizon-.5)/dt)
        acceleration, mean = vertical_windows(vz, dt)
        # The existing managed term masks the first reward after reset. Exclude
        # its first control window from this validation-only proxy as well.
        cost = -np.expm1(-(mean[1:] / 5.)**2)
        metrics = dict(root_vz_RMS_m_s=float(np.sqrt(np.mean(vz**2))),
            root_vz_peak_abs_m_s=float(abs(vz).max()),
            terminal_vz_m_s=float(vz[-1]),
            last_half_second_peak_abs_vz_m_s=float(abs(vz[hold_start:]).max()),
            last_half_second_COM_z_displacement_m=float(com_z[-1]-com_z[hold_start]),
            COM_z_difference_vs_post_velocity_RMSE_m_s=float(np.sqrt(np.mean((np.diff(com_z)/dt-vz[1:])**2))),
            physical_vertical_acceleration_RMS_m_s2=float(np.sqrt(np.mean(acceleration**2))),
            control_mean_vertical_acceleration_RMS_m_s2=float(np.sqrt(np.mean(mean**2))),
            control_windows=len(mean),
            low_mean_but_large_substep_windows=int(((abs(mean) < 5) & (abs(acceleration).max(-1) > 50)).sum()),
            legacy_proxy_windows_after_initial_reset=len(cost), legacy_proxy_mean_cost=float(cost.mean()),
            legacy_proxy_cost_ge_0p999_fraction=float((cost >= .999).mean()),
            legacy_proxy_scale_m_s2=5., legacy_proxy_weight=cfg['reward_weights']['vertical_acceleration'])
        events = []
        for side in range(2):
            for index in loss_onsets(touch[:, side]):
                row = dict(cell=cell, wheel=('L', 'R')[side], onset_s=float(index*dt),
                    previous_gap_mm=float(gap[index-1, side]*1000), onset_gap_mm=float(gap[index, side]*1000),
                    preceding_interval_gap_rate_m_s=float((gap[index, side]-gap[index-1, side])/dt),
                    preceding_support_normal_velocity_m_s=None if ambiguous[index-1, side] else float(velocity[index-1, side]),
                    preceding_support_point_ambiguous=bool(ambiguous[index-1, side]),
                    preceding_root_vertical_velocity_m_s=float(vz[index-1]))
                events.append(row); event_rows.append(row)
        report['cases'].append(dict(cell=cell, qualification_sha256=sha(qpath),
            source_manifest_sha256=q['source_manifest_sha256'], raw_sha256=run['artifacts'],
            native_geometry_max_error_m=error, metrics=metrics, contact_loss_onsets=events,
            task_success=q['qualification']['task_success_claimed']))
        for name, value in dict(time_s=np.arange(count)*dt, gap_m=gap, touching=touch,
                                support_normal_velocity_m_s=velocity, support_ambiguous=ambiguous,
                                root_vz_m_s=vz, root_COM_z_m=com_z, physical_acceleration_m_s2=acceleration).items():
            arrays[f'case{number}_{name}'] = value
        plotted.append((cell, gap, touch, velocity, ambiguous, vz))
    report['interpretation'] = [
        'Positive clearance rate describes wheel separation, not tangential sliding or proof of friction saturation.',
        'Lowest-point normal velocity is evaluated at the preceding state; impulses and actuation within the next step can change it.',
        'A control-interval acceleration can cancel opposing physical substep accelerations.',
        'A near-one bounded cost has little amplitude discrimination; this does not establish a causal failure mechanism.',
        'No candidate reward has been installed by this audit; the active comparison and success gates are unchanged.']
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'signals.npz', **arrays)
    with (a.output / 'liftoff_events.csv').open('w', newline='') as stream:
        fields = ['cell', 'wheel', 'onset_s', 'previous_gap_mm', 'onset_gap_mm',
                  'preceding_interval_gap_rate_m_s', 'preceding_support_normal_velocity_m_s',
                  'preceding_support_point_ambiguous', 'preceding_root_vertical_velocity_m_s']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(event_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(plotted), figsize=(5.3*len(plotted), 8), squeeze=False,
                             sharey='row', constrained_layout=True)
    for column, (cell, gap, touch, velocity, ambiguous, vz) in enumerate(plotted):
        time = np.arange(len(vz))*.0025
        for side, color in enumerate(('#2677ad', '#db772a')):
            label = ('Left', 'Right')[side]
            axes[0, column].plot(time, gap[:, side]*1000, color=color, label=label, lw=.8)
            values = np.where(ambiguous[:, side], np.nan, velocity[:, side])
            axes[1, column].plot(time, values, color=color, lw=.6)
            axes[1, column].scatter(time[~touch[:, side]], values[~touch[:, side]], color=color, s=5)
        axes[0, column].axhline(.1, color='black', ls='--', lw=.7)
        axes[0, column].set(title=cell, ylabel='Authored clearance [mm]')
        axes[1, column].axhline(0, color='black', lw=.5)
        axes[1, column].set(ylabel='Lowest-point normal velocity [m/s]')
        axes[2, column].plot(time, vz, color='#595959', lw=.7)
        axes[2, column].set(xlabel='Time [s]', ylabel='Base COM vertical velocity [m/s]')
        for ax in axes[:, column]: ax.grid(alpha=.2); ax.set_xlim(0, horizon)
    axes[0, 0].legend()
    fig.suptitle('Recorded PhysX trajectories | native contact unchanged | normal motion is not lateral slip')
    fig.savefig(a.output / 'normal_motion_4x1.png', dpi=140); plt.close(fig)
    report['artifacts'] = {name: sha(a.output/name) for name in ('signals.npz', 'liftoff_events.csv', 'normal_motion_4x1.png')}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps([dict(cell=row['cell'], **row['metrics']) for row in report['cases']]))


if __name__ == '__main__':
    main()
