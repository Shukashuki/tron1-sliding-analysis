"""Native wheel friction capacity before the original PPO's first 45-degree tilt.

Read existing recorded PhysX forces. Do not rerun physics, change success gates,
infer normal forces at friction anchors, or count an unloaded wheel as unsaturated.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, sha, reconstruct_contacts, derive_signals, segments, q_matrix


def point_speeds(trace, raw, signals):
    """Force-weighted velocity at loaded normal points, using post-step rigid twist."""
    shape = signals['normal_fz'].shape
    out = {k: np.full(shape, np.nan) for k in ('longitudinal_rms', 'tangential_rms')}
    out['dissipative_tangential_anchor_count'] = np.zeros(shape, dtype=int)
    names = trace['body_names'].astype(str).tolist()
    origin = trace['environment_origin_w']
    for side, name in enumerate(('wheel_L_Link', 'wheel_R_Link')):
        bid = names.index(name)
        axles = q_matrix(trace['body_quat'][:, bid])[..., :, 1]
        for row, i in enumerate(raw['physical_sample_indices']):
            count = int(raw[f'w{side}_normal_buffer_4'][row].item())
            start = int(raw[f'w{side}_normal_buffer_5'][row].item())
            fn = raw[f'w{side}_normal_buffer_0'][row, start:start+count].reshape(-1).astype(float)
            points = raw[f'w{side}_normal_buffer_1'][row, start:start+count].astype(float)-origin
            com, vel, omega = [trace[k][i, bid].astype(float) for k in ('body_com_pos', 'body_com_vel', 'body_omega')]
            loaded = fn > 5.
            if loaded.any():
                normals = raw[f'w{side}_normal_buffer_2'][row, start:start+count].astype(float)[loaded]
                normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
                lateral = axles[i]-(normals@axles[i])[:, None]*normals
                lateral /= np.linalg.norm(lateral, axis=-1, keepdims=True)
                forward = np.cross(lateral, normals)
                pv = vel+np.cross(omega, points[loaded]-com)
                out['longitudinal_rms'][i, side] = np.sqrt(np.average(np.sum(pv*forward, axis=-1)**2, weights=fn[loaded]))
                tangent = pv-np.sum(pv*normals, axis=-1)[:, None]*normals
                out['tangential_rms'][i, side] = np.sqrt(np.average(np.sum(tangent**2, axis=-1), weights=fn[loaded]))
                lateral_rms = np.sqrt(np.average(np.sum(pv*lateral, axis=-1)**2, weights=fn[loaded]))
                if abs(lateral_rms-signals['normal_point_lateral_rms'][i, side]) > 1e-7:
                    raise ValueError('Wheel-plane velocity disagrees with frozen lateral-slip reconstruction')
            count = int(raw[f'w{side}_friction_buffer_2'][row].item())
            start = int(raw[f'w{side}_friction_buffer_3'][row].item())
            ft = raw[f'w{side}_friction_buffer_0'][row, start:start+count].astype(float)
            anchors = raw[f'w{side}_friction_buffer_1'][row, start:start+count].astype(float)-origin
            av = vel+np.cross(omega, anchors-com)
            out['dissipative_tangential_anchor_count'][i, side] = np.sum(
                (np.linalg.norm(av[:, :2], axis=-1) > .05) & (np.sum(ft*av, axis=-1) < 0))
    return out


def read_case(cell, case_id):
    directory = cell/'eval-suite'/case_id
    run = json.loads((directory/'report.json').read_text())
    suite = json.loads((cell/'analyze/report.json').read_text())
    entry = next(x for x in suite['cases'] if x['id'] == case_id)
    qualification_path = cell/'analyze'/(case_id+'.json')
    if sha(qualification_path) != entry['report_sha256']:
        raise ValueError('Qualification changed')
    qualification = json.loads(qualification_path.read_text())
    if sha(directory/'report.json') != qualification['run_report_sha256']:
        raise ValueError('Recorded run identity changed')
    for name, digest in qualification['input_sha256'].items():
        if sha(directory/name) != digest:
            raise ValueError('Recorded input changed')
    material = run['material_audit']
    if not (material['live_physx_materials_verified'] and material['ground_material_usd_verified']
            and np.allclose(material['wheel_mu_range'], [.7, .7], atol=1e-6, rtol=0)
            and all(g['friction'] == 1 and g['combine'] == 'multiply' for g in material['ground_colliders'])):
        raise ValueError('Requires the frozen mu=0.7 flat-ground validation')
    trace = load(directory/'final_rollout.npz')
    raw = load(directory/'native_contact_buffers.npz')
    contact, audit = reconstruct_contacts(trace, raw, .7)
    s = derive_signals(trace, contact)
    archived = load(cell/'analyze'/(case_id+'.signals.npz'))
    differences = {}
    for key, value in s.items():
        # Remote NumPy 1.x / local NumPy 2.x differ by <8e-6 deg in beta
        # subtraction. Preserve strict checks on every force and other field.
        tolerance = 1e-5 if key in ('body_beta_deg', 'wheel_directed_beta_deg', 'wheel_axial_beta_deg') else 1e-10
        error = float(np.nanmax(abs(value.astype(float)-archived[key].astype(float))))
        differences[key] = dict(maximum_error=error, absolute_tolerance=tolerance)
        if not np.allclose(value, archived[key], atol=tolerance, rtol=1e-10, equal_nan=True):
            raise ValueError('New native reconstruction differs: '+key)
    audit['archived_signal_comparison'] = differences
    s.update(point_speeds(trace, raw, s))
    s['capacity_N'] = .7*s['normal_fz']
    s['tangent_force_N'] = np.linalg.norm(s['friction_force'][:, :, :2], axis=-1)
    theta = np.deg2rad(s['wheel_heading_deg'].astype(float))
    fx, fy = s['friction_force'][:, :, 0], s['friction_force'][:, :, 1]
    s['longitudinal_force_N'] = fx*np.cos(theta)+fy*np.sin(theta)
    s['lateral_force_N'] = -fx*np.sin(theta)+fy*np.cos(theta)
    if not np.allclose(np.hypot(s['longitudinal_force_N'], s['lateral_force_N']), s['tangent_force_N'], atol=1e-9):
        raise ValueError('Tangent-force basis does not preserve magnitude')
    s['loaded'] = (s['normal_fz'] > 5) & s['native_sample_valid'][:, None]
    s['rho'] = np.full(s['capacity_N'].shape, np.nan)
    np.divide(s['tangent_force_N'], s['capacity_N'], out=s['rho'], where=s['loaded'])
    s['radial_margin_N'] = np.where(s['loaded'], s['capacity_N']-s['tangent_force_N'], np.nan)
    s['near_limit'] = s['loaded'] & (s['rho'] >= .9) & (s['rho'] <= 1.05)
    s['corroborated_lateral_slip'] = s['near_limit'] & (s['normal_point_lateral_rms'] > .05)
    s['corroborated_lateral_slip'] &= (s['anchor_power'] < -.05) & (s['dissipative_slipping_anchor_count'] >= 1)
    s['corroborated_tangent_slip'] = s['near_limit'] & (s['tangential_rms'] > .05)
    s['corroborated_tangent_slip'] &= (s['anchor_power'] < -.05) & (s['dissipative_tangential_anchor_count'] >= 1)
    return dict(run=run, qualification=qualification, signals=s, native_audit=audit,
                provenance=dict(run_report_sha256=sha(directory/'report.json'),
                    qualification_sha256=sha(qualification_path), recorded_artifacts=qualification['input_sha256']))


def statistics(s, start, end):
    t = s['time_s']; dt = float(np.median(np.diff(t)))
    mask = s['native_sample_valid'] & (t-dt >= start-1e-9) & (t <= end+1e-9)
    if not mask.any():
        return None
    def summary(x):
        x = np.asarray(x); x = x[np.isfinite(x)]
        return dict(median=float(np.median(x)), p05=float(np.quantile(x, .05)),
                    p95=float(np.quantile(x, .95)), maximum=float(np.max(x))) if len(x) else None
    result = dict(requested_interval_s=[float(start), float(end)], interval_sample_count=int(mask.sum()),
        observed_duration_s=float(mask.sum()*dt), anywheel_support_fraction=float(s['loaded'][mask].any(-1).mean()),
        combined_capacity_N=summary(s['capacity_N'][mask].sum(-1)), wheels=[])
    for k, side in enumerate(('left', 'right')):
        loaded = mask & s['loaded'][:, k]
        row = dict(side=side, loaded_samples=int(loaded.sum()), loaded_fraction=float(loaded.sum()/mask.sum()),
            capacity_all_samples_N=summary(s['capacity_N'][mask, k]),
            capacity_loaded_samples_N=summary(s['capacity_N'][loaded, k]),
            normal_load_loaded_N=summary(s['normal_fz'][loaded, k]),
            tangent_force_loaded_N=summary(s['tangent_force_N'][loaded, k]),
            abs_longitudinal_force_loaded_N=summary(abs(s['longitudinal_force_N'][loaded, k])),
            abs_lateral_force_loaded_N=summary(abs(s['lateral_force_N'][loaded, k])),
            utilization_loaded=summary(s['rho'][loaded, k]),
            radial_margin_loaded_N=summary(s['radial_margin_N'][loaded, k]),
            near_limit_fraction_of_loaded=float(s['near_limit'][loaded, k].mean()) if loaded.any() else None,
            above_1p05_fraction_of_loaded=float((s['rho'][loaded, k] > 1.05).mean()) if loaded.any() else None,
            anchor_sum_utilization_loaded=summary(s['anchor_force_sum_utilization'][loaded, k]))
        for key, field in (('lateral', 'corroborated_lateral_slip'), ('tangent', 'corroborated_tangent_slip')):
            events = segments(t, mask & s[field][:, k], interval_samples=True)
            row[key+'_saturated_dissipative_slip'] = dict(events=events,
                total_duration_s=float((mask & s[field][:, k]).sum()*dt),
                longest_duration_s=max((e['duration_s'] for e in events), default=0.))
        normal_valid = loaded & (s['loaded_normal_points'][:, k] > 0)
        row['normal_point_lateral_rms_m_s'] = summary(s['normal_point_lateral_rms'][normal_valid, k])
        row['normal_point_tangent_rms_m_s'] = summary(s['tangential_rms'][normal_valid, k])
        result['wheels'].append(row)
    result['support_gap_events'] = segments(t, mask & ~s['loaded'].any(-1), interval_samples=True)
    return result


def plot_case(original, comparison, fall, path, start):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = original['signals']; c = comparison['signals']; t = s['time_s']
    dt = float(np.median(np.diff(t))); end = fall-dt
    mask = s['native_sample_valid'] & (t >= start-1e-9) & (t <= end+1e-9)
    colors = ('#2874b2', '#d77325')
    fig, axes = plt.subplots(1, 4, figsize=(23, 5), constrained_layout=True)
    for k, side in enumerate(('Left', 'Right')):
        axes[k].plot(t[mask], s['capacity_N'][mask, k], color=colors[k], lw=1.7, label='Original capacity: mu Fn')
        axes[k].plot(t[mask], s['tangent_force_N'][mask, k], color='black', lw=1, label='Original actual |Ft|')
        axes[k].plot(t[mask], c['capacity_N'][mask, k], color='#888888', lw=1, ls='--', label='Hockey PPO capacity (same time)')
        axes[k].set(title=side+' wheel: friction capacity and force', ylabel='Force [N]', ylim=(0, None))
        axes[k].legend(fontsize=8)
        axes[2].plot(t[mask], s['rho'][mask, k], color=colors[k], label=side+' original')
        axes[2].plot(t[mask], c['rho'][mask, k], color=colors[k], ls='--', alpha=.55, label=side+' hockey')
        valid = mask & s['loaded'][:, k] & (s['loaded_normal_points'][:, k] > 0)
        lateral = np.where(valid, s['normal_point_lateral_rms'][:, k], np.nan)
        axes[3].plot(t[mask], lateral[mask], color=colors[k], label=side+' original lateral slip')
    axes[2].axhspan(.9, 1.05, color='#bca871', alpha=.16)
    axes[2].axhline(1, color='gray', lw=.7, ls=':')
    axes[2].set(title='Wheel-aggregate utilization', ylabel='|sum Ft| / (mu sum Fn)', ylim=(0, 1.15))
    axes[2].legend(fontsize=8)
    axes[3].axhline(.05, color='gray', lw=.8, ls=':', label='0.05 m/s slip threshold')
    axes[3].set(title='Loaded lateral material slip and tilt', ylabel='Lateral point speed RMS [m/s]', ylim=(0, None))
    tilt = axes[3].twinx()
    tilt.plot(t[mask], s['root_tilt_deg'][mask], color='#9c2b69', lw=1, label='Original tilt')
    tilt.axhline(45, color='#9c2b69', lw=.7, ls='--'); tilt.set(ylabel='Body tilt [deg]', ylim=(0, 50))
    handles, labels = axes[3].get_legend_handles_labels(); h, l = tilt.get_legend_handles_labels()
    axes[3].legend(handles+h, labels+l, fontsize=7, loc='upper left')
    for ax in axes:
        for gap in segments(t, mask & ~s['loaded'].any(-1), interval_samples=True):
            ax.axvspan(max(start, gap['start_s']), gap['end_s'], color='gray', alpha=.15)
        ax.axvline(fall, color='#b22222', lw=1, ls='--')
        ax.set(xlim=(start, fall+.005), xlabel='Time [s]'); ax.grid(alpha=.2)
    fig.suptitle(f"Original PPO, {original['run']['case_id']}, x1 seed42, mu=0.7 | first tilt >=45 deg: {fall:.4f} s\n"
                 'Raw 400 Hz samples; gray bands = neither wheel >5 N; unloaded utilization omitted; dashed forces = matched successful PPO')
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original-cell', 'comparison-cell', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--case', action='append', dest='cases')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = dict(kind='original_ppo_prefall_friction_capacity', cases={}, effective_mu=.7,
        capacity_definition='C_i = mu * sum normal Fz on wheel i; all samples including unloaded',
        utilization_definition='rho_i = norm(sum friction-anchor forces XY) / C_i; defined only when wheel Fz>5 N',
        last_window_definition='Fixed-duration complete substeps ending at the sample before first tilt >=45 deg; exclude threshold-crossing and later samples',
        radial_margin_interpretation='C-|Ft| is a wheel-aggregate radial difference, not guaranteed feasible corrective force or balancing moment',
        limitations=['No pointwise normal force at friction anchors; no pointwise cone certificate.',
            'Net force can hide cancellation between anchors; anchor-sum utilization is also reported.',
            'Completed-step forces paired with post-step velocities give a power proxy.',
            'Capacity/saturation is not proof of a fall cause; required corrective wrench is not measured.',
            'Only nominal and selected failure from one original-speed training seed are examined.'],
        physics_steps_performed=0, success_gates_changed=False, script_sha256=sha(__file__))
    a.output.mkdir(parents=True)
    for case_id in a.cases or ('nominal', 'trial_022'):
        original, comparison = [read_case(cell, case_id) for cell in (a.original_cell, a.comparison_cell)]
        for key in ('seed', 'speed_scale', 'case_id', 'motion_sha256'):
            if original['run'][key] != comparison['run'][key]:
                raise ValueError('Unmatched policy comparison')
        if (original['run']['method'], comparison['run']['method'], original['run']['seed'], original['run']['speed_scale']) != ('beyondmimic', 'hockey', 42, 1.):
            raise ValueError('This diagnostic is scoped to the completed original-speed seed-42 pair')
        if original['qualification']['case'] != comparison['qualification']['case']:
            raise ValueError('Unpaired reset draw')
        s = original['signals']; t = s['time_s']; dt = float(np.median(np.diff(t)))
        indices = np.flatnonzero(s['root_tilt_deg'] >= 45.)
        if not len(indices):
            raise ValueError('Original episode has no 45-degree tilt event')
        fall = float(t[indices[0]]); end = fall-dt
        if abs(fall-original['qualification']['first_45deg_tilt_s']) > 1e-9:
            raise ValueError('Fall diagnostic changed')
        windows = dict(braking=(.4, min(1.8, end)), stop_before_fall=(1.8, end),
                       last_500ms=(end-.5, end), last_100ms=(end-.1, end))
        result['cases'][case_id] = dict(first_45deg_tilt_s=fall, last_pre_fall_sample_s=end,
            original={name: statistics(s, *bounds) for name, bounds in windows.items()},
            hockey_same_absolute_times={name: statistics(comparison['signals'], *bounds) for name, bounds in windows.items()},
            original_native_audit=original['native_audit'], comparison_native_audit=comparison['native_audit'],
            original_provenance=original['provenance'], comparison_provenance=comparison['provenance'])
        if result['cases'][case_id]['original']['last_100ms']['interval_sample_count'] != 40:
            raise ValueError('Expected exactly forty pre-fall 2.5 ms force intervals')
        fields = ['time_s','side','loaded','normal_fz_N','capacity_N','tangent_force_N','longitudinal_force_N',
                  'lateral_force_N','rho','radial_margin_N','lateral_rms_m_s','longitudinal_rms_m_s',
                  'tangent_rms_m_s','near_limit','corroborated_lateral_slip','corroborated_tangent_slip','body_tilt_deg']
        with (a.output/(case_id+'.prefall.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader()
            for i in np.flatnonzero(s['native_sample_valid'] & (t >= .0225-1e-9) & (t <= end+1e-9)):
                for k, side in enumerate(('left', 'right')):
                    row = dict(time_s=t[i], side=side, loaded=bool(s['loaded'][i,k]), normal_fz_N=s['normal_fz'][i,k],
                        lateral_rms_m_s=s['normal_point_lateral_rms'][i,k] if s['loaded_normal_points'][i,k]>0 else None,
                        longitudinal_rms_m_s=s['longitudinal_rms'][i,k], tangent_rms_m_s=s['tangential_rms'][i,k],
                        body_tilt_deg=s['root_tilt_deg'][i])
                    for key in fields[4:10]+fields[13:16]:
                        row[key] = s[key][i,k].item()
                    w.writerow({key: ('' if isinstance(value, float) and not np.isfinite(value) else value) for key,value in row.items()})
        plot_case(original, comparison, fall, a.output/(case_id+'.last500ms_4x1.png'), end-.5)
        plot_case(original, comparison, fall, a.output/(case_id+'.full_prefall_4x1.png'), .02)
    (a.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output), cases=list(result['cases']))))


if __name__ == '__main__':
    main()
