"""Compare recorded direct braking with archived hockey using common metrics.

No policy or physics is changed. New-seed direct braking and selected seed-42
hockey are descriptive comparisons, not a paired multi-seed causal estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_hockey_physx import load, q_matrix, sha
from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_timing import previous_state_trace
from prepare_hockey_equal_distance import persistent_stop
from analyze_ppo_contact_timestep import band_spectrum

ROOT = Path(__file__).resolve().parents[1]


def analyze_case(directory, reference_path, budget, *, brake_start=.4, brake_end=1.8):
    report = json.loads((directory/'report.json').read_text())
    for name, digest in report['artifacts'].items():
        if sha(directory/name) != digest:
            raise ValueError('Incomplete or changed artifact: '+str(directory/name))
    if report['motion_sha256'] != sha(reference_path):
        raise ValueError('Reference identity mismatch')
    tr = load(directory/'final_rollout.npz')
    raw = load(directory/'native_contact_buffers.npz')
    ref = load(reference_path)
    dt = float(tr['physics_dt_s'])
    end = (len(ref['joint_pos'])-1)/float(ref['fps'])
    if not np.isclose(dt, .0025) or not np.isclose(float(tr['policy_dt_s']), .02):
        raise ValueError('Comparison requires native 400/50 Hz recording')
    n = round(end/dt)+1
    t = np.arange(n)*dt
    contact = contact_kinematics(previous_state_trace(tr), raw, .0001)
    # Raw contact row k is geometry of physical state k, not state k+1.
    touch = contact['touching'][1:n+1]
    unknown = contact['contact_unknown'][1:n+1]
    if touch.shape != (n, 2):
        raise ValueError('Terminal geometry evidence missing')
    both = touch.all(-1) & ~unknown.any(-1)
    position = tr['root_pos'][:n, :2]
    direction = np.asarray(budget['direction_xy'])
    normal = np.array([-direction[1], direction[0]])
    displacement = position-position[0]
    forward, lateral = displacement@direction, displacement@normal
    velocity = tr['root_com_vel'][:n]
    speed_xy, speed_xyz = np.linalg.norm(velocity[:, :2], axis=-1), np.linalg.norm(velocity, axis=-1)
    R = q_matrix(tr['root_quat'][:n])
    tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
    rt = np.arange(len(ref['joint_pos']))/float(ref['fps'])
    target_xy = np.column_stack([np.interp(t, rt, ref['body_pos_w'][:, 0, k]) for k in (0, 1)])
    xy_error = np.linalg.norm(position-target_xy, axis=-1)
    if not 0 < brake_start < brake_end < end:
        raise ValueError('Invalid evaluation phase clock')
    core = (t>=brake_start-1e-9)&(t<=brake_end+1e-9)
    eligible = both & (tilt<45)
    stop_xy = persistent_stop(t, speed_xy, eligible, start=brake_start)
    stop_xyz = persistent_stop(t, speed_xyz, eligible, start=brake_start)
    hold = t>=end-.5-1e-9
    target = budget['total_distance_m']
    base = tr['body_names'].astype(str).tolist().index('base_Link')
    com = tr['body_com_pos'][:n, base]

    def stop_details(stop):
        if stop is None:
            return None
        i = stop['index']
        return dict(**stop, forward_from_start_m=float(forward[i]),
                    forward_since_brake_m=float(forward[i]-np.interp(brake_start, t, forward)),
                    COM_forward_from_start_m=float((com[i, :2]-com[0, :2])@direction))

    nonwheel = None
    if 'all_body_net_contact_force_w' in tr:
        ids = [i for i, name in enumerate(tr['contact_sensor_body_names'].astype(str))
               if name not in ('wheel_L_Link', 'wheel_R_Link')]
        force = tr['all_body_net_contact_force_w'][1:n, ids]
        if not np.isfinite(force).all():
            raise ValueError('Nonfinite non-wheel contact force')
        nonwheel = dict(max_force_N=float(np.linalg.norm(force, axis=-1).max()),
                        samples_over_numerical_epsilon=int((np.linalg.norm(force, axis=-1)>1e-6).any(-1).sum()),
                        epsilon_N=1e-6, measured_not_estimated_from_height=True)
    approach = (t>=.0225-1e-9)&(t<brake_start-1e-9)
    entry = approach&(t>=max(0.,brake_start-.1)-1e-9)
    target_v = np.column_stack([np.interp(t, rt, ref['body_lin_vel_w'][:, 0, k]) for k in (0, 1)])
    speed_ratio = float(speed_xy[approach].mean()/np.linalg.norm(target_v[approach], axis=-1).mean())
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    ref_R = q_matrix(ref['body_quat_w'][:, 0])
    target_yaw = np.interp(t, rt, np.unwrap(np.arctan2(ref_R[:, 1, 0], ref_R[:, 0, 0])))
    yaw_error = np.degrees((yaw-target_yaw+np.pi)%(2*np.pi)-np.pi)
    gates = dict(bilateral_all_native_samples=bool(both.all()), no_unknown_contact=not bool(unknown.any()),
        no_fall=bool((tilt<45).all()),
        xyz_stop_and_terminal_hold=bool(stop_xyz is not None and (speed_xyz[hold]<=.1).all()
                                       and (tilt[hold]<=30).all() and both[hold].all()),
        forward_at_stop_within_5cm=bool(stop_xyz is not None and abs(forward[stop_xyz['index']]-target)<=.05),
        maximum_forward_overshoot_within_5cm=bool(forward.max()<=target+.05),
        terminal_forward_within_5cm=bool(abs(forward[-1]-target)<=.05),
        lateral_excursion_within_10cm=bool(abs(lateral).max()<=.1),
        xy_RMSE_within_15cm=bool(np.sqrt(np.mean(xy_error**2))<=.15),
        xy_peak_within_30cm=bool(xy_error.max()<=.3),
        entry_mean_speed_at_least_0p8=bool(speed_xy[entry].mean()>=.8),
        approach_speed_retained=bool(.95<=speed_ratio<=1.1),
        body_heading_RMSE_within_15deg=bool(np.sqrt(np.mean(yaw_error[core]**2))<=15),
        nonwheel_contact_absent=None if nonwheel is None else nonwheel['samples_over_numerical_epsilon']==0)
    material = report['material_audit']
    if not material['live_physx_materials_verified'] or not material['ground_material_usd_verified']:
        raise ValueError('Unverified physical material')
    row = dict(case_id=directory.name, seed=report['seed'], actor_sha256=report['actor_sha256'],
        source_report_sha256=sha(directory/'report.json'), artifacts_sha256=report['artifacts'],
        mu_dynamic=material.get('wheel_mu_dynamic_range', material['wheel_mu_range']),
        mu_static=material.get('wheel_mu_static_range', material['wheel_mu_range']),
        initial_COM_speed_xy_m_s=float(speed_xy[0]), brake_entry_COM_speed_xy_m_s=float(np.interp(brake_start, t, speed_xy)),
        stop_planar=stop_details(stop_xy), stop_xyz=stop_details(stop_xyz),
        terminal_forward_m=float(forward[-1]), max_forward_m=float(forward.max()),
        overshoot_m=float(max(0, forward.max()-target)), max_lateral_excursion_m=float(abs(lateral).max()),
        terminal_planar_speed_m_s=float(speed_xy[-1]), maximum_tilt_deg=float(tilt.max()),
        first_fall_s=float(t[np.flatnonzero(tilt>=45)[0]]) if (tilt>=45).any() else None,
        bilateral_contact_fraction=float(both[:-1].mean()), terminal_bilateral_contact=bool(both[-1]),
        bilateral_interruption_s=float((~both[:-1]).sum()*dt), unknown_contact_samples=int(unknown.any(-1).sum()),
        angular_xy_10_100hz_rms_rad_s=band_spectrum(tr['root_omega'][:n][core, :2], dt)['common_band_rms'],
        angular_acceleration_RMS_rad_s2=float(np.sqrt(np.mean(np.sum((np.diff(tr['root_omega'][:n], axis=0)/dt)**2, axis=-1)))),
        action_change_RMS=float(np.sqrt(np.mean(np.diff(tr['actions'][:len(ref['joint_pos'])], axis=0)**2))),
        xy_RMSE_m=float(np.sqrt(np.mean(xy_error**2))), nonwheel_contact=nonwheel,
        gates=gates, failed_gates=[k for k, v in gates.items() if v is False],
        unverified_gates=[k for k, v in gates.items() if v is None],
        strict_direct_task_pass=all(v is True for v in gates.values()),
        task_failure_established=any(v is False for v in gates.values()),
        posture_target_is_soft=True, absolute_height_gate=False, lateral_slip_gate=False)
    signals = dict(time=t, forward=forward, lateral=lateral, speed=speed_xy, contact=both,
                   tilt=tilt, omega=tr['root_omega'][:n], root_xy=position,
                   root_quat=tr['root_quat'][:n], initial_state={k:tr[k][0] for k in
                   ('root_pos','root_quat','root_com_vel','root_omega','joint_pos','joint_vel')})
    return row, signals


def render(output, cases, signals, budget):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.5))
    colors = ['#2077b4', '#dd7043']
    names = ['Direct brake · seed 43', 'Hockey stop · selected seed 42']
    target = budget['total_distance_m']
    axs[0].axvspan(target-.05, target+.05, color='gray', alpha=.12)
    axs[0].axvline(target, color='gray', ls=':', lw=1)
    for i, (case, signal, name, color) in enumerate(zip(cases, signals, names, colors)):
        axs[0].plot(signal['forward'], signal['lateral'], color=color, label=name)
        axs[1].plot(signal['time'], signal['speed'], color=color)
        axs[2].plot(signal['time'], signal['contact'].astype(float)+i*1.5, color=color, lw=2)
        axs[3].bar(i, case['angular_xy_10_100hz_rms_rad_s'], color=color)
        axs[3].text(i, case['angular_xy_10_100hz_rms_rad_s']+.004,
                    f"{case['angular_xy_10_100hz_rms_rad_s']:.4f}", ha='center')
    axs[0].set(xlabel='Forward distance (m)', ylabel='Lateral displacement (m)', title='Same commanded stopping distance')
    axs[0].set_ylim(-.11, .11)
    axs[0].legend(fontsize=8)
    axs[1].axvline(.4, color='gray', ls=':', label='Braking command starts')
    axs[1].axhline(.1, color='gray', ls='--')
    axs[1].set(xlabel='Time (s)', ylabel='Base COM planar speed (m/s)', title='Actual entry speeds differ after approach')
    axs[2].set(xlabel='Time (s)', yticks=[0,1,1.5,2.5], yticklabels=['D: missing','D: both','H: missing','H: both'],
               ylim=(-.2,2.8), title='Bilateral native contact · 400 Hz')
    axs[3].set(xticks=[0,1], xticklabels=['Direct','Hockey'], ylabel='Angular XY RMS (rad/s)',
               title='Braking jitter · 10–100 Hz')
    for ax in axs:
        ax.grid(alpha=.15)
    fig.suptitle('PhysX · μs/μd = 0.7/0.7 · descriptive single-policy comparison, not matched training seeds', fontsize=12)
    fig.tight_layout()
    fig.savefig(output/'comparison_4x1.png', dpi=170)
    fig.savefig(output/'comparison_4x1.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    old = json.loads((ROOT/'results/hockey_equal_distance_20261102_v1/report.json').read_text())
    budget = old['budget']
    reference = ROOT/'results/hockey_equal_distance_20261102_v1/direct_brake/motion.npz'
    suite = json.loads((a.suite/'report.json').read_text())
    if len(suite['cases']) != 33 or suite['status'] != 'validation_suite_completed_requires_independent_qualification':
        raise ValueError('Expected all 33 predeclared cases')
    results, nominal = [], None
    for entry in suite['cases']:
        directory = a.suite/entry['id']
        if sha(directory/'report.json') != entry['report_sha256']:
            raise ValueError('Suite/case provenance mismatch')
        row, signal = analyze_case(directory, reference, budget)
        results.append(row)
        if entry['id'] == 'nominal':
            nominal = (row, signal)
    archived = old['archived_hockey_cases'][0]
    hockey, hs = analyze_case(ROOT/archived['source_directory'],
        ROOT/'results/hockey_equal_distance_20261102_v1/hockey/motion.npz', budget)
    initial_errors = {k:float(abs(nominal[1]['initial_state'][k]-hs['initial_state'][k]).max())
                      for k in hs['initial_state']}
    if max(initial_errors.values()) > 5e-5:
        raise ValueError('Nominal physical initial states differ')
    summary = dict(cases=33, policy_seeds=1, seed=43,
        bilateral_continuous_contact=sum(r['gates']['bilateral_all_native_samples'] for r in results),
        stopped_and_held=sum(r['gates']['xyz_stop_and_terminal_hold'] for r in results),
        strict_direct_task_pass=sum(r['strict_direct_task_pass'] for r in results),
        failures_by_gate={key:sum(r['gates'][key] is False for r in results) for key in results[0]['gates']},
        medians={key:float(np.median([r[key] for r in results])) for key in
                 ('angular_xy_10_100hz_rms_rad_s','maximum_tilt_deg','max_lateral_excursion_m','overshoot_m')})
    report = dict(kind='direct_brake_vs_archived_hockey_v1', budget=budget,
        same_nominal_initial_state_max_abs_errors=initial_errors,
        direct_summary=summary, direct_nominal=nominal[0], hockey_archived_nominal=hockey,
        direct_cases=results, matched_training_seed_comparison=False,
        matched_actual_brake_entry_speed=False,
        full_multi_seed_conclusion_available=False,
        archived_hockey_pass_claim_scope='Previously qualified selected seed-42 baseline; absent non-wheel telemetry is not silently assumed safe',
        scope='Common forward, speed, contact and jitter metrics; direct operational gates add fixed distance/corridor requirements. No sliding requirement.',
        source_sha256=sha(__file__), suite_report_sha256=sha(a.suite/'report.json'))
    (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    render(a.output, [nominal[0],hockey], [nominal[1],hs], budget)
    print(json.dumps(summary, indent=2))
    print(json.dumps({'direct_nominal':nominal[0], 'hockey_nominal':hockey}, indent=2))


if __name__ == '__main__':
    main()
