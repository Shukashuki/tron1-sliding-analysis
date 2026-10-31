#!/usr/bin/env python3
"""Offline contact-model audit of completed hockey-stop recordings.

Geometry, wheel-resultant capacity, anchor force/slip direction, and net
momentum are distinct checks. This is not a per-contact NCP certificate:
patch loads, solver-stage velocities and stabilization biases were not saved.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_hockey_contact_cone_relation import read_case
from audit_hockey_contact_momentum import momentum_balance, summarize
from evaluate_hockey_physx import load, q_matrix, sha
from wheel_ground_geometry import cylinder_support


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return dict(n=int(x.size), minimum=float(x.min()), median=float(np.median(x)),
                p95=float(np.quantile(x, .95)), maximum=float(x.max()))


def material_velocities(points, c0, c1, r0, r1, v0, v1, w0, w1, dt):
    """Pre/post twists and displacement of the SAME material points.

    Post points are transported, not replaced by the next manifold's points.
    None of these is asserted to equal an internal solver-stage velocity.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite timestep required')
    local = (points-c0) @ r0
    after = c1 + local @ r1.T
    return (v0 + np.cross(w0, points-c0),
            v1 + np.cross(w1, after-c1), (after-points)/dt)


def opposition(force, velocity):
    """Cosine 1 means opposing, 0 perpendicular, -1 assisting slip."""
    f = np.asarray(force, float)[..., :2]
    v = np.asarray(velocity, float)[..., :2]
    denominator = np.linalg.norm(f, axis=-1)*np.linalg.norm(v, axis=-1)
    out = np.full(denominator.shape, np.nan)
    np.divide(-(f*v).sum(-1), denominator, out=out, where=denominator > 1e-12)
    return np.clip(out, -1., 1.)


def direction_consistency(pre, post, interval):
    """Minimum pairwise direction cosine, undefined for any zero speed."""
    return np.minimum.reduce([-opposition(pre, post), -opposition(pre, interval),
                              -opposition(post, interval)])


def cone_metrics(normal, resultant, anchor_norm_sum, mu):
    """Necessary aggregate checks; never distribute normal load to anchors."""
    fn = np.asarray(normal, float)
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError('Positive finite friction coefficient required')
    capacity = mu*fn
    valid = fn > 1e-6
    return {name: np.divide(value, capacity, out=np.full(fn.shape, np.nan), where=valid)
            for name, value in [('resultant', np.asarray(resultant, float)),
                                ('anchor_norm_sum', np.asarray(anchor_norm_sum, float))]}


ANCHOR_COLUMNS = ['time_start_s', 'side', 'row', 'buffer_index', 'force_norm_N',
                  'speed_pre_m_s', 'speed_post_m_s', 'speed_interval_m_s',
                  'opposition_pre', 'opposition_post', 'opposition_interval',
                  'power_pre_W', 'power_post_W', 'power_interval_W',
                  'velocity_direction_consistency', 'bilateral_endpoint_contact']


def anchor_samples(trace, raw, n, touching):
    samples = []
    dt = float(trace['physics_dt_s'])
    origin = trace['environment_origin_w']
    for side, name in enumerate(('wheel_L_Link', 'wheel_R_Link')):
        bid = trace['body_names'].astype(str).tolist().index(name)
        rotations = q_matrix(trace['body_quat'][:, bid])
        for k in range(n):
            count = int(raw[f'w{side}_friction_buffer_2'][k].item())
            start = int(raw[f'w{side}_friction_buffer_3'][k].item())
            sl = slice(start, start+count)
            force = raw[f'w{side}_friction_buffer_0'][k, sl].astype(float)
            points = raw[f'w{side}_friction_buffer_1'][k, sl].astype(float)-origin
            if not count:
                continue
            pre, post, step = material_velocities(points,
                trace['body_com_pos'][k, bid], trace['body_com_pos'][k+1, bid],
                rotations[k], rotations[k+1], trace['body_com_vel'][k, bid],
                trace['body_com_vel'][k+1, bid], trace['body_omega'][k, bid],
                trace['body_omega'][k+1, bid], dt)
            speeds = [np.linalg.norm(v[:, :2], axis=-1) for v in (pre, post, step)]
            cosines = [opposition(force, v) for v in (pre, post, step)]
            powers = [(force[:, :2]*v[:, :2]).sum(-1) for v in (pre, post, step)]
            consistency = direction_consistency(pre, post, step)
            for j in range(count):
                samples.append([k*dt, side, k, start+j, np.linalg.norm(force[j, :2]),
                    *[v[j] for v in speeds], *[v[j] for v in cosines],
                    *[v[j] for v in powers], consistency[j],
                    float(touching[k:k+2].all())])
    return np.asarray(samples, float).reshape(-1, len(ANCHOR_COLUMNS))


def anchor_summary(a, dt, speed=.05, direction_deg=30., alignment_deg=15.):
    force_active = a[:, 4] > .001
    contact = a[:, 15].astype(bool)
    # Direction-change cases are reported separately, not silently discarded.
    sliding = (a[:, 5:8] > speed).all(-1) & force_active & contact
    coherent = sliding & (a[:, 14] >= np.cos(np.radians(direction_deg)))
    out = dict(anchor_records=len(a), force_active_records=int(force_active.sum()),
        sliding_in_all_three_readouts_records=int(sliding.sum()), coherent_sliding_records=int(coherent.sum()),
        sliding_direction_change_records=int((sliding & ~coherent).sum()),
        slow_or_transition_active_records=int((force_active & contact & ~sliding).sum()),
        samples_are_not_independent_trials=True)
    for mask_name, mask in [('all_active', force_active & contact),
                            ('three_readout_sliding', sliding), ('coherent_sliding', coherent)]:
        parts = {}
        for j, name in enumerate(('pre', 'post', 'interval')):
            valid = mask & np.isfinite(a[:, 8+j])
            w = a[valid, 4]*a[valid, 5+j]
            den = w.sum()
            cosine = a[valid, 8+j]
            power = a[valid, 11+j]
            parts[name] = dict(records=int(valid.sum()), opposition=stats(cosine),
                angle_from_opposition_deg=stats(np.degrees(np.arccos(cosine))),
                opposing_fraction=float((cosine > 0).mean()) if valid.any() else None,
                within_alignment_angle_fraction=float((cosine >= np.cos(np.radians(alignment_deg))).mean()) if valid.any() else None,
                force_speed_weighted_opposition=float((w*cosine).sum()/den) if den > 0 else None,
                negative_power_integral_proxy_J=float(np.minimum(power, 0).sum()*dt),
                positive_power_integral_proxy_J=float(np.maximum(power, 0).sum()*dt))
        out[mask_name] = parts
    return out


def dynamic_capacity_screen(a, mask, rho_sum, dt):
    """Wheel-level necessary check when all force-active anchors are sliding.

    Sampling intervals need not be contiguous; no per-anchor normal load is
    inferred. The eligibility mask does not depend on force opposition/ratio.
    """
    active = (a[:, 4] > .001) & a[:, 15].astype(bool)
    coherent = active & (a[:, 5:8] > .05).all(-1) & (a[:, 14] >= np.cos(np.radians(30)))
    counts = np.bincount(a[active, 2].astype(int), minlength=len(mask))
    good = np.bincount(a[coherent, 2].astype(int), minlength=len(mask))
    selected = mask & (counts > 0) & (good == counts) & np.isfinite(rho_sum)
    return dict(all_active_anchors_sliding_intervals=int(selected.sum()),
        sampled_total_s=float(selected.sum()*dt), intervals_need_not_be_contiguous=True,
        anchor_norm_sum_rho_dynamic=stats(rho_sum[selected]),
        near_dynamic_0p9_to_1p05_intervals=int((selected & (rho_sum >= .9) & (rho_sum <= 1.05)).sum()),
        over_dynamic_1p05_intervals=int((selected & (rho_sum > 1.05)).sum()))


def geometry_summary(tr, raw, run, s):
    n = len(s['time_start_s']); output = []; gaps = []
    for side, name in enumerate(('wheel_L_Link', 'wheel_R_Link')):
        bid = tr['body_names'].astype(str).tolist().index(name)
        r = q_matrix(tr['body_quat'][:n+1, bid])
        g = next(x for x in run['geometry_audit']['prototype'] if x['body'].endswith('/'+name))
        transform = np.asarray(g['original_cylinder_to_body_matrix'], float)
        center = tr['body_pos'][:n+1, bid] + r @ transform[3, :3]
        axle = r @ transform[2, :3]  # USD matrix stores the transpose convention.
        support = cylinder_support(center, axle, g['radius_m'], g['total_width_m']/2)
        gap = support['authored_signed_gap_m']; gaps.append(gap)
        native = s['native_gap_states_m'][:, side]
        fn_values = []; sep_values = []
        for k in range(n):
            count = int(raw[f'w{side}_normal_buffer_4'][k].item())
            start = int(raw[f'w{side}_normal_buffer_5'][k].item())
            sl = slice(start, start+count)
            normals = raw[f'w{side}_normal_buffer_2'][k, sl]
            valid = np.linalg.norm(normals, axis=-1) > 1e-8
            fn_values.extend(raw[f'w{side}_normal_buffer_0'][k, sl].reshape(-1)[valid])
            sep_values.extend(raw[f'w{side}_normal_buffer_3'][k, sl].reshape(-1)[valid])
        fn = np.asarray(fn_values); sep = np.asarray(sep_values)
        output.append(dict(wheel=('L', 'R')[side], native_gap_mm=stats(native*1000),
            authored_minimum_gap_mm=stats(gap*1000),
            native_minus_authored_gap_mm=stats((native-gap)*1000),
            geometric_tolerance_sensitivity=[dict(tolerance_m=tol,
                native_touching_state_fraction=float((native <= tol).mean()),
                authored_touching_state_fraction=float((gap <= tol).mean())) for tol in (0., 1e-5, 1e-4, 5e-4)],
            normal_point_count=len(fn), negative_normal_below_minus_1mN=int((fn < -.001).sum()),
            positive_normal_point_separation_mm=stats(sep[fn > .001]*1000),
            positive_separation_normal_impulse_fraction=float(fn[sep > .0001].sum()/fn.sum()),
            force_gap_abs_Nm=stats(abs(fn*sep)),
            force_gap_product_is_not_solver_complementarity_residual=True))
    return output, np.stack(gaps, axis=-1)


def plots(cases, signals, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#277da8', '#da7c30']
    for index, (case, s) in enumerate(zip(cases, signals)):
        fig, axes = plt.subplots(1, 4, figsize=(23, 4.8), constrained_layout=True)
        t = s['time_start_s']; a = s['anchor_samples']
        for side, color in enumerate(colors):
            label = ('L', 'R')[side]
            axes[0].plot(np.r_[t, t[-1]+case['dt_s']], s['native_gap_states_m'][:, side]*1000,
                         color=color, lw=.7, label=label)
            axes[1].plot(t, s['rho_static'][:, side], color=color, lw=.6, label=label+' resultant')
            axes[1].plot(t, s['anchor_sum_rho_static'][:, side], color=color, lw=.6, ls='--', label=label+' sum |anchor Ft|')
            mask = ((a[:, 1] == side) & (a[:, 4] > .001) & (a[:, 5:8] > .05).all(-1)
                    & (a[:, 14] >= np.cos(np.radians(30))) & a[:, 15].astype(bool))
            axes[2].scatter(a[mask, 0], a[mask, 10], color=color, s=3, alpha=.45, label=label)
        axes[0].axhline(.1, color='black', ls=':', label='contact tolerance')
        axes[0].set(title='Geometry (native separation)', ylabel='Gap [mm]')
        axes[1].axhline(1., color='black', ls=':')
        axes[1].set(title='Static capacity: two aggregate checks', ylabel='Utilization')
        axes[2].axhline(np.cos(np.radians(15)), color='black', ls=':', label='15 degree screen')
        axes[2].set(title='Anchor opposition to interval slip', ylabel='Cosine (1 = opposite)', ylim=(-1.05, 1.05))
        axes[3].plot(t, np.linalg.norm(s['momentum_residual_N'], axis=-1), color='#6c5791', lw=.7)
        axes[3].set(title='Whole-body momentum check', ylabel='Residual force [N]')
        for ax in axes:
            ax.axvspan(.4, 1.8, color='#b8c4d0', alpha=.15)
            ax.set(xlabel='Time [s]', xlim=(0, 2.76)); ax.grid(alpha=.2)
        for ax in axes[:3]: ax.legend(fontsize=7, loc='upper right')
        fig.suptitle(case['label']+'\n400 Hz archived data | shaded: braking | direction is a proxy, not a solver NCP certificate')
        fig.savefig(output/f'case{index}_framework_4x1.png', dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases-json', 'reference', 'mass-report', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    import torch
    torch.set_num_threads(2)
    read = lambda p: json.loads(p.read_text())
    selection = read(args.cases_json); mass_report = read(args.mass_report)
    cases = []; saved = {}; signals = []
    for index, source in enumerate(selection):
        base, s = read_case(source, args.cases_json, args.reference)
        cell, stage = source['case'].split(':')
        directory = (args.cases_json.parent/source['evidence']/source['run_name']/cell/stage).resolve()
        run = read(directory/'report.json')
        if run['external_forces'] != 0 or run['worlds'] != 1: raise ValueError('Unexpected external force/world count')
        if run['asset_sha256'] != mass_report['asset_sha256']: raise ValueError('Mass asset mismatch')
        tr = load(directory/'final_rollout.npz'); raw = load(directory/'native_contact_buffers.npz')
        n = len(s['time_start_s']); dt = float(tr['physics_dt_s'])
        geometry, gap = geometry_summary(tr, raw, run, s)
        anchors = anchor_samples(tr, raw, n, s['touching_states'])
        massmap = mass_report['authored_body_masses_kg']
        names = tr['body_names'].astype(str).tolist()
        if set(names) != set(massmap): raise ValueError('Body/mass mismatch')
        masses = np.array([massmap[name] for name in names])
        balance = momentum_balance(tr['body_com_vel'][:n+1], masses,
                                   s['normal_force_N'], tr['friction_force_w'][1:n+1], dt)
        total = np.asarray(tr['friction_force_w'][1:n+1], float).sum(1)
        normal = s['normal_force_N']
        rho_s = cone_metrics(normal, s['tangent_force_N'], s['anchor_force_norm_sum_N'], base['mu_static'])
        s['anchor_sum_rho_static'] = rho_s['anchor_norm_sum']
        s['authored_gap_states_m'] = gap
        s['anchor_samples'] = anchors
        s['momentum_residual_N'] = balance['residual_N']
        phases = {}
        masks = dict(whole_task=np.ones(n, bool), approach=s['time_end_s'] <= .4+1e-9,
                     braking=s['braking'], hold=s['time_start_s'] >= 1.8-1e-9)
        for phase, mask in masks.items():
            anchor_mask = mask[anchors[:, 2].astype(int)]
            wheels = []
            for side in (0, 1):
                a = anchors[anchor_mask & (anchors[:, 1] == side)]
                valid = mask & s['ratio_valid'][:, side]
                wheels.append(dict(wheel=('L', 'R')[side],
                    ratio_valid_intervals=int(valid.sum()),
                    resultant_rho_static=stats(rho_s['resultant'][valid, side]),
                    anchor_norm_sum_rho_static=stats(rho_s['anchor_norm_sum'][valid, side]),
                    resultant_static_over_1p05=int((rho_s['resultant'][valid, side] > 1.05).sum()),
                    anchor_norm_sum_static_over_1p05=int((rho_s['anchor_norm_sum'][valid, side] > 1.05).sum()),
                    contact_zero_normal_intervals=int((mask & s['touching_states'][:-1, side] & (normal[:, side] <= 1e-6)).sum()),
                    zero_normal_nonzero_tangent_intervals=int((mask & (normal[:, side] <= 1e-6) & (s['anchor_force_norm_sum_N'][:, side] > .001)).sum()),
                    anchor_direction=anchor_summary(a, dt),
                    dynamic_capacity_when_all_anchors_slide=dynamic_capacity_screen(
                        a, mask, s['anchor_sum_rho_dynamic'][:, side], dt),
                    speed_threshold_sensitivity={str(v): anchor_summary(a, dt, speed=v)['coherent_sliding'] for v in (.02, .1)}))
            magnitude = s['tangent_force_N'][mask].sum(-1)
            cancellation = 1 - np.divide(np.linalg.norm(total[mask, :2], axis=-1), magnitude,
                                        out=np.ones_like(magnitude), where=magnitude > 1e-6)
            phases[phase] = dict(intervals=int(mask.sum()), wheels=wheels,
                momentum=summarize(balance['residual_N'], mask),
                wheel_resultant_cancellation_fraction=stats(cancellation))
        observed = balance['measured_dP_dt_N']; expected = balance['wheel_force_plus_gravity_N']
        phases['braking']['force_time_shift_negative_controls'] = {
            str(lag): summarize(observed-np.roll(expected, -lag, axis=0), s['braking']) for lag in (-1, 0, 1)}
        qualification = read(directory.parent/'nominal_qualification.json')['qualification']
        case = dict(label=base['label'], cell=cell, dt_s=dt, intervals=n,
            mu_static=base['mu_static'], mu_dynamic=base['mu_dynamic'],
            task_success=base['task_success'], bilateral_interruption_ms=base['bilateral_interruption_ms'],
            xy_RMSE_m=qualification['xy_RMSE_m'],
            original_qualification_slide_total_s=base['original_qualification_slide_total_s'],
            raw_sha256=base['raw_sha256'], qualification_sha256=base['qualification_sha256'],
            source_manifest_sha256=base['source_manifest_sha256'],
            run_report_sha256=sha(directory/'report.json'), actor_sha256=run['actor_sha256'],
            native_force_audit=base['native_force_audit'], flat_plane_audit=base['flat_plane_audit'],
            geometry=geometry, phases=phases)
        cases.append(case); signals.append(s)
        for name, value in s.items(): saved[f'case{index}_{name}'] = value
        print(json.dumps(dict(case=cell, success=case['task_success'],
                             momentum_RMS_N=phases['whole_task']['momentum']['vector_RMS_N'])), flush=True)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'signals.npz', **saved)
    with (args.output/'anchors.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n'); writer.writerow(['case']+ANCHOR_COLUMNS)
        for i, s in enumerate(signals):
            writer.writerows([i, *row] for row in s['anchor_samples'])
    plots(cases, signals, args.output)
    report = dict(kind='hockey_contact_model_framework_audit_v1',
        paper='https://arxiv.org/html/2304.06372v3',
        source_sha256=sha(__file__), selection_sha256=sha(args.cases_json),
        reference_sha256=sha(args.reference/'motion.npz'), mass_report_sha256=sha(args.mass_report),
        dependency_sha256={f: sha(Path(__file__).with_name(f)) for f in (
            'audit_hockey_contact_cone_relation.py', 'audit_hockey_contact_momentum.py',
            'audit_hockey_contact_timing.py', 'evaluate_hockey_physx.py',
            'audit_physx_contact_float32.py', 'analyze_hockey_ground_contact.py',
            'wheel_friction_capacity.py', 'wheel_ground_geometry.py')},
        physics_steps=0, training_changed=False, original_success_gates_changed=False,
        scope='Four complete nominal recordings, t=0 through 2.76 s; not 33 new perturbation audits.',
        thresholds=dict(contact_tolerance_m=.0001, positive_force_N=.001,
            sliding_speed_m_s=.05, within_step_direction_change_deg=30,
            friction_alignment_angle_deg=15, capacity_relative_screen=1.05),
        anchor_columns=ANCHOR_COLUMNS,
        framework_status=dict(geometry='measured; tolerance and authored-cylinder agreement reported',
            normal_unilateral='nonnegative force and gap diagnostics; exact Signorini NOT certified',
            coulomb='wheel-resultant and sum-of-anchor-norm necessary checks; per-patch NOT certified',
            maximum_dissipation='anchor direction and power proxies; exact solver-stage MDP NOT certified',
            net_dynamics='independent linear-momentum balance and temporal negative controls',
            numerical_convergence='NOT TESTED: no matched timestep/iteration sweep of these policies',
            physical_tire_fidelity='NOT TESTED: no measured tire-force/camber validation'),
        limitations=[
            'Rigid Coulomb point-contact consistency is not empirical tire/ice validation.',
            'Geometric touching at 400 Hz endpoints does not prove continuous-time touching.',
            'Native manifold normal loads cannot be assigned to friction anchors without patch IDs.',
            'Summed anchor norms give a necessary aggregate capacity bound under a common mu/normal, not a sufficient patchwise certificate.',
            'Pre/post twists and material displacement do not expose solver iteration velocities or stabilization bias.',
            'Interval-averaged force times any sampled velocity is a power/work proxy, not exact friction energy.',
            'A transported material point can lift during pure rolling; positive material normal velocity is not a separation violation.',
            'Authored mass is checked against asset identity, not a new runtime mass readback.',
            'Four closed-loop nominal trajectories are not identical-action contact-model comparisons.',
            'Angle, speed and cone screens are diagnostic sensitivity choices, not new task success gates.'],
        cases=cases, artifact_sha256={f.name: sha(f) for f in args.output.iterdir()})
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
