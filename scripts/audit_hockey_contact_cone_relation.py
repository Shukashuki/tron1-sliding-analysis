"""Relate geometric contact loss to wheel-resultant friction in archived screens.

No physics, training or task gate is changed. The native geometry at raw row k
describes state k; its force is impulse/dt over [k, k+1]. Ratios are diagnostics,
not per-anchor cone certificates or observed solver stick/slip flags.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_hockey_ground_contact import contact_kinematics
from audit_hockey_contact_liftoff import loss_onsets
from audit_hockey_contact_timing import previous_state_trace, vector_signals
from evaluate_hockey_physx import load, q_matrix, reconstruct_contacts, sha
from wheel_friction_capacity import capacity_ratios


def finite_stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return dict(count=int(len(x)), median=float(np.median(x)),
                p05=float(np.quantile(x, .05)), p95=float(np.quantile(x, .95)),
                minimum=float(x.min()), maximum=float(x.max()))


def interval_contact(touching_states):
    """Require both recorded endpoints; this is not continuous-time proof."""
    touch = np.asarray(touching_states, dtype=bool)
    if touch.ndim != 2 or touch.shape[1] != 2 or len(touch) < 2:
        raise ValueError('Require consecutive two-wheel states including endpoint')
    return touch[:-1] & touch[1:]


def loss_windows(touching_states, dt, lookback_s=.05):
    """Last force interval ends at loss; never include force after separation."""
    touch = np.asarray(touching_states, dtype=bool)
    interval_contact(touch)
    if not dt > 0 or not lookback_s >= dt:
        raise ValueError('Invalid sampling/window duration')
    steps = round(lookback_s / dt)
    return [dict(side=side, state_index=int(i), preceding_interval=int(i-1),
                 window_start=int(max(0, i-steps)), window_stop=int(i))
            for side in range(2) for i in loss_onsets(touch[:, side])]


def anchor_diagnostics(raw, count):
    total = np.zeros((count, 2)); noncontact_fn = np.zeros_like(total)
    max_tangent_z = 0.; max_normal_xy = 0.
    for side in range(2):
        for row in range(count):
            ac = int(raw[f'w{side}_friction_buffer_2'][row].item())
            ast = int(raw[f'w{side}_friction_buffer_3'][row].item())
            ft = raw[f'w{side}_friction_buffer_0'][row, ast:ast+ac].astype(float)
            total[row, side] = np.linalg.norm(ft[:, :2], axis=-1).sum()
            if ac:
                max_tangent_z = max(max_tangent_z, float(abs(ft[:, 2]).max()))
            nc = int(raw[f'w{side}_normal_buffer_4'][row].item())
            ns = int(raw[f'w{side}_normal_buffer_5'][row].item())
            fn = raw[f'w{side}_normal_buffer_0'][row, ns:ns+nc].reshape(-1).astype(float)
            normals = raw[f'w{side}_normal_buffer_2'][row, ns:ns+nc].astype(float)
            sep = raw[f'w{side}_normal_buffer_3'][row, ns:ns+nc].reshape(-1)
            if (fn > 1e-6).any():
                max_normal_xy = max(max_normal_xy, float(abs(normals[fn > 1e-6, :2]).max()))
            noncontact_fn[row, side] = fn[sep > .0001].sum()
    if max_normal_xy > 1e-4 or max_tangent_z > .001:
        raise ValueError('Horizontal-plane force projection invalid')
    return total, noncontact_fn, dict(max_normal_xy=max_normal_xy,
                                      max_anchor_force_z_N=max_tangent_z)


def read_case(source, selection, reference):
    read = lambda p: json.loads(p.read_text())
    bundle = (selection.parent / source['bundle']).resolve()
    evidence = (selection.parent / source['evidence']).resolve()
    cell, stage = source['case'].split(':')
    if stage != 'nominal':
        raise ValueError('Only completed nominal recordings are comparable')
    plan = read(bundle/'plan.json')
    entry = next(c for c in plan['cells'] if c['id'] == cell)
    protocol = bundle/entry['protocol']; cfg = read(protocol)
    directory = evidence/source['run_name']/cell/stage
    run = read(directory/'report.json'); qualification = read(directory.parent/'nominal_qualification.json')
    if (qualification['run_report_sha256'] != sha(directory/'report.json')
            or qualification['protocol_sha256'] != sha(protocol)
            or qualification['source_manifest_sha256'] != sha(bundle/'manifest.json')
            or run['motion_sha256'] != sha(reference/'motion.npz')):
        raise ValueError('Source/qualification identity mismatch')
    for name, digest in run['artifacts'].items():
        if sha(directory/name) != digest:
            raise ValueError('Raw recording changed')
    motion = load(reference/'motion.npz'); horizon = (len(motion['joint_pos'])-1)/float(motion['fps'])
    tr = load(directory/'final_rollout.npz'); raw = load(directory/'native_contact_buffers.npz')
    dt = float(tr['physics_dt_s']); n = round(horizon/dt)
    if not np.isclose(dt, .0025, atol=1e-12, rtol=0):
        raise ValueError('Expected complete 400 Hz recording')
    mu_d = cfg['evaluation_mu']; mu_s = mu_d*cfg['static_dynamic_friction_ratio']
    material = run['material_audit']
    if not material['live_physx_materials_verified'] or not material['ground_material_usd_verified']:
        raise ValueError('Material values unverified')
    np.testing.assert_allclose(material['wheel_mu_dynamic_range'], [mu_d]*2, atol=1e-6, rtol=0)
    np.testing.assert_allclose(material['wheel_mu_static_range'], [mu_s]*2, atol=1e-6, rtol=0)
    if not all(g['friction'] == 1 and g['combine'] == 'multiply' for g in material['ground_colliders']):
        raise ValueError('Unexpected ground material')
    contact = contact_kinematics(previous_state_trace(tr), raw, .0001)
    touch = contact['touching'][1:n+2]
    if touch.shape != (n+1, 2) or contact['contact_unknown'][1:n+2].any():
        raise ValueError('Missing/unknown geometric contact')
    q = qualification['qualification']; gate = q['primary_contact_gate']
    np.testing.assert_allclose(touch[:-1].all(-1).mean(), gate['both_touch_fraction'], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(touch[-1], gate['terminal_touching'])
    force, force_audit = reconstruct_contacts(tr, raw, mu_d, readback_mode='recorded_float32')
    vector = vector_signals(tr, raw, .0001)
    fn = force['normal_fz'][1:n+1]; ft = force['friction_force'][1:n+1]
    ratios = capacity_ratios(fn, ft, touch[:-1], mu_s, mu_d)
    ids = [tr['body_names'].astype(str).tolist().index('wheel_'+s+'_Link') for s in ('L', 'R')]
    lateral = q_matrix(tr['body_quat'][:n, ids])[..., :, 1].copy(); lateral[..., 2] = 0
    lateral /= np.maximum(np.linalg.norm(lateral, axis=-1, keepdims=True), 1e-8)
    forward = np.stack((lateral[..., 1], -lateral[..., 0], np.zeros_like(lateral[..., 0])), axis=-1)
    anchor_sum, noncontact_fn, plane = anchor_diagnostics(raw, n)
    sum_rho = np.full(fn.shape, np.nan)
    np.divide(anchor_sum, mu_d*fn, out=sum_rho, where=ratios['valid'])
    t = np.arange(n)*dt; persistent = interval_contact(touch)
    brake = (t >= cfg['brake_start_s']-1e-9) & (t+dt <= cfg['brake_end_s']+1e-9)
    interval_lateral = abs((vector['slip_velocity_interval'][1:n+1]*lateral).sum(-1))
    pre_lateral = contact['lateral_rms_m_s'][1:n+1]
    kinematic = (pre_lateral > .05) & (interval_lateral > .05)
    near_d = (ratios['rho_dynamic'] >= .9) & (ratios['rho_dynamic'] <= 1.05)
    near_s = (ratios['rho_static'] >= .9) & (ratios['rho_static'] <= 1.05)
    power = vector['anchor_power_pre'][1:n+1]
    dissipative = (power < -.05) & (vector['opposing_lateral_anchors_pre'][1:n+1] >= 1)
    # Deliberately distinct from frozen task gate (no beta/tilt gate here).
    # Both-wheel contact at both interval endpoints strengthens this diagnostic.
    corroborated = persistent.all(-1)[:, None] & kinematic & near_d & dissipative
    signals = dict(time_start_s=t, time_end_s=t+dt, touching_states=touch,
        native_gap_states_m=contact['min_separation_m'][1:n+2], normal_force_N=fn,
        tangent_force_N=np.linalg.norm(ft[..., :2], axis=-1),
        force_longitudinal_N=(ft*forward).sum(-1), force_lateral_N=(ft*lateral).sum(-1),
        anchor_force_norm_sum_N=anchor_sum, anchor_sum_rho_dynamic=sum_rho,
        normal_force_above_contact_tolerance_N=noncontact_fn,
        rho_dynamic=ratios['rho_dynamic'], rho_static=ratios['rho_static'], ratio_valid=ratios['valid'],
        pre_lateral_speed_m_s=pre_lateral, interval_lateral_speed_m_s=interval_lateral,
        anchor_power_pre_W=power, kinematic_lateral_sliding=kinematic,
        near_dynamic=near_d, near_static=near_s, dissipative=dissipative,
        contact_at_both_endpoints=persistent, corroborated_sliding=corroborated, braking=brake)
    result = dict(cell=cell, label=source['label'], mu_static=mu_s, mu_dynamic=mu_d,
        qualification_sha256=sha(directory.parent/'nominal_qualification.json'),
        source_manifest_sha256=sha(bundle/'manifest.json'), raw_sha256=run['artifacts'],
        native_force_audit=force_audit, flat_plane_audit=plane,
        task_success=q['task_success_claimed'], failed_gates=q['failed_gates'],
        bilateral_interruption_ms=gate['bilateral_interruption']['total_s']*1000,
        original_qualification_slide_total_s=q['corroborated_slide_total_s'], wheels=[], loss_events=[],
        force_consistency_anomalies=[])
    for side in range(2):
        valid = brake & ratios['valid'][:, side]
        end_loss = touch[:-1, side] & ~touch[1:, side]
        x = dict(wheel=('L','R')[side], braking_intervals=int(brake.sum()),
            contact_intervals=int((brake & touch[:-1, side]).sum()), ratio_valid_intervals=int(valid.sum()),
            contact_zero_normal_intervals=int((brake & touch[:-1, side] & (fn[:, side] <= 1e-6)).sum()),
            low_force_ratio_intervals_le_1N=int((valid & (fn[:, side] <= 1)).sum()),
            near_dynamic_contact_s=float((brake & near_d[:, side]).sum()*dt),
            near_static_contact_s=float((brake & near_s[:, side]).sum()*dt),
            bilateral_endpoint_corroborated_slide_s=float((brake & corroborated[:, side]).sum()*dt),
            near_dynamic_intervals_ending_contact_loss=int((brake & near_d[:, side] & end_loss).sum()),
            near_dynamic_intervals_retaining_contact=int((brake & near_d[:, side] & persistent[:, side]).sum()),
            ratio_over_1p05_intervals=int((valid & (ratios['rho_dynamic'][:, side] > 1.05)).sum()),
            static_ratio_over_1p05_intervals=int((valid & (ratios['rho_static'][:, side] > 1.05)).sum()),
            zero_normal_nonzero_tangent_intervals=int((brake & touch[:-1,side] & (fn[:,side] <= 1e-6)
                & (signals['tangent_force_N'][:,side] > .001)).sum()),
            near_dynamic_and_kinematic_sliding_s=float((brake & near_d[:, side] & kinematic[:, side]).sum()*dt),
            force_from_positive_separation_ratio=finite_stats(np.divide(noncontact_fn[valid, side], fn[valid, side])))
        for key in ('normal_force_N','tangent_force_N','rho_dynamic','rho_static','anchor_sum_rho_dynamic',
                    'pre_lateral_speed_m_s','interval_lateral_speed_m_s'):
            x[key] = finite_stats(signals[key][valid, side])
        result['wheels'].append(x)
        over_static=valid & (ratios['rho_static'][:,side] > 1.05)
        zero_normal=brake & touch[:-1,side] & (fn[:,side] <= 1e-6) & (signals['tangent_force_N'][:,side] > .001)
        for j in np.flatnonzero(over_static | zero_normal):
            result['force_consistency_anomalies'].append(dict(wheel=('L','R')[side],time_start_s=float(t[j]),
                kind='static_bound_exceedance' if over_static[j] else 'nonzero_friction_with_zero_normal',
                Fn_N=float(fn[j,side]), Ft_N=float(signals['tangent_force_N'][j,side]),
                rho_dynamic=float(ratios['rho_dynamic'][j,side]) if over_static[j] else None,
                rho_static=float(ratios['rho_static'][j,side]) if over_static[j] else None,
                normal_points=int(raw[f'w{side}_normal_buffer_4'][j].item()),
                friction_anchors=int(raw[f'w{side}_friction_buffer_2'][j].item()),
                contact_at_both_endpoints=bool(persistent[j,side])))
    for event in loss_windows(touch, dt):
        side, i, j = event['side'], event['state_index'], event['preceding_interval']
        before = np.arange(event['window_start'], i)
        usable = before[ratios['valid'][before, side]]
        early = before[before < i-round(.01/dt)]
        early = early[ratios['valid'][early, side]]
        def number(value):
            return float(value) if np.isfinite(value) else None
        row = dict(wheel=('L','R')[side], onset_s=i*dt, preceding_interval_start_s=j*dt,
            preceding_interval_ends_at_loss=True, last_contact_normal_N=float(fn[j, side]),
            last_contact_tangent_N=float(signals['tangent_force_N'][j, side]),
            last_contact_rho_dynamic=number(ratios['rho_dynamic'][j, side]),
            last_contact_rho_static=number(ratios['rho_static'][j, side]),
            last_contact_pre_lateral_speed_m_s=number(pre_lateral[j, side]),
            last_contact_interval_lateral_speed_m_s=number(interval_lateral[j, side]),
            last_contact_anchor_power_W=float(power[j, side]),
            preloss_50ms_ratio_valid_intervals=len(usable),
            preloss_50ms_near_dynamic_intervals=int(near_d[usable, side].sum()),
            earlier_50_to_10ms_normal_N=finite_stats(fn[early, side]),
            earlier_50_to_10ms_tangent_N=finite_stats(signals['tangent_force_N'][early, side]),
            previous_gap_mm=number(signals['native_gap_states_m'][i-1, side]*1000),
            loss_gap_mm=number(signals['native_gap_states_m'][i, side]*1000))
        # The final geometric-contact interval can already have zero impulse.
        # Identify the last positive-normal contact interval explicitly instead
        # of assigning rho=0 or silently shifting the timestamp.
        positive=np.flatnonzero(ratios['valid'][:i,side])
        if len(positive):
            k=int(positive[-1])
            row.update(last_positive_interval_index=k,last_positive_interval_start_s=k*dt,
                last_positive_interval_start_to_loss_ms=(i-k)*dt*1000,
                last_positive_interval_end_to_loss_ms=(i-k-1)*dt*1000,
                last_positive_normal_N=float(fn[k,side]),
                last_positive_tangent_N=float(signals['tangent_force_N'][k,side]),
                last_positive_rho_dynamic=float(ratios['rho_dynamic'][k,side]),
                last_positive_rho_static=float(ratios['rho_static'][k,side]),
                last_positive_pre_lateral_speed_m_s=number(pre_lateral[k,side]),
                last_positive_interval_lateral_speed_m_s=number(interval_lateral[k,side]),
                last_positive_dissipative=bool(dissipative[k,side]))
        else:
            raise ValueError('Contact loss has no preceding positive-normal contact interval')
        result['loss_events'].append(row)
    return result, signals


def figures(results, arrays, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#2275ac', '#de7626')
    for row, s in zip(results, arrays):
        fig, axes = plt.subplots(1, 4, figsize=(23, 4.8), constrained_layout=True)
        t = s['time_start_s']; dt = t[1]-t[0]
        # First geometric interruption, selected without inspecting forces.
        first = min(x['onset_s'] for x in row['loss_events'])
        limits = (first-.1, first+.1)
        for side, color in enumerate(colors):
            label = ('L','R')[side]
            axes[0].plot(t, s['normal_force_N'][:,side], color=color, label=label+' Fn')
            axes[0].plot(t, s['tangent_force_N'][:,side], color=color, ls='--', label=label+' |Ft|')
            axes[0].plot(t, row['mu_dynamic']*s['normal_force_N'][:,side], color=color, ls=':', label=label+' mu_d Fn')
            axes[1].plot(t, s['rho_dynamic'][:,side], color=color, label=label+' dynamic')
            if row['mu_static'] != row['mu_dynamic']:
                axes[1].plot(t, s['rho_static'][:,side], color=color, ls=':', label=label+' static')
            axes[2].plot(t, s['pre_lateral_speed_m_s'][:,side], color=color, label=label+' pre')
            axes[2].plot(t, s['interval_lateral_speed_m_s'][:,side], color=color, ls='--', label=label+' interval')
            mark = s['corroborated_sliding'][:,side] & s['braking']
            axes[2].scatter(t[mark], s['interval_lateral_speed_m_s'][mark,side], color=color, s=13)
            axes[3].plot(np.r_[t,t[-1]+dt], s['native_gap_states_m'][:,side]*1000, color=color, label=label)
        for ax in axes:
            ax.set_xlim(limits); ax.axvline(first, color='crimson', lw=1, label='First contact loss')
            ax.grid(alpha=.2); ax.set_xlabel('Interval start / geometry time [s]'); ax.legend(fontsize=7)
        axes[0].set(title='Force and changing capacity', ylabel='Force [N]')
        axes[1].axhspan(.9,1.05,color='gray',alpha=.13)
        inside=(t>=limits[0])&(t<=limits[1])
        top=max(1.4,float(np.nanmax(s['rho_dynamic'][inside]))*1.1)
        axes[1].set(title='Wheel-resultant utilization', ylabel='rho = |sum Ft| / (mu sum Fn)', ylim=(0,top))
        axes[2].axhline(.05,color='gray',ls=':',lw=.8)
        axes[2].set(title='Material lateral slip', ylabel='Speed [m/s]')
        axes[3].axhline(.1,color='black',ls='--',lw=.8)
        axes[3].set(title='Native geometric contact', ylabel='Separation [mm]')
        fig.suptitle(row['label']+' | First-loss window | Solid/dashed force: normal/tangent; slip dots: corroborated\n'
            'Forces average the following 2.5 ms; rho is undefined without contact or positive Fn; no pointwise cone certificate')
        fig.savefig(output/(row['cell']+'_first_loss_4x1.png'),dpi=140); plt.close(fig)
    # Normalized force-plane sections, one 4-panel matched comparison per model.
    plane_limit=max(1.4,max(float(np.nanmax(s['rho_dynamic'][s['braking']])) for s in arrays)*1.08)
    for model in range(3):
        fig, axes = plt.subplots(1,4,figsize=(21,5.3),constrained_layout=True)
        angle = np.linspace(0,2*np.pi,361)
        for col,(index,side) in enumerate(((model*2,0),(model*2,1),(model*2+1,0),(model*2+1,1))):
            row, s = results[index], arrays[index]; ax = axes[col]
            m = s['braking'] & s['ratio_valid'][:,side]
            capacity = row['mu_dynamic']*s['normal_force_N'][m,side]
            x = s['force_longitudinal_N'][m,side]/capacity; y = s['force_lateral_N'][m,side]/capacity
            ax.plot(np.cos(angle),np.sin(angle),'k-',lw=.8,label='Dynamic analytical boundary')
            ratio = row['mu_static']/row['mu_dynamic']
            if ratio != 1:
                ax.plot(ratio*np.cos(angle),ratio*np.sin(angle),'k:',lw=.8,label='Static analytical boundary')
            sc = ax.scatter(x,y,c=s['time_start_s'][m],cmap='viridis',vmin=.4,vmax=1.8,s=7,alpha=.7)
            for event in row['loss_events']:
                if event['wheel'] == ('L','R')[side]:
                    j = event['last_positive_interval_index']
                    if s['braking'][j] and s['ratio_valid'][j,side]:
                        cap = row['mu_dynamic']*s['normal_force_N'][j,side]
                        ax.scatter(s['force_longitudinal_N'][j,side]/cap,s['force_lateral_N'][j,side]/cap,
                                   color='crimson',marker='x',s=50,zorder=5)
            ax.set(title=row['label']+' / '+('L','R')[side],xlabel='F_long / (mu_d Fn)',ylabel='F_lat / (mu_d Fn)',
                   xlim=(-plane_limit,plane_limit),ylim=(-plane_limit,plane_limit),aspect='equal')
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.colorbar(sc,ax=axes,shrink=.7,label='Braking time [s]')
        fig.suptitle('Wheel-resultant friction sections | red x: last positive-normal interval before contact loss\n'
                     'Identical axes, all outliers shown | Static-bound exceedances require a force/solver audit, not clipping')
        fig.savefig(output/f'model{model}_cone_sections_4x1.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(23,5.4),constrained_layout=True)
    labels=['Equal\nv6','Equal\nv7','Unequal\nv6','Unequal\nv7','Slices\nv6','Slices\nv7']
    x=np.arange(len(results))
    axes[0].bar(x,[r['bilateral_interruption_ms'] for r in results],color=['#8195a3','#3e6682']*3)
    axes[0].set(title='Bilateral contact interruption',ylabel='Time [ms]')
    for side,color in enumerate(colors):
        for ax,key in ((axes[1],'near_dynamic_contact_s'),(axes[2],'bilateral_endpoint_corroborated_slide_s'),
                       (axes[3],'static_ratio_over_1p05_intervals')):
            vals=[r['wheels'][side][key] for r in results]
            ax.bar(x+(.18 if side else -.18),vals,width=.36,color=color,label=('L','R')[side])
    axes[1].set(title='Near dynamic capacity while touching',ylabel='Time [s], braking 0.4–1.8 s')
    axes[2].set(title='Sliding with friction corroboration',ylabel='Time [s], bilateral endpoints')
    axes[3].set(title='Static-bound inconsistencies',ylabel='Intervals with rho_static > 1.05')
    for ax in axes:
        ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2)
    for ax in axes[1:]:ax.legend()
    fig.suptitle('Six archived nominal screens | Physics and task gates unchanged | All six fail continuous bilateral contact\n'
                 'Capacity/sliding durations are per wheel; force-consistency outliers remain unresolved and are not counted as corroborated sliding')
    fig.savefig(output/'summary_4x1.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cases-json','reference','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sources=json.loads(a.cases_json.read_text())
    if len(sources)!=6:
        raise ValueError('Require six matched v6/v7 nominal recordings')
    expected=[model+'_smooth_stop_xyz_huber'+suffix+':nominal'
              for model in ('native_equal','native_split','slices9_split') for suffix in ('','_continuity')]
    if [s['case'] for s in sources] != expected:
        raise ValueError('Expected v6/v7 pairs ordered by equal cylinder, unequal cylinder, nine slices')
    results=[]; arrays=[]
    for source in sources:
        result, signals=read_case(source,a.cases_json,a.reference)
        results.append(result); arrays.append(signals)
    report=dict(kind='contact_friction_cone_relation_v1', source_sha256=sha(Path(__file__)),
        selection_sha256=sha(a.cases_json), reference_sha256=sha(a.reference/'motion.npz'),
        dependencies={name:sha(Path(__file__).with_name(name)) for name in
            ('analyze_hockey_ground_contact.py','audit_hockey_contact_liftoff.py','audit_hockey_contact_timing.py',
             'evaluate_hockey_physx.py','audit_physx_contact_float32.py','physx_reference_math.py','wheel_friction_capacity.py')},
        physics_steps=0, training_changed=False, task_gates_changed=False, pointwise_cone_certified=False,
        actual_solver_stick_slip_flag_observed=False, sample_dt_s=.0025,
        contact_tolerance_m=.0001, normal_force_contact_threshold_N=None, ratio_division_epsilon_N=1e-6,
        near_capacity_range=[.9,1.05], slip_threshold_m_s=.05,
        scope='Six nominal seed42 screens, not independent trials or a causal experiment',
        timing='Geometry/pre velocity at t; impulse/dt over [t,t+dt]; same material-point displacement over that interval',
        corroboration='Both wheels touching at both endpoints; pre RMS and interval lateral speed >0.05; rho_d 0.9..1.05; opposing lateral anchor and pre-velocity power <-0.05 W. Diagnostic, distinct from frozen beta/tilt task gate.',
        limitations=['Wheel totals do not recover patch/anchor normal-load allocations.',
            'Impulse/dt times pre velocity is a dissipative proxy, not integrated physical work.',
            'A missing contact makes rho/sliding undefined, not zero.',
            'An interval ending at first loss can contain unloading; its force is not an instantaneous force before loss.',
            'No solver friction flags are inferred from static/dynamic ratios.',
            'If all tangential/normal impulses were paired consistently and individually bounded, their aggregate would obey the static bound too. Missing anchor correspondence alone does not explain an aggregate exceedance.',
            'Static-bound exceedances are unresolved readback/solver-consistency evidence, not proof of physical super-Coulomb friction.',
            'Adjacent samples and repeated events are correlated. No causal or statistical generalization is claimed.'],
        official_sources=['https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#friction-and-restitution',
            'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html'],
        cases=results)
    events=[dict(cell=r['cell'],**e) for r in results for e in r['loss_events']]
    report['event_summary']=dict(contact_loss_events=len(events),
        final_interval_zero_normal_and_tangent=sum(e['last_contact_normal_N']<=1e-6 and e['last_contact_tangent_N']<=1e-6 for e in events),
        last_positive_interval_near_dynamic_capacity=sum(.9<=e['last_positive_rho_dynamic']<=1.05 for e in events),
        last_positive_rho_dynamic=finite_stats([e['last_positive_rho_dynamic'] for e in events]),
        last_positive_interval_start_to_loss_ms=finite_stats([e['last_positive_interval_start_to_loss_ms'] for e in events]),
        last_positive_interval_end_to_loss_ms=finite_stats([e['last_positive_interval_end_to_loss_ms'] for e in events]))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'signals.npz',**{f'case{i}_{k}':v for i,s in enumerate(arrays) for k,v in s.items()})
    with (a.output/'loss_events.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(events[0]),lineterminator='\n');writer.writeheader();writer.writerows(events)
    figures(results,arrays,a.output)
    report['artifact_sha256']={p.name:sha(p) for p in sorted(a.output.iterdir())}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(label=r['label'],gap_ms=r['bilateral_interruption_ms'],events=len(r['loss_events']),
        slide_s=[w['bilateral_endpoint_corroborated_slide_s'] for w in r['wheels']]) for r in results]))


if __name__=='__main__':
    main()
