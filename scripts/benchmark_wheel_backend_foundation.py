"""Policy-free wheel pilot: archived PhysX versus explicitly specified MuJoCo.

This is an engine-response diagnostic, not real-tire identification. The archived
PhysX contact cache and cooked geometry are unavailable. No policy is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

RADIUS = .127
GRAVITY = 9.81
MASS = 1.0800000429153442
INERTIA = [0.004579830449074507, 0.008709659799933434, 0.004579830449074507]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_state():
    return np.array([0., 0., .12701, 1., 0., 0., 0.,
                     np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20)), 0.,
                     0., np.cos(np.deg2rad(20))/RADIUS, 0.])


def xml_model(dt, iterations=100, cone="elliptic", mu=.7):
    if cone not in ("elliptic", "pyramidal") or dt <= 0 or iterations < 1 or mu < 0:
        raise ValueError("Invalid explicit model parameters")
    return f'''<mujoco model="policy_free_axle_wheel">
  <compiler angle="radian"/>
  <option timestep="{dt:.12g}" gravity="0 0 -{GRAVITY}" integrator="Euler"
          solver="Newton" cone="{cone}" iterations="{iterations}" tolerance="1e-10"/>
  <default>
    <joint damping="0" frictionloss="0" armature="0"/>
    <geom condim="{1 if mu == 0 else 3}" friction="{mu} 0 0" margin="0" gap="0"
          solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 .1"/>
    <body name="wheel">
      <inertial pos="0 0 0" mass="{MASS:.17g}" diaginertia="{' '.join(map(str, INERTIA))}"/>
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="z" type="slide" axis="0 0 1"/>
      <joint name="spin" type="hinge" axis="0 1 0"/>
      <geom name="tire" type="cylinder" size="{RADIUS} .025" quat="0.7071067811865476 0.7071067811865476 0 0"/>
    </body>
  </worldbody>
</mujoco>'''


def root_state(data):
    angle = data.qpos[3]
    return np.r_[data.qpos[:3], np.cos(angle/2), 0., np.sin(angle/2), 0.,
                 data.qvel[:3], 0., data.qvel[3], 0.]


def simulate(initial, dt, duration=2., iterations=100, cone="elliptic", mu=.7):
    import mujoco
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (13,) or not np.allclose(initial[3:7], [1, 0, 0, 0]):
        raise ValueError("Fixture starts with an unrotated body")
    if not np.allclose(initial[[10, 12]], 0):
        raise ValueError("X/Z rotations are locked")
    steps = round(duration/dt)
    if not np.isclose(steps*dt, duration, atol=1e-12, rtol=0):
        raise ValueError("Horizon must contain whole physical steps")
    xml = xml_model(dt, iterations, cone, mu)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:3] = initial[:3]
    data.qvel[:] = initial[[7, 8, 9, 11]]
    # mj_step performs forward dynamics; never call mj_forward after a step
    # before reading its contact solution (that would describe a different state).
    states = [root_state(data)]
    forces, counts, distances, rows = [], [], [], []
    tire = model.geom("tire").id
    for k in range(steps):
        pre = states[-1]
        mujoco.mj_step(model, data)
        total = np.zeros(3)
        gaps = []
        for j in range(data.ncon):
            c = data.contact[j]
            if tire not in (c.geom1, c.geom2):
                raise AssertionError("Unexpected contact pair")
            local = np.zeros(6)
            mujoco.mj_contactForce(model, data, j, local)
            world = c.frame.reshape(3, 3).T @ local[:3]
            world *= 1 if c.geom2 == tire else -1
            total += world
            gaps.append(c.dist)
            # Actual solver point, pre-integration material-point velocity.
            velocity = pre[7:10] + np.cross(pre[10:13], c.pos-pre[:3])
            rows.append(np.r_[k, j, c.pos.copy(), c.dist, world, velocity])
        states.append(root_state(data))
        forces.append(total)
        counts.append(data.ncon)
        distances.append(min(gaps) if gaps else np.nan)
    trace = dict(time_s=np.arange(steps+1)*dt, root_state_w=np.asarray(states),
                 interval_force_w_n=np.asarray(forces), contact_count=np.asarray(counts),
                 contact_min_distance_m=np.asarray(distances),
                 contact_records=np.asarray(rows).reshape(-1, 12))
    metadata = dict(engine="MuJoCo CPU", version=mujoco.__version__, dt_s=dt,
                    cone=cone, solver="Newton", integrator="Euler", iterations=iterations,
                    tolerance=model.opt.tolerance, solref=model.geom_solref.tolist(),
                    solimp=model.geom_solimp.tolist(), friction=model.geom_friction.tolist(),
                    condim=model.geom_condim.tolist(),
                    mass_kg=float(model.body_mass[1]), inertia=model.body_inertia[1].tolist(),
                    initial_state=initial.tolist(), hidden_state="fresh MjData, no settling or warm start",
                    contact_record_columns=["interval", "contact", "x", "y", "z", "distance",
                                            "Fx", "Fy", "Fz", "vx_pre", "vy_pre", "vz_pre"],
                    force_timing="contact solution used by interval k, read immediately after mj_step",
                    xml_sha256=sha(xml.encode()))
    return trace, metadata, xml


def load_archived(archive, manifest, hz):
    entries = {r['path']: r for r in json.loads(manifest.read_text())['files']}
    stem = f"single_wheel_fixture_sweep_v2/native_coulomb_{hz}hz/"
    payload = {}
    hashes = {}
    with tarfile.open(archive) as tar:
        for name in ("status.json", "native_trace.npz", "settled_fixture_preflight.json"):
            path = stem+name
            member = tar.getmember(path)
            if not member.isfile():
                raise ValueError("Regular archive member required")
            raw = tar.extractfile(member).read()
            entry = entries[path]
            assert len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], path
            payload[name] = raw
            hashes[path] = sha(raw)
    status = json.loads(payload['status.json'])
    assert status['status'] == 'completed' and status['law'] == 'native_coulomb'
    assert sha(payload['native_trace.npz']) == status['native_trace_sha256']
    with np.load(io.BytesIO(payload['native_trace.npz']), allow_pickle=False) as z:
        raw = {k: z[k].copy() for k in z.files}
    assert np.all(raw['applied_tangent_force_w_n'] == 0)
    preflight = json.loads(payload['settled_fixture_preflight.json'])
    np.testing.assert_allclose(np.diag(np.array(preflight['inertia']).reshape(3, 3)), INERTIA)
    assert np.isclose(status['mass_kg_readback'], MASS)
    # Archive sample k+1 contains the normal impulse for transition k -> k+1.
    force = np.c_[raw['inferred_total_tangent_force_xy_n'], raw['ground_normal_force_w_n'][1:, 2]]
    trace = dict(time_s=raw['time_s'], root_state_w=raw['root_state_w'].astype(float),
                 interval_force_w_n=force.astype(float))
    metadata = dict(engine="PhysX archived", dt_s=status['physics_dt_s'],
                    mu=.7, source_hashes=hashes,
                    force_timing="m delta v_xy / dt plus recorded normal at sample k+1",
                    startup_cache="unknown after settling, pose reset and velocity write",
                    cooked_geometry_readback=False)
    return trace, metadata


def summarize(trace, mu=.7):
    q, t, force = trace['root_state_w'], trace['time_s'], trace['interval_force_w_n']
    dt = float(t[1]-t[0])
    assert len(force) == len(q)-1 and np.isfinite(q).all() and np.isfinite(force).all()
    momentum_force = MASS*np.diff(q[:, 7:10], axis=0)/dt
    momentum_force[:, 2] += MASS*GRAVITY
    residual = momentum_force-force
    gap = q[:, 2]-RADIUS
    geometric_contact = (gap[:-1] <= .0001) & (gap[1:] <= .0001)
    loaded = force[:, 2] > .001
    # Shared analytic bottom-point proxy; not a native friction-anchor claim.
    slip = q[:, 7:9].copy()
    slip[:, 0] -= RADIUS*q[:, 11]
    speed = np.linalg.norm(slip, axis=1)
    coherent = np.sum(slip[:-1]*slip[1:], axis=1) >= np.cos(np.deg2rad(30))*speed[:-1]*speed[1:]
    eligible = geometric_contact & loaded & (speed[:-1] > .05) & (speed[1:] > .05) & coherent
    ratio = np.full(len(force), np.nan)
    if mu > 0:
        ratio[loaded] = np.linalg.norm(force[loaded, :2], axis=1)/(mu*force[loaded, 2])
    below = np.flatnonzero(np.abs(q[:, 8]) <= .05)
    def population(mask):
        r = ratio[mask & np.isfinite(ratio)]
        return dict(intervals=int(mask.sum()), duration_s=float(mask.sum()*dt),
                    ratio_p50=float(np.median(r)) if len(r) else None,
                    ratio_p95=float(np.quantile(r, .95)) if len(r) else None,
                    ratio_max=float(r.max()) if len(r) else None,
                    ratio_over_1p05_intervals=int((r > 1.05).sum()))
    return dict(final_xy_m=q[-1, :2].tolist(), final_velocity_xy_m_s=q[-1, 7:9].tolist(),
                first_abs_vy_below_0p05_s=float(t[below[0]]) if len(below) else None,
                geometric_contact_both_endpoints_fraction=float(geometric_contact.mean()),
                normal_force_active_fraction=float(loaded.mean()),
                bottom_gap_min_max_m=[float(gap.min()), float(gap.max())],
                vertical_momentum_residual_rms_n=float(np.sqrt(np.mean(residual[:, 2]**2))),
                vertical_momentum_residual_peak_n=float(np.abs(residual[:, 2]).max()),
                vertical_momentum_residual_rms_after_20ms_n=float(np.sqrt(np.mean(residual[t[:-1] >= .02, 2]**2))) if t[-1] > .02 else None,
                all_axes_momentum_residual_peak_n=float(np.abs(residual).max()),
                sliding_bottom_proxy=population(eligible),
                sliding_bottom_proxy_after_20ms=population(eligible & (t[:-1] >= .02)))


def trajectory_gap(a, b):
    t = a['time_s']
    b_interp = np.array([np.interp(t, b['time_s'], b['root_state_w'][:, j]) for j in (0, 1, 8)]).T
    difference = a['root_state_w'][:, [0, 1, 8]]-b_interp
    return dict(xy_rmse_m=float(np.sqrt(np.mean(np.sum(difference[:, :2]**2, axis=1)))),
                lateral_velocity_rmse_m_s=float(np.sqrt(np.mean(difference[:, 2]**2))))


def run(a):
    if a.output_dir.exists():
        raise FileExistsError("Fresh output required")
    a.output_dir.mkdir(parents=True)
    traces, metadata, summaries, pairs = {}, {}, {}, []
    for hz in (400, 800):
        px = f"physx_archive_{hz}hz"
        traces[px], metadata[px] = load_archived(a.archive, a.manifest, hz)
        mj = f"mujoco_archive_initial_{hz}hz"
        traces[mj], metadata[mj], xml = simulate(traces[px]['root_state_w'][0], 1/hz)
        (a.output_dir/f'{mj}.xml').write_text(xml)
        pairs.append(dict(physx=px, mujoco=mj, **trajectory_gap(traces[px], traces[mj])))
    # Separate numerical/model sensitivity cohort with exactly one initial state.
    for hz in (200, 400, 800):
        for cone in ('elliptic', 'pyramidal'):
            for iterations in (10, 100):
                key = f'mujoco_canonical_{hz}hz_{cone}_i{iterations}'
                traces[key], metadata[key], xml = simulate(canonical_state(), 1/hz, cone=cone, iterations=iterations)
                (a.output_dir/f'{key}.xml').write_text(xml)
    for key, trace in traces.items():
        summaries[key] = summarize(trace)
        np.savez_compressed(a.output_dir/f'{key}.npz', **trace)
        metadata[key]['trace_sha256'] = sha((a.output_dir/f'{key}.npz').read_bytes())
    sensitivity = []
    for cone in ('elliptic', 'pyramidal'):
        fine = f'mujoco_canonical_800hz_{cone}_i100'
        for hz in (200, 400):
            coarse = f'mujoco_canonical_{hz}hz_{cone}_i100'
            sensitivity.append(dict(kind='time_step', a=coarse, b=fine, **trajectory_gap(traces[coarse], traces[fine])))
        for hz in (200, 400, 800):
            low, high = [f'mujoco_canonical_{hz}hz_{cone}_i{i}' for i in (10, 100)]
            sensitivity.append(dict(kind='iteration_budget', a=low, b=high, **trajectory_gap(traces[low], traces[high])))
    report = dict(status='exploratory_single_wheel_pilot_completed',
        new_mujoco_rollouts=14, reused_physx_rollouts=2, new_physx_rollouts=0,
        archive_sha256=sha(a.archive.read_bytes()), manifest_sha256=sha(a.manifest.read_bytes()),
        source_sha256=sha(Path(__file__).read_bytes()), metadata=metadata, cases=summaries,
        matched_observable_initial_state_engine_pairs=pairs, numerical_sensitivity=sensitivity,
        scope='Unactuated 1.08 kg wheel with free XYZ translation and world Y spin; X/Z rotation locked.',
        limits=[
            'MuJoCo CPU is not MuJoCo-Warp. No new policy training or whole-robot success result.',
            'Same archived observable initial state does not match hidden PhysX contact cache.',
            'PhysX uses a static triangle-mesh floor; MuJoCo an analytic plane. Cooked PhysX wheel geometry is unavailable.',
            'Pinned MuJoCo soft-contact parameters are not identified from measurements or tuned to PhysX.',
            'Archive 400/800 Hz initial heights differ. Only the canonical MuJoCo cohort isolates timestep changes.',
            'PhysX tangential force is inferred from COM momentum, not friction-anchor readback; initial anomalies require native recollection.',
            'Resultant force and geometric bottom slip are diagnostics, not a per-patch maximum-dissipation certificate.',
            '20 ms exclusion is reported alongside full data, not used to erase initial transient failures.',
            'No real-tire accuracy, sim2real gap, or superiority of either model is established.' ])
    (a.output_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), dpi=150)
    for key in list(traces)[:4]:
        tr = traces[key]; q = tr['root_state_w']; t = tr['time_s']; f = tr['interval_force_w_n']
        color = 'tab:blue' if key.startswith('physx') else 'tab:orange'
        kw = dict(color=color, ls='-' if '400hz' in key else '--', label=key.replace('_', ' '))
        axes[0].plot(q[:, 0], q[:, 1], **kw)
        axes[1].plot(t, q[:, 8], **kw)
        axes[2].plot(t[:-1], f[:, 1], **kw)
        axes[3].plot(t[:-1], f[:, 2], **kw)
    axes[0].set(title='2 s unactuated trajectory', xlabel='X (m)', ylabel='Y (m)')
    # Do not use equal aspect here: this diagnostic magnifies small lateral travel.
    axes[1].set(title='Lateral velocity', xlabel='Time (s)', ylabel='Vy (m/s)', xlim=(0, .15))
    axes[2].set(title='Interval lateral force', xlabel='Time (s)', ylabel='Fy (N)', xlim=(0, .15))
    axes[3].set(title='Interval normal force', xlabel='Time (s)', ylabel='Fn (N)', xlim=(0, .15))
    axes[1].legend(fontsize=6)
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle('Policy-free wheel pilot | mu = 0.7 | PhysX archive vs MuJoCo CPU | no real-tire accuracy claim')
    fig.tight_layout(); fig.savefig(a.output_dir/'wheel_pilot_4x1.png'); plt.close(fig)
    print(json.dumps(dict(output=str(a.output_dir), engine_pairs=pairs, cases={k:summaries[k] for k in list(traces)[:4]}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
