"""Publish two completed MPPI attempts without treating tracking as feasibility."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import shutil
import numpy as np
import generate_mppi_contact_reference as g


def run(a):
    if (a.output/'summary.json').exists() or (a.output/'report.html').exists():raise FileExistsError('Preserve completed publication')
    rows=[]
    for case,audit,render,title in [
        ('independent_v1','independent_audit_v2','independent_render_v2','Independent MPPI'),
        ('smooth_v1','smooth_audit_v1','smooth_render_v1','MPPI with predicted command smoothing')]:
        source=a.runs/case;target=a.output/audit
        d=json.loads((target/'analysis.json').read_text());screen=json.loads((target/'sliding_screen.json').read_text())
        r=json.loads((source/'report.json').read_text());solver=json.loads((source/'solver_log.json').read_text())
        if g.sha(source/'rollout.npz')!=d['rollout_sha256'] or g.sha(source/'report.json')!=d['run_report_sha256']:raise ValueError('Evidence hashes differ')
        evidence=target/'evidence';evidence.mkdir()
        for name in ('report.json','rollout.npz','solver_log.json','candidate_motion.npz'):shutil.copyfile(source/name,evidence/name)
        with np.load(source/'rollout.npz') as f:raw={k:f[k].copy() for k in f.files}
        tilt=np.rad2deg(np.arccos(g.quaternion_up_z(raw['root_quaternion_wxyz']).clip(-1,1)))
        def first(mask):
            index=np.flatnonzero(mask);return None if not len(index) else float(raw['time_s'][index[0]])
        ess=[i['ess'] for s in solver for i in s['mppi_iterations'] if 'ess' in i]
        core=d['windows']['core']['tracking'];whole=d['windows']['whole']['tracking'];sup=d['support']
        row=dict(id=case,title=title,core_XY_RMSE_m=core['xy_rmse_m'],core_XY_peak_m=core['xy_peak_m'],whole_XY_RMSE_m=whole['xy_rmse_m'],
            core_speed_retention=core['speed_retention'],core_beta_RMSE_deg=core['beta_rmse_deg'],HF_ratio_to_original_same_engine_PPO=d['core_HF_ratio_to_original_same_engine_PPO'],
            any_wheel_support_sample_fraction=sup['any_wheel_support_sample_fraction'],longest_no_support_s=sup['longest_no_support_observed_s'],
            maximum_root_tilt_deg=float(tilt.max()),first_tilt_gt45_s=first(tilt>45),first_nonwheel_floor_force_gt5N_s=first(raw['nonwheel_ground_contact_force_norm_n']>5),
            first_tilt_is_not_automatic_fall_time=True,nine_numeric_checks=d['checks'],nine_numeric_checks_pass=d['nine_numeric_checks_pass'],
            sustained_lateral_sliding_observed=screen['sustained_lateral_sliding_observed'],slip_screen=screen['rows'],
            sliding_duration_threshold_s=.05,lateral_slip_threshold_m_s=.05,independent_L_R=True,
            candidate_motion_sha256=g.sha(source/'candidate_motion.npz'),learned_policy_used=False,CEM_seed_used=False,
            controller_fallbacks=sum(int(s['fallback_used']) for s in solver),candidate_evaluations=sum(s['candidate_evaluations'] for s in solver),
            failed_candidate_evaluations=sum(s['failed_candidates'] for s in solver),ESS_p05_p50_p95=np.quantile(ess,[.05,.5,.95]).tolist(),
            compute_ms_p50_p95_max=np.quantile([s['compute_time_s']*1000 for s in solver],[.5,.95,1]).tolist(),
            wall_s=r['wall_s'],exact_continuous_replay_errors=d['replay_max_abs_error_qpos_qvel_Fz'],
            audit=audit,render=render,artifact_sha256={name:g.sha(evidence/name) for name in ('report.json','rollout.npz','solver_log.json','candidate_motion.npz')})
        rows.append(row)
    material=a.output/'inputs';material.mkdir()
    original=g.parser().parse_args(['--output-dir','unused'])
    shutil.copyfile(original.material_report,material/'original_native_material_report.json')
    # The original model contains absolute mesh paths and is an existing asset,
    # not silently bundled as a new model or changed physical representation.
    inputs={name:dict(path=str(path.relative_to(g.ROOT)),sha256=g.sha(path)) for name,path in
        [('model',original.model),('reference',original.reference),('native_library',original.native_library)]}
    result=dict(kind='two_native_MPPI_reference_generation_attempts',status='completed_both_attempts_no_feasible_reference_certified',
        primary_task='traj4 large turn with original nonzero beta',mu=.7,physics_dt_s=.0025,control_dt_s=.02,
        planner='63 Gaussian samples plus weighted-mean verification,3 iterations,H30/B10;fixed covariance;final mean deployed',
        model='MuJoCo3.8.1 original rigid cylinders; no nonlinear tire, contact force optimization or hidden root forces',
        no_PPO_training_or_inference=True,no_CEM_seed=True,no_new_native_compilation=True,PhysX_validation_performed=False,
        absolute_height_cost=False,wheel_XY_cost=False,source_sha256=g.sha(__file__),inputs=inputs,rows=rows,
        policy_training_reference_approved=False,nominal_task_success_claimed=False,
        limitations=['N=1,seed42; no controller-wide or statistical ranking.',
            'New planning cost is not the PPO reward; do not pool with earlier equal-reward controller comparisons.',
            'The filter is the sole intended difference between these two attempts; plant motor parameters and MPPI budget are unchanged.',
            'Prediction cold-resets each20ms, whereas execution and verification replay integrate continuously.',
            'MPPI importance ESS is near1 for many updates; no well-mixed-search or real-time claim.',
            'Tilt>45deg is a diagnostic, not an automatic fall-time label; whole-body video must also be reviewed.',
            'Contact continuity and slip must hold on the same wheel; airborne samples and geometric tangency are not sliding evidence.',
            'Point normal force>5N is the slip-estimator validity gate; wheel summed Fz>5N is the separate support gate.',
            'Force-cone saturation corroborates friction demand but cannot alone distinguish rolling/static contact from sliding.',
            'Both full-body recordings show loss of posture, so neither candidate should be used as a successful imitation demonstration.'])
    g.write_json(a.output/'summary.json',result)
    table=''.join('<tr>'+''.join('<td>'+v+'</td>' for v in [html.escape(r['title']),f"{r['core_XY_RMSE_m']:.3f}",f"{100*r['core_speed_retention']:.1f}%",f"{r['core_beta_RMSE_deg']:.1f}",f"{r['HF_ratio_to_original_same_engine_PPO']:.3f}",f"{100*r['any_wheel_support_sample_fraction']:.1f}%",f"{1000*r['longest_no_support_s']:.1f}",'FAIL'])+'</tr>' for r in rows)
    media=''.join(f'<h2>{html.escape(r["title"])}</h2><video controls preload="metadata" src="{r["render"]}/mppi_reference_4x1.mp4"></video><p><a href="{r["render"]}/mppi_reference_4x1.mp4">Video</a> · <a href="{r["audit"]}/analysis.json">Audit</a> · <a href="{r["audit"]}/sliding_screen.json">Sliding events</a> · <a href="{r["audit"]}/evidence/candidate_motion.npz">Failed candidate motion NPZ</a></p><img src="{r["render"]}/overview_4x1.png">' for r in rows)
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><title>MPPI contact reference attempts</title>
<style>body{font:17px system-ui;max-width:1500px;margin:35px auto;padding:0 24px;line-height:1.55}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #ccc;padding:9px}video,img{width:100%;height:auto}code,pre{background:#f1f3f5}pre{padding:15px;overflow:auto}.fail{color:#a32323;font-weight:bold}</style>
<h1>MPPI-generated TRON1 turning trajectories</h1><p class="fail">Both searches completed. Neither produced a stable-contact, task-feasible reference.</p>
<p>MuJoCo3.8.1, μ=0.7,400Hz physics and50Hz held commands. Six leg commands and two wheel torques are independent; world wheel camber and heading are measured from the resulting full-body pose. Original car XY,COM speed and nonzero body β are tracking targets. No absolute height or wheel XY target is added. Contact forces are simulated outputs, never free planning variables.</p>
<p>The second arm adds a causal first-order command filter (α=0.25 legs,0.5 wheels) inside both prediction and execution. All search budgets, costs and physical motors are otherwise held fixed. This is a new trajectory-planning cost, not the earlier matched-PPO reward experiment. No PPO actor, CEM action seed, GPU or new native compilation was used.</p>
<table><tr><th>Attempt</th><th>Core XY RMSE [m]</th><th>Core speed retention</th><th>β RMSE [deg]</th><th>Core HF/original PPO</th><th>Any-wheel support</th><th>Longest gap [ms]</th><th>All9 checks</th></tr>'''+table+'''</table>
<p>Core:1.44–4.00s; support:0.02–6.96s,actual upward wheel-floor Fz&gt;5N at every2.5ms sample. A single lifted wheel is allowed. HF means10&lt;f&lt;100Hz world angular XY RMS relative to the original same-engine400Hz PPO. The original task scoring reference is unchanged. Full-body videos show loss of posture in both attempts despite small XY error.</p>
<p>Continuous, open-loop action replay reproduces all recorded qpos,qvel and floor-normal loads exactly (maximum errors0). Each attempt has349 decisions and2793 native states. No state correction or reset splice is applied.</p>
<h2>What counts as sliding?</h2><p>Instantaneous slip is actual material-point velocity relative to the floor projected into its tangent plane: vₜ=(I−nnᵀ)[vCOM+ω×(p−pCOM)−vground]. Lateral slip is its component along the projected wheel axle. Per-wheel RMS over actual loaded contact points avoids sign cancellation. No loaded contact means unobserved slip, not zero-slip rolling.</p>
<p>A sustained lateral-sliding event additionally requires the same wheel to remain force-supported, moving and above0.05m/s lateral RMS continuously for at least50ms. Geometry-point identities may change during rolling. No unsupported gaps are bridged. These are diagnostic thresholds, not universal physical phase boundaries. Neither attempt has such a sustained event in the turn core.</p>
<p>Friction saturation is a separate diagnostic. Circular/elliptic utilization uses sqrt[Σ(Fti/μiFn)²]. This experiment uses MuJoCo condim3 pyramidal contact, whose physical force section instead obeys Σ|Fti|/(μiFn)≤1. Its L1 diamond is inscribed in the elliptic cone; the Euclidean ratio can be below1 on the actual pyramid boundary. Velocity plus sustained contact identifies sliding; proximity to the correct force boundary supports interpretation but does not replace velocity measurement. <a href="https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones">MuJoCo primary documentation</a>.</p>
'''+media+'''<h2>Reproduction</h2><p>Use the hash-bound existing model and MuJoCo3.8.1 native ABI3 library from summary.json. Source MuJoCo/C++ bridge is already in the repository; no implicit build is performed. The published material-report copy contains configuration and reset state only, no model weights or credentials. Fresh output directories are required.</p>
<pre>python scripts/generate_mppi_contact_reference.py --material-report results/mppi_contact_reference_20261025_v1/inputs/original_native_material_report.json --output-dir outputs/mppi_reproduction_plain
python scripts/generate_mppi_smooth_contact_reference.py --config config/mppi_smooth_contact_reference_v1.json --material-report results/mppi_contact_reference_20261025_v1/inputs/original_native_material_report.json --output-dir outputs/mppi_reproduction_smooth
python scripts/audit_mppi_contact_reference.py --run outputs/mppi_reproduction_plain --output-dir outputs/mppi_reproduction_plain_audit
python scripts/screen_mppi_sliding.py --run outputs/mppi_reproduction_plain --audit outputs/mppi_reproduction_plain_audit --output outputs/mppi_reproduction_plain_audit/sliding.json
python scripts/render_mppi_contact_reference.py --run outputs/mppi_reproduction_plain --audit outputs/mppi_reproduction_plain_audit --output-dir outputs/mppi_reproduction_plain_render</pre>
<p>All measurements,source/input hashes,solver logs,full replay recordings and failed candidate references are retained beside this report. Tests cover native first-interval parity,causal filter parity,reference streams,and rejection of false sustained-slip events across gaps or alternating wheels. The cameras keep a fixed1.9m distance and root-relative offset; XY scale is fixed throughout. Neither NPZ is approved as a successful policy-training reference.</p></html>'''
    (a.output/'report.html').write_text(document)
    print(json.dumps([dict(id=r['id'],XY=r['core_XY_RMSE_m'],support=r['any_wheel_support_sample_fraction'],pass_all9=r['nine_numeric_checks_pass']) for r in rows]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
