"""Compare body beta with estimated loaded-wheel surface sliding in six traces.

Read-only diagnostic of the historical frozen K actor, not the new fresh K/M.
Uses reported aggregate contact points and post-solve rigid-body velocities;
this does not resolve individual patches or motion within a physical timestep.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

import analyze_physx_tron1_wheel_models as native
from nonlinear_tire_contact import wheel_frame


def calculate(tire):
    q = tire["wheel_quaternion_wxyz"]
    w, x, y, z = np.moveaxis(q, -1, 0)
    axle = np.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), axis=-1)
    forward, lateral, _ = wheel_frame(axle, [0., 0., 1.])
    velocity = tire["COM_velocity_w_m_s"].astype(float)
    # Runtime subtracts native point/COM buffers before the force-law helper
    # promotes its arguments to float64. Preserve that exact operation order.
    offset = (tire["preceding_contact_position_w_m"]-tire["COM_position_w_m"]).astype(float)
    point_velocity = velocity+np.cross(tire["angular_velocity_w_rad_s"].astype(float), offset)
    side = np.sum(point_velocity*lateral, axis=-1)
    longitudinal = np.sum(point_velocity*forward, axis=-1)
    hub_longitudinal = np.sum(velocity*forward, axis=-1)
    denominator = np.maximum.reduce((abs(hub_longitudinal), abs(hub_longitudinal-longitudinal),
                                    np.full_like(side, .2)))
    normalized = np.stack((longitudinal, side), axis=-1)/denominator[..., None]
    load = tire["preceding_normal_load_n"]
    # Independent kinematic reconstruction must agree with the native runtime's
    # cached slip diagnostics before any statistics are accepted.
    active = load > 0.
    np.testing.assert_allclose(normalized[active], tire["normalized_contact_slip"][active], rtol=1e-10, atol=1e-10)
    t = tire["time_before_step_s"]
    core = (t >= 1.44-1e-9) & (t <= 4.+1e-9)
    finite = np.isfinite(point_velocity).all(-1)
    loaded = core[:, None] & (load > 5.) & finite
    moving = loaded & (np.linalg.norm(velocity[..., :2], axis=-1) >= .3)
    rows = {}
    for i, name in enumerate(("left", "right")):
        selection = moving[:, i]
        values = abs(side[selection, i])
        assert len(values) > 0
        rows[name] = dict(core_loaded_samples=int(loaded[:, i].sum()), loaded_moving_samples=int(selection.sum()),
            abs_lateral_contact_speed_m_s_median=float(np.median(values)),
            abs_lateral_contact_speed_m_s_p95=float(np.quantile(values, .95)),
            loaded_moving_fraction_abs_lateral_speed_above_0p1_m_s=float(np.mean(values > .1)))
    return rows


def run(args):
    root, output = args.result, args.output_dir
    assert not output.exists()
    published = json.loads((root/"analysis.json").read_text())
    rows, _, _, contacts, _ = native.load_cases(root/"evidence",
        native.base.metrics.reference_data(root/"reference.npz"), json.loads((root/"protocol.json").read_text()),
        original_metrics=published["original_baseline_metrics"])
    assert rows == published["rows"]
    result_rows = []
    for row in rows:
        result_rows.append(dict(case=row["id"], geometry=row["geometry"], law=row["law"],
            core_body_beta_rmse_deg=row["metrics"]["windows"]["core"]["tracking"]["beta_rmse_deg"],
            wheels=calculate(contacts[row["id"]]), input_sha256=row["input_sha256"]))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    summary = dict(kind="historical_frozen_K_loaded_surface_slip_diagnostic", actor_sha256=native.ACTOR,
        new_fresh_K_M_policies_evaluated=False, physics_jobs_launched=0, task_success_claimed=False,
        new_failure_or_acceptance_gate=False, evaluation_mu=.7, core_interval_s=[1.44, 4.],
        load_threshold_strict_n=5., selected_hub_XY_speed_min_m_s=.3,
        descriptive_slip_speed_threshold_m_s=.1, analyzer_sha256=sha(__file__),
        frozen_native_analysis_sha256=sha(root/"analysis.json"),
        all_six_existing_native_cases_reverified=True, cached_contact_slip_reconstructed=True,
        limitations=["Aggregate reported contact point, not per-patch or per-slice identification.",
            "Uses instantaneous post-solve body velocities at each physical sample; within-step slip is not resolved.",
            "Normal load is the preceding solver sample, paired with the recorded pre-next-step kinematics.",
            "The0.1m/s cutoff and0.3m/s hub-speed selection describe these data and are not task acceptance criteria.",
            "These are frozen historical K model transfers, not this round's fresh K/M policies.",
            "Beta quality alone cannot establish sustained loaded-wheel sliding; these numbers alone do not classify a successful drift."],
        rows=result_rows)
    output.mkdir(parents=True)
    (output/"analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    table = []
    for row in result_rows:
        left, right = row["wheels"]["left"], row["wheels"]["right"]
        values = [row["case"], f'{row["core_body_beta_rmse_deg"]:.2f}',
            f'{left["abs_lateral_contact_speed_m_s_median"]:.4f} / {right["abs_lateral_contact_speed_m_s_median"]:.4f}',
            f'{100*left["loaded_moving_fraction_abs_lateral_speed_above_0p1_m_s"]:.1f} / {100*right["loaded_moving_fraction_abs_lateral_speed_above_0p1_m_s"]:.1f}']
        table.append("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in values)+"</tr>")
    report='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Body beta versus loaded-wheel surface slip</title>
<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:auto;padding:24px}td,th{border:1px solid #ccd4df;padding:8px}table{border-collapse:collapse}</style>
<h1>Body beta does not specify loaded-wheel sliding</h1>
<p>Read-only reconstruction of six existing native PhysX episodes atmu0.7,using the same historical frozen K actor. These are not the new fresh K/M policies. No controller,reward,physics or acceptance criterion changes.</p>
<table><tr><th>Geometry / law</th><th>Core body-beta RMSE(degrees)</th><th>Loaded surface lateral speed median,L/R(m/s)</th><th>Selected loaded samples with |lateral speed|&gt;0.1m/s,L/R(%)</th></tr>TABLE</table>
<p>Selection: core1.44–4.00s,normal load strictly above5N,finite aggregate contact point,and wheel COM XY speed at least0.3m/s. Speed fractions use those selected per-wheel samples; they are not percentages of all episode time or a new pass/fail gate. The0.1m/s descriptive cutoff is not a sensor-identified stick/slip boundary.</p>
<pre>v_surface = v_wheel_COM + omega_wheel_world cross (reported_contact_point - wheel_COM)
v_lateral = dot(v_surface, wheel_lateral_direction_on_flat_floor)</pre>
<p>Wheel angular velocity includes spin. Ground velocity is zero. The reconstruction matches the runtime's cached normalized longitudinal/lateral slip to1e-10 tolerance. All six source/actor/initial-state and force-application records are reverified against the published analysis before computing these additional metrics.</p>
<p>In the original-cylinder/native case,the selected loaded samples have median absolute lateral surface speeds0.00946/0.00754m/s,while beta RMSE is3.77degrees. Only14.3/8.6% exceed the descriptive0.1m/s cutoff. This shows why body-beta tracking alone is insufficient evidence of sustained wheel sliding. It does not establish that there is no within-step slip: the reported point aggregates patches,and velocity is sampled after the solver. Active leg motion,load alternation and solver impulses all matter.</p>
<p>For a direct sliding task,define desired loaded-wheel lateral surface velocity or wheel-local slip angle explicitly,alongside support,speed,path and body integrity. Wheel speed/torque can control longitudinal slip; lateral motion and balance also require whole-body/yaw dynamics. A force-based nonlinear tire model changes the slip-to-force relation,it does not make a desired slip an instantaneous actuator command. The current fresh K/M comparison stays frozen.</p>
<p><a href="analysis.json">Exact values,counts,hashes and limitations</a> · <a href="../../scripts/diagnose_loaded_wheel_slip.py">Reproduction code</a> · <a href="../physx_tron1_wheel_models_20261025_v1/report.html">Original six-case comparison and animations</a>.</p></html>'''
    (output/"report.html").write_text(report.replace("TABLE", "".join(table)))
    print(json.dumps(result_rows[0], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--result", type=Path, default=Path("results/physx_tron1_wheel_models_20261025_v1"))
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
