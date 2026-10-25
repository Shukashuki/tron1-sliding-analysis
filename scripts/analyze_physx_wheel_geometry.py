"""Recompute a six-case single-wheel contact diagnostic from native traces.

No robot task, policy, or identified real tire is implied. Contact positions
and friction vectors aggregate each body's patches, not individual slices.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from nonlinear_tire_contact import TireParameters, tangent_wrench


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze_case(folder, *, source_rows):
    report = json.loads((folder/"status.json").read_text())
    assert report["kind"] == "PhysX_single_wheel_uniform_slice_geometry_force_factorial"
    assert report["status"] == "completed" and report["physical_steps"] == 800 and report["native_states"] == 801
    assert report["parameters"] == asdict(TireParameters()) and report["physics_dt_s"] == .0025
    assert report["native_trace_sha256"] == sha(folder/"native_trace.npz")
    for name, digest in report["source_sha256"].items():
        assert source_rows["scripts/"+Path(name).name]["sha256"] == digest
    with np.load(folder/"native_trace.npz", allow_pickle=False) as archive:
        raw = {key: archive[key].copy() for key in archive.files}
    state, normal, point = (raw[k] for k in ("root_state_w", "ground_normal_force_w_n", "contact_position_w_m"))
    assert state.shape == (801,13) and normal.shape == point.shape == (801,3)
    assert np.isfinite(state).all() and np.isfinite(normal).all() and np.min(normal[:,2]) >= -1e-4
    np.testing.assert_allclose(raw["time_s"], np.arange(801)*.0025, rtol=0, atol=1e-12)
    force, moment = raw["applied_tangent_force_w_n"], raw["applied_tangent_moment_w_nm"]
    native = raw["native_sum_patch_tangent_force_w_n"]
    assert force.shape == moment.shape == (800,3) and native.shape == (801,3)
    assert np.isfinite(force).all() and np.isfinite(moment).all() and np.isfinite(native).all()
    geometry = report["compound_geometry"]
    assert geometry["approximation_mode"] == "smooth" and geometry["slices"] in (1,9)
    assert geometry["original_collision_API_removed"] and geometry["rigid_bodies"] == 1
    parser = json.loads((folder/"parser_audit.json").read_text())
    assert parser["body_max_shapes"] == geometry["slices"]
    assert geometry["mass_kg"] == 1.08 and geometry["radius_m"] == .127 and geometry["width_m"] == .05
    np.testing.assert_allclose(geometry["runtime_inertia_kg_m2"],
        np.diag(geometry["diagonal_inertia_kg_m2"]), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(geometry["runtime_COM_and_principal_frame"][:3], 0., rtol=0, atol=1e-8)
    mass = report["mass_kg_readback"]
    assert abs(mass-1.08) < 1e-5
    fn = np.maximum(normal[:,2],0)
    np.testing.assert_allclose(raw["preceding_sample_load_n"], fn[:-1], rtol=0, atol=1e-8)
    loaded = fn > 1e-3
    assert np.isfinite(point[loaded]).all()
    q = state[:,3:7]; w,x,y,z = np.moveaxis(q,-1,0)
    axle = np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
    slip = np.full((801,3), np.nan)
    slip[loaded] = state[loaded,7:10] + np.cross(state[loaded,10:13], point[loaded]-state[loaded,:3])
    power = np.zeros(800)
    law = report["law"]
    active = np.flatnonzero(loaded[:-1])
    if law == "native_coulomb":
        np.testing.assert_array_equal(force, np.zeros((800,3)))
        np.testing.assert_array_equal(moment, np.zeros((800,3)))
        assert report["external_normal_load_delay_physical_steps"] == 0
    else:
        assert law in ("external_coulomb_regularized", "combined_slip_magic_formula")
        assert report["external_normal_load_delay_physical_steps"] == 1
        recomputed = tangent_wrench(state[active,7:10], state[active,10:13],
            point[active]-state[active,:3], axle[active], [0,0,1], fn[active], law=law)
        np.testing.assert_allclose(force[active], recomputed["force_w_n"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(moment[active], recomputed["torque_about_COM_w_nm"], rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(force[~loaded[:-1]], 0.)
        power[active] = recomputed["contact_power_w"]
        assert np.max(power) <= 1e-10
        assert np.max(np.linalg.norm(native,axis=-1)) < 1e-3, "Native tangent force was not disabled"
    # Translational axes are unactuated and unlocked. This checks applied-wrench
    # coupling against actual acceleration, separately from the tire equation.
    inferred = mass*np.diff(state[:,7:9],axis=0)/.0025
    # The native normal-force vector has measurable XY components, even on
    # this authored flat triangle floor. Omitting it falsely looks like an
    # external-wrench coupling error. Keep the whole measured normal vector.
    omitted_normal_balance = inferred - native[1:,:2] - force[:,:2]
    balance = omitted_normal_balance - normal[1:,:2]
    assert np.max(np.abs(balance)) < 1e-3, "Horizontal impulse balance does not reproduce"
    speed = np.linalg.norm(state[:,7:9],axis=-1)
    valid_angle = speed >= .2
    hub_angle = np.full(801,np.nan)
    # World Z rotation is fixture-locked at zero; spin Y does not steer the axle.
    hub_angle[valid_angle] = np.rad2deg(np.arctan2(state[valid_angle,8],np.abs(state[valid_angle,7])))
    utilization = np.full(801,np.nan)
    utilization[loaded] = np.linalg.norm(native[loaded,:2],axis=-1)/(.7*fn[loaded])
    return {"law":law,"slices":geometry["slices"],"native_states":801,
        "end_xy_m":state[-1,:2].tolist(),"path_length_m":float(np.linalg.norm(np.diff(state[:,:2],axis=0),axis=-1).sum()),
        "initial_hub_side_angle_deg":float(hub_angle[0]),"final_speed_m_s":float(speed[-1]),
        "normal_force_mean_n":float(fn.mean()),"normal_force_peak_n":float(fn.max()),
        "support_fraction_above5N":float((fn>5).mean()),
        "normal_force_increment_rms_n":float(np.sqrt(np.mean(np.diff(fn)**2))),
        "native_normal_force_xy_peak_n":float(np.max(np.abs(normal[:,:2]))),
        "loaded_lateral_contact_speed_rms_m_s":float(np.sqrt(np.mean(slip[loaded,1]**2))),
        "native_patch_count_max":int(raw["native_friction_patch_count"].max()),
        "native_aggregate_utilization_peak":float(np.nanmax(utilization)),
        "external_wrench_energy_at_preceding_contact_j":float(power.sum()*.0025),
        "external_force_when_current_load_below1mN_samples":int(((np.linalg.norm(force,axis=-1)>1e-6)&(fn[1:]<=1e-3)).sum()),
        "xy_momentum_balance_residual_rms_n":float(np.sqrt(np.mean(balance**2))),
        "xy_momentum_balance_residual_peak_n":float(np.max(np.abs(balance))),
        "report_sha256":sha(folder/"status.json"),"native_trace_sha256":sha(folder/"native_trace.npz")}, {
        "time":raw["time_s"],"state":state,"normal":fn,"hub_angle":hub_angle,
        "slip":slip,"native":native,"external":force,"balance":balance}


def run(a):
    driver = json.loads((a.input_dir/"status.json").read_text())
    assert driver["status"] == "selected_single_wheel_cases_completed_require_analysis"
    assert driver["geometry_filter"] == "smooth" and driver["expected_cases"] == len(driver["cases"]) == 6
    manifest = json.loads(a.source_manifest.read_text())
    assert sha(a.source_manifest) == driver["source_manifest_sha256"]
    sources = {r["path"]:r for r in manifest["files"]}
    assert all(sha(a.repository/name) == r["sha256"] for name,r in sources.items())
    assert not a.output_dir.exists()
    rows, traces = [], {}
    for c in driver["cases"]:
        assert c["status"] == "completed_verified" and c["returncode"] == 0
        folder = a.input_dir/c["id"]
        assert sha(folder/"status.json") == c["report_sha256"]
        row, trace = analyze_case(folder,source_rows=sources)
        rows.append(dict(id=c["id"],**row));traces[c["id"]] = trace
    a.output_dir.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(16,8),layout="constrained",sharex="row",sharey="row")
    laws = ("native_coulomb","external_coulomb_regularized","combined_slip_magic_formula")
    for j,law in enumerate(laws):
        for n,color in ((1,"#2271b2"),(9,"#d55e00")):
            key = ("single_smooth" if n==1 else "sliced9_smooth")+"__"+law
            tr = traces[key];label=f"{n} cylinder"+("s" if n>1 else "")
            axes[0,j].plot(tr["time"],tr["state"][:,1]*1000,color=color,label=label)
            axes[1,j].plot(tr["time"],tr["normal"],color=color,label=label,alpha=.8)
        axes[0,j].set(title=law,xlabel="Time (s)",ylabel="Lateral displacement Y (mm)")
        axes[1,j].set(xlabel="Time (s)",ylabel="Ground normal force (N)")
        axes[0,j].legend();axes[1,j].grid(alpha=.2)
    fig.suptitle("PhysX single-wheel diagnostic | mu=0.7 | initial side angle=20 deg | matched mass/inertia")
    fig.savefig(a.output_dir/"comparison.png",dpi=150);plt.close(fig)
    value=dict(kind="single_wheel_smooth_geometry_force_diagnostic",status="completed_analysis",
        analyzer_sha256=sha(__file__),input_status_sha256=sha(a.input_dir/"status.json"),
        source_manifest_sha256=sha(a.source_manifest),rows=rows,
        limitations=["Not a TRON1 or car controller result; no policy was trained.",
            "Initial20deg hub side angle is not a prescribed sustained drift.",
            "Native friction is aggregated over patches; pointwise cones are not identified.",
            "External laws use the preceding load and contact point, one physical step late.",
            "External tire force lies in worldXY and uses worldZ load; measured native normal force also has XY components, included in impulse balance.",
            "Uniform axial slicing is rigid compound collision geometry, not tire deformation.",
            "Authored contact/rest offsets remain automatic; effective cooked offsets were not read back, so a purely manifold-only slicing effect is not identified.",
            "Convex-mode setup failures are preserved separately and excluded from these six traces."],
        task_success_claimed=False)
    (a.output_dir/"analysis.json").write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    print(json.dumps(value))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--source-manifest",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--repository",type=Path,default=Path(__file__).resolve().parents[1])
    run(p.parse_args())
