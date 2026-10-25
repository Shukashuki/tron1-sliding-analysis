"""Write the verified independent-camber reference synthesis report."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

import generate_camber_reference as reference


def run(root):
    root=Path(root);target=root/"report.html";assert not target.exists()
    read=lambda p:json.loads(p.read_text())
    d=read(root/"independent/generation.json");v=read(root/"verification_v2.json");review=read(root/"still_review_v1.json")
    assert v["status"]=="verified_kinematic_reference_artifacts" and v["frames_decoded"]==349
    assert v["source_sha256"]==reference.sha(Path(__file__).parent/"verify_camber_reference.py")
    assert v["generation_sha256"]==reference.sha(root/"independent/generation.json")
    assert review["inspected_frame_indices"]==[72,112,150,200,348]
    for n,h in review["frame_sha256"].items():assert reference.sha(root/"render"/n)==h
    tests=read(root/"tests_v1.json")
    assert tests["exit_code"]==0 and "17 passed" in tests["actual_stdout"]
    assert tests["test_source_sha256"]==reference.sha(Path(__file__).parents[1]/"tests/test_camber_reference.py")
    fits={r["id"]:r for r in d["fits"]};table=[]
    for row in d["rows"]:
        wheels=row["metrics"]["wheels"]
        med=[w["core_touching_signed_lateral_velocity_m_s_median"] for w in wheels]
        f=fits.get(row["id"])
        error="—" if f is None else " / ".join(f"{x:.5f}" for x in f["core_target_RMSE_m_s_L_R"])
        cols=[row["title"],f"{med[0]:+.4f} / {med[1]:+.4f}",error,
              f'{max(w["whole_max_abs_camber_deg"] for w in wheels):.2f}',
              f'{row["metrics"]["max_wheel_center_XY_displacement_from_old_reference_m"]*100:.2f}']
        table.append("<tr>"+"".join("<td>"+html.escape(c)+"</td>" for c in cols)+"</tr>")
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Independent wheel-camber reference synthesis</title>
<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}video,img{max-width:100%}pre{white-space:pre-wrap;background:#f3f6f8;padding:14px}.status{background:#fff1d2;padding:14px}</style>
<h1>Independent left/right camber as a kinematic trajectory generator</h1>
<p class="status">Two separate camber curves can shape each wheel's geometric surface lateral velocity while retaining the original car-mapped body path, speed and heading. This is a reference-generation result. Stable loaded contact and successful dynamic turning remain unproven; no reward or policy was trained or changed in this stage.</p>
<video controls playsinline preload="metadata" poster="render/comparison.png" src="render/camber_reference_4x1.mp4"></video>
<p>Four actual generated references, original opaque meshes, 349 frames at 50 Hz. A common fixed-distance camera follows translation only, with fixed world angles and no zoom. The lower maps share the same world scale and show both wheel paths and plane headings. All four body paths coincide by construction.</p>
<table><tr><th>Reference</th><th>Core signed surface lateral velocity median L/R (m/s)</th><th>Target RMSE L/R (m/s)</th><th>Peak absolute camber (deg)</th><th>Maximum wheel-center XY shift from old reference (cm)</th></tr>TABLE</table>
<p>Core: 1.44–4.00 s. The 90%/90% target reduces the magnitude of each original signed lateral velocity by10%;the 110%/90% target increases the left magnitude by10% while reducing the right by10%. These ratios are illustrative prospective reference commands,not measured improvements in actual tire grip or closed-loop task performance. Medians summarize time series;the optimizer fits every core sample and reports separate per-wheel errors.</p>
<img src="render/camber_and_slip.png" alt="Independent signed camber curves and corresponding geometric surface lateral velocities">
<h2>Model and reference construction</h2>
<pre>gamma_i = asin(n_ground dot axle_localY_world_i)
gamma_i(t) = C2_envelope(t) * sum_k Bernstein_k(t) * c_ik
i in {left,right};each wheel has four independent coefficients c_ik in[-15,+15]deg

v_surface_i = J_material_point_i(q) * qdot
v_lateral_i = tangent_lateral_i dot v_surface_i

minimize over8 independent coefficients:
  sum_core,i (v_lateral_i - target_i)^2 + sum_i,k (0.001*c_ik/15)^2</pre>
<p>Gamma0 means a vertical wheel plane:its angle to the ground is90deg. Positive/negative gamma uses the authored local+Y axle convention for both wheels. Thus left+10/right−10 is an opposed tilt. These are independent camber directions,not newly introduced wheel-steering joints.</p>
<p>Abduction controls gamma. An analytic two-link hip/knee reconstruction preserves each wheel's original body-longitudinal coordinate and places each ideal cylindrical support on the floor. Both generated geometric support gaps are zero to numerical precision. Lateral wheel-center positions are allowed to change and are explicitly exported;old static wheel-XY targets must be replaced when learning from these references. This reconstruction applies to the car task and is not a replacement for human-motion GMR retargeting.</p>
<p>The original body XY, yaw, speed and nominal reference Z are retained. Retaining nominal reference Z introduces no new absolute-height policy reward or task gate. Wheel-spin rates come from the longitudinal rolling Jacobian,then trapezoidal phase integration and pose differentiation. The finite50Hz discretization leaves up to0.00495m/s longitudinal material-point speed in these four references;exact longitudinal rolling is not claimed.</p>
<h2>What changes the sliding term</h2>
<p>A fixed nonzero camber does not change the core lateral velocity in this yaw-only reference. Changing gamma over time moves the legs and wheel centers,which changes the material-point Jacobian velocity. The fitted effect is therefore a finite-duration whole-body kinematic modulation,not a demonstrated steady-state camber-thrust law. The largest fitted camber is about7.10deg;the15deg bounds apply to the Bernstein coefficients.</p>
<p>The initial abduction-only forward model also tested parallel tilts:at+10deg the opposite wheel developed a1.756cm geometric gap. A restricted common-positive-angle fit used only the right geometric support and left the other wheel lifted. Those preliminary results remain in <a href="generated/generation.json">the forward-only diagnostic</a> and <a href="shooting/fit.json">the restricted fit</a>;the four-panel result above uses the later independent reconstruction with both geometric supports.</p>
<h2>Contact discontinuities and physical limits</h2>
<p>At zero camber the bottom of an ideal rigid cylinder is a support line. At nonzero camber the lowest support is on a rim. The generator preserves this set-valued branch change;it does not smooth it into a rubber tread. In the differential fit,the branch changes near2.94s and5.14s. Although the commanded camber envelope is smooth,the geometric rim/line transition can introduce a cusp in reconstructed hip/knee motion. At50Hz the largest normal material-point speed in that fit is0.00228m/s. These are kinematic sampling diagnostics,not measured PhysX contact jumps or loaded-contact guarantees.</p>
<p>No friction coefficient enters the kinematic fit. Geometry alone does not supply normal loads,tire-force saturation,load transfer or motor torques. The current external combined-slip tire model explicitly omits camber thrust;adding a fitted camber-force term would be a separate model change requiring calibration. PhysX's vehicle tire model exposes camber-force parameters,but those are not automatically part of an ordinary rigid cylinder contact:see <a href="https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Vehicles.html">the official vehicle model documentation</a>.</p>
<h2>Verification and reproduction</h2>
<p>Seventeen CPU tests pass. Generation checks the authored joint limits. Independent checks verify all349 frames per case,source/model/config hashes,body-longitudinal wheel coordinates,both ideal support heights,base path/speed/orientation,pose-derived velocities and the COM-plus-angular-cross-offset material-point velocity. All349 video frames decode,and common fixed relative camera translation is checked numerically. Selected stills at1.44,2.24,3.00,4.00 and6.96s and the camber/slip plot were inspected;full manual video playback is not claimed.</p>
<pre>python scripts/independent_camber_reference.py \
  --reference results/contact_wheel_axis_20261025_v1/reference.npz \
  --model outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml \
  --config config/independent_camber_kinematic_reference_v1.json \
  --output-dir /tmp/tron1-independent-camber-reproduction

python scripts/render_camber_reference.py \
  --references /tmp/tron1-independent-camber-reproduction \
  --model outputs/car_failure_sideslip_20261015/raw/mujoco_model.xml \
  --output-dir /tmp/tron1-independent-camber-render</pre>
<p>Use an environment with the repository requirements and a working OSMesa renderer. The executed environment used MuJoCo3.8.1 and SciPy1.17.1. Robot XML/meshes are caller supplied at the declared hash;no physical asset,model weight or credential is added here. Existing reference provenance and acquisition live in the repository's earlier car experiment. Fresh output directories are required to preserve earlier evidence.</p>
<p><a href="independent/generation.json">All independent references and optimizer results</a> · <a href="verification_v2.json">Current FK and artifact verification</a> · <a href="still_review_v1.json">Still review</a> · <a href="tests_v1.json">CPU test record</a> · <a href="../../scripts/independent_camber_reference.py">Generator</a> · <a href="../../config/independent_camber_kinematic_reference_v1.json">Configuration</a> · <a href="../contact_wheel_axis_20261025_v1/report.html">Completed preceding3D-orientation reward ablation</a>.</p>
<p>The next stage is a separately labeled native dynamic test using these new wheel/body references,followed by contact/slip reward learning if necessary. Keep mu0.7 first,original actuator limits,high-speed/nonzero-beta requirements and all original nominal gates. Geometric touch must never substitute for measured upward floor-normal force.</p></html>'''
    target.write_text(text.replace("TABLE","".join(table)))
    print(str(target))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--result",type=Path,required=True);run(p.parse_args().result)
