"""Publish measured PhysX diagnostics with raw hashes, videos and explicit limits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import tarfile


ROOT=Path(__file__).resolve().parents[1]
COLLECTIONS={
    "single_wheel_bench_collection_v1":("4ef056a6fbfbd124f7037acc714630543f5003a118968997440f9d798cede092","b8c9455012b2904f4f9d45e298ec72dc7ca722da0c2dd5d031b4c4706a444b7e"),
    "tron1_tire_collection_v1":("79ef2e6ceecc094914417fc57bf4246904d54b23c07f46f735ac7242e9c4e885","d959fef1e3f33bf145159a6a7141f567cf33815d0bf9056f510613eb658a8bbb"),
    "tron1_tire_800hz_collection_v1":("ee38010b4a4644f905e316f856ac266cf3307660b0024e6c952a0d6670f6b72f","cda8aa1b5cee4c2e069585af06ef9795abb2237e1c51bac87d1c19423b4dd9ba"),
    "car_tire_collection_v1":("4cf7bfbd5305f8a06426dfc743cecf7c726bda0d01c7edd73d43e29d7ef2df53","967582b2dff8ddbb535c4681e719d892b509d36f527f2de5edc8cc31b2f48fac"),
    "vehicle_sdk_collection_v1":("2bd47b344b05808a55e7f0cfbd876ee68965318838b1ab9c6f613faa294b26a5","0db2d4c2ba992ca10840dfda98c391d19c857aa41d274026ca2a124d7dcb5139")}
SOURCES=("scripts/nonlinear_tire_contact.py","tests/test_nonlinear_tire_contact.py",
    "scripts/probe_physx_tire_contact.py","scripts/probe_physx_tire_contact_fixture.py","scripts/run_physx_tire_bench_sweep.py",
    "scripts/analyze_physx_tire_bench.py","scripts/analyze_tire_reference_scaling.py","scripts/physx_tire_runtime.py",
    "scripts/eval_physx_tire_tron1.py","scripts/analyze_physx_tire_tron1.py","scripts/analyze_physx_tire_timestep.py",
    "scripts/extract_tire_collection.py","scripts/prepare_physx_car_reference.py","scripts/run_physx_car_tire_surrogate.py",
    "scripts/run_physx_car_tire_sweep.py","scripts/analyze_physx_car_tires.py","scripts/run_physx_vehicle_sdk_probe.py",
    "scripts/analyze_physx_vehicle_sdk.py","scripts/package_physx_tire_study.py",
    "config/physx_car_surrogate_v1.json","config/physx_tire_comparison_v1.json","config/physx_tire_comparison_v2.json",
    "config/roller_amp_drift_assessment_v1.json",
    "scripts/analyze_ppo_smoothness.py","scripts/analyze_ppo_backend_400hz_matrix.py","scripts/controller_trace.py",
    "scripts/controller_metrics.py","scripts/analyze_cem_ground_contact_v2.py","scripts/analyze_ppo_contact_stability.py",
    "scripts/analyze_ppo_contact_timestep.py","scripts/analyze_ppo_ground_support_failure.py",
    "scripts/nmpc_tracking_objective.py","scripts/ppo_backend_400hz_contract.py")


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):return json.loads(Path(p).read_text())


def run(a):
    assert not a.output_dir.exists()
    study=a.study_root
    for name,(archive,manifest) in COLLECTIONS.items():
        assert sha(study/(name+".tar.gz"))==archive and sha(study/(name+".manifest.json"))==manifest
        value=read(study/(name+".manifest.json"))
        assert value["archive_sha256"]==archive
        assert all(sha(study/name/x["path"])==x["sha256"] for x in value["files"])
    reference=read(study/"reference_preflight_v2/analysis.json")
    bench=read(study/"single_wheel_analysis_v1/analysis.json")
    robot=read(study/"tron1_analysis_v1/analysis.json")
    clocks=read(study/"tron1_timestep_analysis_v1/analysis.json")
    car=read(study/"car_analysis_v1/analysis.json")
    vehicle=read(study/"vehicle_sdk_analysis_v2/analysis.json")
    assert len(bench["cases"])==6 and len(robot["rows"])==3 and len(clocks["rows"])==6 and len(car["rows"])==6 and len(vehicle["rows"])==2
    assert robot["baseline_instrumentation_matches_previous_raw_recording_exactly"]
    assert car["same_initial_mechanical_state_all_six_verified"] and car["same_control_and_torque_targets_all_six_verified"]
    assert vehicle["same_normalized_source_controls_as_rigid_surrogate_verified"]
    assert sha(a.motion)=="da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254"
    output=a.output_dir;output.mkdir(parents=True)
    def copy(source,name):
        target=output/name;assert source.is_file() and not source.is_symlink() and not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha(source)==sha(target)
    for name in COLLECTIONS:
        for suffix in (".tar.gz",".manifest.json"):copy(study/(name+suffix),"raw/"+name+suffix)
    dirs={"reference_preflight_v2":"reference_preflight","single_wheel_analysis_v1":"single_wheel",
        "tron1_analysis_v1":"tron1","tron1_timestep_analysis_v1":"timestep",
        "car_analysis_v1":"car_rigid_axle","vehicle_sdk_analysis_v2":"car_vehicle"}
    for source,name in dirs.items():
        for path in sorted((study/source).iterdir()):
            assert path.suffix in (".json",".png",".mp4",".npz",".framemd5")
            copy(path,name+"/"+path.name)
    for name in ("study_environment_v1.json","study_diagnostic_index_v1.json"):
        copy(study/"vehicle_sdk_collection_v1"/name,name)
    for name in ("car_source_reference_v1.npz","car_source_reference_v1.json"):copy(study/name,"reference/"+name)
    copy(a.motion,"reference/tron1_motion.npz")
    for name in ("physx_tire_comparison_v2.json","physx_car_surrogate_v1.json","roller_amp_drift_assessment_v1.json"):
        copy(ROOT/"config"/name,"protocol/"+name)
    for name in ("decode_verification_v1.json","vehicle_decode_verification_v1.json","manual_review_v1.json"):
        copy(study/name,name)
    source_ledger={name:sha(ROOT/name) for name in SOURCES}
    with tarfile.open(output/"analysis_source.tar.gz","x:gz") as tar:
        for name in SOURCES:
            b=(ROOT/name).read_bytes();info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o644;info.mtime=0
            tar.addfile(info,io.BytesIO(b))
    columns=("law","physics_hz","core_XY_RMSE_m","core_beta_RMSE_deg","core_speed_ratio","any_wheel_support_fraction")
    with (output/"tron1_metrics.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in clocks["rows"]:
            m=row["core"];writer.writerow(dict(law=row["law"],physics_hz=row["physics_hz"],core_XY_RMSE_m=m["xy_rmse_m"],
                core_beta_RMSE_deg=m["beta_rmse_deg"],core_speed_ratio=m["speed_retention"],any_wheel_support_fraction=row["support_fraction"]))
    status={"kind":"PhysX_tire_model_comparison_first_phase","status":"measured_comparisons_completed_with_model_and_coupling_limits",
        "completed_measured_episodes":{"single_wheel":6,"TRON1_three_laws_two_validation_timesteps":6,"car_rigid_axle_three_laws_two_mu":6,"car_Vehicle_SDK_two_mu":2},
        "original_nominal_TRON1_continuous_support_task_achieved":False,"exact_original_car_plant_reproduced":False,
        "nonlinear_tire_accuracy_identified":False,"human_prior_policy_trained":False,"new_reward_weight_sweep_started":False,
        "critical_findings":["Frozen PPO transfer depends on the tire law and strongly on external-wrench coupling.",
            "Native cylinder instrumentation reproduced the previous PhysX trajectory byte for byte.",
            "Reference retargeting preserves geometric beta while reducing acceleration/gravity demand by50x.",
            "Vehicle suspension removes the rigid-axle contact-loss pattern at mu=.7 but does not recover the source path; mu3.5 includes rollover.",
            "Human priors are a coordination hypothesis, not an established drift improvement or tire-identification method."],
        "source_sha256":source_ledger,"task_success_claimed":False}
    (output/"status.json").write_text(json.dumps(status,indent=2)+chr(10))
    reproduction={"collections":{n:{"archive_sha256":v[0],"manifest_sha256":v[1]} for n,v in COLLECTIONS.items()},
        "instructions":["Use scripts/extract_tire_collection.py with the listed archive/manifest hashes and a fresh directory for each collection.",
            "Run the recorded NumPy-only analyzers using reference/tron1_motion.npz or reference/car_source_reference_v1.npz; no simulator or actor is needed for offline metrics/rendering.",
            "Reproduce rollout execution only in the recorded installed Isaac environment, with fresh GPU audit and the exact source snapshots embedded in the raw collections.",
            "TRON1 evaluation requires the completed M normalized actor identified by SHA; weights are not redistributed. Existing M training protocol and source provide its from-scratch reproduction.",
            "Vehicle SDK must enable scene-query support; sleeping is_on_ground=false is not a stationary unloaded-wheel label."],
        "offline_commands":[
            ["python","scripts/analyze_physx_tire_bench.py","--collection","<extracted_single_wheel>","--output-dir","<fresh_bench_analysis>"],
            ["python","scripts/analyze_physx_tire_tron1.py","--collection","<extracted_tron1_400>","--motion","reference/tron1_motion.npz","--output-dir","<fresh_tron1_analysis>","--render"],
            ["python","scripts/analyze_physx_tire_timestep.py","--coarse","<extracted_tron1_400>","--fine","<extracted_tron1_800>","--motion","reference/tron1_motion.npz","--output-dir","<fresh_dt_analysis>"],
            ["python","scripts/analyze_physx_car_tires.py","--collection","<extracted_car>","--reference","reference/car_source_reference_v1.npz","--output-dir","<fresh_car_analysis>","--render"],
            ["python","scripts/analyze_physx_vehicle_sdk.py","--sdk-collection","<extracted_Vehicle>","--rigid-collection","<extracted_car>","--reference","reference/car_source_reference_v1.npz","--output-dir","<fresh_vehicle_analysis>"]],
        "new_simulation_or_training_performed_by_packager":False,"models_weights_credentials_or_native_binaries_published":False}
    (output/"reproduction.json").write_text(json.dumps(reproduction,indent=2)+chr(10))
    write_html(output,reference,clocks,car,vehicle)
    files=[{"path":str(p.relative_to(output)),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(output.rglob("*")) if p.is_file()]
    manifest={"kind":status["kind"],"files":files,"task_success_claimed":False,"contains_models_weights_credentials_native_binaries":False}
    (output/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+chr(10))
    print(json.dumps({"output":str(output),"files_including_manifest":len(files)+1,"artifact_manifest_sha256":sha(output/"artifact_manifest.json")}))


def write_html(output,reference,clocks,car,vehicle):
    esc=html.escape
    tr="".join(f"<tr><td>{esc(r['law'])}</td><td>{r['physics_hz']}</td><td>{r['core']['xy_rmse_m']:.3f}</td><td>{r['core']['beta_rmse_deg']:.2f}</td><td>{100*r['core']['speed_retention']:.1f}%</td><td>{100*r['support_fraction']:.1f}%</td></tr>" for r in clocks["rows"])
    cv="".join(f"<tr><td>Vehicle SDK</td><td>{r['mu']}</td><td>{r['core_XY_RMSE_m']:.2f}</td><td>{r['core_beta_RMSE_deg']:.2f}</td><td>{100*r['core_speed_ratio']:.1f}%</td><td>{100*r['core_any_wheel_suspension_Fup_gt5N_fraction']:.1f}%</td><td>{r['core_maximum_body_tilt_deg']:.1f}</td></tr>" for r in vehicle["rows"])
    cr="".join(f"<tr><td>{esc(r['law'])}</td><td>{r['mu']}</td><td>{r['core_XY_RMSE_m']:.2f}</td><td>{r['core_beta_RMSE_deg']:.2f}</td><td>{100*r['core_speed_ratio']:.1f}%</td><td>{100*r['core_any_wheel_support_fraction']:.1f}%</td><td>{r['core_maximum_body_tilt_deg']:.1f}</td></tr>" for r in car["rows"])
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>PhysX tire contact comparison</title>
<style>body{{font:17px/1.55 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#18212c}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border:1px solid #ccd2d8;text-align:left}}video,img{{width:100%;height:auto}}.note{{padding:16px;background:#fff1d9}}code{{font-size:14px}}</style>
<h1>PhysX wheel contact and nonlinear tire laws</h1><p>Measured first-phase comparison, 24 October 2026. Twenty diagnostic episodes completed: six single-wheel tests, six frozen-PPO TRON1 episodes, six rigid-axle car episodes and two native Vehicle SDK episodes. This does not establish task success or real-tire accuracy.</p>
<p class="note">Main result: changing the tire curve alone did not improve this frozen policy. External force coupling is a substantial confound, and the car source plant is not identified. Human-motion priors remain a testable coordination hypothesis.</p>
<h2>1. TRON1: same frozen PPO, geometry and reference</h2><p>The three400Hz arms share exactly the same initial mechanical state. Instrumenting native friction reproduced the previous PhysX trajectory byte for byte. The external arms disable native wheel tangential friction, retain native normal contact, and apply force plus COM moment using the preceding physical sample's load. Other body contacts retain their original material.</p>
<table><tr><th>Law</th><th>Physics Hz</th><th>Core XY RMSE m</th><th>Beta RMSE deg</th><th>Core speed/reference</th><th>Any-wheel support</th></tr>{tr}</table>
<p>Core:1.44–4s. Support:0.02–6.96s, max(FzL,FzR)&gt;5N, with single-wheel support allowed. The actor remains50Hz and was trained at400Hz. The800Hz rows are evaluation-only numerical refinement. Two resolutions do not prove convergence. No arm meets the original continuous-support requirement.</p>
<video controls preload="metadata" poster="tron1/top_view.png" src="tron1/top_view.mp4"></video><p>Fixed world XY bounds for every frame and panel. Blue: body; orange: velocity; opaque red/purple: wheel planes. Green dots indicate support at the display sample; all eight physical samples remain in the raw data. Reference contacts are unobserved.</p>
<img src="tron1/comparison.png" alt="TRON1 trajectories beta speed and force ratios">
<p>The native-to-external Coulomb difference combines regularization, load delay and force application. External Coulomb-to-Magic-Formula is the cleaner curve-shape comparison within that same coupling. Current-load and lagged-load force ratios are stored separately. Aggregate utilization is not pointwise friction-cone identification.</p>
<h2>2. Original car reference and car-system diagnostics</h2><p>The reference is a modified CARLA0.9.5 simulated trajectory, not a real-car measurement. Original timestamps, individual wheel orientations/RPM and loads are absent. The clock is reconstructed. Four-wheel mass, dimensions, steering and torque assumptions are explicit in the protocol. Identical normalized source controls and torque targets were verified across the six rigid-axle arms.</p>
<p>Original core speed:{reference['source_core_speed_range_m_s'][0]:.1f}–{reference['source_core_speed_range_m_s'][1]:.1f}m/s; original body beta:{reference['source_core_beta_range_deg'][0]:.1f}–{reference['source_core_beta_range_deg'][1]:.1f}deg. These source values do not identify tires or capture friction.</p>
<table><tr><th>Car model/law</th><th>Mu</th><th>Core XY RMSE m</th><th>Beta RMSE deg</th><th>Speed/reference</th><th>Any-wheel support</th><th>Max tilt deg</th></tr>{cr}{cv}</table>
<p class="note">Rigid-axle outcomes include contact losses and rollover. Vehicle SDK adds suspension/road queries and its default tire model, and uses CPU PhysX rather than the GPU articulation path. It is a component-model diagnostic, not a curve-only ablation or exact CARLA reproduction. Atmu0.7 it maintains support but misses the source turn. Atmu3.5 it rolls over, so lower XY/beta error is not task success.</p>
<video controls preload="metadata" poster="car_vehicle/top_view_mu0p7.png" src="car_vehicle/top_view_mu0p7.mp4"></video>
<video controls preload="metadata" poster="car_vehicle/top_view_mu3p5.png" src="car_vehicle/top_view_mu3p5.mp4"></video>
<p>Panels: original source; rigid cylinder/Coulomb; rigid cylinder/external Magic Formula; native Vehicle suspension/tire model. Source wheel orientations are not invented. Vehicle wheel centers use measured root, steering and suspension jounce. All panels and bothmu conditions share fixed XY bounds.</p>
<p>The Vehicle SDK default tire is not Pacejka. Its filtered tire load is not exposed by the installed wheel-state binding. Tire force divided by raw suspension-up force is a diagnostic proxy, not actual friction-cone utilization.</p>
<h2>3. Retarget scaling and human-motion priors</h2><img src="reference_preflight/scaling_and_tire_laws.png" alt="Source and retarget scaling audit">
<p>Space0.08 and time2 preserve geometric beta but scale velocity by0.04 and acceleration by0.02. The source core's estimated lateral acceleration/g median is2.24, versus0.0447 after uniform scaling. Kinematic intent therefore does not preserve the friction budget or physical leaning requirement. These are estimated-clock kinematic derivatives, not measured tire forces.</p>
<p><a href="https://arxiv.org/html/2607.10815v1">Learning Roller-Skating Motions of Humanoid Robots Based on Adversarial Motion Priors</a> uses a passive-wheel humanoid, nine cylinder slices and GMR-derived AMP priors. Its reported forward tests do not establish high-speed large-beta turning. The wheel geometry lesson is relevant; cylinder slicing does not identify a nonlinear tire law.</p>
<p>Our assessment: use human data as a posture/load-transfer/support-timing prior only after validating contact and acceleration compatibility. TRON1's wheel motors remain active; wheel spin should not be an imitation target. Test task-only PPO, phase-aligned posture imitation and AMP under identical task rewards, physics, budgets, DR and reset distributions. No AMP policy was trained here, so an improvement has not been demonstrated. Straight gliding is not relabeled as drift data.</p>
<p><a href="protocol/roller_amp_drift_assessment_v1.json">Full assessment and fair-ablation specification</a>. The inspected project did not provide a located code/motion release; its website prohibits media redistribution. No website media was copied.</p>
<h2>Evidence and reproduction</h2><p><a href="status.json">Study status</a> · <a href="tron1_metrics.csv">TRON1 metrics CSV</a> · <a href="reproduction.json">Reproduction commands and raw collection hashes</a> · <a href="artifact_manifest.json">Artifact manifest</a> · <a href="study_environment_v1.json">Environment</a> · <a href="study_diagnostic_index_v1.json">Preserved harness failures</a></p>
<p>Primary sources: <a href="https://arxiv.org/pdf/2001.01377">Source drifting study</a>; <a href="https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/Vehicles.html">PhysX Vehicle model architecture</a>; <a href="https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html">Installed-generation wheel telemetry API</a>. Open source curve shape is pinned through the existing <code>upstream/xcar-rlgpu</code> submodule. Raw-source provenance remains in the existing <code>upstream/drift_drl</code> submodule.</p></html>'''
    (output/"report.html").write_text(page)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for n in ("study-root","motion","output-dir"):p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
