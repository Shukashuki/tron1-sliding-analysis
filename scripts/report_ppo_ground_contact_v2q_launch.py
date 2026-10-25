"""Publish a dated native-smoke/launch observation; never claim trained success."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import html
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(data):return hashlib.sha256(data).hexdigest()


def run(args):
    source=args.input_root.resolve();out=args.output_dir.resolve()
    archive=source/'launch_observation_v1.tar.gz'
    assert sha(archive.read_bytes())==args.collection_sha256
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members})==len(members)
        data={m.name:tar.extractfile(m).read() for m in members}
    read=lambda n:json.loads(data[n])
    observation_manifest=read('observation_manifest.json')
    assert observation_manifest['models_weights_credentials_binaries_included'] is False
    for name,row in observation_manifest['files'].items():
        assert sha(data[name])==row['sha256'] and len(data[name])==row['bytes']
    assert observation_manifest['observer_source_sha256']==sha((ROOT/'scripts/observe_ppo_ground_contact_v2q.py').read_bytes())
    executed=read('source_snapshot_v1.manifest.json')
    assert sha((source/'source_snapshot_v1.tar.gz').read_bytes())==executed['archive_sha256']
    assert executed==json.loads((source/'source_snapshot_v1.manifest.json').read_text())
    for row in executed['files']:assert sha((ROOT/row['path']).read_bytes())==row['sha256']
    smoke=read('smoke_v1_strict_verification.json');runtime=smoke['contact_runtime_audit']
    assert smoke['status']=='completed_verified' and smoke['transitions']==1536
    assert smoke['initial_policy_state_sha256']=='7dcdf78ffeb61806eeefcb31c4739d707c1caef934d152f2296ba2ee488b4d5d'
    startup=read('formal_v1/ground_contact_v2q_startup_audit.json')['runtime']
    import ground_contact_reward_v2q as definition
    definition.verify_geometry_runtime(runtime,expected_envs=32,expected_windows=56)
    definition.verify_geometry_runtime(startup,expected_envs=512,expected_windows=8)
    assert runtime['geometry_physics_samples']==448
    assert runtime['positive_geometry_cost_environment_windows']>0
    observed=read('process_observation.json')
    assert observed['task_success_claimed'] is False
    assert observed['training_driver_status']=='running_formal'
    assert observed['nominal_driver_status']=='waiting_for_exact_training_driver'
    assert all(v and v['state'] in ('R','S','D') for v in observed['processes'].values())
    tests=(source/'source_snapshot_v1_tests.log').read_text()
    assert '151 passed, 56 subtests passed' in tests
    out.mkdir(exist_ok=False,parents=True)
    files={}
    def save(name,payload):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        files[name]=dict(sha256=sha(payload),bytes=len(payload))
    for name,payload in data.items():save('evidence/'+name,payload)
    for name in ('source_snapshot_v1.tar.gz','source_snapshot_v1.manifest.json','source_snapshot_v1_tests.log','native_K_geometry_signal_check.json'):
        save(name,(source/name).read_bytes())
    stamp=datetime.datetime.fromtimestamp(observed['observed_unix_time'],datetime.timezone.utc).isoformat()
    iteration=observed['last_logged_iteration']
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Q unsupported-gap reward: native smoke and launch</title>
<style>body{{font:17px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 20px}}pre{{background:#eff3f6;padding:16px;white-space:pre-wrap}}td,th{{border-bottom:1px solid #ccc;padding:8px;text-align:left}}</style>
<h1>Q contact reward: add a geometric signal while wheel load is absent</h1>
<p>This is a dated launch observation at {html.escape(stamp)}, not a completed training or nominal result. PhysX training was confirmed live at iteration {iteration[0]}/{iteration[1]}; the nominal driver was waiting for that exact process. No learned support improvement or task success is claimed.</p>
<h2>Only change from the tracking-valid K reward</h2><pre>missing_i = max(Fz_left_i, Fz_right_i) &lt;= 5 N
gap_i = max(min(authored_gap_left_i, authored_gap_right_i), 0)
new_cost = max_over_8_actual_substeps(missing_i * (1 - exp(-gap_i / 0.02 m)))
raw_Q = raw_K - new_cost
grounded_reward = 0.4 * raw_Q   # unchanged policy-step dt factor follows</pre>
<p>The 20 mm exponential length shapes the reward; it is not a contact tolerance. One wheel exceeding5 N disables the new cost at that physical sample. The other wheel may lift or alternate. No absolute torso-height target, bilateral requirement, zero-slip target, new impact cost or model change is introduced. The geometric input is the original finite cylinder's distance to the flat floor, not the native manifold's separation or point count.</p>
<p>In the completed K diagnosis,288 of291 below5N PhysX samples had positive minimum authored wheel gap. Batched Torch float32 geometry agrees with the independent saved CPU geometry to2.50e-8 m over all2792 captured native K states. This checks the input signal; it does not predict learning outcomes. <a href="native_K_geometry_signal_check.json">Signal check</a> · <a href="../native_contact_manifold_20261025_v1/report.html">Native manifold diagnosis</a></p>
<h2>Executed checks</h2><table><tr><th>Check</th><th>Evidence</th></tr>
<tr><td>CPU arithmetic, lifecycle, trainer and provenance</td><td>151 tests +56 subtests passed</td></tr>
<tr><td>Native smoke</td><td>32 worlds;1536 training transitions;56 reward windows;448 paired native force/geometry samples per world</td></tr>
<tr><td>Smoke reward signal observed</td><td>{runtime['positive_geometry_cost_environment_windows']} positive-cost world/windows; mean raw new cost {runtime['mean_unsupported_geometry_gap_cost']:.6f}</td></tr>
<tr><td>Formal startup</td><td>All512 worlds' original wheel colliders and local frames checked;8 setup reward windows;64 paired physical samples</td></tr>
<tr><td>Startup independent CPU/Torch parity</td><td>{startup['authored_geometry_audit']['startup_CPU_Torch_max_gap_difference_m']:.3g} m maximum error</td></tr>
<tr><td>Initialization</td><td>Same seed42 from-scratch network as K; no checkpoint resume</td></tr>
<tr><td>GPU selection</td><td>GPU3; fresh complete compute/graphics process, memory and PCI-order audit before smoke and formal; one GPU job at a time</td></tr></table>
<h2>Fixed recipe and pending evaluation</h2><p>512 environments ×24 steps ×1000 iterations =12,288,000 transitions. Original14 other reward terms, beta+2/std0.12, action-rate−0.5,400Hz physics/50Hz policy, actuator limits and gains, original rigid cylinders, original friction/encoder/COM/push DR and reference remain unchanged. The snapshot contains31 code/config/test files (no assets, neural weights, native binaries or credentials).</p>
<p>After source-bound completed training, the same normalized actor is evaluated sequentially at nominalμ=0.7 in native CPU MuJoCo and PhysX. PhysX waits for a freshly confirmed fully idle GPU. Both native400Hz recordings and fixed-relative-camera full-body replay are required. Original XY, speed, intended-beta and HF gates remain; at least one wheel must exceed5N at every accepted native sample. Friction-range qualification and model-specific nonlinear/sliced-cylinder comparisons follow nominal qualification. This launch does not establish that any of those criteria have passed.</p>
<p><a href="evidence/smoke_v1_strict_verification.json">Native smoke proof</a> · <a href="evidence/formal_v1/ground_contact_v2q_startup_audit.json">All-world formal startup</a> · <a href="evidence/process_observation.json">Exact process observation</a> · <a href="source_snapshot_v1.manifest.json">Executed source manifest</a> · <a href="../ground_contact_v2p_ppo_nominal_20261025_v1/report.html">Previous impact-cost failure</a></p></html>'''
    save('report.html',text.encode())
    sources={row['path']:row['sha256'] for row in executed['files']}
    for name in ('scripts/observe_ppo_ground_contact_v2q.py','scripts/report_ppo_ground_contact_v2q_launch.py'):
        sources[name]=sha((ROOT/name).read_bytes())
    save('manifest.json',(json.dumps(dict(kind='Q_native_smoke_and_formal_launch',files=files,
        source_sha256=sources,task_success_claimed=False,observed_unix_time=observed['observed_unix_time'],
        source_collection_sha256=args.collection_sha256),indent=2)+'\n').encode())
    print(json.dumps(dict(output=str(out),status='dated_launch_observation_published',task_success_claimed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--collection-sha256',required=True);run(p.parse_args())
