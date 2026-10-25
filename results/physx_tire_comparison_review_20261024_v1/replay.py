"""Read-only replay of20 measured cases from the published diagnostic archives."""
from pathlib import Path
import encodings.cp437
import gzip
import hashlib
import io
import json
import sys
import tarfile
import zipfile
import numpy as np
import numpy.ma

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
import analyze_ppo_smoothness as smooth
from controller_trace import native_trace

PUBLIC=ROOT/"results/physx_tire_comparison_20261024_v1"
manifest=json.loads((PUBLIC/"artifact_manifest.json").read_text())
allowed={str((PUBLIC/x["path"]).resolve()) for x in manifest["files"]}|{str((PUBLIC/"artifact_manifest.json").resolve())}
proof=Path(__file__).with_name("replay_verification.json")
assert not proof.exists()
reads=set()
def guard(event,args):
    if event.startswith(("socket.","subprocess.")) or event=="os.system":raise RuntimeError("No network/subprocess")
    if event=="import" and str(args[0]).split(".")[0] in ("mujoco","torch","isaaclab","isaacsim"):
        raise RuntimeError("No simulation/inference imports")
    if event=="open" and isinstance(args[0],(str,bytes)):
        path=str(Path(args[0]).resolve());mode=args[1] or ""
        writing=any(c in str(mode) for c in "wax+")
        if writing:
            assert path==str(proof.resolve()),"No artifact writes"
        else:
            assert path in allowed,"Undeclared input: "+path
            reads.add(path)
sys.addaudithook(guard)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for entry in manifest["files"]:
    p=PUBLIC/entry["path"];assert p.stat().st_size==entry["bytes"] and sha(p)==entry["sha256"]
def read(name):return json.loads((PUBLIC/name).read_text())
def arrays(data):
    with np.load(io.BytesIO(data),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def collection(name):
    meta=read("raw/"+name+".manifest.json");archive=PUBLIC/"raw"/(name+".tar.gz")
    assert sha(archive)==meta["archive_sha256"]
    entries={x["path"]:x for x in meta["files"]};payload={}
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes()),mode="r:gz") as tar:
        members=tar.getmembers();assert len(members)==len(entries) and {m.name for m in members}==set(entries)
        for m in members:
            assert m.isfile();b=tar.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==entries[m.name]["sha256"]
            payload[m.name]=b
    return payload
def near(x,y):
    assert np.allclose(x,y,atol=1e-11,rtol=1e-11),"Published numeric result differs"
done=[]
reference=smooth._load_reference(PUBLIC/"reference/tron1_motion.npz")
for name,hz in (("tron1_tire_collection_v1",400),("tron1_tire_800hz_collection_v1",800)):
    payload=collection(name)
    for row in read("timestep/analysis.json")["rows"]:
        if row["physics_hz"]!=hz:continue
        path=next(n for n,b in payload.items() if n.endswith("/tire_study_status.json") and json.loads(b)["law"]==row["law"])
        raw=arrays(payload[path.rsplit("/",1)[0]+"/physics_trajectory.npz"]);trace=native_trace(raw,"policy_isaac")
        result=smooth._tracking(trace["time_s"],trace["position"],trace["com_velocity"],trace["quaternion"],reference,(1.44,4.))
        for key in ("xy_rmse_m","xy_peak_m","speed_retention","beta_rmse_deg"):near(result[key],row["core"][key])
        mask=(trace["time_s"]>=.02-1e-9)&(trace["time_s"]<=6.96+1e-9)
        near((trace["contact_force"][mask,:,2]>5).any(axis=-1).mean(),row["support_fraction"])
        done.append("TRON1/"+row["law"]+"/"+str(hz))
ref=arrays((PUBLIC/"reference/car_source_reference_v1.npz").read_bytes())
for name,analysis,sdk in (("car_tire_collection_v1","car_rigid_axle/analysis.json",False),("vehicle_sdk_collection_v1","car_vehicle/analysis.json",True)):
    payload=collection(name)
    for row in read(analysis)["rows"]:
        case=("vehicle_sdk_mu"+str(row["mu"]).replace(".","p")+"_v1") if sdk else "car_tire_sweep_v1/"+row["case"]
        raw=arrays(payload[case+"/native_trace.npz"]);state=raw["root_state_source_Zup" if sdk else "root_state_w"];t=raw["time_s"]
        core=(t>=ref["core_time_bounds_s_estimated"][0])&(t<=ref["core_time_bounds_s_estimated"][1])
        desired=np.stack([np.interp(t,ref["time_s_estimated"],ref["position_xy_m"][:,k]) for k in range(2)],axis=-1)
        error=np.linalg.norm(state[:,:2]-desired,axis=-1)
        near(np.sqrt(np.mean(error[core]**2)),row["core_XY_RMSE_m"])
        near(np.sqrt(np.mean(error**2)),row["whole_XY_RMSE_m"])
        target=np.interp(t,ref["time_s_estimated"],np.linalg.norm(ref["velocity_xy_m_s"],axis=-1))
        near(np.linalg.norm(state[:,7:9],axis=-1)[core].mean()/target[core].mean(),row["core_speed_ratio"])
        done.append("car/"+case)
payload=collection("single_wheel_bench_collection_v1")
for row in read("single_wheel/analysis.json")["cases"]:
    raw=arrays(payload["single_wheel_fixture_sweep_v2/"+row["id"]+"/native_trace.npz"])
    near(raw["root_state_w"][-1,:2],row["final_xy_m"])
    near(raw["root_state_w"][-1,7:9],row["final_velocity_xy_m_s"])
    done.append("single_wheel/"+row["id"])
assert len(done)==20
result={"status":"published_bytes_and20_numeric_cases_replayed","public_artifact_files_verified":len(manifest["files"])+1,
    "public_manifest_sha256":sha(PUBLIC/"artifact_manifest.json"),"cases_replayed":done,
    "undeclared_reads":0,"network_subprocess_or_physics_or_inference":False,"artifact_mutations":False,
    "numeric_tolerance":{"atol":1e-11,"rtol":1e-11},"manual_movie_review_claimed":False}
proof.write_text(json.dumps(result,indent=2)+chr(10))
print(json.dumps(result))
