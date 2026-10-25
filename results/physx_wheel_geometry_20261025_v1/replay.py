"""Native metric and tire-wrench replay; no Isaac, actors or physical models."""
import argparse,hashlib,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((root/"manifest.json").read_text())
for name,row in m["files"].items():assert sha(root/name)==row["sha256"],name
for name,digest in m["source_sha256"].items():assert sha(a.repository/name)==digest,name
sys.path.insert(0,str(a.repository/"scripts"))
import analyze_physx_wheel_geometry as audit
sources={v["path"]:v for v in json.loads((root/"evidence/source_snapshot_v5.manifest.json").read_text())["files"]}
summary=json.loads((root/"analysis.json").read_text())
for expected in summary["rows"]:
    actual,_=audit.analyze_case(root/"evidence/smooth_geometry_force_factorial_v5"/expected["id"],source_rows=sources)
    assert dict(id=expected["id"],**actual)==expected,expected["id"]
print(json.dumps(dict(native_cases_recomputed=6,external_wrenches_recomputed=True,xy_impulse_balance_checked=True,new_physics_or_training_run=False,task_success_claimed=False)))
