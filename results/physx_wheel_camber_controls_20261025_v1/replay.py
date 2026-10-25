import argparse,hashlib,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True);a=p.parse_args();r=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((r/"manifest.json").read_text())
for n,v in m["files"].items():assert sha(r/n)==v["sha256"],n
for n,v in m["source_sha256"].items():assert sha(a.repository/n)==v,n
sys.path.insert(0,str(a.repository/"scripts"));from analyze_physx_wheel_camber_controls import recompute
rows,_=recompute(r/"evidence");assert rows==json.loads((r/"analysis.json").read_text())["rows"]
print(json.dumps(dict(eight_measurements_exact=True,native_impulse_balance_verified=True,physics_or_policy_execution=False,task_success_claimed=False)))
