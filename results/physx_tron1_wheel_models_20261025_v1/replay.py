import argparse,hashlib,json,pathlib,sys
p=argparse.ArgumentParser();p.add_argument("--repository",type=pathlib.Path,required=True);a=p.parse_args();r=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((r/"manifest.json").read_text())
for n,v in m["files"].items():assert sha(r/n)==v["sha256"],n
for n,v in m["source_sha256"].items():assert sha(a.repository/n)==v,n
sys.path.insert(0,str(a.repository/"scripts"));import analyze_physx_tron1_wheel_models as audit
summary=json.loads((r/"analysis.json").read_text());protocol=json.loads((r/"protocol.json").read_text())
reference=audit.base.metrics.reference_data(r/"reference.npz")
rows,_,_,_,_=audit.load_cases(r/"evidence",reference,protocol,original_metrics=summary["original_baseline_metrics"])
assert rows==summary["rows"]
print(json.dumps(dict(six_native_metric_sets_exact=True,all_external_wrenches_exact=True,frozen_nominal_screens_exact=True,physics_or_policy_execution=False,task_success_claimed=False)))
